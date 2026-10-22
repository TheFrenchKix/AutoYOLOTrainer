"""AutoLabeler: predict, write YOLO labels and triage images by confidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.config.schema import LabelingConfig
from src.datasets.yolo_format import write_labels
from src.labeler.router import ConfidenceRouter, Decision
from src.model.types import Detection, Detector
from src.utils.files import list_images, move_with_sidecars, read_json, write_json
from src.utils.images import read_image
from src.utils.logging import get_logger

log = get_logger("dataset")


@dataclass
class LabelResult:
    image: Path
    decision: Decision
    score: float | None
    detections: list[Detection]


class AutoLabeler:
    """Turn raw captures into labeled images sorted in accepted/review/rejected.

    For every image:

    1. predict (or reuse the predictions saved at capture time),
    2. write ``image.txt`` in YOLO format,
    3. update the ``image.json`` metadata with the decision,
    4. move the image and its sidecars to the destination folder.
    """

    def __init__(
        self,
        detector: Detector | None,
        config: LabelingConfig,
        destinations: Mapping[str, Path],
        model_version: str | None = None,
    ) -> None:
        if detector is None and not config.use_saved_predictions:
            raise ValueError("Un détecteur est requis si use_saved_predictions=false")
        self.detector = detector
        self.config = config
        self.router = ConfidenceRouter(config)
        self.destinations = {Decision(k): Path(v) for k, v in destinations.items()}
        self.model_version = model_version

    def _detections(self, image_path: Path) -> list[Detection]:
        metadata = read_json(image_path.with_suffix(".json"), default=None)
        if self.config.use_saved_predictions and metadata and "detections" in metadata:
            return [Detection.from_dict(d) for d in metadata["detections"]]
        if self.detector is None:
            raise RuntimeError(f"Pas de prédictions sauvegardées pour {image_path.name}")
        image = read_image(image_path)
        if image is None:
            raise OSError(f"Image illisible: {image_path}")
        return [d for d in self.detector.predict(image) if d.confidence >= self.config.predict_confidence]

    def label_image(self, image_path: Path) -> LabelResult:
        detections = self._detections(image_path)
        confidences = [d.confidence for d in detections]
        decision = self.router.decide(confidences)
        score = self.router.score(confidences)

        write_labels(image_path.with_suffix(".txt"), [d.to_yolo() for d in detections])
        meta_path = image_path.with_suffix(".json")
        metadata = read_json(meta_path, default={}) or {}
        metadata.update({
            "detections": [d.to_dict() for d in detections],
            "labeling": {
                "decision": decision.value,
                "score": score,
                "labeled_at": datetime.now().isoformat(timespec="seconds"),
                "model_version": self.model_version,
            },
        })
        write_json(meta_path, metadata)

        destination = move_with_sidecars(image_path, self.destinations[decision])
        return LabelResult(destination, decision, score, detections)

    def process_folder(self, folder: Path, limit: int | None = None) -> Counter[str]:
        """Label every image of ``folder``; unreadable images are moved to rejected."""
        images = list_images(folder)[:limit]
        stats: Counter[str] = Counter()
        for image in tqdm(images, desc="Auto-labeling", unit="img", disable=len(images) < 2):
            try:
                result = self.label_image(image)
                stats[result.decision.value] += 1
            except Exception as exc:  # one bad file must not stop a weeks-long run
                log.error(f"Échec auto-labeling {image.name}: {exc}")
                if image.exists():
                    move_with_sidecars(image, self.destinations[Decision.REJECTED])
                stats["errors"] += 1
        if images:
            log.info(f"Auto-labeling de {len(images)} images: {dict(stats)}")
        return stats
