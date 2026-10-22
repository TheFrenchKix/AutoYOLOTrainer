"""CaptureManager: decide whether a frame with detections is worth saving, then save it.

Saved files (same stem ``YYYYMMDD_HHMMSS_ffffff``)::

    captures/20261022_143015_123456.png    the frame
    captures/20261022_143015_123456.txt    YOLO labels predicted by the model
    captures/20261022_143015_123456.json   date, time, classes, confidences, bboxes, AL score

Every saved capture is also appended to ``<state>/captures_index.csv``.
"""

from __future__ import annotations

import csv
import random
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from src.capture.active_learning import ActiveLearningSelector, InterestScore
from src.capture.rate_limiter import RateLimiter
from src.config.schema import ActiveLearningConfig, CaptureConfig
from src.datasets.yolo_format import write_labels
from src.model.types import Detection
from src.utils.files import write_json
from src.utils.hashing import HashIndex, phash
from src.utils.images import write_image
from src.utils.logging import get_logger

log = get_logger("capture")

INDEX_COLUMNS = ["file", "timestamp", "n_detections", "classes", "min_confidence", "max_confidence", "score"]
_SSIM_SIZE = (160, 90)


class CaptureManager:
    """Filter (rate limit, interest, duplicates) and persist detection frames."""

    def __init__(
        self,
        config: CaptureConfig,
        al_config: ActiveLearningConfig,
        captures_dir: Path,
        state_dir: Path,
        selector: ActiveLearningSelector | None = None,
        model_version: str | None = None,
        clock: Callable[[], float] = time.time,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.al_config = al_config
        self.captures_dir = captures_dir
        self.index_file = state_dir / "captures_index.csv"
        self.selector = selector or ActiveLearningSelector(al_config)
        self.model_version = model_version
        self.clock = clock
        self.rng = rng or random.Random()
        self.limiter = RateLimiter(config.max_images_per_hour, config.min_interval_seconds, clock)
        dedup = al_config.dedup
        self.hashes = HashIndex(
            dedup.max_hamming_distance, dedup.memory_size, dedup.hash_size, storage=state_dir / "capture_hashes.json"
        ) if dedup.enabled else None
        self._last_thumb: np.ndarray | None = None
        self._lock = threading.Lock()
        self.stats: Counter[str] = Counter()
        captures_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- filters
    def _is_interesting(self, interest: InterestScore) -> bool:
        if self.config.save_every_detection or not self.al_config.enabled:
            return True
        if interest.score >= self.al_config.score_threshold:
            return True
        return self.rng.random() < self.al_config.exploration_rate

    def _is_near_identical(self, frame: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """SSIM against the last saved frame (catches static scenes pHash may miss)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        thumb = cv2.resize(gray, _SSIM_SIZE, interpolation=cv2.INTER_AREA)
        if self._last_thumb is None:
            return False, thumb
        similarity = structural_similarity(self._last_thumb, thumb, data_range=255)
        return similarity >= self.al_config.dedup.ssim_threshold, thumb

    # -------------------------------------------------------------------- main
    def on_detection(self, frame: np.ndarray, detections: Sequence[Detection]) -> Path | None:
        """Entry point called by the application after each inference.

        Returns:
            The saved image path, or ``None`` if the frame was filtered out.
        """
        with self._lock:
            self.stats["frames"] += 1
            detections = [d for d in detections if d.confidence >= self.config.min_confidence]
            if not detections:
                return None
            self.stats["frames_with_detections"] += 1

            if not self.limiter.allow():
                self.stats["skipped_rate_limit"] += 1
                return None

            interest = self.selector.score(detections)
            if not self._is_interesting(interest):
                self.stats["skipped_low_interest"] += 1
                return None

            thumb = None
            if self.hashes is not None:
                duplicate, thumb = self._is_near_identical(frame)
                frame_hash = phash(frame, self.al_config.dedup.hash_size)
                if duplicate or self.hashes.is_duplicate(frame_hash):
                    self.stats["skipped_duplicate"] += 1
                    return None
                self.hashes.add(frame_hash)

            path = self._save(frame, detections, interest)
            self.limiter.record()
            self.selector.update(detections)
            self._last_thumb = thumb
            self.stats["saved"] += 1
            if self.hashes is not None and self.stats["saved"] % 50 == 0:
                self.hashes.save()
            return path

    def _unique_path(self, now: datetime) -> Path:
        stem = now.strftime("%Y%m%d_%H%M%S_%f")
        path = self.captures_dir / f"{stem}.{self.config.image_format}"
        counter = 1
        while path.exists():
            path = self.captures_dir / f"{stem}_{counter}.{self.config.image_format}"
            counter += 1
        return path

    def _save(self, frame: np.ndarray, detections: Sequence[Detection], interest: InterestScore) -> Path:
        now = datetime.fromtimestamp(self.clock())
        path = self._unique_path(now)
        # Sidecars first, image last: watchers react to the image and must find its labels.
        if self.config.save_labels:
            write_labels(path.with_suffix(".txt"), [d.to_yolo() for d in detections])
        if self.config.save_metadata:
            write_json(path.with_suffix(".json"), {
                "file": path.name,
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S.%f"),
                "timestamp": now.isoformat(),
                "model_version": self.model_version,
                "detections": [d.to_dict() for d in detections],
                "active_learning": {"score": interest.score, "components": interest.components,
                                    "reasons": interest.reasons},
            })
        write_image(path, frame)
        self._append_index(path, now, detections, interest)
        log.debug(f"Capture {path.name}: {len(detections)} objets, score={interest.score}")
        return path

    def _append_index(
        self, path: Path, now: datetime, detections: Sequence[Detection], interest: InterestScore
    ) -> None:
        new_file = not self.index_file.exists()
        confidences = [d.confidence for d in detections]
        with self.index_file.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(INDEX_COLUMNS)
            writer.writerow([
                path.name, now.isoformat(), len(detections),
                "|".join(sorted({d.class_name for d in detections})),
                f"{min(confidences):.4f}", f"{max(confidences):.4f}", interest.score,
            ])

    def close(self) -> None:
        if self.hashes is not None:
            self.hashes.save()
        log.info(f"Capture terminée: {dict(self.stats)}")
