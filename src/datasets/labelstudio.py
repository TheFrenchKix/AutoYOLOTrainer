"""Label Studio <-> YOLO conversion (pure functions, no network access).

Label Studio stores rectangles as percentages of the image size with the
top-left corner as origin; YOLO uses normalised centre coordinates.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from xml.sax.saxutils import quoteattr

from src.model.types import Detection, YoloBox

# Label Studio prefixes uploaded files with an 8-char hash: "1a2b3c4d-name.png".
_UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-")


@dataclass
class ConvertedTask:
    """One Label Studio task converted to YOLO."""

    task_id: int
    image_ref: str
    image_path: Path | None
    boxes: list[YoloBox] = field(default_factory=list)


def build_label_config(class_names: list[str]) -> str:
    """Label Studio labeling interface for bounding boxes."""
    labels = "\n".join(f"    <Label value={quoteattr(name)}/>" for name in class_names)
    return (
        "<View>\n"
        '  <Image name="image" value="$image" zoom="true" zoomControl="true"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f"{labels}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )


def ls_rect_to_yolo(value: dict[str, Any], class_id: int) -> YoloBox:
    """Convert a Label Studio rectangle ``value`` (percentages) to YOLO."""
    x, y = float(value["x"]) / 100, float(value["y"]) / 100
    w, h = float(value["width"]) / 100, float(value["height"]) / 100
    # Clip to the image: annotators sometimes drag slightly outside.
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(1.0, x + w), min(1.0, y + h)
    return YoloBox(class_id, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def yolo_to_ls_rect(box: YoloBox) -> dict[str, float]:
    """Inverse of :func:`ls_rect_to_yolo` (percent, top-left origin)."""
    return {
        "x": (box.x_center - box.width / 2) * 100,
        "y": (box.y_center - box.height / 2) * 100,
        "width": box.width * 100,
        "height": box.height * 100,
        "rotation": 0,
    }


def detections_to_ls_prediction(detections: Iterable[Detection], model_version: str) -> dict[str, Any]:
    """Build a Label Studio pre-annotation from model detections."""
    results = []
    detections = list(detections)
    for i, det in enumerate(detections):
        results.append({
            "id": f"pred{i}",
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "original_width": det.img_w,
            "original_height": det.img_h,
            "score": det.confidence,
            "value": {**yolo_to_ls_rect(det.to_yolo()), "rectanglelabels": [det.class_name]},
        })
    score = min((d.confidence for d in detections), default=0.0)
    return {"model_version": model_version, "score": score, "result": results}


def resolve_image_ref(ref: str, document_root: Path | None, search_dirs: Iterable[Path]) -> Path | None:
    """Find the local file behind a Label Studio image reference.

    Handles local-storage URLs (``/data/local-files/?d=...``), uploads
    (``/data/upload/<project>/<hash>-name.png``), absolute paths and remote URLs
    (matched by filename in ``search_dirs``).
    """
    parsed = urlparse(ref)
    if parsed.path.endswith("/data/local-files/") or "local-files" in parsed.path:
        rel = parse_qs(parsed.query).get("d", [""])[0]
        if rel and document_root is not None:
            candidate = document_root / unquote(rel)
            if candidate.exists():
                return candidate

    direct = Path(unquote(ref))
    if direct.is_absolute() and direct.exists():
        return direct

    name = Path(unquote(parsed.path)).name
    candidates = [name, _UPLOAD_PREFIX.sub("", name)]
    for folder in search_dirs:
        for candidate_name in candidates:
            candidate = folder / candidate_name
            if candidate.exists():
                return candidate
    return None


def _latest_annotation(task: dict[str, Any]) -> dict[str, Any] | None:
    valid = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not valid:
        return None
    return max(valid, key=lambda a: a.get("updated_at") or a.get("created_at") or "")


def convert_tasks(
    tasks: list[dict[str, Any]],
    class_names: list[str],
    document_root: Path | None = None,
    search_dirs: Iterable[Path] = (),
) -> tuple[list[ConvertedTask], list[str]]:
    """Convert a Label Studio JSON export to YOLO boxes.

    Tasks without a (non-cancelled) annotation are skipped. An annotation
    with no rectangle is kept: it is a valid "background" image.

    Returns:
        The converted tasks and a list of human-readable warnings.
    """
    class_ids = {name: i for i, name in enumerate(class_names)}
    search_dirs = list(search_dirs)
    converted: list[ConvertedTask] = []
    warnings: list[str] = []

    for task in tasks:
        annotation = _latest_annotation(task)
        if annotation is None:
            continue
        data = task.get("data", {})
        ref = str(data.get("image") or next(iter(data.values()), ""))
        item = ConvertedTask(
            task_id=int(task.get("id", -1)),
            image_ref=ref,
            image_path=resolve_image_ref(ref, document_root, search_dirs),
        )
        if item.image_path is None:
            warnings.append(f"Tâche {item.task_id}: image introuvable ({ref})")
        for result in annotation.get("result", []):
            if result.get("type") != "rectanglelabels":
                continue
            value = result["value"]
            for label in value.get("rectanglelabels", []):
                if label not in class_ids:
                    warnings.append(f"Tâche {item.task_id}: classe inconnue '{label}' ignorée")
                    continue
                item.boxes.append(ls_rect_to_yolo(value, class_ids[label]))
        converted.append(item)
    return converted, warnings
