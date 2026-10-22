"""Reading and writing YOLO dataset files (label ``.txt`` and ``data.yaml``)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

from src.model.types import YoloBox
from src.utils.files import atomic_write_text

SPLITS: tuple[str, ...] = ("train", "val")


def read_labels(path: Path) -> list[YoloBox]:
    """Parse a YOLO label file. A missing file means "no object"."""
    if not path.exists():
        return []
    return [
        YoloBox.from_line(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_labels(path: Path, boxes: Iterable[YoloBox]) -> None:
    """Write a YOLO label file (an empty file is a valid background image)."""
    lines = [box.to_line() for box in boxes]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def label_path_for(image_path: Path) -> Path:
    """Map ``.../images/<split>/x.png`` to ``.../labels/<split>/x.txt``.

    Images outside an ``images`` folder get their label next to them.
    """
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def write_data_yaml(dataset_dir: Path, class_names: list[str]) -> Path:
    """Write the ``data.yaml`` consumed by Ultralytics."""
    content = {
        "path": str(dataset_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    path = dataset_dir / "data.yaml"
    atomic_write_text(path, yaml.safe_dump(content, sort_keys=False, allow_unicode=True))
    return path


def read_class_names(dataset_dir: Path) -> list[str]:
    path = dataset_dir / "data.yaml"
    if not path.exists():
        return []
    names = yaml.safe_load(path.read_text(encoding="utf-8")).get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)
