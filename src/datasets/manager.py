"""DatasetManager: the single entry point to manipulate YOLO datasets.

The *working* dataset lives in ``datasets/current``. Every training run uses
an immutable snapshot ``datasets/vN`` created by :meth:`create_version`
(hard links, so snapshots cost almost no disk space).
"""

from __future__ import annotations

import shutil
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.datasets.labelstudio import convert_tasks
from src.datasets.versioning import VersionRegistry
from src.datasets.yolo_format import (
    SPLITS,
    label_path_for,
    read_class_names,
    read_labels,
    write_data_yaml,
    write_labels,
)
from src.model.types import YoloBox
from src.utils.files import IMAGE_EXTENSIONS, link_or_copy, list_images, read_json, write_json
from src.utils.hashing import find_duplicates
from src.utils.logging import get_logger

log = get_logger("dataset")


@dataclass
class SplitCounts:
    images: int = 0
    labels: int = 0
    boxes: int = 0
    backgrounds: int = 0
    per_class: dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetCounts:
    splits: dict[str, SplitCounts] = field(default_factory=dict)

    @property
    def images(self) -> int:
        return sum(s.images for s in self.splits.values())

    @property
    def boxes(self) -> int:
        return sum(s.boxes for s in self.splits.values())

    @property
    def per_class(self) -> dict[str, int]:
        total: dict[str, int] = {}
        for split in self.splits.values():
            for name, n in split.per_class.items():
                total[name] = total.get(name, 0) + n
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "boxes": self.boxes,
            "per_class": self.per_class,
            "splits": {k: vars(v) for k, v in self.splits.items()},
        }


@dataclass
class ImportSummary:
    imported: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)


class DatasetManager:
    """Create, fill, clean, merge, report and version YOLO datasets."""

    def __init__(
        self,
        dataset_dir: Path,
        class_names: list[str],
        versions_dir: Path | None = None,
        archive_dir: Path | None = None,
        val_ratio: float = 0.2,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.class_names = list(class_names)
        self.val_ratio = val_ratio
        self.versions = VersionRegistry(versions_dir or dataset_dir.parent, archive_dir)

    # ------------------------------------------------------------------ layout
    def init_structure(self) -> Path:
        """Create ``images/{train,val}``, ``labels/{train,val}`` and ``data.yaml``."""
        for split in SPLITS:
            (self.dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        return write_data_yaml(self.dataset_dir, self.class_names)

    @property
    def data_yaml(self) -> Path:
        return self.dataset_dir / "data.yaml"

    def images(self, split: str | None = None) -> list[Path]:
        splits = [split] if split else list(SPLITS)
        return [img for s in splits for img in list_images(self.dataset_dir / "images" / s)]

    def choose_split(self, name: str) -> str:
        """Deterministic train/val split from the file name (stable across runs)."""
        bucket = zlib.crc32(name.encode("utf-8")) % 1000
        return "val" if bucket < self.val_ratio * 1000 else "train"

    # ---------------------------------------------------------------- adding
    def add_image(
        self,
        image: Path,
        boxes: Iterable[YoloBox],
        split: str | None = None,
        move: bool = False,
    ) -> Path:
        """Add one image and its labels to the working dataset.

        A name collision with a different file gets a numeric suffix.
        """
        split = split or self.choose_split(image.name)
        target_dir = self.dataset_dir / "images" / split
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / image.name
        counter = 1
        while target.exists():
            target = target_dir / f"{image.stem}_{counter}{image.suffix}"
            counter += 1
        if move:
            shutil.move(str(image), target)
        else:
            shutil.copy2(image, target)
        write_labels(label_path_for(target), boxes)
        return target

    def import_labelstudio(
        self,
        tasks: list[dict[str, Any]],
        document_root: Path | None = None,
        search_dirs: Iterable[Path] = (),
    ) -> ImportSummary:
        """Import a Label Studio JSON export into the working dataset.

        Images already present (same file name) are updated in place, so the
        import can be re-run after annotators fix labels.
        """
        self.init_structure()
        converted, warnings = convert_tasks(tasks, self.class_names, document_root, search_dirs)
        summary = ImportSummary(warnings=warnings)
        existing = {img.name: img for img in self.images()}
        for item in tqdm(converted, desc="Import Label Studio", unit="img"):
            if item.image_path is None:
                summary.skipped += 1
                continue
            if item.image_path.name in existing:
                write_labels(label_path_for(existing[item.image_path.name]), item.boxes)
            else:
                self.add_image(item.image_path, item.boxes)
            summary.imported += 1
        for warning in warnings:
            log.warning(warning)
        log.info(f"Import Label Studio: {summary.imported} importées, {summary.skipped} ignorées")
        return summary

    def import_folder(self, folder: Path, move: bool = True) -> int:
        """Import a flat folder of images with sidecar ``.txt`` labels (e.g. ``accepted/``)."""
        self.init_structure()
        count = 0
        for image in list_images(folder):
            boxes = read_labels(image.with_suffix(".txt"))
            self.add_image(image, boxes, move=move)
            if move:
                for suffix in (".txt", ".json"):
                    image.with_suffix(suffix).unlink(missing_ok=True)
            count += 1
        log.info(f"{count} images importées depuis {folder}")
        return count

    def merge(self, sources: Iterable[Path]) -> int:
        """Merge other YOLO datasets into the working dataset.

        Class ids are remapped by class *name* when the source has a
        ``data.yaml``; unknown classes are appended to the class list.
        """
        self.init_structure()
        total = 0
        for source in sources:
            names = read_class_names(source) or self.class_names
            for name in names:
                if name not in self.class_names:
                    self.class_names.append(name)
            remap = {i: self.class_names.index(n) for i, n in enumerate(names)}
            images = list_images(source / "images", recursive=True) or list_images(source)
            for image in tqdm(images, desc=f"Fusion {source.name}", unit="img"):
                boxes = [
                    YoloBox(remap.get(b.class_id, b.class_id), b.x_center, b.y_center, b.width, b.height)
                    for b in read_labels(label_path_for(image))
                ]
                split = image.parent.name if image.parent.name in SPLITS else None
                self.add_image(image, boxes, split=split)
                total += 1
            log.info(f"Fusion de {source}: {len(images)} images")
        write_data_yaml(self.dataset_dir, self.class_names)
        return total

    # ---------------------------------------------------------------- quality
    def remove_duplicates(self, max_distance: int = 6, dry_run: bool = False) -> list[Path]:
        """Delete near-duplicate images (perceptual hash) and their labels."""
        duplicates = find_duplicates(tqdm(self.images(), desc="pHash", unit="img"), max_distance)
        removed = []
        for duplicate, original in duplicates:
            log.debug(f"Doublon: {duplicate.name} ~ {original.name}")
            if not dry_run:
                duplicate.unlink(missing_ok=True)
                label_path_for(duplicate).unlink(missing_ok=True)
            removed.append(duplicate)
        log.info(f"{len(removed)} doublons {'détectés' if dry_run else 'supprimés'}")
        return removed

    def validate(self) -> list[str]:
        """Return a list of problems (invalid boxes, unknown classes, orphan labels)."""
        problems: list[str] = []
        for image in self.images():
            label = label_path_for(image)
            try:
                boxes = read_labels(label)
            except ValueError as exc:
                problems.append(f"{label.name}: {exc}")
                continue
            for box in boxes:
                if not 0 <= box.class_id < len(self.class_names):
                    problems.append(f"{label.name}: class_id {box.class_id} hors limites")
                if not (0 < box.width <= 1 and 0 < box.height <= 1):
                    problems.append(f"{label.name}: boîte dégénérée {box}")
        for split in SPLITS:
            for label in (self.dataset_dir / "labels" / split).glob("*.txt"):
                if not any((self.dataset_dir / "images" / split / f"{label.stem}{e}").exists()
                           for e in IMAGE_EXTENSIONS):
                    problems.append(f"{label.name}: label sans image")
        return problems

    # ------------------------------------------------------------------ stats
    def count(self, dataset_dir: Path | None = None) -> DatasetCounts:
        root = dataset_dir or self.dataset_dir
        counts = DatasetCounts()
        for split in SPLITS:
            stats = SplitCounts(per_class={name: 0 for name in self.class_names})
            for image in list_images(root / "images" / split):
                stats.images += 1
                label = label_path_for(image)
                boxes = read_labels(label)
                stats.labels += int(label.exists())
                stats.boxes += len(boxes)
                stats.backgrounds += int(not boxes)
                for box in boxes:
                    name = self.class_names[box.class_id] if box.class_id < len(self.class_names) else str(box.class_id)
                    stats.per_class[name] = stats.per_class.get(name, 0) + 1
            counts.splits[split] = stats
        return counts

    def report(self, output_dir: Path | None = None) -> Path:
        """Write a JSON + HTML report with class distribution chart."""
        from src.datasets.report import write_dataset_report

        output_dir = output_dir or self.dataset_dir
        return write_dataset_report(self.count(), self.validate(), output_dir)

    # ------------------------------------------------------------- versioning
    def create_version(self, note: str = "") -> tuple[int, Path]:
        """Snapshot the working dataset into ``datasets/vN`` (hard links)."""
        version, path = self.versions.create()
        for split in SPLITS:
            for image in self.images(split):
                link_or_copy(image, path / "images" / split / image.name)
                label = label_path_for(image)
                if label.exists():
                    link_or_copy(label, path / "labels" / split / label.name)
        write_data_yaml(path, self.class_names)
        manifest = {
            "version": version,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": note,
            "classes": self.class_names,
            "counts": self.count(path).to_dict(),
        }
        write_json(path / "manifest.json", manifest)
        self.versions.set_active(version)
        log.info(f"Dataset versionné: {path.name} ({manifest['counts']['images']} images)")
        return version, path

    def version_manifest(self, version: int) -> dict[str, Any]:
        return read_json(self.versions.path(version) / "manifest.json", default={})
