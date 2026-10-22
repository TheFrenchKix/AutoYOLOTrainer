"""Offline data augmentation with Albumentations (+ custom Mosaic and MixUp).

Augmented copies are written into the *train split of a dataset snapshot*,
never into ``datasets/current``, so the working dataset stays pristine and
validation images are never augmented.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

from src.config.schema import AugmentationConfig, TransformConfig
from src.datasets.yolo_format import label_path_for, read_labels, write_labels
from src.model.types import YoloBox
from src.utils.files import list_images
from src.utils.images import read_image, write_image
from src.utils.logging import get_logger

log = get_logger("dataset")

AUG_PREFIX = "aug_"


@dataclass
class Sample:
    image: np.ndarray
    boxes: list[YoloBox]


# Factories turning a TransformConfig into an Albumentations transform.
_FACTORIES: dict[str, Callable[[TransformConfig], A.BasicTransform]] = {
    "horizontal_flip": lambda c: A.HorizontalFlip(p=c.p),
    "vertical_flip": lambda c: A.VerticalFlip(p=c.p),
    "rotation": lambda c: A.Rotate(limit=c.params.get("limit", 15), border_mode=cv2.BORDER_CONSTANT, p=c.p),
    "blur": lambda c: A.Blur(blur_limit=(3, max(3, int(c.params.get("blur_limit", 5)))), p=c.p),
    "brightness": lambda c: A.RandomBrightnessContrast(
        brightness_limit=c.params.get("brightness_limit", 0.2),
        contrast_limit=c.params.get("contrast_limit", 0.2), p=c.p),
    "hsv": lambda c: A.HueSaturationValue(
        hue_shift_limit=c.params.get("hue", 10), sat_shift_limit=c.params.get("saturation", 20),
        val_shift_limit=c.params.get("value", 20), p=c.p),
    "noise": lambda c: A.GaussNoise(
        std_range=(c.params.get("std_min", 0.02), c.params.get("std_max", 0.08)), p=c.p),
    # BBox-safe crop: never cuts an annotated object, output size varies.
    "random_crop": lambda c: A.BBoxSafeRandomCrop(erosion_rate=1 - c.params.get("scale_min", 0.6), p=c.p),
    "perspective": lambda c: A.Perspective(scale=(0.01, c.params.get("scale_max", 0.06)), p=c.p),
}


def _to_arrays(boxes: list[YoloBox]) -> tuple[list[list[float]], list[int]]:
    return [[b.x_center, b.y_center, b.width, b.height] for b in boxes], [b.class_id for b in boxes]


class AugmentationPipeline:
    """Build and apply the augmentation pipeline described in the config."""

    def __init__(self, config: AugmentationConfig, seed: int | None = None) -> None:
        self.config = config
        self.rng = random.Random(seed)
        transforms = []
        for name, transform_cfg in config.transforms.items():
            if name not in _FACTORIES:
                raise ValueError(f"Augmentation inconnue: {name} (connues: {sorted(_FACTORIES)})")
            if transform_cfg.enabled:
                transforms.append(_FACTORIES[name](transform_cfg))
        self.transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_ids"], min_visibility=0.3, clip=True),
            seed=seed,
        )

    def augment(self, sample: Sample) -> Sample:
        """Apply the Albumentations pipeline to one sample."""
        bboxes, class_ids = _to_arrays(sample.boxes)
        out = self.transform(image=sample.image, bboxes=bboxes, class_ids=class_ids)
        boxes = [YoloBox(int(c), *map(float, b)) for b, c in zip(out["bboxes"], out["class_ids"], strict=True)]
        return Sample(out["image"], boxes)

    @staticmethod
    def mosaic(samples: list[Sample], size: int = 640) -> Sample:
        """Combine 4 samples in a 2x2 grid of ``size x size`` pixels."""
        if len(samples) != 4:
            raise ValueError("Mosaic attend exactement 4 images")
        half = size // 2
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        boxes: list[YoloBox] = []
        for index, sample in enumerate(samples):
            ox, oy = (index % 2) * half, (index // 2) * half
            canvas[oy:oy + half, ox:ox + half] = cv2.resize(sample.image, (half, half))
            for b in sample.boxes:
                boxes.append(YoloBox(
                    b.class_id, (b.x_center * half + ox) / size, (b.y_center * half + oy) / size,
                    b.width / 2, b.height / 2,
                ))
        return Sample(canvas, boxes)

    @staticmethod
    def mixup(first: Sample, second: Sample, alpha: float = 0.5) -> Sample:
        """Blend two samples; boxes of both are kept (normalised coords survive the resize)."""
        h, w = first.image.shape[:2]
        other = cv2.resize(second.image, (w, h))
        image = cv2.addWeighted(first.image, 1 - alpha, other, alpha, 0)
        return Sample(image, first.boxes + second.boxes)

    # ------------------------------------------------------------------ disk
    def augment_split(self, dataset_dir: Path, split: str = "train", copies: int | None = None) -> int:
        """Write augmented copies of every image of ``split`` (prefixed ``aug_``)."""
        copies = self.config.offline_copies if copies is None else copies
        images = [p for p in list_images(dataset_dir / "images" / split) if not p.name.startswith(AUG_PREFIX)]
        if not images or copies <= 0:
            return 0

        def load(path: Path) -> Sample:
            return Sample(read_image(path), read_labels(label_path_for(path)))

        written = 0
        for path in tqdm(images, desc="Augmentation", unit="img"):
            base = load(path)
            if base.image is None:
                log.warning(f"Image illisible ignorée: {path}")
                continue
            for k in range(copies):
                sample = self._random_combination(base, images, load)
                sample = self.augment(sample)
                target = path.with_name(f"{AUG_PREFIX}{k}_{path.stem}.jpg")
                write_image(target, sample.image)
                write_labels(label_path_for(target), sample.boxes)
                written += 1
        log.info(f"{written} images augmentées générées dans {dataset_dir / 'images' / split}")
        return written

    def _random_combination(self, base: Sample, pool: list[Path], load: Callable[[Path], Sample]) -> Sample:
        """Optionally turn ``base`` into a Mosaic or MixUp sample."""
        cfg = self.config
        if cfg.mosaic.enabled and len(pool) >= 4 and self.rng.random() < cfg.mosaic.p:
            others = [load(p) for p in self.rng.sample(pool, 3)]
            if all(o.image is not None for o in others):
                size = int(max(base.image.shape[:2]))
                return self.mosaic([base, *others], size=size)
        if cfg.mixup.enabled and len(pool) >= 2 and self.rng.random() < cfg.mixup.p:
            other = load(self.rng.choice(pool))
            if other.image is not None:
                return self.mixup(base, other, float(cfg.mixup.params.get("alpha", 0.5)))
        return base
