"""Core value types shared by the detector, the capture and the labeler."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class YoloBox:
    """A YOLO label: class id + normalised centre/size (all in ``[0, 1]``)."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )

    @classmethod
    def from_line(cls, line: str) -> YoloBox:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Ligne YOLO invalide: {line!r}")
        return cls(int(parts[0]), *(float(v) for v in parts[1:]))

    @classmethod
    def from_xyxy(
        cls, class_id: int, x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
    ) -> YoloBox:
        """Build from absolute pixel corners, clipping to the image."""
        x1, x2 = sorted((max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))))
        y1, y2 = sorted((max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))))
        return cls(
            class_id=class_id,
            x_center=(x1 + x2) / 2 / img_w,
            y_center=(y1 + y2) / 2 / img_h,
            width=(x2 - x1) / img_w,
            height=(y2 - y1) / img_h,
        )

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Absolute pixel corners ``(x1, y1, x2, y2)``."""
        half_w, half_h = self.width * img_w / 2, self.height * img_h / 2
        cx, cy = self.x_center * img_w, self.y_center * img_h
        return cx - half_w, cy - half_h, cx + half_w, cy + half_h


@dataclass(frozen=True)
class Detection:
    """One object predicted by a detector, in absolute pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    img_w: int
    img_h: int

    def to_yolo(self) -> YoloBox:
        return YoloBox.from_xyxy(self.class_id, self.x1, self.y1, self.x2, self.y2, self.img_w, self.img_h)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Detection:
        return cls(**data)


@runtime_checkable
class Detector(Protocol):
    """Anything able to turn an image into detections (YOLO, ONNX, fake...)."""

    class_names: list[str]

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Run inference on a BGR image (OpenCV convention)."""
        ...
