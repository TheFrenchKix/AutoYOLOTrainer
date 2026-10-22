"""Ultralytics YOLO11 implementation of the :class:`Detector` protocol."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from src.model.types import Detection


class YoloDetector:
    """Thin wrapper around ``ultralytics.YOLO`` with hot-reload support.

    The capture loop keeps a single detector alive for weeks; when the
    trainer promotes a new model, :meth:`reload` swaps the weights without
    restarting the application.
    """

    def __init__(
        self,
        weights: Path | str,
        confidence: float = 0.25,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self._lock = threading.Lock()
        self.weights = Path(weights)
        self._model = self._load(weights)

    @staticmethod
    def _load(weights: Path | str):  # noqa: ANN205 - ultralytics type
        from ultralytics import YOLO  # lazy: heavy import (torch)

        return YOLO(str(weights))

    @property
    def class_names(self) -> list[str]:
        names = self._model.names
        return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    def reload(self, weights: Path) -> None:
        """Atomically switch to new weights."""
        model = self._load(weights)
        with self._lock:
            self._model, self.weights = model, Path(weights)

    def predict(self, image: np.ndarray) -> list[Detection]:
        with self._lock:
            results = self._model.predict(
                image, conf=self.confidence, imgsz=self.imgsz, device=self.device, verbose=False
            )
        detections: list[Detection] = []
        names = self.class_names
        for result in results:
            h, w = result.orig_shape
            boxes = result.boxes
            if boxes is None:
                continue
            arrays = (boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())
            for xyxy, conf, cls in zip(*arrays, strict=True):
                class_id = int(cls)
                detections.append(Detection(
                    class_id=class_id,
                    class_name=names[class_id] if class_id < len(names) else str(class_id),
                    confidence=float(conf),
                    x1=float(xyxy[0]), y1=float(xyxy[1]), x2=float(xyxy[2]), y2=float(xyxy[3]),
                    img_w=int(w), img_h=int(h),
                ))
        return detections

    def export_onnx(self, output_dir: Path | None = None, dynamic: bool = False) -> Path:
        """Export the current weights to ONNX (for ONNX Runtime deployment)."""
        with self._lock:
            exported = Path(self._model.export(format="onnx", imgsz=self.imgsz, dynamic=dynamic, simplify=True))
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / exported.name
            exported.replace(target)
            return target
        return exported
