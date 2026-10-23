"""Run a detector on a centred square of the frame only: far fewer pixels to process, so more FPS."""

from __future__ import annotations

import inspect
import math

import cv2
import numpy as np

from src.model.types import Detection, Detector

STRIDE = 32  # YOLO input sizes are multiples of the network stride
PAD_VALUE = 114  # grey used by the Ultralytics letterbox


class ZoneDetector:
    """Wraps a detector so that it only looks at the ``size`` x ``size`` square at the centre of each frame.

    The zone keeps the object scale the model sees on a full frame (``imgsz / longest side``): a model trained
    on screens shrunk to 640 px would otherwise meet objects several times larger than in training. The
    detections come back in full-frame coordinates, as if the whole frame had been analysed.

    ``reference`` is the longest side of the full frame when the source already delivers only the zone
    (``ScreenSource(zone=...)``); the frames are then used as they are, without any further crop.
    """

    def __init__(self, detector: Detector, size: int = 200, reference: int | None = None) -> None:
        if size < 1:
            raise ValueError("La zone doit mesurer au moins 1 pixel")
        self.detector = detector
        self.size = size
        self.reference = reference
        # only detectors whose ``predict`` accepts ``imgsz`` (YoloDetector) can be fed a small input
        self._sized = "imgsz" in inspect.signature(detector.predict).parameters

    @property
    def class_names(self) -> list[str]:
        return self.detector.class_names

    @property
    def confidence(self) -> float:
        return self.detector.confidence  # type: ignore[attr-defined]

    @confidence.setter
    def confidence(self, value: float) -> None:
        self.detector.confidence = value  # type: ignore[attr-defined]

    def zone_box(self, height: int, width: int) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) of the zone in a ``width`` x ``height`` frame (shrunk if the frame is smaller)."""
        side = min(self.size, height, width)
        left, top = (width - side) // 2, (height - side) // 2
        return left, top, left + side, top + side

    def predict(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        left, top, right, bottom = self.zone_box(height, width)
        crop = image[top:bottom, left:right]
        scale, extent = 1.0, right - left
        if self._sized:
            reference = getattr(self.detector, "imgsz", None)
            if reference:
                scale = min(1.0, reference / (self.reference or max(height, width)))
            extent = max(1, round((right - left) * scale))
            if extent != crop.shape[0]:
                crop = cv2.resize(crop, (extent, extent), interpolation=cv2.INTER_AREA)
            side = math.ceil(extent / STRIDE) * STRIDE
            canvas = np.full((side, side, 3), PAD_VALUE, dtype=np.uint8)
            canvas[:extent, :extent] = crop
            found = self.detector.predict(canvas, imgsz=side)  # type: ignore[call-arg]
        else:
            found = self.detector.predict(crop)

        detections = []
        for d in found:
            x1, x2 = (min(max(v, 0.0), extent) / scale + left for v in (d.x1, d.x2))
            y1, y2 = (min(max(v, 0.0), extent) / scale + top for v in (d.y1, d.y2))
            if x2 - x1 >= 1 and y2 - y1 >= 1:  # drop boxes that lay in the padding
                detections.append(Detection(d.class_id, d.class_name, d.confidence, x1, y1, x2, y2, width, height))
        return detections
