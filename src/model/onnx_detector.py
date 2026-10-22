"""ONNX Runtime implementation of the :class:`Detector` protocol.

Lets you deploy an exported model (``python main.py export``) without PyTorch.
Supports the standard YOLO11 detection head output ``(1, 4 + nc, N)``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.model.types import Detection


class OnnxDetector:
    def __init__(
        self,
        model_path: Path,
        class_names: list[str],
        confidence: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> None:
        import onnxruntime as ort

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = class_names
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        h, w = image.shape[:2]
        scale = min(self.imgsz / h, self.imgsz / w)
        nh, nw = round(h * scale), round(w * scale)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
        return canvas, scale, (left, top)

    def predict(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        padded, scale, (pad_x, pad_y) = self._letterbox(image)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = self.session.run(None, {self.input_name: blob})[0][0].T  # (N, 4 + nc)

        scores = output[:, 4:]
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)
        keep = confidences >= self.confidence
        boxes, class_ids, confidences = output[keep, :4], class_ids[keep], confidences[keep]
        if not len(boxes):
            return []

        # cx, cy, w, h (letterboxed) -> x, y, w, h (original image)
        xywh = np.empty_like(boxes)
        xywh[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / scale
        xywh[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / scale
        xywh[:, 2] = boxes[:, 2] / scale
        xywh[:, 3] = boxes[:, 3] / scale
        indices = cv2.dnn.NMSBoxesBatched(xywh.tolist(), confidences.tolist(), class_ids.tolist(),
                                          self.confidence, self.iou)
        detections = []
        for i in np.array(indices).flatten():
            x, y, bw, bh = xywh[i]
            cid = int(class_ids[i])
            detections.append(Detection(
                class_id=cid,
                class_name=self.class_names[cid] if cid < len(self.class_names) else str(cid),
                confidence=float(confidences[i]),
                x1=float(max(0, x)), y1=float(max(0, y)),
                x2=float(min(w, x + bw)), y2=float(min(h, y + bh)),
                img_w=w, img_h=h,
            ))
        return detections
