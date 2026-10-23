"""Live preview of a detector: a small window with the boxes, labels and confidences drawn on the source.

Meant to try a model by eye (``python main.py preview``). The detector sees the full-resolution frame; only
the picture shown is shrunk (letterboxed, aspect ratio kept) to ``size`` x ``size`` pixels.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

import cv2
import numpy as np

from src.capture.sources import FrameSource
from src.model.types import Detection, Detector

DEFAULT_SIZE = 300
PADDING_COLOR = (30, 26, 26)  # BGR
TEXT_COLOR = (0, 0, 0)
NOTE_COLOR = (230, 230, 230)
ZONE_COLOR = (255, 215, 120)  # BGR, light blue: stands out from the grey text and the class colours
PALETTE = [(120, 200, 80), (255, 160, 60), (60, 160, 255), (220, 90, 200), (230, 220, 70), (90, 90, 255)]  # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.38
CONFIDENCE_STEP = 0.05
QUIT_KEYS = {ord("q"), ord("Q"), 27}  # q or Esc


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Fit ``frame`` in a ``size`` x ``size`` canvas. Returns (canvas, scale, left offset, top offset)."""
    height, width = frame.shape[:2]
    scale = size / max(height, width)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
    canvas = np.full((size, size, 3), PADDING_COLOR, dtype=np.uint8)
    left, top = (size - new_w) // 2, (size - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, scale, left, top


def render_preview(
    frame: np.ndarray,
    detections: Sequence[Detection],
    size: int = DEFAULT_SIZE,
    note: str = "",
    zone: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """The frame fitted in ``size`` x ``size`` with one box per detection, labelled ``name conf``.

    ``zone`` = (left, top, right, bottom) in frame pixels is outlined, to show what the detector looks at.
    """
    canvas, scale, left, top = letterbox(frame, size)
    if zone is not None:
        z1, z2 = (round(zone[0] * scale) + left, round(zone[1] * scale) + top), (
            round(zone[2] * scale) + left - 1, round(zone[3] * scale) + top - 1)
        cv2.rectangle(canvas, z1, z2, ZONE_COLOR, 1)
    for detection in detections:
        color = PALETTE[detection.class_id % len(PALETTE)]
        x1, y1 = round(detection.x1 * scale) + left, round(detection.y1 * scale) + top
        x2, y2 = round(detection.x2 * scale) + left, round(detection.y2 * scale) + top
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

        label = f"{detection.class_name} {detection.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        above = y1 - text_h - baseline - 2 >= 0  # keep the label inside the picture
        text_y = y1 - baseline - 1 if above else y1 + text_h + 1
        text_x = min(max(0, x1), max(0, size - text_w - 2))
        cv2.rectangle(canvas, (text_x, text_y - text_h - 1), (text_x + text_w + 2, text_y + baseline), color, -1)
        cv2.putText(canvas, label, (text_x + 1, text_y), FONT, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA)
    if note:
        cv2.putText(canvas, note, (4, size - 5), FONT, FONT_SCALE, TEXT_COLOR, 3, cv2.LINE_AA)
        cv2.putText(canvas, note, (4, size - 5), FONT, FONT_SCALE, NOTE_COLOR, 1, cv2.LINE_AA)
    return canvas


class PreviewWindow(Protocol):
    def show(self, image: np.ndarray) -> None: ...

    def poll(self, delay_ms: int) -> str | None:
        """Wait up to ``delay_ms``; return "quit" (q, Esc or window closed), "+", "-", "view" (z) or ``None``."""
        ...

    def close(self) -> None: ...


class CvWindow:
    """An OpenCV window sized exactly like the picture it shows."""

    def __init__(self, title: str) -> None:
        self.title = title
        self._shown = False
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def show(self, image: np.ndarray) -> None:
        cv2.imshow(self.title, image)
        self._shown = True

    def poll(self, delay_ms: int) -> str | None:
        key = cv2.waitKey(max(1, delay_ms)) & 0xFF
        if key in QUIT_KEYS or (self._shown and cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1):
            return "quit"
        if key in (ord("+"), ord("=")):
            return "+"
        if key in (ord("-"), ord("_")):
            return "-"
        if key in (ord("z"), ord("Z")):
            return "view"
        return None

    def close(self) -> None:
        cv2.destroyAllWindows()


def run_preview(
    source: FrameSource,
    detector: Detector,
    window: PreviewWindow,
    size: int = DEFAULT_SIZE,
    fps: float = 0.0,
    duration: float | None = None,
    max_frames: int | None = None,
) -> int:
    """Show the detections frame after frame until q/Esc, the end of the source or the limits. Returns the frames shown.

    ``fps`` caps the rate (0 = as fast as possible). ``+`` / ``-`` raise or lower the detector's confidence
    threshold while it runs. If the detector has a detection zone (``zone_box``), the picture shows the zone
    enlarged; ``z`` switches to the whole frame with the zone outlined. When a finite source (video, folder)
    ends, the last picture stays on screen until the window is closed.
    """
    period = 1.0 / fps if fps > 0 else 0.0
    started = time.monotonic()
    frames = 0
    rate = 0.0
    last: np.ndarray | None = None
    ended = False  # the source ran out of frames
    zoomed = True  # picture = the detection zone enlarged (when there is one)
    try:
        while True:
            tick = time.monotonic()
            if duration is not None and tick - started >= duration:
                break
            frame = source.read()
            if frame is None:
                ended = True
                break
            detections = detector.predict(frame)
            threshold = getattr(detector, "confidence", None)
            zone_box = getattr(detector, "zone_box", None)
            zone = zone_box(*frame.shape[:2]) if zone_box else None
            note = f"{len(detections)} det"
            if threshold is not None:
                note += f" | conf>={threshold:.2f}"
            note += f" | {rate:.0f} fps"
            if zone is not None and zoomed:
                left, top, right, bottom = zone
                shifted = [replace(d, x1=d.x1 - left, y1=d.y1 - top, x2=d.x2 - left, y2=d.y2 - top) for d in detections]
                last = render_preview(frame[top:bottom, left:right], shifted, size, note)
            else:
                last = render_preview(frame, detections, size, note, zone)
            window.show(last)
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
            command = window.poll(int((period - (time.monotonic() - tick)) * 1000))
            if command == "quit":
                break
            if command == "view":
                zoomed = not zoomed
            if command in ("+", "-") and threshold is not None:
                step = CONFIDENCE_STEP if command == "+" else -CONFIDENCE_STEP
                detector.confidence = min(0.95, max(0.05, round(threshold + step, 2)))  # type: ignore[attr-defined]
            elapsed = time.monotonic() - tick
            if elapsed > 0:
                rate = 1.0 / elapsed if rate == 0 else 0.8 * rate + 0.2 / elapsed  # smoothed
        if ended and last is not None:  # finite source over: keep the last picture visible
            while window.poll(100) != "quit":
                pass
    finally:
        source.close()
        window.close()
    return frames
