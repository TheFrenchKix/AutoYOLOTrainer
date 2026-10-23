import sys
import threading
import time

import numpy as np
import pytest

from src.capture.preview import ZONE_COLOR, run_preview
from src.capture.sources import BackgroundSource, FrameSource, ScreenSource
from src.model.types import Detection
from src.model.zone import PAD_VALUE, ZoneDetector


def box(x1: float, y1: float, x2: float, y2: float, size: int = 0) -> Detection:
    return Detection(0, "player", 0.9, x1, y1, x2, y2, size, size)


class SizedDetector:
    """Like YoloDetector: ``predict`` accepts ``imgsz``. Records what it was given."""

    class_names = ["player"]
    imgsz = 640
    confidence = 0.25

    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections
        self.calls: list[tuple[tuple[int, ...], int | None]] = []

    def predict(self, image: np.ndarray, imgsz: int | None = None) -> list[Detection]:
        self.calls.append((image.shape, imgsz))
        return self.detections


class PlainDetector:
    class_names = ["player"]

    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections
        self.seen: list[tuple[int, ...]] = []

    def predict(self, image: np.ndarray) -> list[Detection]:
        self.seen.append(image.shape)
        return self.detections


def frame(width: int = 1680, height: int = 1050) -> np.ndarray:
    return np.full((height, width, 3), 90, dtype=np.uint8)


def test_zone_box_is_centred_and_clamped() -> None:
    zone = ZoneDetector(PlainDetector([]), 200)
    assert zone.zone_box(1050, 1680) == (740, 425, 940, 625)
    assert zone.zone_box(100, 150) == (25, 0, 125, 100)  # frame smaller than the zone: largest centred square


def test_zone_keeps_the_full_frame_object_scale_and_maps_boxes_back() -> None:
    inner = SizedDetector([box(10, 20, 50, 60)])
    zone = ZoneDetector(inner, 200)
    result = zone.predict(frame())
    scale = 640 / 1680
    extent = round(200 * scale)  # 76 px: what 200 native pixels weigh on a 640 px input
    side = 96  # padded to a multiple of 32, no further rescaling
    assert inner.calls == [((side, side, 3), side)]
    (d,) = result
    assert (d.x1, d.y1) == pytest.approx((10 / scale + 740, 20 / scale + 425))
    assert (d.x2, d.y2) == pytest.approx((50 / scale + 740, 60 / scale + 425))
    assert (d.img_w, d.img_h) == (1680, 1050) and extent < side
    assert 740 <= d.x1 < d.x2 <= 940 and 425 <= d.y1 < d.y2 <= 625  # inside the zone


def test_boxes_are_clipped_to_the_zone_and_padding_is_ignored() -> None:
    inner = SizedDetector([box(-20, -20, 30, 30), box(80, 80, 95, 95)])  # 2nd one only lies in the padding
    result = ZoneDetector(inner, 200).predict(frame())
    assert len(result) == 1 and (result[0].x1, result[0].y1) == (740, 425)


def test_frame_already_cropped_by_the_source_uses_the_reference_side() -> None:
    inner = SizedDetector([box(10, 10, 40, 40)])
    zone = ZoneDetector(inner, 200, reference=1680)
    (d,) = zone.predict(frame(200, 200))
    assert inner.calls[0][1] == 96  # same input as when the crop is cut from the full frame
    assert (d.x1, d.y1) == pytest.approx((10 / (640 / 1680), 10 / (640 / 1680)))
    assert (d.img_w, d.img_h) == (200, 200)


def test_detector_without_imgsz_gets_the_native_crop() -> None:
    inner = PlainDetector([box(5, 5, 25, 25)])
    (d,) = ZoneDetector(inner, 200).predict(frame())
    assert inner.seen == [(200, 200, 3)]
    assert (d.x1, d.y1, d.x2, d.y2) == (745, 430, 765, 450)


def test_confidence_and_names_are_delegated() -> None:
    inner = SizedDetector([])
    zone = ZoneDetector(inner, 200)
    zone.confidence = 0.4
    assert inner.confidence == 0.4 and zone.confidence == 0.4 and zone.class_names == ["player"]
    with pytest.raises(ValueError):
        ZoneDetector(inner, 0)
    assert PAD_VALUE == 114


# ------------------------------------------------------------------------ preview with a zone
class ListSource(FrameSource):
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = list(frames)

    def read(self) -> np.ndarray | None:
        return self.frames.pop(0) if self.frames else None


class Window:
    def __init__(self, commands: list[str | None]) -> None:
        self.commands = list(commands)
        self.shown: list[np.ndarray] = []

    def show(self, image: np.ndarray) -> None:
        self.shown.append(image)

    def poll(self, delay_ms: int) -> str | None:
        return self.commands.pop(0) if self.commands else "quit"

    def close(self) -> None:
        pass


def test_preview_shows_the_zone_then_the_whole_frame_with_the_zone_outlined() -> None:
    detector = ZoneDetector(PlainDetector([box(5, 5, 25, 25)]), 200)
    window = Window(["view", "quit"])
    run_preview(ListSource([frame(), frame()]), detector, window, size=300)
    zoomed, full = window.shown
    assert zoomed.shape == full.shape == (300, 300, 3)
    outline = np.all(full == ZONE_COLOR, axis=2)
    assert not np.any(np.all(zoomed == ZONE_COLOR, axis=2)) and outline.sum() > 20
    assert not np.array_equal(zoomed, full)


# --------------------------------------------------------------------------- sources
@pytest.mark.skipif(sys.platform != "win32", reason="monitor selection is Windows-only")
def test_screen_source_grabs_only_the_centred_zone() -> None:
    from src.capture.windows import enable_dpi_awareness, list_screens

    enable_dpi_awareness()
    primary = next(s for s in list_screens() if s.primary)
    source = ScreenSource(zone=200)  # no monitor given: the primary one
    left, top, right, bottom = primary.bbox
    assert source.reference == max(right - left, bottom - top)
    zone_left, zone_top = left + (right - left - 200) // 2, top + (bottom - top - 200) // 2
    assert source.bbox == (zone_left, zone_top, zone_left + 200, zone_top + 200)
    assert source.read().shape == (200, 200, 3)
    assert ScreenSource(zone=10**6).read().shape[:2] == (min(right - left, bottom - top),) * 2


def test_zone_of_a_region_does_not_need_windows() -> None:
    source = ScreenSource(region=[100, 100, 500, 300], zone=100)
    assert source.bbox == (250, 150, 350, 250) and source.reference == 400


class SlowSource(FrameSource):
    """Frames numbered by their pixel value; each read takes ``delay`` seconds."""

    def __init__(self, count: int, delay: float, fail_at: int | None = None) -> None:
        self.count, self.delay, self.fail_at, self.index = count, delay, fail_at, 0
        self.closed = threading.Event()

    def read(self) -> np.ndarray | None:
        time.sleep(self.delay)
        if self.fail_at == self.index:
            raise RuntimeError("capture cassée")
        if self.index >= self.count:
            return None
        self.index += 1
        return np.full((2, 2, 3), self.index, dtype=np.uint8)

    def close(self) -> None:
        self.closed.set()


def test_background_source_returns_the_newest_frame_and_ends() -> None:
    source = BackgroundSource(SlowSource(count=6, delay=0.02))
    time.sleep(0.2)  # the consumer is slower than the producer: it gets the latest frame, not frame 1
    first = source.read()
    assert first is not None and int(first[0, 0, 0]) == 6
    assert source.read() is None  # source over
    source.close()
    assert source._source.closed.is_set() and not source._thread.is_alive()


def test_background_source_hands_over_errors() -> None:
    source = BackgroundSource(SlowSource(count=5, delay=0.01, fail_at=1))
    with pytest.raises(RuntimeError, match="cassée"):
        while source.read() is not None:
            pass
    source.close()


def test_zone_option_is_known_to_the_parser() -> None:
    from src.cli.parser import build_parser

    parser = build_parser()
    assert parser.parse_args(["preview"]).zone == 200
    assert parser.parse_args(["preview", "--zone", "0", "--fps", "30"]).zone == 0
