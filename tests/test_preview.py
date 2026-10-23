import sys

import cv2
import numpy as np
import pytest

from src.capture.preview import CONFIDENCE_STEP, PADDING_COLOR, letterbox, render_preview, run_preview
from src.capture.sources import FrameSource, create_source
from src.config.schema import SourceConfig
from src.model.types import Detection


def detection(
    x1=100.0, y1=60.0, x2=300.0, y2=180.0, conf=0.87, class_id=0, name="player", size=(640, 360)
) -> Detection:
    return Detection(class_id, name, conf, x1, y1, x2, y2, *size)


class ListSource(FrameSource):
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = list(frames)
        self.closed = False

    def read(self) -> np.ndarray | None:
        return self.frames.pop(0) if self.frames else None

    def close(self) -> None:
        self.closed = True


class FakeWindow:
    """Records the pictures; ``commands`` are the answers of successive ``poll`` calls."""

    def __init__(self, commands: list[str | None] | None = None) -> None:
        self.commands = list(commands or [])
        self.shown: list[np.ndarray] = []
        self.polls = 0
        self.closed = False

    def show(self, image: np.ndarray) -> None:
        self.shown.append(image)

    def poll(self, delay_ms: int) -> str | None:
        self.polls += 1
        return self.commands.pop(0) if self.commands else "quit"

    def close(self) -> None:
        self.closed = True


class Detector:
    class_names = ["player"]

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.detections = detections if detections is not None else [detection()]
        self.confidence = 0.25

    def predict(self, image: np.ndarray) -> list[Detection]:
        return self.detections


def frame(width: int = 640, height: int = 360, value: int = 90) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_letterbox_keeps_aspect_ratio_and_centres() -> None:
    canvas, scale, left, top = letterbox(frame(640, 360), 300)
    assert canvas.shape == (300, 300, 3)
    assert scale == pytest.approx(300 / 640)
    assert left == 0 and top == (300 - round(360 * scale)) // 2
    assert tuple(canvas[0, 0]) == PADDING_COLOR  # padding bar above the picture
    assert tuple(canvas[150, 150]) == (90, 90, 90)  # picture


def test_render_draws_box_and_label_in_the_requested_size() -> None:
    plain, _, _, _ = letterbox(frame(), 300)
    image = render_preview(frame(), [detection()], 300)
    assert image.shape == (300, 300, 3) and image.dtype == np.uint8
    changed = np.any(image != plain, axis=2)
    assert changed.sum() > 100  # box outline + label background and text
    # the box outline sits at the scaled x1 (100 px * 300/640) on the box height
    assert np.any(changed[:, round(100 * 300 / 640)])


def test_render_other_sizes_and_empty_detections() -> None:
    assert render_preview(frame(), [], 200).shape == (200, 200, 3)
    assert render_preview(frame(100, 400), [detection(size=(100, 400), x2=90, y2=380)], 300).shape == (300, 300, 3)
    plain, _, _, _ = letterbox(frame(), 300)
    assert np.array_equal(render_preview(frame(), [], 300), plain)


def test_label_stays_inside_the_picture() -> None:
    right_edge = detection(x1=600, y1=100, x2=640, y2=150)
    assert render_preview(frame(), [right_edge], 300).shape == (300, 300, 3)  # label clipped, no exception
    # box touching the top edge of a square picture: the label background goes below the edge, inside the box
    square = frame(300, 300)
    canvas = render_preview(square, [detection(x1=0, y1=0, x2=50, y2=40, size=(300, 300))], 300)
    assert np.any(np.all(canvas[3:8, 5:45] == (120, 200, 80), axis=2))


def test_note_is_drawn() -> None:
    without = render_preview(frame(), [], 300)
    with_note = render_preview(frame(), [], 300, note="1 det | 5 fps")
    assert not np.array_equal(without, with_note)


def test_preview_loop_shows_every_frame_then_holds_until_closed() -> None:
    source, window, detector = ListSource([frame(), frame()]), FakeWindow([None, None, None, "quit"]), Detector()
    frames = run_preview(source, detector, window, size=300, fps=1000)
    assert frames == 2 and len(window.shown) == 2
    assert all(image.shape == (300, 300, 3) for image in window.shown)
    assert window.polls == 4  # 2 frames + hold loop (source over) until "quit"
    assert source.closed and window.closed


def test_quit_stops_immediately() -> None:
    source, window = ListSource([frame()] * 5), FakeWindow(["quit"])
    assert run_preview(source, Detector(), window, fps=1000) == 1
    assert source.closed and window.closed and window.polls == 1


def test_limits_and_confidence_keys() -> None:
    detector = Detector()
    run_preview(ListSource([frame()] * 4), detector, FakeWindow(["+", "+", "-", "quit"]), fps=1000)
    assert detector.confidence == pytest.approx(0.25 + CONFIDENCE_STEP)

    detector.confidence = 0.06
    run_preview(ListSource([frame()] * 3), detector, FakeWindow(["-", "quit"]), fps=1000)
    assert detector.confidence == 0.05  # lower bound

    assert run_preview(ListSource([frame()] * 9), Detector(), FakeWindow([None] * 9), fps=1000, max_frames=3) == 3
    assert run_preview(ListSource([frame()] * 9), Detector(), FakeWindow([None] * 9), fps=1000, duration=0) == 0


def test_window_source_requires_a_title() -> None:
    with pytest.raises(ValueError, match="window"):
        create_source(SourceConfig(type="window"))


def test_preview_command_is_known_to_the_parser() -> None:
    from src.cli.parser import build_parser

    argv = ["preview", "--source", "window", "--window", "Jeu", "--size", "250", "--conf", "0.4"]
    args = build_parser().parse_args(argv)
    assert (args.source, args.window, args.size, args.conf) == ("window", "Jeu", 250, 0.4)
    assert cv2.__version__  # OpenCV with GUI support is importable


@pytest.mark.skipif(sys.platform != "win32", reason="monitor selection is Windows-only")
def test_screen_source_selects_a_monitor() -> None:
    from src.capture.sources import ScreenSource
    from src.capture.windows import enable_dpi_awareness, list_screens

    enable_dpi_awareness()
    screens = list_screens()
    shot = ScreenSource(screen=len(screens) - 1).read()
    width, height = screens[-1].size
    assert shot.shape == (height, width, 3)
    with pytest.raises(ValueError, match="inexistant"):
        ScreenSource(screen=len(screens))


def test_screen_option_is_known_to_the_parser() -> None:
    from src.cli.parser import build_parser

    args = build_parser().parse_args(["capture", "--source", "screen", "--screen", "2"])
    assert (args.source, args.screen) == ("screen", 2)
