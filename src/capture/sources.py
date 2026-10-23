"""Frame sources: screen, window, camera, video file or image folder (all yield BGR arrays)."""

from __future__ import annotations

import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab

from src.config.schema import SourceConfig
from src.utils.files import list_images
from src.utils.images import read_image


class FrameSource(ABC):
    """Iterable of BGR frames. ``None`` from :meth:`read` means "no more frames"."""

    @abstractmethod
    def read(self) -> np.ndarray | None: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""

    def __iter__(self) -> Iterator[np.ndarray]:
        while (frame := self.read()) is not None:
            yield frame


class ScreenSource(FrameSource):
    """Screenshots of a monitor, a region or the whole desktop (native on Windows).

    ``region`` = [left, top, right, bottom]; ``screen`` = monitor number as listed by
    ``scripts/capture.py --list`` (0 = every monitor). Without either, every monitor is captured.
    ``zone`` keeps only the centred square of that area and grabs nothing else (much faster than a full
    screenshot); with a zone and no area given, the primary monitor is used. :attr:`reference` is then the
    longest side of the whole area, so that a detector can keep the object scale of a full frame.
    """

    def __init__(self, region: list[int] | None = None, screen: int | None = None, zone: int | None = None) -> None:
        self.bbox = tuple(region) if region else None
        self.reference: int | None = None
        if self.bbox is None and (screen is not None or zone):
            self.bbox = self._monitor_bbox(screen)
        if zone and self.bbox is not None:
            left, top, right, bottom = self.bbox
            self.reference = max(right - left, bottom - top)
            side = min(zone, right - left, bottom - top)
            left, top = left + (right - left - side) // 2, top + (bottom - top - side) // 2
            self.bbox = (left, top, left + side, top + side)

    @staticmethod
    def _monitor_bbox(screen: int | None) -> tuple[int, int, int, int]:
        """Rectangle of monitor ``screen``; the primary one when ``screen`` is ``None``."""
        if sys.platform != "win32":
            raise RuntimeError("Le choix de l'écran fonctionne uniquement sous Windows")
        from src.capture import windows  # lazy: Win32 bindings

        windows.enable_dpi_awareness()  # so monitor rectangles match the pixels that are grabbed
        screens = windows.list_screens()
        if screen is None:
            return next((s for s in screens if s.primary), screens[-1]).bbox
        if not 0 <= screen < len(screens):
            raise ValueError(f"Écran {screen} inexistant (écrans 0 à {len(screens) - 1})")
        return screens[screen].bbox

    def read(self) -> np.ndarray:
        if self.bbox is not None and sys.platform == "win32":
            from src.capture import windows

            image = windows.grab_region(self.bbox)  # only this rectangle is copied
        else:
            # all_screens: bbox is in virtual-desktop coordinates, which can be negative
            image = ImageGrab.grab(bbox=self.bbox, all_screens=True)
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


class BackgroundSource(FrameSource):
    """Reads a live source in a thread so that capture and inference overlap.

    :meth:`read` waits for a frame newer than the last one returned, so a slow consumer always gets the
    most recent picture and never a backlog. Meant for live sources (screen, window, camera), not files.
    """

    def __init__(self, source: FrameSource) -> None:
        self._source = source
        self._ready = threading.Condition()
        self._frame: np.ndarray | None = None
        self._produced = 0
        self._consumed = 0
        self._finished = False
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="frame-reader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self._source.read()
                with self._ready:
                    if frame is None:
                        break
                    self._frame, self._produced = frame, self._produced + 1
                    self._ready.notify_all()
        except BaseException as exc:  # handed over to the thread that calls read()
            self._error = exc
        finally:
            with self._ready:
                self._finished = True
                self._ready.notify_all()

    def read(self) -> np.ndarray | None:
        with self._ready:
            while self._produced <= self._consumed and not self._finished:
                self._ready.wait(0.5)  # short waits so that Ctrl+C is not blocked
            if self._produced > self._consumed:
                self._consumed = self._produced
                return self._frame
            if self._error is not None:
                raise self._error
            return None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._source.close()


class WindowSource(FrameSource):
    """Content of the first visible window whose title contains ``title`` (Windows only).

    The window is rendered by Windows itself, so other windows covering it do not show up. If it
    disappears (closed or minimised) the source waits a few seconds before reporting the end.
    """

    RETRIES = 10
    RETRY_DELAY = 0.5

    def __init__(self, title: str) -> None:
        if sys.platform != "win32":
            raise RuntimeError("La source window fonctionne uniquement sous Windows")
        from src.capture import windows  # lazy: Win32 bindings

        windows.enable_dpi_awareness()
        self._windows = windows
        self.title = title
        if self._find() is None:
            raise RuntimeError(f"Aucune fenêtre visible ne correspond à {title!r}")

    def _find(self):  # noqa: ANN202
        matches = self._windows.find_windows(self.title, self._windows.list_windows())
        return matches[0] if matches else None

    def read(self) -> np.ndarray | None:
        for _ in range(self.RETRIES):
            window = self._find()
            if window is not None:
                image = self._windows.capture_window(window)
                return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            time.sleep(self.RETRY_DELAY)
        return None


class VideoSource(FrameSource):
    """Webcam (integer index) or video file through OpenCV."""

    def __init__(self, target: int | str) -> None:
        # CAP_DSHOW opens Windows webcams much faster than the default backend.
        backend = cv2.CAP_DSHOW if isinstance(target, int) and sys.platform == "win32" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(target, backend)
        if not self.capture.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source vidéo {target!r}")

    def read(self) -> np.ndarray | None:
        ok, frame = self.capture.read()
        return frame if ok else None

    def close(self) -> None:
        self.capture.release()


class FolderSource(FrameSource):
    """Replay the images of a folder (handy for tests and offline batches)."""

    def __init__(self, folder: Path) -> None:
        self._files = iter(list_images(folder))

    def read(self) -> np.ndarray | None:
        for path in self._files:
            frame = read_image(path)
            if frame is not None:
                return frame
        return None


def create_source(config: SourceConfig, zone: int | None = None) -> FrameSource:
    """Source of the configuration. ``zone`` (centred square, in pixels) only applies to the screen source."""
    kind = config.type.lower()
    if kind == "screen":
        return ScreenSource(config.region, config.screen, zone)
    if kind == "window":
        if not config.window:
            raise ValueError("capture.source.window est requis pour type=window")
        return WindowSource(config.window)
    if kind == "camera":
        return VideoSource(config.camera_index)
    if kind == "video":
        if not config.path:
            raise ValueError("capture.source.path est requis pour type=video")
        return VideoSource(config.path)
    if kind == "folder":
        if not config.path:
            raise ValueError("capture.source.path est requis pour type=folder")
        return FolderSource(Path(config.path))
    raise ValueError(f"Source inconnue: {config.type} (screen | window | camera | video | folder)")
