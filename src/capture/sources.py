"""Frame sources: screen, camera, video file or image folder (all yield BGR arrays)."""

from __future__ import annotations

import sys
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
    """Screenshots via Pillow's ``ImageGrab`` (native on Windows)."""

    def __init__(self, region: list[int] | None = None) -> None:
        self.bbox = tuple(region) if region else None

    def read(self) -> np.ndarray:
        image = ImageGrab.grab(bbox=self.bbox, all_screens=self.bbox is None)
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


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


def create_source(config: SourceConfig) -> FrameSource:
    kind = config.type.lower()
    if kind == "screen":
        return ScreenSource(config.region)
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
    raise ValueError(f"Source inconnue: {config.type} (screen | camera | video | folder)")
