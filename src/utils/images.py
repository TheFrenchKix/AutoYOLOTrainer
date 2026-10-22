"""Unicode-safe image I/O.

``cv2.imread``/``cv2.imwrite`` fail silently on Windows when the path contains
non-ASCII characters (e.g. ``C:\\Users\\Jérôme``); going through NumPy avoids it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray | None:
    """Read an image as a BGR array, or ``None`` if unreadable."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    """Write a BGR array; the format is deduced from the file extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"Encodage impossible: {path}")
    buffer.tofile(str(path))
