"""Perceptual hashing and near-duplicate detection.

Hashes are stored as 64-bit integers so that Hamming distances against a whole
index can be computed in one vectorised NumPy operation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from src.utils.files import read_json, write_json

ImageLike = Path | Image.Image | np.ndarray


def _to_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        # OpenCV frames are BGR; channel order barely matters for a pHash of
        # the luminance, but convert anyway for consistency.
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1]
        return Image.fromarray(np.ascontiguousarray(image))
    with Image.open(image) as img:
        return img.convert("RGB")


def phash(image: ImageLike, hash_size: int = 8) -> int:
    """Perceptual hash of an image as an integer (``hash_size**2`` bits)."""
    value = imagehash.phash(_to_pil(image), hash_size=hash_size)
    return int(str(value), 16)


def hamming_many(value: int, others: np.ndarray) -> np.ndarray:
    """Hamming distances between one hash and an array of ``uint64`` hashes."""
    if others.size == 0:
        return np.empty(0, dtype=np.int64)
    xor = np.bitwise_xor(others, np.uint64(value & 0xFFFFFFFFFFFFFFFF))
    return np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


class HashIndex:
    """Bounded memory of recently seen hashes, optionally persisted to JSON.

    Only 64-bit hashes (``hash_size=8``) are vectorised; this is the default
    and gives a good precision/speed trade-off for duplicate detection.
    """

    def __init__(
        self,
        max_distance: int = 6,
        memory_size: int = 5000,
        hash_size: int = 8,
        storage: Path | None = None,
    ) -> None:
        if hash_size != 8:
            raise ValueError("HashIndex ne supporte que hash_size=8 (hash 64 bits)")
        self.max_distance = max_distance
        self.hash_size = hash_size
        self.storage = storage
        self._hashes: deque[int] = deque(maxlen=memory_size)
        if storage is not None:
            self._hashes.extend(int(h) for h in read_json(storage, default=[]))

    def __len__(self) -> int:
        return len(self._hashes)

    def _array(self) -> np.ndarray:
        return np.fromiter(self._hashes, dtype=np.uint64, count=len(self._hashes))

    def is_duplicate(self, value: int) -> bool:
        distances = hamming_many(value, self._array())
        return bool(distances.size and distances.min() <= self.max_distance)

    def add(self, value: int) -> None:
        self._hashes.append(value)

    def check_and_add(self, image: ImageLike) -> bool:
        """Return ``True`` if ``image`` is new (and remember it), else ``False``."""
        value = phash(image, self.hash_size)
        if self.is_duplicate(value):
            return False
        self.add(value)
        return True

    def save(self) -> None:
        if self.storage is not None:
            write_json(self.storage, [str(h) for h in self._hashes])


def find_duplicates(
    images: Iterable[Path], max_distance: int = 6, hash_size: int = 8
) -> list[tuple[Path, Path]]:
    """Return ``(duplicate, original)`` pairs among ``images``.

    The first image of a group of near-identical images is kept as the
    original; every later one is reported as its duplicate.
    """
    kept_paths: list[Path] = []
    kept_hashes: list[int] = []
    duplicates: list[tuple[Path, Path]] = []
    for path in images:
        value = phash(path, hash_size)
        distances = hamming_many(value, np.array(kept_hashes, dtype=np.uint64))
        if distances.size and distances.min() <= max_distance:
            duplicates.append((path, kept_paths[int(distances.argmin())]))
        else:
            kept_paths.append(path)
            kept_hashes.append(value)
    return duplicates
