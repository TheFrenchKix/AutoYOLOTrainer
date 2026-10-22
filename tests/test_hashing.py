from pathlib import Path

import numpy as np

from src.utils.hashing import HashIndex, find_duplicates, hamming_many, phash
from src.utils.images import read_image, write_image
from tests.conftest import random_image


def test_hamming_many() -> None:
    distances = hamming_many(0b1011, np.array([0b1011, 0b0000, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64))
    assert distances.tolist() == [0, 3, 61]


def test_phash_stable_on_small_changes() -> None:
    img = random_image(1)
    brighter = (img.astype(int) + 5).clip(0, 255).astype(np.uint8)
    assert bin(phash(img) ^ phash(brighter)).count("1") <= 4
    assert bin(phash(img) ^ phash(random_image(2))).count("1") > 10


def test_find_duplicates(tmp_path: Path) -> None:
    paths = []
    for i, seed in enumerate([1, 2, 1, 3]):
        path = tmp_path / f"{i}.png"
        write_image(path, random_image(seed))
        paths.append(path)
    duplicates = find_duplicates(paths)
    assert duplicates == [(paths[2], paths[0])]


def test_hash_index_persistence(tmp_path: Path) -> None:
    storage = tmp_path / "hashes.json"
    index = HashIndex(storage=storage)
    assert index.check_and_add(random_image(1))
    assert not index.check_and_add(random_image(1))
    assert index.check_and_add(random_image(2))
    index.save()

    reloaded = HashIndex(storage=storage)
    assert len(reloaded) == 2
    assert not reloaded.check_and_add(random_image(2))


def test_hash_index_memory_is_bounded() -> None:
    index = HashIndex(memory_size=3)
    for seed in range(10):
        index.add(phash(random_image(seed)))
    assert len(index) == 3


def test_unicode_path_io(tmp_path: Path) -> None:
    path = tmp_path / "éèà" / "image_é.png"
    write_image(path, random_image(4))
    assert read_image(path) is not None
