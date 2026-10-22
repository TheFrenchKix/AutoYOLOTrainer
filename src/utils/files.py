"""Filesystem helpers shared by every module."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(folder: Path, recursive: bool = False) -> list[Path]:
    """Return image files of ``folder`` sorted by name (empty if missing)."""
    if not folder.exists():
        return []
    iterator: Iterator[Path] = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in iterator if is_image(p))


def atomic_write_text(path: Path, content: str) -> None:
    """Write a file atomically (write to a temp file then rename).

    Atomic replacement also means a file hard-linked into a dataset snapshot is
    never modified in place: the snapshot keeps the old content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False, default=str))


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link ``src`` to ``dst`` (no extra disk space), fall back to a copy.

    Hard links work on NTFS as long as both paths are on the same volume.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def move_with_sidecars(image: Path, dest_dir: Path, suffixes: tuple[str, ...] = (".txt", ".json")) -> Path:
    """Move an image and its sidecar files (same stem) into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / image.name
    shutil.move(str(image), target)
    for suffix in suffixes:
        sidecar = image.with_suffix(suffix)
        if sidecar.exists():
            shutil.move(str(sidecar), dest_dir / sidecar.name)
    return target


def folder_size(path: Path) -> int:
    """Total size in bytes of every file below ``path``."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"
