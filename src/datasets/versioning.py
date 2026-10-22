"""Generic ``v1/ v2/ v3/...`` version registry, used for datasets and models.

Layout::

    <base>/v1/  <base>/v2/ ...  <base>/active.json

``active.json`` points to the version currently in use; every version folder
is kept (or zipped into an archive folder) so the full history is preserved.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

from src.utils.files import read_json, write_json

_VERSION_DIR = re.compile(r"^v(\d+)$")


class VersionRegistry:
    """Manage numbered version folders inside ``base_dir``."""

    ACTIVE_FILE = "active.json"

    def __init__(self, base_dir: Path, archive_dir: Path | None = None) -> None:
        self.base_dir = base_dir
        self.archive_dir = archive_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def name(version: int) -> str:
        return f"v{version}"

    def path(self, version: int) -> Path:
        return self.base_dir / self.name(version)

    def versions(self) -> list[int]:
        """Existing versions (folders and archives), ascending."""
        found = {
            int(m.group(1))
            for p in self.base_dir.iterdir()
            if p.is_dir() and (m := _VERSION_DIR.match(p.name))
        }
        if self.archive_dir and self.archive_dir.exists():
            found |= {
                int(m.group(1))
                for p in self.archive_dir.glob("v*.zip")
                if (m := _VERSION_DIR.match(p.stem))
            }
        return sorted(found)

    def latest(self) -> int | None:
        versions = self.versions()
        return versions[-1] if versions else None

    def create(self) -> tuple[int, Path]:
        """Reserve the next version number and create its folder."""
        version = (self.latest() or 0) + 1
        path = self.path(version)
        path.mkdir(parents=True, exist_ok=False)
        return version, path

    # -- active pointer ------------------------------------------------------
    def active(self) -> int | None:
        data = read_json(self.base_dir / self.ACTIVE_FILE, default=None)
        return int(data["version"]) if data else None

    def active_path(self) -> Path | None:
        version = self.active()
        return self.path(version) if version is not None else None

    def set_active(self, version: int) -> None:
        if not self.path(version).exists():
            raise FileNotFoundError(f"La version {self.name(version)} n'existe pas")
        write_json(
            self.base_dir / self.ACTIVE_FILE,
            {"version": version, "updated_at": datetime.now().isoformat(timespec="seconds")},
        )

    # -- archiving -----------------------------------------------------------
    def archive(self, version: int) -> Path:
        """Zip a version into ``archive_dir`` and delete the folder."""
        if self.archive_dir is None:
            raise RuntimeError("Aucun dossier d'archive configuré")
        if version == self.active():
            raise ValueError("Impossible d'archiver la version active")
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        source = self.path(version)
        archive = shutil.make_archive(str(self.archive_dir / self.name(version)), "zip", source)
        shutil.rmtree(source)
        return Path(archive)

    def archive_old(self, keep: int) -> list[Path]:
        """Archive every unzipped version except the ``keep`` newest and the active one."""
        on_disk = [v for v in self.versions() if self.path(v).exists()]
        active = self.active()
        candidates = [v for v in on_disk[:-keep] if v != active] if keep > 0 else []
        return [self.archive(v) for v in candidates]
