"""Cross-process lock file (prevents two trainings from running at once)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import TracebackType


class LockError(RuntimeError):
    pass


class FileLock:
    """Exclusive lock based on atomic file creation.

    A lock older than ``stale_after`` seconds is considered abandoned (crash,
    power loss) and is taken over. We deliberately do not probe the PID: on
    Windows ``os.kill(pid, 0)`` would *terminate* the process.
    """

    def __init__(self, path: Path, stale_after: float = 48 * 3600) -> None:
        self.path = path
        self.stale_after = stale_after
        self._held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and time.time() - self.path.stat().st_mtime > self.stale_after:
            self.path.unlink(missing_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise LockError(f"Verrou déjà pris: {self.path}") from exc
        with os.fdopen(fd, "w") as handle:
            handle.write(f"{os.getpid()} {time.time()}")
        self._held = True

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    @property
    def locked(self) -> bool:
        return self.path.exists()

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
