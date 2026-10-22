"""Watch the captures folder and auto-label images as soon as they appear."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.labeler.auto_labeler import AutoLabeler
from src.utils.files import IMAGE_EXTENSIONS
from src.utils.logging import get_logger

log = get_logger("dataset")


class _ImageEventHandler(FileSystemEventHandler):
    def __init__(self, pending: queue.Queue[Path]) -> None:
        self.pending = pending

    def _submit(self, raw_path: object, is_directory: bool) -> None:
        path = Path(str(raw_path))
        if not is_directory and path.suffix.lower() in IMAGE_EXTENSIONS:
            self.pending.put(path)

    def on_created(self, event: FileSystemEvent) -> None:
        self._submit(event.src_path, event.is_directory)

    def on_moved(self, event: FileSystemEvent) -> None:
        # Files written atomically (temp file + rename) only emit a "moved" event.
        self._submit(event.dest_path, event.is_directory)


class CaptureWatcher:
    """Run the :class:`AutoLabeler` on every new image of ``folder``.

    Images are processed after a short ``settle_seconds`` delay so that the
    capture has finished writing the image *and* its sidecar files.
    """

    def __init__(
        self,
        folder: Path,
        labeler: AutoLabeler,
        settle_seconds: float = 1.0,
        on_labeled: Callable[[], None] | None = None,
    ) -> None:
        self.folder = folder
        self.labeler = labeler
        self.settle_seconds = settle_seconds
        self.on_labeled = on_labeled
        self._pending: queue.Queue[Path] = queue.Queue()
        self._stop = threading.Event()
        self._observer = Observer()

    def start(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        # Images captured while the watcher was down are labeled first.
        self.labeler.process_folder(self.folder)
        self._observer.schedule(_ImageEventHandler(self._pending), str(self.folder), recursive=False)
        self._observer.start()
        threading.Thread(target=self._worker, name="autolabel-worker", daemon=True).start()
        log.info(f"Surveillance de {self.folder}")

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                path = self._pending.get(timeout=0.5)
            except queue.Empty:
                continue
            time.sleep(self.settle_seconds)
            if not path.exists():
                continue
            try:
                self.labeler.label_image(path)
                if self.on_labeled:
                    self.on_labeled()
            except Exception as exc:
                log.error(f"Auto-labeling échoué pour {path.name}: {exc}")

    def stop(self) -> None:
        self._stop.set()
        self._observer.stop()
        self._observer.join(timeout=5)
