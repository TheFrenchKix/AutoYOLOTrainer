"""Capture loop: source -> detector -> CaptureManager, at a fixed frame rate."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from src.capture.capture_manager import CaptureManager
from src.capture.sources import FrameSource
from src.model.types import Detector
from src.utils.logging import get_logger

log = get_logger("capture")


class CaptureLoop:
    """Run inference on every frame and hand detections to the CaptureManager.

    Errors on a single frame are logged and skipped so the loop can run for
    weeks; :meth:`stop` (or Ctrl+C) ends it cleanly.
    """

    def __init__(
        self,
        source: FrameSource,
        detector: Detector,
        manager: CaptureManager,
        fps: float = 5.0,
        on_frame: Callable[[int], None] | None = None,
    ) -> None:
        self.source = source
        self.detector = detector
        self.manager = manager
        self.period = 1.0 / fps if fps > 0 else 0.0
        self.on_frame = on_frame
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self, max_frames: int | None = None, duration: float | None = None) -> None:
        start = time.monotonic()
        frames = 0
        consecutive_errors = 0
        log.info("Boucle de capture démarrée")
        try:
            while not self._stop.is_set():
                tick = time.monotonic()
                if duration is not None and tick - start >= duration:
                    break
                frame = self.source.read()
                if frame is None:
                    log.info("Source épuisée")
                    break
                try:
                    self.manager.on_detection(frame, self.detector.predict(frame))
                    consecutive_errors = 0
                except Exception as exc:
                    consecutive_errors += 1
                    log.error(f"Erreur sur la frame {frames}: {exc}")
                    if consecutive_errors >= 100:
                        raise RuntimeError("100 erreurs consécutives, arrêt de la capture") from exc
                frames += 1
                if self.on_frame:
                    self.on_frame(frames)
                if max_frames is not None and frames >= max_frames:
                    break
                self._stop.wait(max(0.0, self.period - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            log.info("Interruption utilisateur")
        finally:
            self.source.close()
            self.manager.close()
