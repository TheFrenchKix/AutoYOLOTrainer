"""Background scheduler that periodically retrains when enough data is available."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from src.trainer.trainer_manager import TrainerManager, TrainingOutcome
from src.utils.logging import get_logger

log = get_logger("training")


class RetrainScheduler:
    """Check every ``interval_seconds`` whether a retraining is due.

    ``on_promoted`` receives the new active weights, typically
    ``YoloDetector.reload`` so the running capture uses the new model.
    """

    def __init__(
        self,
        trainer: TrainerManager,
        interval_seconds: float,
        on_promoted: Callable[[Path], None] | None = None,
    ) -> None:
        self.trainer = trainer
        self.interval = interval_seconds
        self.on_promoted = on_promoted
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="retrain-scheduler", daemon=True)

    def start(self) -> None:
        self._thread.start()
        log.info(f"Planificateur de ré-entraînement actif (toutes les {self.interval / 60:.0f} min)")

    def stop(self) -> None:
        self._stop.set()

    def check_now(self) -> TrainingOutcome | None:
        try:
            outcome = self.trainer.maybe_retrain()
        except Exception as exc:  # keep the scheduler alive whatever happens
            log.exception(f"Ré-entraînement échoué: {exc}")
            return None
        if outcome and outcome.promoted and self.on_promoted:
            weights = self.trainer.registry.active_weights()
            if weights:
                self.on_promoted(weights)
        return outcome

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.check_now()
