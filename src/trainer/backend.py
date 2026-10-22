"""Training backends. The TrainerManager only depends on :class:`TrainingBackend`,
which keeps it testable without a GPU and open to other frameworks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.config.schema import TrainingConfig
from src.trainer.metrics import Metrics


@dataclass
class TrainResult:
    run_dir: Path

    @property
    def weights_dir(self) -> Path:
        return self.run_dir / "weights"


class TrainingBackend(Protocol):
    def train(self, data_yaml: Path, weights: str, project_dir: Path, name: str, epochs: int) -> TrainResult: ...

    def evaluate(self, weights: Path, data_yaml: Path, project_dir: Path, name: str) -> Metrics: ...


class UltralyticsBackend:
    """Train/evaluate YOLO11 with Ultralytics (AMP, auto batch, cache, TensorBoard)."""

    def __init__(self, config: TrainingConfig, online_augmentation: dict[str, Any] | None = None) -> None:
        self.config = config
        self.online_augmentation = dict(online_augmentation or {})

    def _enable_tensorboard(self) -> None:
        from ultralytics import settings

        if self.config.tensorboard and not settings.get("tensorboard", False):
            settings.update({"tensorboard": True})

    def train(self, data_yaml: Path, weights: str, project_dir: Path, name: str, epochs: int) -> TrainResult:
        from ultralytics import YOLO

        self._enable_tensorboard()
        cfg = self.config
        model = YOLO(weights)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            amp=cfg.amp,
            cache=cfg.cache,
            workers=cfg.workers,
            device=cfg.device,
            patience=cfg.patience,
            seed=cfg.seed,
            project=str(project_dir),
            name=name,
            exist_ok=False,
            plots=True,
            **self.online_augmentation,
        )
        return TrainResult(run_dir=Path(model.trainer.save_dir))

    def evaluate(self, weights: Path, data_yaml: Path, project_dir: Path, name: str) -> Metrics:
        from ultralytics import YOLO

        results = YOLO(str(weights)).val(
            data=str(data_yaml),
            split="val",
            imgsz=self.config.imgsz,
            batch=16,  # auto-batch (-1) is a training-only option
            device=self.config.device,
            workers=self.config.workers,
            project=str(project_dir),
            name=name,
            plots=False,
            verbose=False,
        )
        return Metrics.from_ultralytics(results)
