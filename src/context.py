"""Application context: builds every component from the configuration.

The CLI, the scripts and the dashboard all go through :class:`AppContext`, so
wiring (paths, class names, thresholds) is written once.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from src.config import PROJECT_ROOT, AppConfig, Workspace, load_config
from src.datasets.manager import DatasetManager
from src.model.registry import ModelRegistry
from src.utils.logging import setup_logging


@dataclass
class AppContext:
    config: AppConfig
    workspace: Workspace

    @classmethod
    def create(cls, overrides: Iterable[str] | None = None, root: Path = PROJECT_ROOT) -> AppContext:
        config = load_config(overrides)
        workspace = Workspace.from_config(config, root)
        workspace.ensure()
        setup_logging(workspace.logs, config.logging)
        return cls(config, workspace)

    @cached_property
    def datasets(self) -> DatasetManager:
        return DatasetManager(
            self.workspace.current_dataset, self.config.class_names,
            self.workspace.datasets, self.workspace.archive, self.config.training.val_ratio,
        )

    @cached_property
    def registry(self) -> ModelRegistry:
        return ModelRegistry(self.workspace.models)

    @cached_property
    def trainer(self):  # noqa: ANN201 - lazy import keeps `main.py stats` fast
        from src.trainer.trainer_manager import TrainerManager

        return TrainerManager(self.config, self.workspace, datasets=self.datasets, registry=self.registry)

    def detector(self, weights: Path | None = None, confidence: float | None = None):  # noqa: ANN201
        """YOLO detector on the active model (or explicit ``weights``)."""
        from src.model.yolo_detector import YoloDetector

        weights = weights or self.registry.active_weights()
        if weights is None:
            raise RuntimeError("Aucun modèle actif. Lancez d'abord: python main.py train")
        return YoloDetector(
            weights,
            confidence=confidence if confidence is not None else self.config.capture.min_confidence,
            imgsz=self.config.training.imgsz,
            device=self.config.training.device,
        )

    def auto_labeler(self, detector=None):  # noqa: ANN001, ANN201
        from src.labeler.auto_labeler import AutoLabeler

        lab = self.config.labeling
        if detector is None and not lab.use_saved_predictions:
            detector = self.detector(confidence=lab.predict_confidence)
        version = self.registry.active_version()
        return AutoLabeler(detector, lab, self.workspace.triage_dirs, f"v{version}" if version else None)

    def capture_manager(self):  # noqa: ANN201
        from src.capture.active_learning import ActiveLearningSelector
        from src.capture.capture_manager import CaptureManager

        # Seed rarity statistics with the current dataset class distribution.
        per_class = self.datasets.count().per_class
        names = self.config.class_names
        selector = ActiveLearningSelector(
            self.config.active_learning,
            {names.index(n): c for n, c in per_class.items() if n in names},
        )
        version = self.registry.active_version()
        return CaptureManager(
            self.config.capture, self.config.active_learning, self.workspace.captures,
            self.workspace.state, selector, f"v{version}" if version else None,
        )
