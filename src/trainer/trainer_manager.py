"""TrainerManager: initial training, automatic retraining and model promotion.

Retraining cycle::

    accepted/ --import--> datasets/current --snapshot--> datasets/vN (+ offline augmentation)
        --train--> runs/<name> --evaluate candidate & active on vN/val--> compare
        --register--> models/vM --(if better)--> active

The candidate and the active model are evaluated on the *same* validation set
so the comparison is fair even though the dataset grows between versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.config.schema import AppConfig
from src.config.workspace import Workspace
from src.datasets.augmentation import AugmentationPipeline
from src.datasets.manager import DatasetManager
from src.model.registry import ModelRegistry
from src.trainer.backend import TrainingBackend, UltralyticsBackend
from src.trainer.metrics import Metrics, is_better
from src.trainer.report import write_evaluation_report
from src.utils.files import list_images, read_json, write_json
from src.utils.lock import FileLock
from src.utils.logging import get_logger

log = get_logger("training")


@dataclass
class TrainingOutcome:
    dataset_version: int
    model_version: int
    candidate: Metrics
    reference: Metrics | None
    reference_version: int | None
    promoted: bool
    run_dir: Path
    report: Path | None = None
    notes: list[str] = field(default_factory=list)


class TrainerManager:
    STATE_FILE = "trainer_state.json"

    def __init__(
        self,
        config: AppConfig,
        workspace: Workspace,
        backend: TrainingBackend | None = None,
        datasets: DatasetManager | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.backend = backend or UltralyticsBackend(config.training, config.augmentation.yolo_online)
        self.datasets = datasets or DatasetManager(
            workspace.current_dataset, config.class_names, workspace.datasets,
            workspace.archive, config.training.val_ratio,
        )
        self.registry = registry or ModelRegistry(workspace.models)
        self.lock = FileLock(workspace.state / "training.lock")
        self.state_path = workspace.state / self.STATE_FILE

    # ------------------------------------------------------------------ state
    def _state(self) -> dict:
        return read_json(self.state_path, default={}) or {}

    def _update_state(self, **values: object) -> None:
        write_json(self.state_path, {**self._state(), **values})

    def pending_images(self) -> int:
        """New images not yet used for training (in accepted/ or imported since last run)."""
        return len(list_images(self.workspace.accepted)) + int(self._state().get("imported_since_training", 0))

    def should_retrain(self) -> bool:
        return self.config.training.auto_train and self.pending_images() >= self.config.training.min_new_images

    # --------------------------------------------------------------- training
    def train_initial(self, epochs: int | None = None, base_model: str | None = None) -> TrainingOutcome:
        """First training (YOLO11n by default) on the manually annotated dataset."""
        if not self.datasets.images():
            raise RuntimeError(
                f"Dataset vide ({self.workspace.current_dataset}). "
                "Annotez des images puis lancez scripts/import_labelstudio.py"
            )
        return self._run_cycle(
            weights=base_model or self.config.training.base_model,
            epochs=epochs or self.config.training.epochs,
            note="initial",
            force_promote=self.registry.active_version() is None,
        )

    def retrain(self, force: bool = False) -> TrainingOutcome | None:
        """Import accepted images, train a candidate and promote it if better."""
        pending = self.pending_images()
        if not force and pending < self.config.training.min_new_images:
            log.info(f"Pas de ré-entraînement: {pending}/{self.config.training.min_new_images} nouvelles images")
            return None
        imported = self.datasets.import_folder(self.workspace.accepted, move=True)
        self._update_state(imported_since_training=int(self._state().get("imported_since_training", 0)) + imported)

        active = self.registry.active_weights()
        tcfg = self.config.training
        weights = str(active) if (active and tcfg.fine_tune_from_active) else tcfg.base_model
        return self._run_cycle(weights=weights, epochs=tcfg.retrain_epochs, note=f"retrain +{pending} images")

    def maybe_retrain(self) -> TrainingOutcome | None:
        """Retrain only if enough new images exist and no training is running."""
        if not self.should_retrain() or self.lock.locked:
            return None
        return self.retrain()

    def _run_cycle(self, weights: str, epochs: int, note: str, force_promote: bool = False) -> TrainingOutcome:
        with self.lock:
            dataset_version, dataset_dir = self.datasets.create_version(note=note)
            data_yaml = dataset_dir / "data.yaml"
            aug = self.config.augmentation
            if aug.enabled and aug.offline_copies > 0:
                AugmentationPipeline(aug, seed=self.config.training.seed).augment_split(dataset_dir)

            name = f"train_d{dataset_version}_{datetime.now():%Y%m%d_%H%M%S}"
            log.info(f"Entraînement {name}: poids={weights}, epochs={epochs}, dataset=v{dataset_version}")
            result = self.backend.train(data_yaml, weights, self.workspace.runs, name, epochs)
            outcome = self._evaluate_and_register(result.run_dir, data_yaml, dataset_version, name, force_promote)
            outcome.notes.append(note)
            self._update_state(
                imported_since_training=0,
                last_training=datetime.now().isoformat(timespec="seconds"),
                last_model_version=outcome.model_version,
                last_promoted=outcome.promoted,
            )
            return outcome

    def _evaluate_and_register(
        self, run_dir: Path, data_yaml: Path, dataset_version: int, name: str, force_promote: bool
    ) -> TrainingOutcome:
        eval_dir = self.workspace.runs / "eval"
        candidate = self.backend.evaluate(run_dir / "weights" / "best.pt", data_yaml, eval_dir, f"{name}_candidate")

        reference_version = self.registry.active_version()
        active_weights = self.registry.active_weights()
        reference = (
            self.backend.evaluate(active_weights, data_yaml, eval_dir, f"{name}_active")
            if active_weights else None
        )
        ecfg = self.config.evaluation
        promoted = force_promote or is_better(candidate, reference, ecfg.primary_metric, ecfg.min_improvement)

        model_version = self.registry.register(
            run_dir / "weights", candidate.to_dict(), dataset_version, run_dir,
            extra={"reference_version": reference_version,
                   "reference_metrics": reference.to_dict() if reference else None},
        )
        if promoted:
            self.registry.promote(model_version)
        verdict = "PROMU" if promoted else f"rejeté (v{reference_version} conservé)"
        log.info(f"Modèle v{model_version} {verdict} | candidat={candidate.to_dict()} "
                 f"| actif={reference.to_dict() if reference else None}")

        report = write_evaluation_report(
            self.registry.versions.path(model_version) / "report.html",
            model_version, candidate, reference_version, reference, promoted,
            self.registry.history(), ecfg.primary_metric,
        )
        return TrainingOutcome(dataset_version, model_version, candidate, reference,
                               reference_version, promoted, run_dir, report)
