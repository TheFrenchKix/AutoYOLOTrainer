"""TrainerManager: initial training, automatic retraining and model promotion.

Retraining cycle::

    accepted/ --import--> datasets/current --snapshot--> datasets/vN (+ offline augmentation)
        --train--> runs/<name> --evaluate candidate & active on vN/val--> compare
        --register--> models/vM --(if better)--> active

The candidate and the active model are evaluated on the *same* validation set
so the comparison is fair even though the dataset grows between versions.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    def train_models(self, models: Sequence[str] | None = None, epochs: int | None = None) -> list[TrainingOutcome]:
        """Train one model per entry of ``models`` on the *same* snapshot of datasets/current, then compare them.

        Each candidate is evaluated against the model active at that moment, so the best one ends up active.
        Without ``models`` the configured ``training.base_model`` is trained.
        """
        if not self.datasets.images():
            raise RuntimeError(
                f"Dataset vide ({self.workspace.current_dataset}). "
                "Annotez des images puis lancez scripts/import_labelstudio.py"
            )
        weights = list(dict.fromkeys(models or [self.config.training.base_model]))
        note = "initial" if len(weights) == 1 else f"comparaison de {len(weights)} modèles"
        return self._run_cycle(
            weights, epochs or self.config.training.epochs, note,
            force_promote=self.registry.active_version() is None,
        )

    def train_initial(self, epochs: int | None = None, base_model: str | None = None) -> TrainingOutcome:
        """First training (YOLO11n by default) on the manually annotated dataset."""
        return self.train_models([base_model] if base_model else None, epochs)[0]

    def retrain_models(
        self, force: bool = False, models: Sequence[str] | None = None, epochs: int | None = None
    ) -> list[TrainingOutcome]:
        """Import accepted images, train the candidate(s) and promote the best one if it beats the active model.

        Without ``models`` the active model is fine-tuned (``training.fine_tune_from_active``). With ``models``
        each architecture is trained from its pretrained weights, to compare them on the same data.
        """
        pending = self.pending_images()
        if not force and pending < self.config.training.min_new_images:
            log.info(f"Pas de ré-entraînement: {pending}/{self.config.training.min_new_images} nouvelles images")
            return []
        imported = self.datasets.import_folder(self.workspace.accepted, move=True)
        self._update_state(imported_since_training=int(self._state().get("imported_since_training", 0)) + imported)

        tcfg = self.config.training
        if models:
            weights = list(dict.fromkeys(models))
        else:
            active = self.registry.active_weights()
            weights = [str(active) if (active and tcfg.fine_tune_from_active) else tcfg.base_model]
        return self._run_cycle(weights, epochs or tcfg.retrain_epochs, f"retrain +{pending} images")

    def retrain(self, force: bool = False) -> TrainingOutcome | None:
        """Retrain (fine-tuning the active model) if enough new images were accepted; ``None`` otherwise."""
        outcomes = self.retrain_models(force)
        return outcomes[0] if outcomes else None

    def maybe_retrain(self) -> TrainingOutcome | None:
        """Retrain only if enough new images exist and no training is running."""
        if not self.should_retrain() or self.lock.locked:
            return None
        return self.retrain()

    def benchmark_models(self, versions: Sequence[int] | None = None) -> dict[int, dict[str, float]]:
        """Measure the inference speed of registered models, one after the other, and store it in their info."""
        images = self._benchmark_images()
        if not images:
            raise RuntimeError("Aucune image de validation pour mesurer la vitesse (créez un dataset d'abord)")
        results: dict[int, dict[str, float]] = {}
        for version in versions or self.registry.versions.versions():
            weights = self.registry.versions.path(version) / "best.pt"
            if not weights.exists():
                continue
            speed = self.backend.benchmark(weights, images)
            if speed:
                self.registry.update_info(version, speed_ms=speed["inference_ms"], params_m=speed["params_m"])
                results[version] = speed
        return results

    def _benchmark_images(self) -> list[Path]:
        latest = self.datasets.versions.latest()
        folder = (self.datasets.versions.path(latest) if latest else self.workspace.current_dataset) / "images" / "val"
        return list_images(folder)

    def _measure_speed(self, weights: Path, images_dir: Path) -> dict[str, float]:
        """Speed of a fresh candidate; a failed measurement must not lose a finished training."""
        try:
            speed = self.backend.benchmark(weights, list_images(images_dir))
        except Exception as exc:
            log.warning(f"Mesure de vitesse impossible: {exc}")
            return {}
        return {"speed_ms": speed["inference_ms"], "params_m": speed["params_m"]} if speed else {}

    def _architecture(self, weights: str) -> str:
        """``yolo26s`` for pretrained weights; a fine-tuned registry model inherits its parent's architecture."""
        path = Path(weights)
        parent = path.parent
        in_registry = parent.parent == self.registry.versions.base_dir
        if in_registry and parent.name.startswith("v") and parent.name[1:].isdigit():
            return self.registry.architecture(int(parent.name[1:]))
        return path.stem

    def _run_cycle(
        self, weights: Sequence[str], epochs: int, note: str, force_promote: bool = False
    ) -> list[TrainingOutcome]:
        with self.lock:
            dataset_version, dataset_dir = self.datasets.create_version(note=note)
            data_yaml = dataset_dir / "data.yaml"
            aug = self.config.augmentation
            if aug.enabled and aug.offline_copies > 0:
                AugmentationPipeline(aug, seed=self.config.training.seed).augment_split(dataset_dir)

            outcomes: list[TrainingOutcome] = []
            for start in weights:
                name = f"train_d{dataset_version}_{datetime.now():%Y%m%d_%H%M%S}"
                if len(weights) > 1:
                    name += f"_{Path(start).stem}"
                log.info(f"Entraînement {name}: poids={start}, epochs={epochs}, dataset=v{dataset_version}")
                result = self.backend.train(data_yaml, start, self.workspace.runs, name, epochs)
                outcome = self._evaluate_and_register(
                    result.run_dir, data_yaml, dataset_version, name,
                    force_promote and self.registry.active_version() is None, self._architecture(start),
                )
                outcome.notes.append(note)
                outcomes.append(outcome)
            self._update_state(
                imported_since_training=0,
                last_training=datetime.now().isoformat(timespec="seconds"),
                last_model_version=outcomes[-1].model_version,
                last_promoted=any(o.promoted for o in outcomes),
            )
            return outcomes

    def _evaluate_and_register(
        self, run_dir: Path, data_yaml: Path, dataset_version: int, name: str, force_promote: bool,
        architecture: str = "inconnue",
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

        speed = self._measure_speed(run_dir / "weights" / "best.pt", data_yaml.parent / "images" / "val")
        model_version = self.registry.register(
            run_dir / "weights", candidate.to_dict(), dataset_version, run_dir,
            extra={"reference_version": reference_version,
                   "reference_metrics": reference.to_dict() if reference else None,
                   "architecture": architecture, **speed},
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
