"""Typed configuration schema.

The YAML file in ``config/config.yaml`` is merged on top of these dataclasses by
OmegaConf, which gives us type validation and sensible defaults for every key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProjectConfig:
    name: str = "autoyolo"
    classes: list[str] = field(default_factory=lambda: ["object"])


@dataclass
class PathsConfig:
    datasets: str = "datasets"
    current_dataset: str = "datasets/current"
    archive: str = "datasets/archive"
    raw_images: str = "datasets/raw"
    captures: str = "captures"
    accepted: str = "accepted"
    review: str = "review"
    rejected: str = "rejected"
    models: str = "models"
    runs: str = "runs"
    logs: str = "logs"
    state: str = ".autoyolo"


@dataclass
class SourceConfig:
    type: str = "screen"
    camera_index: int = 0
    path: str | None = None
    region: list[int] | None = None
    fps: float = 5.0


@dataclass
class CaptureConfig:
    save_every_detection: bool = True
    max_images_per_hour: int = 1000
    min_interval_seconds: float = 0.5
    min_confidence: float = 0.25
    image_format: str = "png"
    save_labels: bool = True
    save_metadata: bool = True
    source: SourceConfig = field(default_factory=SourceConfig)


@dataclass
class DedupConfig:
    enabled: bool = True
    hash_size: int = 8
    max_hamming_distance: int = 6
    memory_size: int = 5000
    ssim_threshold: float = 0.97


@dataclass
class ActiveLearningWeights:
    rarity: float = 1.0
    small_objects: float = 1.0
    low_confidence: float = 1.5
    new_position: float = 1.0


@dataclass
class ActiveLearningConfig:
    enabled: bool = True
    score_threshold: float = 0.3
    exploration_rate: float = 0.02
    small_object_area: float = 0.01
    uncertainty_low: float = 0.25
    uncertainty_high: float = 0.85
    position_grid: int = 8
    weights: ActiveLearningWeights = field(default_factory=ActiveLearningWeights)
    dedup: DedupConfig = field(default_factory=DedupConfig)


@dataclass
class LabelingConfig:
    confidence_accept: float = 0.97
    confidence_review: float = 0.85
    aggregation: str = "min"
    empty_policy: str = "rejected"
    use_saved_predictions: bool = False
    predict_confidence: float = 0.25


@dataclass
class TrainingConfig:
    auto_train: bool = True
    min_new_images: int = 500
    base_model: str = "yolo11n.pt"
    epochs: int = 100
    retrain_epochs: int = 50
    imgsz: int = 640
    batch: Any = -1
    amp: bool = True
    cache: Any = True
    workers: int = 4
    device: str | None = None
    patience: int = 30
    val_ratio: float = 0.2
    fine_tune_from_active: bool = True
    check_interval_minutes: float = 30.0
    tensorboard: bool = True
    seed: int = 42


@dataclass
class EvaluationConfig:
    primary_metric: str = "map50_95"
    min_improvement: float = 0.0


@dataclass
class TransformConfig:
    enabled: bool = True
    p: float = 0.5
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    enabled: bool = True
    offline_copies: int = 1
    transforms: dict[str, TransformConfig] = field(default_factory=dict)
    mosaic: TransformConfig = field(default_factory=lambda: TransformConfig(p=0.3))
    mixup: TransformConfig = field(default_factory=lambda: TransformConfig(p=0.1))
    yolo_online: dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelStudioConfig:
    url: str = "http://localhost:8080"
    port: int = 8080
    username: str = "admin@autoyolo.local"
    password: str = "autoyolo-change-me"
    api_token: str = "autoyolo-dev-token-change-me"
    project_title: str = "AutoYOLO"
    review_project_title: str = "AutoYOLO - Review"
    venv_dir: str = ".venv-labelstudio"
    data_dir: str = ".labelstudio"


@dataclass
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class LoggingConfig:
    level: str = "INFO"
    rotation: str = "10 MB"
    retention: str = "30 days"


@dataclass
class CleanConfig:
    rejected_retention_days: int = 7
    keep_runs: int = 5
    keep_dataset_versions: int = 5


@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    labelstudio: LabelStudioConfig = field(default_factory=LabelStudioConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    clean: CleanConfig = field(default_factory=CleanConfig)

    @property
    def class_names(self) -> list[str]:
        return list(self.project.classes)
