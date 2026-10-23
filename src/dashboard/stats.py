"""Project-wide statistics, shared by ``main.py stats`` and the dashboard."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.schema import AppConfig
from src.config.workspace import Workspace
from src.datasets.manager import DatasetManager
from src.model.registry import ModelRegistry
from src.utils.files import folder_size, human_size, list_images, read_json


@dataclass
class ProjectStats:
    dataset_images: int = 0
    dataset_annotations: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    dataset_versions: list[int] = field(default_factory=list)
    captures_pending: int = 0
    captures_total: int = 0
    accepted: int = 0
    review: int = 0
    rejected: int = 0
    pending_for_training: int = 0
    min_new_images: int = 0
    active_model: int | None = None
    best_map50: float | None = None
    best_map50_95: float | None = None
    trainings: list[dict[str, Any]] = field(default_factory=list)
    disk_usage: dict[str, str] = field(default_factory=dict)
    disk_free: str = ""
    dataset_growth: list[dict[str, Any]] = field(default_factory=list)
    last_training: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def captures_per_day(index_file: Path) -> pd.DataFrame:
    """Daily capture counts from the CSV index written by the CaptureManager."""
    if not index_file.exists():
        return pd.DataFrame(columns=["day", "captures"])
    frame = pd.read_csv(index_file, parse_dates=["timestamp"])
    counts = frame.groupby(frame["timestamp"].dt.date).size().rename("captures")
    counts.index.name = "day"
    return counts.reset_index()


def collect_stats(config: AppConfig, workspace: Workspace, heavy: bool = True) -> ProjectStats:
    """Gather every metric shown by the CLI and the dashboard.

    Args:
        heavy: also compute folder sizes (walks the whole tree).
    """
    datasets = DatasetManager(workspace.current_dataset, config.class_names, workspace.datasets, workspace.archive)
    registry = ModelRegistry(workspace.models)
    counts = datasets.count()
    history = registry.history()

    stats = ProjectStats(
        dataset_images=counts.images,
        dataset_annotations=counts.boxes,
        per_class=counts.per_class,
        dataset_versions=datasets.versions.versions(),
        captures_pending=len(list_images(workspace.captures)),
        accepted=len(list_images(workspace.accepted)),
        review=len(list_images(workspace.review)),
        rejected=len(list_images(workspace.rejected)),
        min_new_images=config.training.min_new_images,
        active_model=registry.active_version(),
        trainings=history[-10:][::-1],
    )
    index_file = workspace.state / "captures_index.csv"
    if index_file.exists():
        stats.captures_total = max(0, sum(1 for _ in index_file.open(encoding="utf-8")) - 1)

    state = read_json(workspace.state / "trainer_state.json", default={}) or {}
    stats.pending_for_training = stats.accepted + int(state.get("imported_since_training", 0))
    stats.last_training = state.get("last_training")

    metrics = [h.get("metrics") or {} for h in history]
    stats.best_map50 = max((m.get("map50", 0) for m in metrics), default=None)
    stats.best_map50_95 = max((m.get("map50_95", 0) for m in metrics), default=None)

    for version in stats.dataset_versions:
        manifest = datasets.version_manifest(version)
        if manifest:
            stats.dataset_growth.append({
                "version": version, "images": manifest["counts"]["images"], "created_at": manifest["created_at"],
            })

    if heavy:
        for name in ("datasets", "captures", "accepted", "review", "rejected", "models", "runs"):
            stats.disk_usage[name] = human_size(folder_size(getattr(workspace, name)))
    stats.disk_free = human_size(shutil.disk_usage(workspace.root).free)
    return stats
