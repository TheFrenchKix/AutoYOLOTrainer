"""Model registry: ``models/vN/{best.pt,last.pt,info.json}`` + active pointer."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.datasets.versioning import VersionRegistry
from src.utils.files import read_json, write_json


class ModelRegistry:
    """Keep every trained model; one of them is *active* (used for inference)."""

    WEIGHTS = ("best.pt", "last.pt")

    def __init__(self, models_dir: Path) -> None:
        self.versions = VersionRegistry(models_dir)

    def register(
        self,
        weights_dir: Path,
        metrics: dict[str, float],
        dataset_version: int | None = None,
        run_dir: Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> int:
        """Copy the weights of a training run into a new ``models/vN`` folder."""
        best = weights_dir / "best.pt"
        if not best.exists():
            raise FileNotFoundError(f"Poids introuvables: {best}")
        version, path = self.versions.create()
        for name in self.WEIGHTS:
            if (weights_dir / name).exists():
                shutil.copy2(weights_dir / name, path / name)
        write_json(path / "info.json", {
            "version": version,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_version": dataset_version,
            "run_dir": str(run_dir) if run_dir else None,
            "metrics": metrics,
            "promoted": False,
            **(extra or {}),
        })
        return version

    def promote(self, version: int) -> None:
        info_path = self.versions.path(version) / "info.json"
        info = read_json(info_path, default={})
        info["promoted"] = True
        info["promoted_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(info_path, info)
        self.versions.set_active(version)

    def active_version(self) -> int | None:
        return self.versions.active()

    def active_weights(self) -> Path | None:
        path = self.versions.active_path()
        return path / "best.pt" if path and (path / "best.pt").exists() else None

    def info(self, version: int) -> dict[str, Any]:
        return read_json(self.versions.path(version) / "info.json", default={})

    def update_info(self, version: int, **values: Any) -> None:
        """Merge ``values`` into ``models/vN/info.json`` (e.g. a speed measurement made after training)."""
        path = self.versions.path(version) / "info.json"
        write_json(path, {**read_json(path, default={}), **values})

    def architecture(self, version: int, _depth: int = 0) -> str:
        """Architecture of a version (``yolo11n``...), even for models trained before it was recorded.

        A model fine-tuned from another version inherits its architecture: the chain is followed through
        the ``model`` argument Ultralytics saved in the training run.
        """
        info = self.info(version)
        if info.get("architecture"):
            return str(info["architecture"])
        args_file = Path(info["run_dir"]) / "args.yaml" if info.get("run_dir") else None
        if args_file is None or not args_file.exists() or _depth > 20:
            return "inconnue"
        model = Path(str((yaml.safe_load(args_file.read_text(encoding="utf-8")) or {}).get("model", "")))
        parent = model.parent
        if parent.name.startswith("v") and parent.name[1:].isdigit() and parent.parent == self.versions.base_dir:
            return self.architecture(int(parent.name[1:]), _depth + 1)
        return model.stem or "inconnue"

    def active_metrics(self) -> dict[str, float] | None:
        version = self.active_version()
        return self.info(version).get("metrics") if version is not None else None

    def history(self) -> list[dict[str, Any]]:
        """Info of every model version, oldest first."""
        active = self.active_version()
        return [
            {**self.info(v), "version": v, "active": v == active, "architecture": self.architecture(v)}
            for v in self.versions.versions()
            if self.versions.path(v).exists()
        ]
