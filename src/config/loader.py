"""Configuration loading through Hydra's compose API.

We use ``compose`` rather than ``@hydra.main`` so that Hydra does not take over
the CLI or change the working directory: the configuration stays a plain,
typed :class:`AppConfig` object that every module can receive explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.config.schema import AppConfig

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR: Path = PROJECT_ROOT / "config"


def load_config(
    overrides: Iterable[str] | None = None,
    config_dir: Path | None = None,
    config_name: str = "config",
) -> AppConfig:
    """Load and validate the YAML configuration.

    Args:
        overrides: Hydra-style overrides, e.g. ``["training.epochs=10"]``.
        config_dir: Directory containing ``<config_name>.yaml``.
        config_name: Name of the root config file, without extension.

    Returns:
        A fully typed :class:`AppConfig` instance.
    """
    directory = (config_dir or DEFAULT_CONFIG_DIR).resolve()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        raw = compose(config_name=config_name, overrides=list(overrides or []))

    schema = OmegaConf.structured(AppConfig)
    merged = OmegaConf.merge(schema, raw)
    config = OmegaConf.to_object(merged)
    assert isinstance(config, AppConfig)
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    """Cross-field checks that the type system cannot express."""
    lab = config.labeling
    if not 0.0 <= lab.confidence_review <= lab.confidence_accept <= 1.0:
        raise ValueError(
            "labeling: il faut 0 <= confidence_review <= confidence_accept <= 1 "
            f"(reçu review={lab.confidence_review}, accept={lab.confidence_accept})"
        )
    if lab.aggregation not in {"min", "mean"}:
        raise ValueError(f"labeling.aggregation invalide: {lab.aggregation}")
    if lab.empty_policy not in {"accepted", "review", "rejected"}:
        raise ValueError(f"labeling.empty_policy invalide: {lab.empty_policy}")
    if not 0.0 < config.training.val_ratio < 1.0:
        raise ValueError("training.val_ratio doit être dans ]0, 1[")
    if not config.project.classes:
        raise ValueError("project.classes ne peut pas être vide")


def to_yaml(config: AppConfig) -> str:
    """Serialize a configuration back to YAML (useful for run snapshots)."""
    return OmegaConf.to_yaml(OmegaConf.structured(config))
