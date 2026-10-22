"""Configuration: typed schema, Hydra loader and workspace layout."""

from src.config.loader import PROJECT_ROOT, load_config
from src.config.schema import AppConfig
from src.config.workspace import Workspace

__all__ = ["AppConfig", "PROJECT_ROOT", "Workspace", "load_config"]
