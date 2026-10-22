"""Training, evaluation, comparison and promotion of YOLO models."""

from src.trainer.metrics import Metrics, is_better
from src.trainer.trainer_manager import TrainerManager, TrainingOutcome

__all__ = ["Metrics", "TrainerManager", "TrainingOutcome", "is_better"]
