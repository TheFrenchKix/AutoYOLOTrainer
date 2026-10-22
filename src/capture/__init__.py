"""Automatic capture of detection frames with active-learning filtering."""

from src.capture.active_learning import ActiveLearningSelector, InterestScore
from src.capture.capture_manager import CaptureManager

__all__ = ["ActiveLearningSelector", "CaptureManager", "InterestScore"]
