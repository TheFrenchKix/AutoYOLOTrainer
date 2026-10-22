"""Auto-labeling: confidence triage of captured images."""

from src.labeler.auto_labeler import AutoLabeler, LabelResult
from src.labeler.router import ConfidenceRouter, Decision

__all__ = ["AutoLabeler", "ConfidenceRouter", "Decision", "LabelResult"]
