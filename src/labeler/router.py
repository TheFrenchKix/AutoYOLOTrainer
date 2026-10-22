"""Confidence-based triage decision (pure logic, trivially testable)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from statistics import fmean

from src.config.schema import LabelingConfig


class Decision(StrEnum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    REJECTED = "rejected"


class ConfidenceRouter:
    """Map the detection confidences of an image to a :class:`Decision`.

    * ``score > confidence_accept``                       -> accepted
    * ``confidence_review < score <= confidence_accept``  -> review
    * otherwise                                           -> rejected

    ``score`` is the min (conservative, default) or mean of the confidences:
    with ``min`` one doubtful box is enough to send the image to review.
    """

    def __init__(self, config: LabelingConfig) -> None:
        self.config = config

    def score(self, confidences: Sequence[float]) -> float | None:
        if not confidences:
            return None
        return min(confidences) if self.config.aggregation == "min" else fmean(confidences)

    def decide(self, confidences: Sequence[float]) -> Decision:
        score = self.score(confidences)
        if score is None:
            return Decision(self.config.empty_policy)
        if score > self.config.confidence_accept:
            return Decision.ACCEPTED
        if score > self.config.confidence_review:
            return Decision.REVIEW
        return Decision.REJECTED
