"""Evaluation metrics and model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

METRIC_NAMES: tuple[str, ...] = ("map50", "map50_95", "precision", "recall", "f1")


@dataclass(frozen=True)
class Metrics:
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    def get(self, name: str) -> float:
        if name not in METRIC_NAMES:
            raise KeyError(f"Métrique inconnue: {name} (connues: {METRIC_NAMES})")
        return float(getattr(self, name))

    def to_dict(self) -> dict[str, float]:
        return {name: round(self.get(name), 5) for name in METRIC_NAMES}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metrics:
        return cls(**{k: float(data.get(k, 0.0)) for k in ("map50", "map50_95", "precision", "recall")})

    @classmethod
    def from_ultralytics(cls, results: Any) -> Metrics:
        """Build from the ``DetMetrics`` object returned by ``YOLO.val()``."""
        box = results.box
        return cls(map50=float(box.map50), map50_95=float(box.map),
                   precision=float(box.mp), recall=float(box.mr))


def is_better(
    candidate: Metrics,
    reference: Metrics | None,
    primary_metric: str = "map50_95",
    min_improvement: float = 0.0,
) -> bool:
    """``True`` if ``candidate`` beats ``reference`` by at least ``min_improvement``.

    Ties on the primary metric are broken by F1 so that an equal mAP with a
    better precision/recall balance is still promoted.
    """
    if reference is None:
        return True
    delta = candidate.get(primary_metric) - reference.get(primary_metric)
    if delta > min_improvement:
        return True
    return delta == 0 and min_improvement == 0 and candidate.f1 > reference.f1
