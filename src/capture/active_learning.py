"""Active-learning scoring: how *useful* would this frame be for training?

Four signals, each in ``[0, 1]``, combined with configurable weights:

* **rarity**        - the frame contains a class under-represented so far;
* **small_objects** - it contains small objects (hard for the detector);
* **low_confidence**- the model is uncertain (inside the uncertainty band);
* **new_position**  - an object appears in a grid cell rarely seen for its class.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from src.config.schema import ActiveLearningConfig
from src.model.types import Detection


@dataclass
class InterestScore:
    score: float
    components: dict[str, float] = field(default_factory=dict)

    @property
    def reasons(self) -> list[str]:
        """Signals that significantly contributed (for metadata/debugging)."""
        return [name for name, value in self.components.items() if value >= 0.5]


class ActiveLearningSelector:
    def __init__(self, config: ActiveLearningConfig, class_counts: Mapping[int, int] | None = None) -> None:
        self.config = config
        self.class_counts: Counter[int] = Counter(class_counts or {})
        self.position_counts: Counter[tuple[int, int, int]] = Counter()

    def _cell(self, det: Detection) -> tuple[int, int, int]:
        box = det.to_yolo()
        n = self.config.position_grid
        gx = min(n - 1, int(box.x_center * n))
        gy = min(n - 1, int(box.y_center * n))
        return det.class_id, gx, gy

    def _rarity(self, detections: Sequence[Detection]) -> float:
        if not self.class_counts:
            return 1.0
        most = max(self.class_counts.values())
        return max(1.0 - self.class_counts.get(d.class_id, 0) / most for d in detections)

    def _small(self, detections: Sequence[Detection]) -> float:
        threshold = self.config.small_object_area
        return float(any(d.to_yolo().area < threshold for d in detections))

    def _uncertainty(self, detections: Sequence[Detection]) -> float:
        low, high = self.config.uncertainty_low, self.config.uncertainty_high
        span = max(high - low, 1e-6)
        # Confidence == low -> 1 (very uncertain), == high -> 0, outside the band -> 0.
        values = [1.0 - (d.confidence - low) / span for d in detections if low <= d.confidence <= high]
        return max(values, default=0.0)

    def _novelty(self, detections: Sequence[Detection]) -> float:
        return max(1.0 / (1 + self.position_counts[self._cell(d)]) for d in detections)

    def score(self, detections: Sequence[Detection]) -> InterestScore:
        if not detections:
            return InterestScore(0.0)
        w = self.config.weights
        components = {
            "rarity": self._rarity(detections),
            "small_objects": self._small(detections),
            "low_confidence": self._uncertainty(detections),
            "new_position": self._novelty(detections),
        }
        weights = {"rarity": w.rarity, "small_objects": w.small_objects,
                   "low_confidence": w.low_confidence, "new_position": w.new_position}
        total_weight = sum(weights.values()) or 1.0
        score = sum(components[k] * weights[k] for k in components) / total_weight
        return InterestScore(round(score, 4), {k: round(v, 4) for k, v in components.items()})

    def update(self, detections: Sequence[Detection]) -> None:
        """Record a *saved* frame so future similar frames score lower."""
        for det in detections:
            self.class_counts[det.class_id] += 1
            self.position_counts[self._cell(det)] += 1
