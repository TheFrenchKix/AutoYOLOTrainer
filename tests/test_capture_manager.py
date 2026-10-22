import json
import random
import re

import pytest

from src.capture.active_learning import ActiveLearningSelector
from src.capture.capture_manager import CaptureManager
from src.capture.rate_limiter import RateLimiter
from src.config.schema import ActiveLearningConfig, CaptureConfig, DedupConfig
from src.datasets.yolo_format import read_labels
from tests.conftest import detection, random_image


class Clock:
    def __init__(self, start: float = 1_790_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> Clock:
    return Clock()


def make_manager(workspace, clock, capture: CaptureConfig | None = None,  # noqa: ANN001
                 al: ActiveLearningConfig | None = None, seed: int = 0) -> CaptureManager:
    return CaptureManager(capture or CaptureConfig(min_interval_seconds=0), al or ActiveLearningConfig(),
                          workspace.captures, workspace.state, clock=clock, rng=random.Random(seed))


def test_saves_image_labels_and_metadata(workspace, clock) -> None:  # noqa: ANN001
    manager = make_manager(workspace, clock)
    path = manager.on_detection(random_image(1), [detection(0.9, 1)])

    assert path is not None and path.exists()
    assert re.fullmatch(r"\d{8}_\d{6}_\d{6}\.png", path.name)
    assert read_labels(path.with_suffix(".txt"))[0].class_id == 1
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["detections"][0]["class_name"] == "dog"
    assert meta["detections"][0]["confidence"] == pytest.approx(0.9)
    assert {"date", "time", "active_learning"} <= meta.keys()
    assert (workspace.state / "captures_index.csv").read_text(encoding="utf-8").count("\n") == 2


def test_ignores_frames_without_confident_detections(workspace, clock) -> None:  # noqa: ANN001
    manager = make_manager(workspace, clock)
    assert manager.on_detection(random_image(1), []) is None
    assert manager.on_detection(random_image(1), [detection(0.1)]) is None
    assert manager.stats["saved"] == 0


def test_rejects_duplicates(workspace, clock) -> None:  # noqa: ANN001
    manager = make_manager(workspace, clock)
    assert manager.on_detection(random_image(1), [detection(0.9)]) is not None
    clock.advance(10)
    assert manager.on_detection(random_image(1), [detection(0.9)]) is None
    assert manager.stats["skipped_duplicate"] == 1
    clock.advance(10)
    assert manager.on_detection(random_image(2), [detection(0.9)]) is not None


def test_rate_limit_per_hour_and_interval(workspace, clock) -> None:  # noqa: ANN001
    manager = make_manager(workspace, clock, CaptureConfig(max_images_per_hour=2, min_interval_seconds=5))
    assert manager.on_detection(random_image(1), [detection(0.9)])
    assert manager.on_detection(random_image(2), [detection(0.9)]) is None  # too soon
    clock.advance(6)
    assert manager.on_detection(random_image(3), [detection(0.9)])
    clock.advance(6)
    assert manager.on_detection(random_image(4), [detection(0.9)]) is None  # hourly quota reached
    clock.advance(3600)
    assert manager.on_detection(random_image(5), [detection(0.9)])
    assert manager.stats["skipped_rate_limit"] == 2


def test_low_interest_frames_skipped_when_not_saving_everything(workspace, clock) -> None:  # noqa: ANN001
    al = ActiveLearningConfig(score_threshold=0.9, exploration_rate=0.0)
    manager = make_manager(workspace, clock, CaptureConfig(save_every_detection=False, min_interval_seconds=0), al)
    manager.selector = ActiveLearningSelector(al, {0: 1000, 1: 1000})
    assert manager.on_detection(random_image(1), [detection(0.99, box=(0, 0, 100, 90))]) is None
    assert manager.stats["skipped_low_interest"] == 1


def test_unique_names_within_same_microsecond(workspace, clock) -> None:  # noqa: ANN001
    manager = make_manager(workspace, clock, al=ActiveLearningConfig(dedup=DedupConfig(enabled=False)))
    first = manager.on_detection(random_image(1), [detection(0.9)])
    second = manager.on_detection(random_image(2), [detection(0.9)])
    assert first != second and first.exists() and second.exists()


class TestActiveLearning:
    def test_signals(self) -> None:
        selector = ActiveLearningSelector(ActiveLearningConfig(), {0: 100, 1: 1})
        rare_small_uncertain = selector.score([detection(0.3, 1, box=(0, 0, 5, 5))])
        common_big_confident = selector.score([detection(0.99, 0, box=(0, 0, 120, 90))])
        assert rare_small_uncertain.score > common_big_confident.score
        assert set(rare_small_uncertain.reasons) >= {"rarity", "small_objects", "low_confidence"}

    def test_novelty_decreases_after_update(self) -> None:
        selector = ActiveLearningSelector(ActiveLearningConfig())
        dets = [detection(0.9)]
        before = selector.score(dets).components["new_position"]
        selector.update(dets)
        assert selector.score(dets).components["new_position"] < before

    def test_empty(self) -> None:
        assert ActiveLearningSelector(ActiveLearningConfig()).score([]).score == 0.0


def test_rate_limiter_standalone() -> None:
    clock = Clock()
    limiter = RateLimiter(max_per_hour=1, clock=clock)
    assert limiter.allow()
    limiter.record()
    assert not limiter.allow()
    clock.advance(3601)
    assert limiter.allow()
