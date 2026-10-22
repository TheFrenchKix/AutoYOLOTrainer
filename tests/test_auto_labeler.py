import json
from pathlib import Path

import pytest

from src.config.schema import LabelingConfig
from src.datasets.yolo_format import read_labels
from src.labeler.auto_labeler import AutoLabeler
from src.labeler.router import ConfidenceRouter, Decision
from src.utils.files import write_json
from tests.conftest import FakeDetector, detection


@pytest.mark.parametrize(
    ("confidences", "expected"),
    [
        ([0.99], Decision.ACCEPTED),
        ([0.97], Decision.REVIEW),       # strictly greater than accept is required
        ([0.90], Decision.REVIEW),
        ([0.85], Decision.REJECTED),     # strictly greater than review is required
        ([0.50], Decision.REJECTED),
        ([0.99, 0.90], Decision.REVIEW),  # min aggregation: one doubtful box is enough
    ],
)
def test_router_thresholds(confidences: list[float], expected: Decision) -> None:
    assert ConfidenceRouter(LabelingConfig()).decide(confidences) == expected


def test_router_mean_aggregation_and_custom_thresholds() -> None:
    router = ConfidenceRouter(LabelingConfig(aggregation="mean", confidence_accept=0.9, confidence_review=0.5))
    assert router.decide([0.99, 0.95]) == Decision.ACCEPTED
    assert router.decide([0.99, 0.2]) == Decision.REVIEW


@pytest.mark.parametrize("policy", ["accepted", "review", "rejected"])
def test_router_empty_policy(policy: str) -> None:
    assert ConfidenceRouter(LabelingConfig(empty_policy=policy)).decide([]) == Decision(policy)


def _labeler(workspace, detections, **cfg) -> AutoLabeler:  # noqa: ANN001
    return AutoLabeler(FakeDetector(detections), LabelingConfig(**cfg), workspace.triage_dirs, "v1")


def test_label_image_writes_and_moves(workspace, make_image) -> None:  # noqa: ANN001
    image = make_image("a.png", 1, workspace.captures)
    result = _labeler(workspace, [detection(0.99, 1)]).label_image(image)

    assert result.decision == Decision.ACCEPTED
    assert result.image == workspace.accepted / "a.png"
    assert not image.exists()
    labels = read_labels(workspace.accepted / "a.txt")
    assert len(labels) == 1 and labels[0].class_id == 1
    meta = json.loads((workspace.accepted / "a.json").read_text(encoding="utf-8"))
    assert meta["labeling"]["decision"] == "accepted" and meta["labeling"]["model_version"] == "v1"


@pytest.mark.parametrize(("conf", "folder"), [(0.99, "accepted"), (0.9, "review"), (0.3, "rejected")])
def test_triage_folders(workspace, make_image, conf: float, folder: str) -> None:  # noqa: ANN001
    image = make_image("b.png", 2, workspace.captures)
    _labeler(workspace, [detection(conf)]).label_image(image)
    assert (getattr(workspace, folder) / "b.png").exists()
    assert (getattr(workspace, folder) / "b.txt").exists()


def test_predictions_below_predict_confidence_are_dropped(workspace, make_image) -> None:  # noqa: ANN001
    image = make_image("c.png", 3, workspace.captures)
    result = _labeler(workspace, [detection(0.99), detection(0.1)], predict_confidence=0.25).label_image(image)
    assert result.decision == Decision.ACCEPTED and len(result.detections) == 1


def test_use_saved_predictions(workspace, make_image) -> None:  # noqa: ANN001
    image = make_image("d.png", 4, workspace.captures)
    write_json(image.with_suffix(".json"), {"detections": [detection(0.9).to_dict()]})
    labeler = AutoLabeler(None, LabelingConfig(use_saved_predictions=True), workspace.triage_dirs)
    assert labeler.label_image(image).decision == Decision.REVIEW


def test_requires_detector_without_saved_predictions(workspace) -> None:  # noqa: ANN001
    with pytest.raises(ValueError):
        AutoLabeler(None, LabelingConfig(), workspace.triage_dirs)


def test_process_folder_counts_and_isolates_errors(workspace, make_image) -> None:  # noqa: ANN001
    for i in range(3):
        make_image(f"img{i}.png", i, workspace.captures)
    (workspace.captures / "broken.png").write_bytes(b"not an image")
    stats = _labeler(workspace, [detection(0.99)]).process_folder(workspace.captures)
    assert stats["accepted"] == 3 and stats["errors"] == 1
    assert (workspace.rejected / "broken.png").exists()
    assert not any(Path(workspace.captures).iterdir())
