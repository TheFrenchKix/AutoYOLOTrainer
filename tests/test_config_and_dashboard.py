import pytest
from fastapi.testclient import TestClient

from src.config import load_config
from src.dashboard.app import create_app
from src.datasets.manager import DatasetManager
from src.model.types import YoloBox


def test_default_config_matches_spec() -> None:
    cfg = load_config()
    assert cfg.capture.save_every_detection is True
    assert cfg.capture.max_images_per_hour == 1000
    assert cfg.training.auto_train is True and cfg.training.min_new_images == 500
    assert cfg.labeling.confidence_accept == 0.97 and cfg.labeling.confidence_review == 0.85
    assert cfg.augmentation.enabled is True
    assert cfg.training.base_model == "yolo11n.pt" and cfg.training.epochs == 100
    assert set(cfg.augmentation.transforms) >= {"horizontal_flip", "rotation", "blur", "noise", "perspective"}


def test_overrides() -> None:
    cfg = load_config(["training.epochs=3", "labeling.confidence_accept=0.9"])
    assert cfg.training.epochs == 3 and cfg.labeling.confidence_accept == 0.9


def test_invalid_thresholds_rejected() -> None:
    with pytest.raises(ValueError):
        load_config(["labeling.confidence_review=0.99"])


def test_dashboard(config, workspace, make_image) -> None:  # noqa: ANN001
    dm = DatasetManager(workspace.current_dataset, config.class_names, workspace.datasets)
    dm.init_structure()
    dm.add_image(make_image("a.png", 1), [YoloBox(0, 0.5, 0.5, 0.1, 0.1)])
    make_image("r.png", 2, workspace.review)

    client = TestClient(create_app(config, workspace))
    stats = client.get("/api/stats").json()
    assert stats["dataset_images"] == 1 and stats["dataset_annotations"] == 1
    assert stats["review"] == 1 and stats["active_model"] is None
    page = client.get("/")
    assert page.status_code == 200 and "AutoYOLO Trainer" in page.text
    assert client.get("/api/health").json() == {"status": "ok"}
