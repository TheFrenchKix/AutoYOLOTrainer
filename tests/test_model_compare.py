import json
from pathlib import Path

import pytest

from src.cli.parser import build_parser
from src.dashboard.actions import build_command, catalog
from src.model.catalog import CATALOG, DEFAULT_WEIGHTS, weights_names
from src.model.registry import ModelRegistry
from src.trainer.metrics import Metrics
from src.trainer.trainer_manager import TrainerManager
from tests.conftest import FakeBackend
from tests.test_trainer_manager import BOX, GOOD, WORSE, add_accepted, seed_dataset  # noqa: F401

FAST_BUT_WEAK = Metrics(map50=0.6, map50_95=0.4, precision=0.6, recall=0.5)
ACCURATE = Metrics(map50=0.9, map50_95=0.7, precision=0.9, recall=0.8)


@pytest.fixture
def trainer(config, workspace, make_image):  # noqa: ANN001, ANN201
    def _make(metrics: list[Metrics]) -> tuple[TrainerManager, FakeBackend]:
        config.training.min_new_images = 3
        backend = FakeBackend(metrics)
        manager = TrainerManager(config, workspace, backend=backend)
        seed_dataset(manager, make_image)
        return manager, backend

    return _make


# ---------------------------------------------------------------------------- catalog
def test_catalog_names_are_real_ultralytics_checkpoints() -> None:
    from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

    for weights in weights_names():
        assert weights.removesuffix(".pt") in GITHUB_ASSETS_STEMS, weights
    assert len(set(weights_names())) == len(CATALOG) and DEFAULT_WEIGHTS in weights_names()


def test_dashboard_offers_the_catalog_for_training_and_retraining() -> None:
    items = {i["id"]: i for g in catalog() for i in g["items"] if i["type"] == "action"}
    for action in ("train", "retrain"):
        field = next(f for f in items[action]["fields"] if f["kind"] == "multichoice")
        assert tuple(field["choices"]) == weights_names() and len(field["labels"]) == len(CATALOG)
    command = build_command("train", {"models": ["yolo26n.pt", "yolo11s.pt"], "epochs": 20}, [])
    assert command.argv == ["main.py", "train", "--model=yolo26n.pt", "--model=yolo11s.pt", "--epochs=20"]
    assert build_command("retrain", {"force": True, "models": []}, []).argv == ["main.py", "retrain", "--force"]
    assert build_command("benchmark", {}, []).argv == ["main.py", "benchmark"]


def test_cli_accepts_several_models() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--model", "yolo26n.pt", "--model", "yolo11n.pt", "--epochs", "5"])
    assert args.model == ["yolo26n.pt", "yolo11n.pt"] and args.epochs == 5
    args = parser.parse_args(["retrain", "--force", "--model", "yolo26s.pt", "--epochs", "10"])
    assert (args.model, args.epochs, args.force) == (["yolo26s.pt"], 10, True)
    assert parser.parse_args(["benchmark", "--version", "2", "--version", "3"]).version == [2, 3]
    assert parser.parse_args(["train"]).model is None


# --------------------------------------------------------------------------- training
def test_several_models_share_one_dataset_and_the_best_is_promoted(trainer, workspace) -> None:  # noqa: ANN001
    manager, backend = trainer([FAST_BUT_WEAK, ACCURATE, GOOD])
    outcomes = manager.train_models(["yolo26n.pt", "yolo11s.pt", "yolov8n.pt"], epochs=7)

    assert [o.model_version for o in outcomes] == [1, 2, 3]
    assert {o.dataset_version for o in outcomes} == {1}  # one snapshot, identical data for everyone
    assert [w for w, _ in backend.trained] == ["yolo26n.pt", "yolo11s.pt", "yolov8n.pt"]
    assert all(epochs == 7 for _, epochs in backend.trained)
    assert [o.promoted for o in outcomes] == [True, True, False]  # 1st forced, 2nd beats it, 3rd does not
    registry = ModelRegistry(workspace.models)
    assert registry.active_version() == 2
    assert [registry.architecture(v) for v in (1, 2, 3)] == ["yolo26n", "yolo11s", "yolov8n"]
    runs = {o.run_dir.name for o in outcomes}
    assert len(runs) == 3 and any(name.endswith("_yolo26n") for name in runs)


def test_speed_is_recorded_and_shown_in_the_history(trainer, workspace) -> None:  # noqa: ANN001
    manager, _ = trainer([GOOD])
    manager.train_initial()
    info = ModelRegistry(workspace.models).history()[0]
    assert info["speed_ms"] == 5.0 and info["params_m"] == 2.6 and info["architecture"] == "yolo11n"


def test_a_failed_speed_measurement_does_not_lose_the_training(trainer, workspace) -> None:  # noqa: ANN001
    manager, backend = trainer([GOOD])

    def broken(weights, images):  # noqa: ANN001, ANN202
        raise RuntimeError("CUDA indisponible")

    backend.benchmark = broken
    outcome = manager.train_initial()
    assert outcome.promoted and "speed_ms" not in ModelRegistry(workspace.models).info(outcome.model_version)


def test_retrain_without_models_fine_tunes_and_keeps_the_architecture(trainer, workspace, make_image) -> None:  # noqa: ANN001
    manager, backend = trainer([GOOD, ACCURATE])
    manager.train_models(["yolo26s.pt"])
    add_accepted(workspace, make_image, 3)
    (outcome,) = manager.retrain_models()
    assert backend.trained[-1][0].endswith("best.pt")  # fine-tunes the active model
    assert ModelRegistry(workspace.models).architecture(outcome.model_version) == "yolo26s"


def test_retrain_with_models_trains_from_pretrained_weights(trainer, workspace, make_image) -> None:  # noqa: ANN001
    manager, backend = trainer([GOOD, ACCURATE, FAST_BUT_WEAK])
    manager.train_initial()
    add_accepted(workspace, make_image, 3)
    outcomes = manager.retrain_models(models=["yolo26n.pt", "yolo26s.pt"], epochs=4)
    assert [w for w, _ in backend.trained[1:]] == ["yolo26n.pt", "yolo26s.pt"]
    assert all(e == 4 for _, e in backend.trained[1:]) and len(outcomes) == 2
    assert manager.retrain_models() == []  # nothing pending any more: below the threshold


def test_benchmark_models_updates_every_version(trainer, workspace) -> None:  # noqa: ANN001
    manager, backend = trainer([GOOD, ACCURATE])
    manager.train_models(["yolo11n.pt", "yolo26n.pt"])
    backend.benchmarked.clear()
    results = manager.benchmark_models()
    assert set(results) == {1, 2} and len(backend.benchmarked) == 2
    registry = ModelRegistry(workspace.models)
    assert registry.info(1)["speed_ms"] == 5.0 and registry.info(2)["speed_ms"] == 6.0
    assert set(manager.benchmark_models([2])) == {2}


# ---------------------------------------------------------------------------- registry
def test_architecture_of_models_trained_before_it_was_recorded(tmp_path: Path) -> None:
    models = tmp_path / "models"
    registry = ModelRegistry(models)
    for version in (1, 2):
        (models / f"v{version}").mkdir(parents=True)
    runs = {1: ("yolo11n.pt", tmp_path / "run1"), 2: (str(models / "v1" / "best.pt"), tmp_path / "run2")}
    for version, (model, run) in runs.items():
        run.mkdir()
        (run / "args.yaml").write_text(f"model: {json.dumps(model)}\nepochs: 5\n", encoding="utf-8")
        (models / f"v{version}" / "info.json").write_text(json.dumps({"run_dir": str(run)}), encoding="utf-8")
    assert registry.architecture(1) == "yolo11n" and registry.architecture(2) == "yolo11n"  # v2 = fine-tune of v1
    assert registry.architecture(9) == "inconnue"
