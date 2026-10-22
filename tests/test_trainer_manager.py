import pytest

from src.datasets.yolo_format import write_labels
from src.model.types import YoloBox
from src.trainer.metrics import Metrics, is_better
from src.trainer.trainer_manager import TrainerManager
from tests.conftest import FakeBackend

GOOD = Metrics(map50=0.8, map50_95=0.6, precision=0.8, recall=0.7)
BETTER = Metrics(map50=0.85, map50_95=0.65, precision=0.82, recall=0.75)
WORSE = Metrics(map50=0.7, map50_95=0.5, precision=0.7, recall=0.6)
BOX = YoloBox(0, 0.5, 0.5, 0.2, 0.2)


@pytest.fixture
def trainer_factory(config, workspace):  # noqa: ANN001, ANN201
    def _factory(metrics: list[Metrics], min_new_images: int = 3) -> tuple[TrainerManager, FakeBackend]:
        config.training.min_new_images = min_new_images
        backend = FakeBackend(metrics)
        return TrainerManager(config, workspace, backend=backend), backend

    return _factory


def seed_dataset(trainer: TrainerManager, make_image, n: int = 4) -> None:  # noqa: ANN001
    trainer.datasets.init_structure()
    for i in range(n):
        trainer.datasets.add_image(make_image(f"seed{i}.png", 100 + i), [BOX])


def add_accepted(workspace, make_image, n: int, offset: int = 0) -> None:  # noqa: ANN001
    for i in range(n):
        image = make_image(f"acc{offset + i}.png", 200 + offset + i, workspace.accepted)
        write_labels(image.with_suffix(".txt"), [BOX])


def test_metrics_f1_and_comparison() -> None:
    assert GOOD.f1 == pytest.approx(2 * 0.8 * 0.7 / 1.5)
    assert is_better(BETTER, GOOD) and not is_better(WORSE, GOOD)
    assert is_better(GOOD, None)
    assert not is_better(BETTER, GOOD, min_improvement=0.1)
    assert is_better(BETTER, GOOD, primary_metric="recall")
    tie = Metrics(map50=0.8, map50_95=0.6, precision=0.9, recall=0.9)
    assert is_better(tie, GOOD)
    with pytest.raises(KeyError):
        GOOD.get("accuracy")


def test_train_initial_requires_data(trainer_factory) -> None:  # noqa: ANN001
    trainer, _ = trainer_factory([GOOD])
    with pytest.raises(RuntimeError, match="Dataset vide"):
        trainer.train_initial()


def test_train_initial_creates_v1(trainer_factory, workspace, make_image) -> None:  # noqa: ANN001
    trainer, backend = trainer_factory([GOOD])
    seed_dataset(trainer, make_image)
    outcome = trainer.train_initial()

    assert outcome.promoted and outcome.model_version == 1 and outcome.dataset_version == 1
    assert backend.trained == [("yolo11n.pt", 100)]
    for name in ("best.pt", "last.pt", "info.json", "report.html"):
        assert (workspace.models / "v1" / name).exists()
    assert trainer.registry.active_version() == 1
    assert (workspace.datasets / "v1" / "data.yaml").exists()


def test_retrain_skipped_below_threshold(trainer_factory, workspace, make_image) -> None:  # noqa: ANN001
    trainer, backend = trainer_factory([GOOD], min_new_images=5)
    seed_dataset(trainer, make_image)
    add_accepted(workspace, make_image, 2)
    assert trainer.pending_images() == 2 and not trainer.should_retrain()
    assert trainer.retrain() is None
    assert backend.trained == []


def test_retrain_promotes_better_model(trainer_factory, workspace, make_image) -> None:  # noqa: ANN001
    trainer, backend = trainer_factory([GOOD, BETTER])
    seed_dataset(trainer, make_image)
    trainer.train_initial()
    add_accepted(workspace, make_image, 3)
    assert trainer.should_retrain()

    outcome = trainer.maybe_retrain()
    assert outcome is not None and outcome.promoted
    assert outcome.model_version == 2 and outcome.reference_version == 1
    assert outcome.reference == GOOD  # active model re-evaluated on the new validation set
    assert trainer.registry.active_version() == 2
    assert backend.trained[1] == (str(workspace.models / "v1" / "best.pt"), 50)  # fine-tune from active
    assert trainer.datasets.count().images == 7
    assert trainer.pending_images() == 0


def test_retrain_keeps_old_model_when_worse(trainer_factory, workspace, make_image) -> None:  # noqa: ANN001
    trainer, _ = trainer_factory([GOOD, WORSE])
    seed_dataset(trainer, make_image)
    trainer.train_initial()
    add_accepted(workspace, make_image, 3)

    outcome = trainer.retrain()
    assert outcome is not None and not outcome.promoted
    assert trainer.registry.active_version() == 1
    assert (workspace.models / "v2" / "best.pt").exists()  # history kept anyway
    history = trainer.registry.history()
    assert [h["active"] for h in history] == [True, False]


def test_force_retrain(trainer_factory, make_image) -> None:  # noqa: ANN001
    trainer, _ = trainer_factory([GOOD, BETTER], min_new_images=1000)
    seed_dataset(trainer, make_image)
    trainer.train_initial()
    assert trainer.retrain(force=True) is not None


def test_failed_training_keeps_pending_count(trainer_factory, workspace, make_image) -> None:  # noqa: ANN001
    trainer, backend = trainer_factory([GOOD])
    seed_dataset(trainer, make_image)
    trainer.train_initial()
    add_accepted(workspace, make_image, 3)
    backend.candidate_metrics = []  # next train() raises IndexError

    with pytest.raises(IndexError):
        trainer.retrain()
    assert trainer.pending_images() == 3  # images imported, still counted for the next attempt
    assert not trainer.lock.locked
