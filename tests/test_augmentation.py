from src.config import load_config
from src.datasets.augmentation import AUG_PREFIX, AugmentationPipeline, Sample
from src.datasets.manager import DatasetManager
from src.datasets.yolo_format import label_path_for, read_labels
from src.model.types import YoloBox
from tests.conftest import random_image

BOX = YoloBox(0, 0.5, 0.5, 0.3, 0.3)


def pipeline() -> AugmentationPipeline:
    return AugmentationPipeline(load_config().augmentation, seed=0)


def _valid(boxes: list[YoloBox]) -> bool:
    return all(0 <= b.x_center <= 1 and 0 <= b.y_center <= 1 and 0 < b.width <= 1 and 0 < b.height <= 1
               for b in boxes)


def test_augment_keeps_valid_boxes() -> None:
    aug = pipeline()
    for seed in range(20):
        out = aug.augment(Sample(random_image(seed), [BOX]))
        assert out.image.ndim == 3 and _valid(out.boxes)


def test_mosaic() -> None:
    out = AugmentationPipeline.mosaic([Sample(random_image(i), [BOX]) for i in range(4)], size=200)
    assert out.image.shape == (200, 200, 3) and len(out.boxes) == 4
    assert _valid(out.boxes)
    assert {(round(b.x_center, 2), round(b.y_center, 2)) for b in out.boxes} == {
        (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)}


def test_mixup() -> None:
    out = AugmentationPipeline.mixup(Sample(random_image(1), [BOX]), Sample(random_image(2, (50, 60)), [BOX]))
    assert out.image.shape == random_image(1).shape and len(out.boxes) == 2


def test_augment_split_only_train(workspace, make_image) -> None:  # noqa: ANN001
    dm = DatasetManager(workspace.current_dataset, ["cat", "dog"], workspace.datasets)
    dm.init_structure()
    for i in range(5):
        dm.add_image(make_image(f"t{i}.png", i), [BOX], split="train")
    dm.add_image(make_image("v.png", 9), [BOX], split="val")

    written = pipeline().augment_split(dm.dataset_dir, copies=2)
    assert written == 10
    augmented = [p for p in dm.images("train") if p.name.startswith(AUG_PREFIX)]
    assert len(augmented) == 10 and all(label_path_for(p).exists() for p in augmented)
    assert all(_valid(read_labels(label_path_for(p))) for p in augmented)
    assert len(dm.images("val")) == 1
