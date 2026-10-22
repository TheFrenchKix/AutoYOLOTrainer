from pathlib import Path

import pytest

from src.config import Workspace
from src.datasets.manager import DatasetManager
from src.datasets.yolo_format import label_path_for, read_labels, write_data_yaml, write_labels
from src.model.types import YoloBox
from src.utils.images import read_image, write_image
from tests.conftest import CLASSES

BOX = YoloBox(0, 0.5, 0.5, 0.2, 0.2)


@pytest.fixture
def manager(workspace: Workspace) -> DatasetManager:
    dm = DatasetManager(workspace.current_dataset, list(CLASSES), workspace.datasets, workspace.archive, 0.2)
    dm.init_structure()
    return dm


def test_init_structure(manager: DatasetManager) -> None:
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        assert (manager.dataset_dir / sub).is_dir()
    assert manager.data_yaml.exists()


def test_split_is_deterministic(manager: DatasetManager) -> None:
    names = [f"img_{i}.png" for i in range(500)]
    splits = [manager.choose_split(n) for n in names]
    assert splits == [manager.choose_split(n) for n in names]
    assert 0.1 < splits.count("val") / len(splits) < 0.3


def test_add_image_and_name_collision(manager: DatasetManager, make_image) -> None:
    image = make_image("a.png", 1)
    first = manager.add_image(image, [BOX], split="train")
    second = manager.add_image(image, [], split="train")
    assert first.name == "a.png" and second.name == "a_1.png"
    assert read_labels(label_path_for(first)) == [BOX]
    assert image.exists()  # copy by default


def test_count(manager: DatasetManager, make_image) -> None:
    manager.add_image(make_image("a.png", 1), [BOX, YoloBox(1, 0.2, 0.2, 0.1, 0.1)], split="train")
    manager.add_image(make_image("b.png", 2), [], split="val")
    counts = manager.count()
    assert counts.images == 2 and counts.boxes == 2
    assert counts.per_class == {"cat": 1, "dog": 1}
    assert counts.splits["val"].backgrounds == 1


def test_import_labelstudio_and_reimport_updates(manager: DatasetManager, workspace: Workspace, make_image) -> None:
    make_image("x.png", 3, workspace.raw_images)
    tasks = [{"id": 1, "data": {"image": "/data/local-files/?d=datasets/raw/x.png"},
              "annotations": [{"result": [{"type": "rectanglelabels", "value": {
                  "x": 10, "y": 10, "width": 20, "height": 20, "rectanglelabels": ["dog"]}}]}]}]
    summary = manager.import_labelstudio(tasks, document_root=workspace.root)
    assert summary.imported == 1 and manager.count().per_class["dog"] == 1

    tasks[0]["annotations"][0]["result"][0]["value"]["rectanglelabels"] = ["cat"]
    manager.import_labelstudio(tasks, document_root=workspace.root)
    counts = manager.count()
    assert counts.images == 1 and counts.per_class == {"cat": 1, "dog": 0}


def test_import_folder_moves(manager: DatasetManager, workspace: Workspace, make_image) -> None:
    image = make_image("c.png", 4, workspace.accepted)
    write_labels(image.with_suffix(".txt"), [BOX])
    image.with_suffix(".json").write_text("{}")
    assert manager.import_folder(workspace.accepted) == 1
    assert not any(workspace.accepted.iterdir())
    assert manager.count().boxes == 1


def test_merge_remaps_classes(manager: DatasetManager, tmp_path: Path) -> None:
    other = tmp_path / "other"
    write_data_yaml(other, ["dog", "bird"])
    from tests.conftest import random_image
    write_image(other / "images" / "train" / "o.png", random_image(9))
    write_labels(other / "labels" / "train" / "o.txt", [YoloBox(0, 0.5, 0.5, 0.1, 0.1), YoloBox(1, 0.3, 0.3, 0.1, 0.1)])

    assert manager.merge([other]) == 1
    assert manager.class_names == ["cat", "dog", "bird"]
    labels = read_labels(manager.dataset_dir / "labels" / "train" / "o.txt")
    assert [b.class_id for b in labels] == [1, 2]


def test_remove_duplicates(manager: DatasetManager, make_image) -> None:
    original = make_image("orig.png", 5)
    near_copy = original.with_name("copy.png")
    write_image(near_copy, (read_image(original).astype(int) + 3).clip(0, 255).astype("uint8"))
    manager.add_image(original, [BOX], split="train")
    manager.add_image(near_copy, [BOX], split="train")
    manager.add_image(make_image("other.png", 6), [BOX], split="train")

    assert len(manager.remove_duplicates(dry_run=True)) == 1
    removed = manager.remove_duplicates()
    assert [p.name for p in removed] == ["orig.png"] or [p.name for p in removed] == ["copy.png"]
    assert manager.count().images == 2
    assert not label_path_for(removed[0]).exists()


def test_validate_detects_problems(manager: DatasetManager, make_image) -> None:
    img = manager.add_image(make_image("v.png", 7), [YoloBox(5, 0.5, 0.5, 0.0, 0.2)], split="train")
    (manager.dataset_dir / "labels" / "train" / "orphan.txt").write_text("")
    problems = manager.validate()
    assert any("hors limites" in p for p in problems)
    assert any("dégénérée" in p for p in problems)
    assert any("orphan" in p for p in problems)
    assert img.exists()


def test_create_version_snapshot(manager: DatasetManager, workspace: Workspace, make_image) -> None:
    image = manager.add_image(make_image("s.png", 8), [BOX], split="train")
    version, path = manager.create_version(note="test")
    assert version == 1 and path == workspace.datasets / "v1"
    assert (path / "images" / "train" / "s.png").exists()
    assert manager.version_manifest(1)["counts"]["images"] == 1
    assert manager.versions.active() == 1

    # Rewriting labels in current must not alter the snapshot (atomic replace + hard link).
    write_labels(label_path_for(image), [])
    assert read_labels(path / "labels" / "train" / "s.txt") == [BOX]
    assert manager.create_version()[0] == 2


def test_report(manager: DatasetManager, make_image) -> None:
    manager.add_image(make_image("r.png", 10), [BOX], split="train")
    report = manager.report()
    assert report.exists() and (report.parent / "report.json").exists()
    assert "Rapport du dataset" in report.read_text(encoding="utf-8")


def test_archive_old_versions(manager: DatasetManager, workspace: Workspace, make_image) -> None:
    manager.add_image(make_image("z.png", 11), [BOX], split="train")
    for _ in range(3):
        manager.create_version()
    archived = manager.versions.archive_old(keep=1)
    assert [p.name for p in archived] == ["v1.zip", "v2.zip"]
    assert manager.versions.versions() == [1, 2, 3]
    assert not (workspace.datasets / "v1").exists()
