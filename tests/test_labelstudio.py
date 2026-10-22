from pathlib import Path
from xml.etree import ElementTree

import pytest

from src.datasets.labelstudio import (
    build_label_config,
    convert_tasks,
    detections_to_ls_prediction,
    ls_rect_to_yolo,
    resolve_image_ref,
    yolo_to_ls_rect,
)
from src.datasets.labelstudio_client import local_file_url
from src.model.types import YoloBox
from tests.conftest import CLASSES, detection


def rect(label: str, x: float = 10, y: float = 20, w: float = 30, h: float = 40) -> dict:
    return {"type": "rectanglelabels", "from_name": "label", "to_name": "image",
            "original_width": 640, "original_height": 480,
            "value": {"x": x, "y": y, "width": w, "height": h, "rotation": 0, "rectanglelabels": [label]}}


def task(task_id: int, image: str, results: list[dict], cancelled: bool = False) -> dict:
    return {"id": task_id, "data": {"image": image},
            "annotations": [{"result": results, "was_cancelled": cancelled, "updated_at": "2026-01-01"}]}


def test_ls_rect_to_yolo() -> None:
    box = ls_rect_to_yolo({"x": 10, "y": 20, "width": 30, "height": 40}, 1)
    assert box.class_id == 1
    assert (box.x_center, box.y_center, box.width, box.height) == pytest.approx((0.25, 0.4, 0.3, 0.4))


def test_ls_rect_clipped_to_image() -> None:
    box = ls_rect_to_yolo({"x": 90, "y": -5, "width": 20, "height": 20}, 0)
    assert box.x_center + box.width / 2 == pytest.approx(1.0)
    assert box.y_center - box.height / 2 == pytest.approx(0.0)


def test_round_trip_yolo_ls() -> None:
    box = YoloBox(0, 0.4, 0.6, 0.2, 0.1)
    back = ls_rect_to_yolo(yolo_to_ls_rect(box), 0)
    assert (back.x_center, back.y_center, back.width, back.height) == pytest.approx((0.4, 0.6, 0.2, 0.1))


def test_convert_local_files(tmp_path: Path, make_image) -> None:
    make_image("a.png", 1, tmp_path / "datasets" / "raw")
    tasks = [task(1, "/data/local-files/?d=datasets/raw/a.png", [rect("cat"), rect("dog", 50, 50, 10, 10)])]
    converted, warnings = convert_tasks(tasks, CLASSES, document_root=tmp_path)
    assert warnings == []
    assert converted[0].image_path == tmp_path / "datasets" / "raw" / "a.png"
    assert [b.class_id for b in converted[0].boxes] == [0, 1]


def test_convert_upload_prefix_and_unknown_class(tmp_path: Path, make_image) -> None:
    make_image("b.png", 2, tmp_path)
    tasks = [task(2, "/data/upload/3/1a2b3c4d-b.png", [rect("cat"), rect("bird")])]
    converted, warnings = convert_tasks(tasks, CLASSES, search_dirs=[tmp_path])
    assert converted[0].image_path == tmp_path / "b.png"
    assert len(converted[0].boxes) == 1
    assert any("bird" in w for w in warnings)


def test_skips_unannotated_and_cancelled_keeps_background() -> None:
    tasks = [
        {"id": 1, "data": {"image": "x.png"}, "annotations": []},
        task(2, "y.png", [rect("cat")], cancelled=True),
        task(3, "z.png", []),
    ]
    converted, warnings = convert_tasks(tasks, CLASSES)
    assert [c.task_id for c in converted] == [3]
    assert converted[0].boxes == []
    assert any("introuvable" in w for w in warnings)


def test_resolve_missing_returns_none(tmp_path: Path) -> None:
    assert resolve_image_ref("/data/upload/1/zzzzzzzz-nope.png", tmp_path, [tmp_path]) is None


def test_label_config_and_prediction() -> None:
    tree = ElementTree.fromstring(build_label_config(["cat", 'd"og']))
    assert [label.get("value") for label in tree.iter("Label")] == ["cat", 'd"og']
    assert tree.find("RectangleLabels") is not None
    prediction = detections_to_ls_prediction([detection(0.9)], "v1")
    assert prediction["model_version"] == "v1"
    assert prediction["result"][0]["value"]["rectanglelabels"] == ["cat"]


def test_local_file_url(tmp_path: Path) -> None:
    assert local_file_url(tmp_path / "review" / "a b.png", tmp_path) == "/data/local-files/?d=review/a b.png"
