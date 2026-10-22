from pathlib import Path

import pytest

from src.datasets.yolo_format import label_path_for, read_class_names, read_labels, write_data_yaml, write_labels
from src.model.types import YoloBox


def test_xyxy_round_trip() -> None:
    box = YoloBox.from_xyxy(1, 20, 40, 60, 100, img_w=200, img_h=200)
    assert box == YoloBox(1, 0.2, 0.35, 0.2, 0.3)
    assert box.to_xyxy(200, 200) == pytest.approx((20, 40, 60, 100))


def test_from_xyxy_clips_and_orders() -> None:
    box = YoloBox.from_xyxy(0, 120, -10, 80, 50, img_w=100, img_h=100)
    x1, y1, x2, y2 = box.to_xyxy(100, 100)
    assert (x1, y1, x2, y2) == pytest.approx((80, 0, 100, 50))


def test_line_round_trip() -> None:
    box = YoloBox(3, 0.5, 0.25, 0.1, 0.2)
    assert YoloBox.from_line(box.to_line()) == box


def test_invalid_line() -> None:
    with pytest.raises(ValueError):
        YoloBox.from_line("0 0.5 0.5")


def test_write_and_read_labels(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    boxes = [YoloBox(0, 0.5, 0.5, 0.2, 0.2), YoloBox(1, 0.1, 0.1, 0.05, 0.05)]
    write_labels(path, boxes)
    assert read_labels(path) == boxes
    write_labels(path, [])
    assert path.read_text() == "" and read_labels(path) == []
    assert read_labels(tmp_path / "missing.txt") == []


def test_label_path_for() -> None:
    assert label_path_for(Path("ds/images/train/x.png")) == Path("ds/labels/train/x.txt")
    assert label_path_for(Path("captures/x.png")) == Path("captures/x.txt")


def test_data_yaml(tmp_path: Path) -> None:
    write_data_yaml(tmp_path, ["cat", "dog"])
    assert read_class_names(tmp_path) == ["cat", "dog"]
