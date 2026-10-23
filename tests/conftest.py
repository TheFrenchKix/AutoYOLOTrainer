"""Shared fixtures: isolated workspace, synthetic images, fake detector/backend.

No test needs a GPU, Ultralytics weights or a running Label Studio.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import AppConfig, Workspace, load_config  # noqa: E402
from src.model.types import Detection  # noqa: E402
from src.trainer.backend import TrainResult  # noqa: E402
from src.trainer.metrics import Metrics  # noqa: E402
from src.utils.images import write_image  # noqa: E402

CLASSES = ["cat", "dog"]


@pytest.fixture
def config() -> AppConfig:
    cfg = load_config([f"project.classes=[{','.join(CLASSES)}]"])
    cfg.augmentation.enabled = False  # keep tests fast; augmentation is tested separately
    return cfg


@pytest.fixture
def workspace(config: AppConfig, tmp_path: Path) -> Workspace:
    ws = Workspace.from_config(config, tmp_path)
    ws.ensure()
    return ws


def random_image(seed: int, size: tuple[int, int] = (96, 128)) -> np.ndarray:
    """Distinct images per seed (blocky noise -> very different perceptual hashes)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    return np.kron(small, np.ones((size[0] // 8, size[1] // 8, 1), dtype=np.uint8))


@pytest.fixture
def make_image(tmp_path: Path) -> Callable[..., Path]:
    def _make(name: str, seed: int, folder: Path | None = None) -> Path:
        path = (folder or tmp_path / "images") / name
        write_image(path, random_image(seed))
        return path

    return _make


def detection(conf: float, class_id: int = 0, box: tuple[float, float, float, float] = (10, 10, 50, 60),
              img: tuple[int, int] = (128, 96)) -> Detection:
    return Detection(class_id, CLASSES[class_id], conf, *box, img_w=img[0], img_h=img[1])


class FakeDetector:
    """Returns a fixed list of detections for every image."""

    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections
        self.class_names = CLASSES
        self.calls = 0

    def predict(self, image: np.ndarray) -> list[Detection]:
        self.calls += 1
        h, w = image.shape[:2]
        return [Detection(d.class_id, d.class_name, d.confidence, d.x1, d.y1, d.x2, d.y2, w, h)
                for d in self.detections]


class FakeBackend:
    """Training backend producing dummy weights and scripted metrics."""

    def __init__(self, candidate_metrics: list[Metrics]) -> None:
        self.candidate_metrics = list(candidate_metrics)
        self.trained: list[tuple[str, int]] = []
        self.scores: dict[str, Metrics] = {}
        self.benchmarked: list[Path] = []

    def train(self, data_yaml: Path, weights: str, project_dir: Path, name: str, epochs: int) -> TrainResult:
        run_dir = project_dir / name
        (run_dir / "weights").mkdir(parents=True)
        best = run_dir / "weights" / "best.pt"
        best.write_text(name)
        (run_dir / "weights" / "last.pt").write_text(name)
        self.scores[name] = self.candidate_metrics.pop(0)
        self.trained.append((weights, epochs))
        return TrainResult(run_dir)

    def evaluate(self, weights: Path, data_yaml: Path, project_dir: Path, name: str) -> Metrics:
        # The weights file content is the run name, which maps to its metrics.
        return self.scores[Path(weights).read_text()]

    def benchmark(self, weights: Path, images: list[Path]) -> dict[str, float]:
        self.benchmarked.append(Path(weights))
        return {"inference_ms": 4.0 + len(self.benchmarked), "params_m": 2.6}


@pytest.fixture
def fake_detector() -> Callable[[list[Detection]], FakeDetector]:
    return FakeDetector
