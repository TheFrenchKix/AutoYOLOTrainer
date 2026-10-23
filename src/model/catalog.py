"""YOLO architectures offered for training, to find the best compromise between speed and accuracy.

Only Ultralytics detection checkpoints that the installed version can download by name are listed
(checked by ``tests/test_model_catalog.py``). The nano/small sizes come first: for a detector that
must run in real time they are the usual candidates, the medium ones serve as an accuracy reference.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    weights: str  # name Ultralytics resolves (and downloads) itself
    label: str
    note: str


CATALOG: tuple[ModelChoice, ...] = (
    ModelChoice("yolo26n.pt", "YOLO26 nano", "conçu pour l'inférence rapide, sans NMS (post-traitement allégé)"),
    ModelChoice("yolo26s.pt", "YOLO26 small", "même famille, un peu plus précis et plus lent"),
    ModelChoice("yolo26m.pt", "YOLO26 medium", "référence de précision, nettement plus lent"),
    ModelChoice("yolo11n.pt", "YOLO11 nano", "modèle de départ actuel du projet"),
    ModelChoice("yolo11s.pt", "YOLO11 small", "compromis vitesse / précision"),
    ModelChoice("yolo11m.pt", "YOLO11 medium", "plus précis, plus lent"),
    ModelChoice("yolov10n.pt", "YOLOv10 nano", "sans NMS, très rapide"),
    ModelChoice("yolov10s.pt", "YOLOv10 small", "sans NMS, un peu plus précis"),
    ModelChoice("yolov8n.pt", "YOLOv8 nano", "référence classique, très répandue"),
    ModelChoice("yolov8s.pt", "YOLOv8 small", "référence classique, un peu plus précise"),
    ModelChoice("yolo12n.pt", "YOLO12 nano", "architecture à attention, souvent plus gourmande à l'inférence"),
)

DEFAULT_WEIGHTS = "yolo11n.pt"


def weights_names() -> tuple[str, ...]:
    return tuple(choice.weights for choice in CATALOG)


def weights_labels() -> tuple[tuple[str, str], ...]:
    """(weights, text for the form) pairs."""
    return tuple((choice.weights, f"{choice.label} - {choice.note}") for choice in CATALOG)
