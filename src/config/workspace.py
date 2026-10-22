"""Resolved filesystem layout of a project.

Every component receives a :class:`Workspace` instead of building paths by
hand, so that the folder layout is defined in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.schema import AppConfig


@dataclass(frozen=True)
class Workspace:
    """Absolute paths of all project folders."""

    root: Path
    datasets: Path
    current_dataset: Path
    archive: Path
    raw_images: Path
    captures: Path
    accepted: Path
    review: Path
    rejected: Path
    models: Path
    runs: Path
    logs: Path
    state: Path

    @classmethod
    def from_config(cls, config: AppConfig, root: Path) -> Workspace:
        root = root.resolve()
        p = config.paths

        def resolve(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else root / path

        return cls(
            root=root,
            datasets=resolve(p.datasets),
            current_dataset=resolve(p.current_dataset),
            archive=resolve(p.archive),
            raw_images=resolve(p.raw_images),
            captures=resolve(p.captures),
            accepted=resolve(p.accepted),
            review=resolve(p.review),
            rejected=resolve(p.rejected),
            models=resolve(p.models),
            runs=resolve(p.runs),
            logs=resolve(p.logs),
            state=resolve(p.state),
        )

    @property
    def triage_dirs(self) -> dict[str, Path]:
        """Destination folders of the auto-labeler, keyed by decision."""
        return {"accepted": self.accepted, "review": self.review, "rejected": self.rejected}

    def ensure(self) -> None:
        """Create every folder of the layout (idempotent)."""
        for path in (
            self.datasets, self.current_dataset, self.archive, self.raw_images,
            self.captures, self.accepted, self.review, self.rejected,
            self.models, self.runs, self.logs, self.state,
        ):
            path.mkdir(parents=True, exist_ok=True)
