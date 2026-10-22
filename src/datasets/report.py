"""Dataset report (JSON + HTML with class distribution chart)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless: reports are generated from background threads
import matplotlib.pyplot as plt  # noqa: E402

from src.utils import html  # noqa: E402
from src.utils.files import atomic_write_text, write_json  # noqa: E402

if TYPE_CHECKING:
    from src.datasets.manager import DatasetCounts


def write_dataset_report(counts: DatasetCounts, problems: list[str], output_dir: Path) -> Path:
    """Write ``report.json`` and ``report.html`` into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **counts.to_dict(),
        "problems": problems,
    }
    write_json(output_dir / "report.json", data)

    per_class = counts.per_class
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(list(per_class), list(per_class.values()), color="#2b6cb0")
    ax.set_title("Annotations par classe")
    ax.set_ylabel("Boîtes")
    ax.tick_params(axis="x", rotation=30)

    split_rows = [
        (name, s.images, s.boxes, s.backgrounds) for name, s in counts.splits.items()
    ]
    body = (
        html.cards([("Images", counts.images), ("Annotations", counts.boxes),
                    ("Classes", len(per_class)), ("Problèmes", len(problems))])
        + html.table(["Split", "Images", "Boîtes", "Sans objet"], split_rows)
        + "<h2>Distribution</h2>" + html.figure_to_img(fig)
        + "<h2>Problèmes</h2>"
        + (html.table(["Description"], [(p,) for p in problems[:500]]) if problems else "<p>Aucun</p>")
    )
    path = output_dir / "report.html"
    atomic_write_text(path, html.page("Rapport du dataset", body))
    return path
