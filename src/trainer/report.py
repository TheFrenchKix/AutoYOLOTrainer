"""HTML evaluation report: candidate vs active model + history of all versions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.trainer.metrics import METRIC_NAMES, Metrics  # noqa: E402
from src.utils import html  # noqa: E402
from src.utils.files import atomic_write_text  # noqa: E402


def _comparison_table(candidate: Metrics, reference: Metrics | None) -> str:
    rows = []
    for name in METRIC_NAMES:
        new = candidate.get(name)
        if reference is None:
            rows.append((name, "-", f"{new:.4f}", "-"))
            continue
        old = reference.get(name)
        delta = new - old
        css = "ok" if delta > 0 else "bad" if delta < 0 else "muted"
        rows.append((name, f"{old:.4f}", f"{new:.4f}", f'<span class="{css}">{delta:+.4f}</span>'))
    return html.table(["Métrique", "Modèle actif", "Candidat", "Δ"], rows, raw_cells=True)


def _history_chart(frame: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name in METRIC_NAMES:
        if name in frame:
            ax.plot(frame["version"], frame[name], marker="o", label=name)
    ax.set_xlabel("Version du modèle")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(ncol=5, fontsize=8, loc="lower right")
    return html.figure_to_img(fig)


def history_frame(history: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten the registry history into a DataFrame (one row per model)."""
    rows = [
        {"version": h["version"], "dataset": h.get("dataset_version"), "created_at": h.get("created_at"),
         "active": h.get("active", False), **(h.get("metrics") or {})}
        for h in history
    ]
    return pd.DataFrame(rows)


def write_evaluation_report(
    output: Path,
    candidate_version: int,
    candidate: Metrics,
    reference_version: int | None,
    reference: Metrics | None,
    promoted: bool,
    history: list[dict[str, Any]],
    primary_metric: str,
) -> Path:
    verdict = (
        f'<p class="ok"><b>v{candidate_version} promu</b> comme modèle actif.</p>' if promoted
        else f'<p class="bad"><b>v{candidate_version} non promu</b>: '
             f'le modèle actif v{reference_version} est conservé.</p>'
    )
    frame = history_frame(history)
    history_rows = frame.fillna("-").round(4).astype(str).values.tolist() if not frame.empty else []
    body = (
        f'<p class="muted">Généré le {datetime.now():%Y-%m-%d %H:%M} · '
        f"métrique principale: <b>{primary_metric}</b> · comparaison sur le même jeu de validation</p>"
        + verdict
        + "<h2>Comparaison</h2>" + _comparison_table(candidate, reference)
        + "<h2>Historique</h2>" + (_history_chart(frame) if len(frame) else "")
        + html.table(list(frame.columns), history_rows)
    )
    atomic_write_text(output, html.page(f"Évaluation du modèle v{candidate_version}", body))
    return output
