"""FastAPI dashboard.

Run with ``python main.py dashboard`` then open http://127.0.0.1:8000.
``/api/stats`` returns the same data as JSON for external monitoring.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.config.schema import AppConfig
from src.config.workspace import Workspace
from src.dashboard.stats import ProjectStats, captures_per_day, collect_stats
from src.utils import html


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _render(stats: ProjectStats, per_day: list[tuple[Any, int]]) -> str:
    progress = min(100, round(100 * stats.pending_for_training / max(1, stats.min_new_images)))
    body = html.cards([
        ("Images du dataset", stats.dataset_images),
        ("Annotations", stats.dataset_annotations),
        ("Modèle actif", f"v{stats.active_model}" if stats.active_model else "aucun"),
        ("Meilleure mAP50", _fmt(stats.best_map50)),
        ("Meilleure mAP50-95", _fmt(stats.best_map50_95)),
        ("Captures (total)", stats.captures_total),
        ("Captures à traiter", stats.captures_pending),
        ("En attente de validation", stats.review),
        ("Acceptées", stats.accepted),
        ("Rejetées", stats.rejected),
        ("Prochain entraînement", f"{stats.pending_for_training}/{stats.min_new_images} ({progress}%)"),
        ("Espace disque libre", stats.disk_free),
    ])
    trainings = [
        (f"v{t['version']}" + (" ★" if t.get("active") else ""), t.get("created_at", "-"),
         f"v{t.get('dataset_version')}", *(_fmt((t.get("metrics") or {}).get(k))
                                            for k in ("map50", "map50_95", "precision", "recall", "f1")))
        for t in stats.trainings
    ]
    growth = [(f"v{g['version']}", g["created_at"], g["images"]) for g in stats.dataset_growth]
    body += (
        "<h2>Derniers entraînements</h2>"
        + html.table(["Modèle", "Date", "Dataset", "mAP50", "mAP50-95", "Precision", "Recall", "F1"], trainings)
        + "<h2>Progression du dataset</h2>"
        + html.table(["Version", "Date", "Images"], growth)
        + "<h2>Annotations par classe</h2>"
        + html.table(["Classe", "Boîtes"], sorted(stats.per_class.items(), key=lambda kv: -kv[1]))
        + "<h2>Captures par jour</h2>"
        + html.table(["Jour", "Captures"], per_day[-14:][::-1])
        + "<h2>Espace disque utilisé</h2>"
        + html.table(["Dossier", "Taille"], stats.disk_usage.items())
    )
    return html.page("AutoYOLO Trainer", body, refresh_seconds=30)


def create_app(config: AppConfig, workspace: Workspace) -> FastAPI:
    app = FastAPI(title="AutoYOLO Trainer", version="0.1.0")

    @app.get("/api/stats")
    def api_stats() -> dict[str, Any]:
        return collect_stats(config, workspace).to_dict()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        per_day = captures_per_day(workspace.state / "captures_index.csv")
        rows = [(str(r.day), int(r.captures)) for r in per_day.itertuples()]
        return _render(collect_stats(config, workspace), rows)

    return app
