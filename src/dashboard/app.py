"""FastAPI dashboard.

Run with ``python main.py dashboard`` then open http://127.0.0.1:8000. The page is a menu with
sub-menus that launches every command of the project as a background job and shows its live output.
``/api/stats`` returns the project statistics as JSON for external monitoring.

Launching jobs and listing screens or windows (titles can be private) is only allowed when the server
listens on the loopback interface, for requests addressed to a loopback host name and sent as JSON
(which a foreign web page cannot do).
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.config.loader import PROJECT_ROOT
from src.config.schema import AppConfig
from src.config.workspace import Workspace
from src.dashboard.actions import build_command, catalog
from src.dashboard.jobs import JobConflict, JobManager
from src.dashboard.stats import captures_per_day, collect_stats
from src.model.registry import ModelRegistry

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
INDEX_FILE = Path(__file__).parent / "static" / "index.html"
SECRET_KEYS = {"password", "api_token"}


class JobRequest(BaseModel):
    action: str
    values: dict[str, Any] = Field(default_factory=dict)
    overrides: list[str] = Field(default_factory=list)


def _hostname(header: str) -> str:
    """Host name of a ``Host`` header (port removed, IPv6 brackets handled)."""
    if header.startswith("["):
        return header[1 : header.find("]")]
    return header.split(":")[0]


def _public_config(config: AppConfig) -> dict[str, Any]:
    data = asdict(config)
    for section in data.values():
        if isinstance(section, dict):
            for key in SECRET_KEYS & section.keys():
                section[key] = "••••••••"
    return data


def create_app(config: AppConfig, workspace: Workspace, jobs: JobManager | None = None) -> FastAPI:
    manager = jobs or JobManager(workspace.state / "jobs", PROJECT_ROOT)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        manager.shutdown()

    app = FastAPI(title="AutoYOLO Trainer", version="0.1.0", lifespan=lifespan)

    def local_json_only(request: Request) -> None:
        if config.dashboard.host not in LOOPBACK_HOSTS:
            raise HTTPException(403, "Actions désactivées: le dashboard n'écoute pas uniquement en local")
        if _hostname(request.headers.get("host", "")) not in LOOPBACK_HOSTS:
            raise HTTPException(403, "Hôte non autorisé")
        if not request.headers.get("content-type", "").startswith("application/json"):
            raise HTTPException(415, "Requête JSON attendue")

    @app.get("/api/stats")
    def api_stats() -> dict[str, Any]:
        return collect_stats(config, workspace).to_dict()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/captures-per-day")
    def api_captures_per_day() -> list[dict[str, Any]]:
        frame = captures_per_day(workspace.state / "captures_index.csv")
        return [{"day": str(row.day), "captures": int(row.captures)} for row in frame.itertuples()]

    @app.get("/api/models")
    def api_models() -> list[dict[str, Any]]:
        return ModelRegistry(workspace.models).history()[::-1]

    @app.get("/api/config")
    def api_config() -> dict[str, Any]:
        return _public_config(config)

    @app.get("/api/catalog")
    def api_catalog() -> list[dict[str, Any]]:
        return catalog()

    @app.get("/api/screens", dependencies=[Depends(local_json_only)])
    def api_screens() -> list[dict[str, Any]]:
        """Monitors for the drop-downs (index 0 = every monitor). Empty outside Windows."""
        if sys.platform != "win32":
            return []
        from src.capture import windows

        windows.enable_dpi_awareness()
        return [
            {"index": s.index, "width": s.size[0], "height": s.size[1], "primary": s.primary}
            for s in windows.list_screens()
        ]

    @app.get("/api/windows", dependencies=[Depends(local_json_only)])
    def api_windows() -> list[dict[str, Any]]:
        """Visible windows with a title, front-most first (duplicated titles listed once). Empty outside Windows."""
        if sys.platform != "win32":
            return []
        from src.capture import windows

        windows.enable_dpi_awareness()
        seen: set[str] = set()
        result = []
        for window in windows.list_windows():
            if window.title not in seen:
                seen.add(window.title)
                result.append({"title": window.title, "width": window.size[0], "height": window.size[1]})
        return result

    @app.get("/api/jobs")
    def api_jobs() -> list[dict[str, Any]]:
        return [job.to_dict() for job in manager.history()]

    @app.post("/api/jobs", dependencies=[Depends(local_json_only)])
    def api_start_job(request: JobRequest) -> dict[str, Any]:
        try:
            command = build_command(request.action, request.values, request.overrides)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            job = manager.start(command.action.id, command.action.label, command.argv, command.display)
        except JobConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str, offset: int = 0) -> dict[str, Any]:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "Tâche introuvable")
        text, next_offset = manager.read_log(job_id, offset)
        return {**job.to_dict(), "log": text, "offset": next_offset}

    @app.post("/api/jobs/{job_id}/stop", dependencies=[Depends(local_json_only)])
    def api_stop_job(job_id: str) -> dict[str, Any]:
        job = manager.stop(job_id)
        if job is None:
            raise HTTPException(404, "Tâche introuvable")
        return job.to_dict()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_FILE.read_text(encoding="utf-8")

    return app
