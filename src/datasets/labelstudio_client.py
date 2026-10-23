"""Minimal Label Studio REST client (projects, local storage, import, export).

Authentication accepts both kinds of Label Studio tokens:

* legacy tokens -> ``Authorization: Token <token>``
* personal access tokens (JWT refresh tokens, LS >= 1.15) -> exchanged on
  ``/api/token/refresh`` for a short-lived ``Bearer`` access token.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

from src.utils.logging import get_logger

log = get_logger("dataset")

IMAGE_REGEX = r".*\.(png|jpe?g|bmp|webp|PNG|JPE?G|BMP|WEBP)$"


class LabelStudioError(RuntimeError):
    pass


class LabelStudioClient:
    def __init__(self, url: str, token: str, timeout: float = 60.0) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()
        self._authenticated = False

    # ------------------------------------------------------------------ http
    def _authenticate(self) -> None:
        if self._authenticated:
            return
        self.session.headers["Authorization"] = f"Token {self.token}"
        response = self.session.get(f"{self.url}/api/current-user/whoami", timeout=self.timeout)
        if response.status_code == 401 and self.token.count(".") == 2:
            refreshed = self.session.post(f"{self.url}/api/token/refresh",
                                          json={"refresh": self.token}, timeout=self.timeout)
            if refreshed.ok:
                self.session.headers["Authorization"] = f"Bearer {refreshed.json()['access']}"
                response = self.session.get(f"{self.url}/api/current-user/whoami", timeout=self.timeout)
        if not response.ok:
            raise LabelStudioError(
                f"Authentification Label Studio refusée ({response.status_code}). "
                "Vérifiez labelstudio.api_token / la variable LABEL_STUDIO_TOKEN."
            )
        self._authenticated = True

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        self._authenticate()
        response = self.session.request(method, f"{self.url}{path}", timeout=self.timeout, **kwargs)
        if not response.ok:
            raise LabelStudioError(f"{method} {path} -> {response.status_code}: {response.text[:500]}")
        return response.json() if response.content else None

    # ---------------------------------------------------------------- server
    def is_alive(self) -> bool:
        try:
            return requests.get(f"{self.url}/health", timeout=5).ok
        except requests.RequestException:
            return False

    def wait_until_ready(self, timeout: float = 180.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_alive():
                return
            time.sleep(2)
        raise LabelStudioError(f"Label Studio ne répond pas sur {self.url} après {timeout:.0f}s")

    # -------------------------------------------------------------- projects
    def find_project(self, title: str) -> dict[str, Any] | None:
        data = self._request("GET", "/api/projects", params={"page_size": 1000})
        projects = data.get("results", []) if isinstance(data, dict) else data
        return next((p for p in projects if p.get("title") == title), None)

    def ensure_project(self, title: str, label_config: str, description: str = "") -> dict[str, Any]:
        project = self.find_project(title)
        if project is None:
            project = self._request("POST", "/api/projects", json={
                "title": title, "label_config": label_config, "description": description,
            })
            log.info(f"Projet Label Studio créé: {title} (id={project['id']})")
        else:
            # Keep the labeling interface in sync with project.classes.
            self._request("PATCH", f"/api/projects/{project['id']}", json={"label_config": label_config})
        return project

    # --------------------------------------------------------------- storage
    def ensure_local_storage(self, project_id: int, folder: Path, title: str, sync: bool = True) -> int:
        """Attach a local folder (must be under the document root) to the project.

        Label Studio only serves ``/data/local-files/?d=...`` for files inside a folder covered by a storage
        of the project, so every project showing local images needs one. ``sync`` also turns each image of
        the folder into a task: wanted for the raw images, not for the review project, whose tasks (with
        their predictions) are imported separately.
        """
        storages = self._request("GET", "/api/storages/localfiles", params={"project": project_id}) or []
        path = str(folder.resolve())
        storage = next((s for s in storages if Path(s.get("path", "")).resolve() == Path(path)), None)
        if storage is None:
            storage = self._request("POST", "/api/storages/localfiles", json={
                "project": project_id, "path": path, "title": title,
                "regex_filter": IMAGE_REGEX, "use_blob_urls": True,
            })
        if sync:
            self._request("POST", f"/api/storages/localfiles/{storage['id']}/sync")
        return int(storage["id"])

    def import_tasks(self, project_id: int, tasks: list[dict[str, Any]]) -> None:
        if tasks:
            self._request("POST", f"/api/projects/{project_id}/import", json=tasks)

    def export(self, project_id: int) -> list[dict[str, Any]]:
        """Export all annotated tasks as Label Studio JSON."""
        return self._request("GET", f"/api/projects/{project_id}/export", params={"exportType": "JSON"})

    def export_url(self, project_id: int, export_type: str = "YOLO") -> str:
        """Download URL of a native export (YOLO, COCO, ...) for manual use."""
        return f"{self.url}/api/projects/{project_id}/export?exportType={export_type}"


def local_file_url(path: Path, document_root: Path) -> str:
    """URL under which Label Studio serves a local file."""
    rel = path.resolve().relative_to(document_root.resolve()).as_posix()
    return f"/data/local-files/?d={rel}"
