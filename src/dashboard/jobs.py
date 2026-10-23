"""Background jobs of the dashboard: each action runs as a child process, its output goes to a log file."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

MAX_HISTORY = 50
MAX_CHUNK = 256 * 1024
STOP_TIMEOUT_SECONDS = 10


class JobConflict(RuntimeError):
    """The same action is already running."""


@dataclass
class Job:
    id: str
    action: str
    label: str
    command: str
    started_at: str
    finished_at: str | None = None
    status: str = "running"  # running | done | failed | stopped | interrupted
    returncode: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def decode_chunk(data: bytes) -> tuple[str, int]:
    """Decode a log chunk without cutting a UTF-8 character in two. Returns (text, bytes consumed)."""
    try:
        return data.decode("utf-8"), len(data)
    except UnicodeDecodeError as exc:
        if exc.start >= len(data) - 3 and exc.reason == "unexpected end of data":
            return data[: exc.start].decode("utf-8"), exc.start
        return data.decode("utf-8", errors="replace"), len(data)


class JobManager:
    """Starts, follows and stops the child processes launched from the dashboard."""

    def __init__(self, folder: Path, cwd: Path) -> None:
        self.folder = folder
        self.cwd = cwd
        folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._stopping: set[str] = set()
        self._load()

    # ------------------------------------------------------------------ persistence
    def _log_path(self, job_id: str) -> Path:
        return self.folder / f"{job_id}.log"

    def _save(self, job: Job) -> None:
        (self.folder / f"{job.id}.json").write_text(json.dumps(job.to_dict(), ensure_ascii=False), encoding="utf-8")

    def _load(self) -> None:
        for path in sorted(self.folder.glob("*.json")):
            try:
                job = Job(**json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue
            if job.status == "running":  # the dashboard stopped before the job reported its end
                job.status = "interrupted"
                self._save(job)
            self._jobs[job.id] = job

    def _prune(self) -> None:
        finished = [j for j in self._jobs.values() if j.id not in self._processes]
        for job in sorted(finished, key=lambda j: j.id)[: max(0, len(self._jobs) - MAX_HISTORY)]:
            for path in (self.folder / f"{job.id}.json", self._log_path(job.id)):
                path.unlink(missing_ok=True)
            del self._jobs[job.id]

    # ----------------------------------------------------------------------- state
    def _refresh(self, job: Job) -> None:
        process = self._processes.get(job.id)
        if process is None or process.poll() is None:
            return
        job.returncode = process.returncode
        job.finished_at = datetime.now().isoformat(timespec="seconds")
        if job.id in self._stopping:
            job.status = "stopped"
        else:
            job.status = "done" if process.returncode == 0 else "failed"
        del self._processes[job.id]
        self._stopping.discard(job.id)
        self._save(job)

    def history(self) -> list[Job]:
        """Every known job, newest first."""
        with self._lock:
            for job in self._jobs.values():
                self._refresh(job)
            return sorted(self._jobs.values(), key=lambda j: j.id, reverse=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                self._refresh(job)
            return job

    def read_log(self, job_id: str, offset: int = 0) -> tuple[str, int]:
        """Log text from byte ``offset`` and the offset to ask for next."""
        path = self._log_path(job_id)
        if not path.exists():
            return "", 0
        with path.open("rb") as handle:
            handle.seek(max(0, offset))
            text, consumed = decode_chunk(handle.read(MAX_CHUNK))
        return text, max(0, offset) + consumed

    # --------------------------------------------------------------------- control
    def start(self, action: str, label: str, argv: list[str], command: str) -> Job:
        with self._lock:
            if any(j.action == action and j.status == "running" for j in self.history()):
                raise JobConflict(f"« {label} » est déjà en cours")
            job_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            options: dict = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                if sys.platform == "win32" else {"start_new_session": True}
            )
            with self._log_path(job_id).open("wb") as log:
                process = subprocess.Popen(
                    [sys.executable, *argv], cwd=self.cwd, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, **options,
                )
            job = Job(job_id, action, label, command, datetime.now().isoformat(timespec="seconds"))
            self._jobs[job_id] = job
            self._processes[job_id] = process
            self._save(job)
            self._prune()
            return job

    def stop(self, job_id: str) -> Job | None:
        """Stop a running job together with the processes it started."""
        with self._lock:
            job = self.get(job_id)
            process = self._processes.get(job_id)
            if job is None or process is None:
                return job
            self._stopping.add(job_id)
            self._kill_tree(process)
        try:
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        return self.get(job_id)

    @staticmethod
    def _kill_tree(process: subprocess.Popen) -> None:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
        else:
            os.killpg(process.pid, signal.SIGTERM)

    def shutdown(self) -> None:
        """Stop every running job (called when the dashboard exits, so no process is left orphaned)."""
        for job in self.history():
            if job.status == "running":
                self.stop(job.id)
