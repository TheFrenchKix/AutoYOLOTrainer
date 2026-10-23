import copy
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.cli.parser import build_parser
from src.dashboard.actions import GROUPS, Action, Field, build_command, catalog, find_action
from src.dashboard.app import create_app
from src.dashboard.jobs import Job, JobConflict, JobManager, decode_chunk

ACTIONS = [item for group in GROUPS for item in group.items if isinstance(item, Action)]


def test_catalog_is_consistent() -> None:
    ids = [a.id for a in ACTIONS]
    assert len(ids) == len(set(ids))
    for action in ACTIONS:
        names = {f.name for f in action.fields}
        assert len(names) == len(action.fields)
        for field in action.fields:
            assert field.kind in {"text", "int", "float", "bool", "choice", "multichoice", "lines"}
            assert field.flag is None or field.flag.startswith("--")
            assert field.kind != "bool" or field.flag
            assert field.show_if is None or field.show_if[0] in names
    assert {g["id"] for g in catalog()} == {g.id for g in GROUPS}


def _sample(field: Field):  # noqa: ANN202
    return {
        "bool": True, "int": max(1, int(field.minimum or 1)), "float": 5.0,
        "choice": field.choices[0] if field.choices else None, "lines": ["a", "b"], "text": "x",
        "multichoice": list(field.choices[:2]),
    }[field.kind]


@pytest.mark.parametrize("action", [a for a in ACTIONS if a.script == "main.py"], ids=lambda a: a.id)
def test_main_actions_are_accepted_by_the_real_cli_parser(action: Action) -> None:
    values = {f.name: _sample(f) for f in action.fields}
    command = build_command(action.id, values, ["training.epochs=3"])
    args = build_parser().parse_args(command.argv[1:])
    assert args.override == ["training.epochs=3"]


def test_command_layout() -> None:
    train = build_command("train", {"epochs": "5", "other_model": " yolo11s.pt "}, ["a.b=1"])
    assert train.argv == ["main.py", "-o", "a.b=1", "train", "--model=yolo11s.pt", "--epochs=5"]
    script = build_command("labelstudio_import", {"dedup": True, "from_file": "C:/x.json"}, ["a.b=1"])
    assert script.argv == ["scripts/import_labelstudio.py", "--from-file=C:/x.json", "--dedup", "-o", "a.b=1"]
    merge = build_command("dataset_merge", {"sources": "C:/a\n\nC:/b"}, [])
    assert merge.argv == ["main.py", "dataset", "merge", "C:/a", "C:/b"]
    assert build_command("clean", {"dry_run": False}, []).argv == ["main.py", "clean"]
    assert build_command("promote", {"version": 3}, []).argv == ["main.py", "version", "--promote=3"]


def test_screenshots_target_selects_the_right_option() -> None:
    window = build_command("screenshots", {"target": "window", "window": "Jeu", "screen": 2, "count": 10}, [])
    assert window.argv == ["scripts/screenshots.py", "--window=Jeu", "--count=10"]
    screen = build_command("screenshots", {"target": "screen", "window": "", "screen": 0}, [])
    assert screen.argv == ["scripts/screenshots.py", "--screen=0"]
    with pytest.raises(ValueError, match="requis"):
        build_command("screenshots", {"target": "window", "window": ""}, [])


@pytest.mark.parametrize(("action", "values", "overrides"), [
    ("nope", {}, []),
    ("train", {"epochs": 0}, []),
    ("train", {"epochs": "abc"}, []),
    ("train", {"epochs": True}, []),
    ("train", {"epochs": float("nan")}, []),
    ("train", {"other_model": "--help"}, []),
    ("train", {"other_model": "a\nb"}, []),
    ("train", {"models": ["yolo-inconnu.pt"]}, []),
    ("train", {"models": "yolo11n.pt"}, []),
    ("train", {"unknown": 1}, []),
    ("retrain", {"force": "yes"}, []),
    ("capture", {"source": "webcam"}, []),
    ("dataset_merge", {"sources": []}, []),
    ("train", {}, ["not an override"]),
    ("train", {}, ["x=1\ny=2"]),
    ("screenshots", {"target": "screen"}, ["a=1"]),
])
def test_invalid_requests_are_refused(action: str, values: dict, overrides: list[str]) -> None:
    with pytest.raises(ValueError):
        build_command(action, values, overrides)


def test_find_action() -> None:
    assert find_action("train").command == ("train",)


# ----------------------------------------------------------------------------- jobs
def _wait(manager: JobManager, job_id: str, timeout: float = 20) -> Job:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if job and job.status != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("job still running")


def test_job_success_failure_and_log(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs", tmp_path)
    ok = manager.start("a", "A", ["-c", "print('héllo \u00e9')"], "cmd")
    job = _wait(manager, ok.id)
    assert job.status == "done" and job.returncode == 0 and job.finished_at
    text, offset = manager.read_log(ok.id)
    assert "héllo" in text and manager.read_log(ok.id, offset)[0] == ""

    bad = manager.start("b", "B", ["-c", "import sys; print('boom'); sys.exit(3)"], "cmd")
    job = _wait(manager, bad.id)
    assert job.status == "failed" and job.returncode == 3
    assert [j.id for j in manager.history()] == sorted([ok.id, bad.id], reverse=True)


def test_job_conflict_stop_and_restart(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs", tmp_path)
    job = manager.start("long", "Long", ["-c", "import time; time.sleep(60)"], "cmd")
    with pytest.raises(JobConflict):
        manager.start("long", "Long", ["-c", "pass"], "cmd")
    stopped = manager.stop(job.id)
    assert stopped is not None and stopped.status == "stopped"

    running = manager.start("again", "Again", ["-c", "import time; time.sleep(60)"], "cmd")
    reloaded = JobManager(tmp_path / "jobs", tmp_path)  # a new dashboard process sees it as interrupted
    assert reloaded.get(running.id).status == "interrupted"
    manager.shutdown()
    assert manager.get(running.id).status == "stopped"


def test_decode_chunk_keeps_split_characters() -> None:
    data = "aé".encode()
    assert decode_chunk(data) == ("aé", 3)
    assert decode_chunk(data[:2]) == ("a", 1)  # half of "é" is left for the next read
    assert decode_chunk(b"a\xffb")[0] == "a\ufffdb"


# ------------------------------------------------------------------------ endpoints
class FakeJobs:
    def __init__(self) -> None:
        self.started: list[tuple] = []

    def start(self, action, label, argv, command):  # noqa: ANN001, ANN201
        if self.started:
            raise JobConflict("déjà en cours")
        self.started.append((action, label, argv, command))
        return Job("j1", action, label, command, "2026-01-01T00:00:00")

    def get(self, job_id):  # noqa: ANN001, ANN201
        if job_id != "j1":
            return None
        return Job("j1", "train", "Premier entraînement", "main.py train", "2026-01-01T00:00:00")

    def stop(self, job_id):  # noqa: ANN001, ANN201
        return self.get(job_id)

    def history(self):  # noqa: ANN201
        return [self.get("j1")]

    def read_log(self, job_id, offset=0):  # noqa: ANN001, ANN201
        return "ligne\n", offset + 6

    def shutdown(self) -> None:
        pass


@pytest.fixture
def client(config, workspace):  # noqa: ANN001, ANN201
    jobs = FakeJobs()
    test_client = TestClient(create_app(config, workspace, jobs), base_url="http://localhost")
    test_client.jobs = jobs
    return test_client


def test_pages_and_read_endpoints(client) -> None:  # noqa: ANN001
    page = client.get("/")
    assert page.status_code == 200 and "<title>AutoYOLO Trainer</title>" in page.text
    assert {g["id"] for g in client.get("/api/catalog").json()} == {g.id for g in GROUPS}
    config = client.get("/api/config").json()
    assert "change-me" not in str(config["labelstudio"])  # password and API token are masked
    assert client.get("/api/models").json() == []
    assert client.get("/api/captures-per-day").json() == []
    assert client.get("/api/jobs").json()[0]["id"] == "j1"
    detail = client.get("/api/jobs/j1?offset=2").json()
    assert detail["log"] == "ligne\n" and detail["offset"] == 8
    assert client.get("/api/jobs/zzz").status_code == 404


def test_start_and_stop_jobs(client) -> None:  # noqa: ANN001
    response = client.post("/api/jobs", json={"action": "train", "values": {"epochs": 2}, "overrides": ["a.b=1"]})
    assert response.status_code == 200
    assert client.jobs.started[0][2] == ["main.py", "-o", "a.b=1", "train", "--epochs=2"]
    assert client.post("/api/jobs", json={"action": "train"}).status_code == 409
    assert client.post("/api/jobs", json={"action": "train", "values": {"epochs": 0}}).status_code == 422
    assert client.post("/api/jobs/j1/stop", json={}).status_code == 200
    assert client.post("/api/jobs/zzz/stop", json={}).status_code == 404


def test_job_requests_are_guarded(client, config, workspace) -> None:  # noqa: ANN001
    body = {"action": "train"}
    assert client.post("/api/jobs", content="action=train", headers={"content-type": "text/plain"}).status_code == 415
    assert client.post("/api/jobs", json=body, headers={"host": "evil.example"}).status_code == 403
    assert client.jobs.started == []

    exposed = copy.deepcopy(config)
    exposed.dashboard.host = "0.0.0.0"
    other = TestClient(create_app(exposed, workspace, FakeJobs()), base_url="http://localhost")
    assert other.post("/api/jobs", json=body).status_code == 403
    assert other.get("/api/stats").status_code == 200  # reading stays possible


def test_real_job_manager_is_used_by_default(config, workspace) -> None:  # noqa: ANN001
    with TestClient(create_app(config, workspace), base_url="http://localhost") as client:
        assert client.get("/api/jobs").json() == []
    assert (workspace.state / "jobs").is_dir() and sys.executable


# ------------------------------------------------------------- screen / window pickers
def test_source_fields_follow_the_chosen_source() -> None:
    screen = build_command("capture", {"source": "screen", "screen": "2", "window": "ignoré", "path": "ignoré"}, [])
    assert screen.argv == ["main.py", "capture", "--source=screen", "--screen=2"]
    window = build_command("preview", {"source": "window", "window": "Mon jeu", "screen": "1"}, [])
    assert window.argv == ["main.py", "preview", "--source=window", "--window=Mon jeu"]
    video = build_command("run", {"source": "video", "path": "essai.mp4"}, [])
    assert video.argv == ["main.py", "run", "--source=video", "--path=essai.mp4"]
    for values in ({}, {"source": "window"}, {"source": "video"}, {"source": "screen", "screen": "-1"}):
        with pytest.raises(ValueError):
            build_command("capture", values, [])


def test_dynamic_fields_are_declared_for_the_pickers() -> None:
    dynamic = {(a.id, f.name): f.dynamic for a in ACTIONS for f in a.fields if f.dynamic}
    assert dynamic[("preview", "window")] == "windows" and dynamic[("preview", "screen")] == "screens"
    assert dynamic[("screenshots", "window")] == "windows" and dynamic[("screenshots", "screen")] == "screens"
    fields = [f for item in catalog()[1]["items"] for f in item.get("fields", [])]
    assert {f["dynamic"] for f in fields} >= {None, "windows", "screens"}


def test_screen_and_window_listing_endpoints(client) -> None:  # noqa: ANN001
    json_header = {"content-type": "application/json"}
    screens = client.get("/api/screens", headers=json_header)
    windows = client.get("/api/windows", headers=json_header)
    assert screens.status_code == 200 and windows.status_code == 200
    if sys.platform == "win32":
        assert screens.json()[0]["index"] == 0  # whole desktop first
        assert sum(1 for s in screens.json() if s["primary"]) == 1
        titles = [w["title"] for w in windows.json()]
        assert len(titles) == len(set(titles)) and all(titles)
    else:
        assert screens.json() == [] and windows.json() == []


def test_listing_endpoints_are_guarded(client) -> None:  # noqa: ANN001
    assert client.get("/api/windows").status_code == 415  # not a JSON request
    evil = {"content-type": "application/json", "host": "evil.example"}
    assert client.get("/api/screens", headers=evil).status_code == 403


def test_review_action_sends_images_without_blocking() -> None:
    command = build_command("labelstudio_review", {"review": True, "detach": True}, [])
    assert command.argv == ["scripts/start_labelstudio.py", "--review", "--detach"]
    assert find_action("labelstudio_start").id != find_action("labelstudio_review").id  # the server job may be running


def test_captures_per_day_reads_the_real_index_format(client, workspace) -> None:  # noqa: ANN001
    index = workspace.state / "captures_index.csv"
    index.write_text(
        "file,timestamp,n_detections,classes,min_confidence,max_confidence,score\n"
        "a.png,2026-10-22T10:00:00.1,1,player,0.5,0.5,0.6\n"
        "b.png,2026-10-23T09:30:00.2,1,player,0.4,0.4,0.6\n"
        "c.png,2026-10-23T15:13:01.851506,2,player,0.3,0.9,0.7\n",
        encoding="utf-8",
    )
    assert client.get("/api/captures-per-day").json() == [
        {"day": "2026-10-22", "captures": 1}, {"day": "2026-10-23", "captures": 2},
    ]
