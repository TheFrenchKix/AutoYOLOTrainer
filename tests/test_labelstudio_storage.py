from pathlib import Path

from src.datasets.labelstudio_client import LabelStudioClient


class RecordingClient(LabelStudioClient):
    def __init__(self, storages: list[dict] | None = None) -> None:
        super().__init__("http://ls", "token")
        self.storages = storages or []
        self.calls: list[tuple[str, str]] = []

    def _request(self, method: str, path: str, **kwargs):  # noqa: ANN003, ANN202
        self.calls.append((method, path))
        if method == "GET":
            return self.storages
        if method == "POST" and path == "/api/storages/localfiles":
            return {"id": 7, "path": kwargs["json"]["path"]}
        return None


def test_review_storage_is_created_without_importing_its_images(tmp_path: Path) -> None:
    client = RecordingClient()
    assert client.ensure_local_storage(3, tmp_path, "review images", sync=False) == 7
    assert ("POST", "/api/storages/localfiles") in client.calls
    assert not any(path.endswith("/sync") for _, path in client.calls)  # would duplicate the pre-annotated tasks


def test_raw_storage_is_synced_and_reused(tmp_path: Path) -> None:
    client = RecordingClient([{"id": 5, "path": str(tmp_path)}])
    assert client.ensure_local_storage(1, tmp_path, "raw images") == 5
    assert client.calls == [("GET", "/api/storages/localfiles"), ("POST", "/api/storages/localfiles/5/sync")]
