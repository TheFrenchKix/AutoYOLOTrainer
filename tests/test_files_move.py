import shutil
from pathlib import Path

import pytest

from src.utils import files
from src.utils.files import move_with_sidecars


def test_move_waits_for_a_file_that_is_still_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "review" / "a.png"
    image.parent.mkdir()
    image.write_bytes(b"png")
    image.with_suffix(".txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")

    real_move, locked = shutil.move, {"left": 2}

    def flaky_move(src, dst):  # noqa: ANN001, ANN202
        if locked["left"] > 0 and str(src).endswith("a.png"):
            locked["left"] -= 1
            raise PermissionError(32, "fichier utilisé par un autre processus")
        return real_move(src, dst)

    monkeypatch.setattr(files.shutil, "move", flaky_move)
    monkeypatch.setattr(files.time, "sleep", lambda _s: None)
    target = move_with_sidecars(image, tmp_path / "accepted")
    assert target.read_bytes() == b"png" and (tmp_path / "accepted" / "a.txt").exists()
    assert not image.exists() and not image.with_suffix(".txt").exists()


def test_move_gives_up_after_the_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"png")
    calls = []

    def always_locked(src, dst):  # noqa: ANN001, ANN202
        calls.append(src)
        raise PermissionError(32, "verrouillé")

    monkeypatch.setattr(files.shutil, "move", always_locked)
    monkeypatch.setattr(files.time, "sleep", lambda _s: None)
    with pytest.raises(PermissionError):
        move_with_sidecars(image, tmp_path / "accepted")
    assert len(calls) == 5 and image.exists()
