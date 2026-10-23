"""Install (if needed) and start Label Studio, then prepare the annotation project.

    python scripts/start_labelstudio.py            # projet principal (datasets/raw)
    python scripts/start_labelstudio.py --review   # + projet de revue (review/ pré-annoté)
    python scripts/start_labelstudio.py --detach   # laisser tourner en arrière-plan

Label Studio is installed in its own virtual environment (``.venv-labelstudio``)
because its pinned dependencies (Django, numpy...) can conflict with
Ultralytics/PyTorch. Images are served from the project folder through Label
Studio's *local files* storage, so nothing is copied or uploaded.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

import _bootstrap  # noqa: F401

from src.context import AppContext
from src.datasets.labelstudio import build_label_config, detections_to_ls_prediction
from src.datasets.labelstudio_client import LabelStudioClient, local_file_url
from src.model.types import Detection
from src.utils.files import list_images, read_json, write_json

ROOT = _bootstrap.ROOT


def ensure_labelstudio(venv_dir: Path) -> Path:
    """Create the dedicated venv and install Label Studio once."""
    scripts = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    exe = scripts / ("label-studio.exe" if os.name == "nt" else "label-studio")
    if exe.exists():
        return exe
    print(f"Installation de Label Studio dans {venv_dir} (une seule fois, quelques minutes)...")
    venv.create(venv_dir, with_pip=True)
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    subprocess.check_call([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements-labelstudio.txt")])
    return exe


def start_server(ctx: AppContext, exe: Path) -> subprocess.Popen:
    cfg = ctx.config.labelstudio
    env = {
        **os.environ,
        # Serve images straight from the project folder.
        "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
        "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(ROOT),
        "LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN": "true",
        "PYTHONUTF8": "1",  # avoids cp1252 errors on Windows consoles
    }
    data_dir = ROOT / cfg.data_dir
    command = [
        str(exe), "start", "--port", str(cfg.port), "--no-browser",
        "--data-dir", str(data_dir),
        "--username", cfg.username, "--password", cfg.password, "--user-token", cfg.api_token,
    ]
    print(f"Démarrage de Label Studio sur {cfg.url} ...")
    log_file = (ctx.workspace.logs / "labelstudio.log").open("a", encoding="utf-8")
    return subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def push_review_tasks(ctx: AppContext, client: LabelStudioClient, project_id: int) -> int:
    """Import review/ images as tasks pre-annotated with the model predictions."""
    state_file = ctx.workspace.state / "labelstudio_review_imported.json"
    already = set(read_json(state_file, default=[]))
    tasks, names = [], []
    for image in list_images(ctx.workspace.review):
        if image.name in already:
            continue
        meta = read_json(image.with_suffix(".json"), default={}) or {}
        detections = [Detection.from_dict(d) for d in meta.get("detections", [])]
        version = (meta.get("labeling") or {}).get("model_version") or "unknown"
        tasks.append({
            "data": {"image": local_file_url(image, ROOT)},
            "predictions": [detections_to_ls_prediction(detections, version)],
        })
        names.append(image.name)
    client.import_tasks(project_id, tasks)
    write_json(state_file, sorted(already | set(names)))
    return len(tasks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", action="store_true", help="créer/mettre à jour le projet de revue")
    parser.add_argument("--detach", action="store_true", help="ne pas attendre la fin du serveur")
    parser.add_argument("-o", "--override", action="append", default=[])
    args = parser.parse_args()

    ctx = AppContext.create(args.override)
    cfg = ctx.config.labelstudio
    client = LabelStudioClient(cfg.url, cfg.api_token)

    process = None
    if client.is_alive():
        print(f"Label Studio déjà démarré sur {cfg.url}")
    else:
        process = start_server(ctx, ensure_labelstudio(ROOT / cfg.venv_dir))
        client.wait_until_ready()

    label_config = build_label_config(ctx.config.class_names)
    project = client.ensure_project(cfg.project_title, label_config, "Annotation initiale AutoYOLO")
    ctx.workspace.raw_images.mkdir(parents=True, exist_ok=True)
    client.ensure_local_storage(project["id"], ctx.workspace.raw_images, "raw images")
    n_raw = len(list_images(ctx.workspace.raw_images))
    print(f"Projet '{cfg.project_title}': {cfg.url}/projects/{project['id']} "
          f"({n_raw} images dans {ctx.workspace.raw_images})")
    print(f"Export YOLO natif disponible: {client.export_url(project['id'], 'YOLO')}")

    if args.review:
        review = client.ensure_project(cfg.review_project_title, label_config, "Revue des images incertaines")
        # Without a storage covering review/, Label Studio refuses to serve the images (404 on /data/local-files).
        client.ensure_local_storage(review["id"], ctx.workspace.review, "review images", sync=False)
        pushed = push_review_tasks(ctx, client, review["id"])
        print(f"Projet de revue: {cfg.url}/projects/{review['id']} ({pushed} nouvelles tâches pré-annotées)")

    print(f"Identifiants: {cfg.username} / {cfg.password}")
    print("Après annotation: python scripts/import_labelstudio.py")
    if process is not None and not args.detach:
        print("Ctrl+C pour arrêter Label Studio.")
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
