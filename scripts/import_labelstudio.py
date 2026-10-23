"""Fetch Label Studio annotations and convert them to YOLO.

    python scripts/import_labelstudio.py                    # projet principal -> datasets/current
    python scripts/import_labelstudio.py --review           # projet de revue  -> accepted/
    python scripts/import_labelstudio.py --from-file export.json

Re-running is safe: images already in the dataset get their labels updated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from src.context import AppContext
from src.datasets.labelstudio import convert_tasks
from src.datasets.labelstudio_client import LabelStudioClient
from src.datasets.yolo_format import write_labels
from src.utils.files import move_with_sidecars, write_json

ROOT = _bootstrap.ROOT


def fetch_tasks(ctx: AppContext, title: str, from_file: str | None) -> list[dict]:
    if from_file:
        return json.loads(Path(from_file).read_text(encoding="utf-8"))
    cfg = ctx.config.labelstudio
    client = LabelStudioClient(cfg.url, cfg.api_token)
    project = client.find_project(title)
    if project is None:
        raise SystemExit(f"Projet Label Studio introuvable: {title!r} (lancez scripts/start_labelstudio.py)")
    tasks = client.export(project["id"])
    # Keep every raw export for traceability.
    write_json(ctx.workspace.state / "labelstudio_exports" / f"{datetime.now():%Y%m%d_%H%M%S}.json", tasks)
    return tasks


def import_review(ctx: AppContext, tasks: list[dict]) -> tuple[int, int]:
    """Human-validated review images: write corrected labels and move to accepted/.

    A file that cannot be moved (still open elsewhere) is reported and skipped, so the other images are
    still imported; running the import again moves it. Returns (moved, failed).
    """
    converted, warnings = convert_tasks(tasks, ctx.config.class_names, ROOT, [ctx.workspace.review])
    moved = failed = 0
    for item in converted:
        if item.image_path is None or item.image_path.parent != ctx.workspace.review:
            continue
        try:
            write_labels(item.image_path.with_suffix(".txt"), item.boxes)
            move_with_sidecars(item.image_path, ctx.workspace.accepted)
            moved += 1
        except OSError as exc:
            failed += 1
            print(f"! {item.image_path.name}: déplacement impossible ({exc})")
    for warning in warnings:
        print(f"! {warning}")
    return moved, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", action="store_true", help="importer le projet de revue vers accepted/")
    parser.add_argument("--from-file", help="export JSON Label Studio déjà téléchargé")
    parser.add_argument("--dedup", action="store_true", help="supprimer les doublons après import")
    parser.add_argument("-o", "--override", action="append", default=[])
    args = parser.parse_args()

    ctx = AppContext.create(args.override)
    cfg = ctx.config.labelstudio
    title = cfg.review_project_title if args.review else cfg.project_title
    tasks = fetch_tasks(ctx, title, args.from_file)
    print(f"{len(tasks)} tâches récupérées")

    if args.review:
        moved, failed = import_review(ctx, tasks)
        print(f"{moved} images validées déplacées vers {ctx.workspace.accepted}")
        if failed:
            print(f"{failed} image(s) n'ont pas pu être déplacées (fichier utilisé ?): relancez l'import")
        return 1 if failed else 0

    search_dirs = [ctx.workspace.raw_images, ROOT / cfg.data_dir / "media" / "upload"]
    search_dirs += [p for p in (ROOT / cfg.data_dir / "media" / "upload").glob("*") if p.is_dir()]
    summary = ctx.datasets.import_labelstudio(tasks, document_root=ROOT, search_dirs=search_dirs)
    for warning in summary.warnings:
        print(f"! {warning}")
    if args.dedup:
        ctx.datasets.remove_duplicates(ctx.config.active_learning.dedup.max_hamming_distance)
    counts = ctx.datasets.count()
    print(f"{summary.imported} images importées, {summary.skipped} ignorées")
    print(f"Dataset: {counts.images} images, {counts.boxes} boîtes, par classe: {counts.per_class}")
    print(f"Rapport: {ctx.datasets.report()}")
    print("Étape suivante: python scripts/train_initial.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
