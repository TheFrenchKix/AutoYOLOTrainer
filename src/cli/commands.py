"""Implementation of the ``main.py`` sub-commands (one function per command)."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.context import AppContext
from src.utils.files import list_images
from src.utils.logging import get_logger

console = Console()
log = get_logger("training")


def print_outcome(outcome) -> None:  # noqa: ANN001
    if outcome is None:
        return
    table = Table(title=f"Modèle v{outcome.model_version} (dataset v{outcome.dataset_version})")
    table.add_column("Métrique")
    table.add_column("Actif" + (f" v{outcome.reference_version}" if outcome.reference_version else ""))
    table.add_column("Candidat")
    ref = outcome.reference.to_dict() if outcome.reference else {}
    for name, value in outcome.candidate.to_dict().items():
        table.add_row(name, f"{ref[name]:.4f}" if name in ref else "-", f"{value:.4f}")
    console.print(table)
    status = "[green]promu (actif)[/]" if outcome.promoted else "[yellow]non promu, ancien modèle conservé[/]"
    console.print(f"Résultat: {status} · rapport: {outcome.report}")


def print_comparison(ctx: AppContext, outcomes: list) -> None:  # noqa: ANN001
    """Side-by-side table of the models of one training session (accuracy, speed, verdict)."""
    registry = ctx.registry
    table = Table(title="Comparaison des modèles")
    for column in ("Modèle", "Architecture", "mAP50", "mAP50-95", "F1", "ms/image", "FPS", "Résultat"):
        table.add_column(column)
    for outcome in outcomes:
        info = registry.info(outcome.model_version)
        metrics = outcome.candidate.to_dict()
        speed = info.get("speed_ms")
        table.add_row(
            f"v{outcome.model_version}", registry.architecture(outcome.model_version),
            f"{metrics['map50']:.4f}", f"{metrics['map50_95']:.4f}", f"{metrics['f1']:.4f}",
            f"{speed:.1f}" if speed else "-", f"{1000 / speed:.0f}" if speed else "-",
            "[green]promu[/]" if outcome.promoted else "non promu",
        )
    console.print(table)


def print_outcomes(ctx: AppContext, outcomes: list) -> None:  # noqa: ANN001
    for outcome in outcomes:
        print_outcome(outcome)
    if len(outcomes) > 1:
        print_comparison(ctx, outcomes)


# ---------------------------------------------------------------------- train
def cmd_train(ctx: AppContext, args: argparse.Namespace) -> None:
    """Initial training on datasets/current (YOLO11n, 100 epochs by default); several --model compare them."""
    print_outcomes(ctx, ctx.trainer.train_models(args.model, args.epochs))


def cmd_retrain(ctx: AppContext, args: argparse.Namespace) -> None:
    outcomes = ctx.trainer.retrain_models(force=args.force, models=args.model, epochs=args.epochs)
    if not outcomes:
        console.print(f"[yellow]{ctx.trainer.pending_images()}/{ctx.config.training.min_new_images} "
                      "nouvelles images: pas de ré-entraînement (--force pour forcer)[/]")
    print_outcomes(ctx, outcomes)


def cmd_benchmark(ctx: AppContext, args: argparse.Namespace) -> None:
    """Inference speed of the registered models, measured back to back on the same validation images."""
    results = ctx.trainer.benchmark_models(args.version)
    table = Table(title="Vitesse d'inférence (médiane par image, pré/post-traitement inclus)")
    for column in ("Modèle", "Architecture", "Paramètres (M)", "ms/image", "FPS"):
        table.add_column(column)
    for version, speed in results.items():
        table.add_row(f"v{version}", ctx.registry.architecture(version), f"{speed['params_m']:.1f}",
                      f"{speed['inference_ms']:.1f}", f"{1000 / speed['inference_ms']:.0f}")
    console.print(table)


# -------------------------------------------------------------------- capture
def _open_source(ctx: AppContext, args: argparse.Namespace, zone: int | None = None):  # noqa: ANN202
    """Frame source of the configuration, with the ``--source/--path/--window/--screen`` options applied."""
    from src.capture.sources import create_source

    source_cfg = ctx.config.capture.source
    if args.source:
        source_cfg.type = args.source
    if args.path:
        source_cfg.path = args.path
    if args.window:
        source_cfg.window = args.window
    if args.screen is not None:
        source_cfg.screen = args.screen
    return create_source(source_cfg, zone), source_cfg


def _build_capture_loop(ctx: AppContext, args: argparse.Namespace):  # noqa: ANN202
    from src.capture.runner import CaptureLoop

    source, source_cfg = _open_source(ctx, args)
    detector = ctx.detector()
    loop = CaptureLoop(source, detector, ctx.capture_manager(), fps=source_cfg.fps)
    return loop, detector


def cmd_preview(ctx: AppContext, args: argparse.Namespace) -> None:
    """Live window with the detections of a model (boxes, labels, confidences)."""
    from src.capture.preview import CvWindow, run_preview
    from src.capture.sources import BackgroundSource
    from src.model.zone import ZoneDetector

    if args.size < 100:
        raise ValueError("--size doit être au moins 100")
    if args.zone < 0:
        raise ValueError("--zone ne peut pas être négatif")
    model = ctx.detector(Path(args.model) if args.model else None, args.conf)
    source, source_cfg = _open_source(ctx, args, args.zone or None)
    detector = ZoneDetector(model, args.zone, getattr(source, "reference", None)) if args.zone else model
    live = source_cfg.type.lower() in ("screen", "window", "camera")
    if live:
        source = BackgroundSource(source)  # capture the next frame while the model works on this one
    fps = args.fps if args.fps is not None else (0.0 if live else source_cfg.fps)
    zone_note = f", zone centrale {args.zone}px" if args.zone else ""
    keys = "q ou Échap pour fermer, + / - pour le seuil de confiance"
    if args.zone:
        keys += ", z zone / image entière"
    console.print(f"[cyan]Aperçu de {model.weights}{zone_note}. {keys}[/]")
    frames = run_preview(source, detector, CvWindow("AutoYOLO - apercu"), size=args.size, fps=fps,
                         duration=args.duration)
    console.print(f"{frames} image(s) affichée(s)")


def cmd_capture(ctx: AppContext, args: argparse.Namespace) -> None:
    loop, _ = _build_capture_loop(ctx, args)
    console.print("[cyan]Capture en cours, Ctrl+C pour arrêter[/]")
    loop.run(max_frames=args.max_frames, duration=args.duration)
    console.print(f"Statistiques: {dict(loop.manager.stats)}")


def cmd_run(ctx: AppContext, args: argparse.Namespace) -> None:
    """Fully autonomous mode: capture + auto-labeling + periodic retraining."""
    from src.labeler.watcher import CaptureWatcher
    from src.trainer.scheduler import RetrainScheduler

    loop, detector = _build_capture_loop(ctx, args)
    watcher = CaptureWatcher(ctx.workspace.captures, ctx.auto_labeler(detector))
    scheduler = RetrainScheduler(ctx.trainer, ctx.config.training.check_interval_minutes * 60, detector.reload)
    watcher.start()
    scheduler.start()
    console.print("[cyan]Mode autonome démarré (capture + labeling + ré-entraînement). Ctrl+C pour arrêter.[/]")
    try:
        loop.run(duration=args.duration)
    finally:
        scheduler.stop()
        watcher.stop()


# ---------------------------------------------------------------------- label
def cmd_label(ctx: AppContext, args: argparse.Namespace) -> None:
    labeler = ctx.auto_labeler()
    folder = Path(args.folder) if args.folder else ctx.workspace.captures
    stats = labeler.process_folder(folder, limit=args.limit)
    console.print(f"Auto-labeling: {dict(stats)}")
    if args.watch:
        from src.labeler.watcher import CaptureWatcher

        watcher = CaptureWatcher(folder, labeler)
        watcher.start()
        console.print(f"[cyan]Surveillance de {folder}, Ctrl+C pour arrêter[/]")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            watcher.stop()


# ---------------------------------------------------------------------- stats
def cmd_stats(ctx: AppContext, args: argparse.Namespace) -> None:
    from src.dashboard.stats import collect_stats

    stats = collect_stats(ctx.config, ctx.workspace, heavy=not args.fast)
    table = Table(title="AutoYOLO - statistiques", show_header=False)
    rows = [
        ("Images du dataset", stats.dataset_images),
        ("Annotations", stats.dataset_annotations),
        ("Versions de dataset", ", ".join(f"v{v}" for v in stats.dataset_versions) or "-"),
        ("Modèle actif", f"v{stats.active_model}" if stats.active_model else "aucun"),
        ("Meilleure mAP50 / mAP50-95", f"{stats.best_map50 or 0:.4f} / {stats.best_map50_95 or 0:.4f}"),
        ("Captures totales / à traiter", f"{stats.captures_total} / {stats.captures_pending}"),
        ("accepted / review / rejected", f"{stats.accepted} / {stats.review} / {stats.rejected}"),
        ("Avant prochain entraînement", f"{stats.pending_for_training}/{stats.min_new_images}"),
        ("Dernier entraînement", stats.last_training or "-"),
        ("Espace disque libre", stats.disk_free),
    ]
    rows += [(f"Taille {k}/", v) for k, v in stats.disk_usage.items()]
    rows += [(f"Classe '{k}'", v) for k, v in stats.per_class.items()]
    for label, value in rows:
        table.add_row(label, str(value))
    console.print(table)


# -------------------------------------------------------------------- version
def cmd_version(ctx: AppContext, args: argparse.Namespace) -> None:
    from src import __version__

    if args.promote is not None:
        ctx.registry.promote(args.promote)
        console.print(f"[green]Modèle v{args.promote} activé[/]")

    console.print(f"AutoYOLO Trainer {__version__}")
    models = Table(title="Modèles")
    for col in ("Version", "Date", "Dataset", "Architecture", "mAP50", "mAP50-95", "F1", "ms/image", "Actif"):
        models.add_column(col)
    for info in ctx.registry.history():
        m = info.get("metrics") or {}
        speed = info.get("speed_ms")
        models.add_row(f"v{info['version']}", info.get("created_at", "-"), f"v{info.get('dataset_version')}",
                       info["architecture"], f"{m.get('map50', 0):.4f}", f"{m.get('map50_95', 0):.4f}",
                       f"{m.get('f1', 0):.4f}", f"{speed:.1f}" if speed else "-", "★" if info["active"] else "")
    console.print(models)

    datasets = Table(title="Datasets")
    for col in ("Version", "Date", "Images", "Boîtes", "Note"):
        datasets.add_column(col)
    for version in ctx.datasets.versions.versions():
        manifest = ctx.datasets.version_manifest(version)
        counts = manifest.get("counts", {})
        datasets.add_row(f"v{version}", manifest.get("created_at", "archivé"), str(counts.get("images", "-")),
                         str(counts.get("boxes", "-")), manifest.get("note", ""))
    console.print(datasets)


# ---------------------------------------------------------------------- clean
def cmd_clean(ctx: AppContext, args: argparse.Namespace) -> None:
    """Purge old rejected images, prune old runs, archive old dataset versions."""
    cfg, ws = ctx.config.clean, ctx.workspace
    limit = datetime.now() - timedelta(days=cfg.rejected_retention_days)
    old_rejected = [p for p in list_images(ws.rejected) if datetime.fromtimestamp(p.stat().st_mtime) < limit]

    runs = sorted((p for p in ws.runs.glob("train_*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    # Never delete a run referenced by a model version that is still on disk.
    referenced = {str(Path(h["run_dir"]).resolve()) for h in ctx.registry.history() if h.get("run_dir")}
    old_runs = [r for r in runs[:-cfg.keep_runs or None] if str(r.resolve()) not in referenced]

    console.print(f"{len(old_rejected)} images rejetées > {cfg.rejected_retention_days} j, "
                  f"{len(old_runs)} anciens runs")
    if args.dry_run:
        console.print("[yellow]--dry-run: rien n'a été supprimé[/]")
        return
    for image in old_rejected:
        for suffix in (image.suffix, ".txt", ".json"):
            image.with_suffix(suffix).unlink(missing_ok=True)
    for run in old_runs:
        shutil.rmtree(run, ignore_errors=True)
    archived = ctx.datasets.versions.archive_old(cfg.keep_dataset_versions)
    console.print(f"{len(archived)} versions de dataset archivées dans {ws.archive}")
    if args.dedup:
        removed = ctx.datasets.remove_duplicates(ctx.config.active_learning.dedup.max_hamming_distance)
        console.print(f"{len(removed)} doublons supprimés du dataset")
    console.print("[green]Nettoyage terminé[/]")


# -------------------------------------------------------------------- dataset
def cmd_dataset(ctx: AppContext, args: argparse.Namespace) -> None:
    dm = ctx.datasets
    dm.init_structure()
    if args.action == "report":
        console.print(f"Rapport: {dm.report()}")
    elif args.action == "validate":
        problems = dm.validate()
        for problem in problems[:200]:
            console.print(f"[red]- {problem}[/]")
        console.print(f"{len(problems)} problème(s)")
    elif args.action == "dedup":
        removed = dm.remove_duplicates(ctx.config.active_learning.dedup.max_hamming_distance, dry_run=args.dry_run)
        console.print(f"{len(removed)} doublon(s)")
    elif args.action == "merge":
        console.print(f"{dm.merge([Path(p) for p in args.sources])} images fusionnées")
    elif args.action == "import":
        console.print(f"{dm.import_folder(Path(args.sources[0]), move=False)} images importées")
    elif args.action == "snapshot":
        version, path = dm.create_version(note="manual snapshot")
        console.print(f"Dataset v{version}: {path}")


# --------------------------------------------------------- dashboard / export
def cmd_dashboard(ctx: AppContext, args: argparse.Namespace) -> None:
    import uvicorn

    from src.dashboard.app import create_app

    cfg = ctx.config.dashboard
    console.print(f"Dashboard: http://{cfg.host}:{cfg.port}")
    uvicorn.run(create_app(ctx.config, ctx.workspace), host=cfg.host, port=cfg.port, log_level="warning")


def cmd_export(ctx: AppContext, args: argparse.Namespace) -> None:
    detector = ctx.detector()
    target = detector.export_onnx(ctx.registry.versions.active_path(), dynamic=args.dynamic)
    console.print(f"[green]Modèle exporté en ONNX: {target}[/]")
