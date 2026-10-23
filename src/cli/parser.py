"""Argument parser of ``main.py``."""

from __future__ import annotations

import argparse

from src.cli import commands


def _capture_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=["screen", "window", "camera", "video", "folder"],
                        help="surcharge capture.source.type")
    parser.add_argument("--path", help="fichier vidéo ou dossier (source video/folder)")
    parser.add_argument("--window", metavar="TEXTE", help="titre de la fenêtre à capturer (source window)")
    parser.add_argument("--screen", type=int, metavar="N",
                        help="écran à capturer, 0 = tous (source screen, voir scripts/capture.py --list)")
    parser.add_argument("--duration", type=float, help="durée maximale en secondes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="AutoYOLO Trainer - auto-labeling et active learning pour YOLO11",
    )
    parser.add_argument(
        "-o", "--override", action="append", default=[], metavar="CLE=VALEUR",
        help="surcharge Hydra de la configuration (répétable), ex: -o training.epochs=10",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="commande")

    p = sub.add_parser("train", help="premier entraînement, un ou plusieurs modèles à comparer")
    p.add_argument("--epochs", type=int)
    p.add_argument("--model", action="append", metavar="POIDS",
                   help="poids de départ, ex. yolo26n.pt (répétable pour comparer; défaut: training.base_model)")
    p.set_defaults(func=commands.cmd_train)

    p = sub.add_parser("capture", help="capture automatique des détections")
    _capture_options(p)
    p.add_argument("--max-frames", type=int)
    p.set_defaults(func=commands.cmd_capture)

    p = sub.add_parser("preview", help="aperçu en direct des détections du modèle (fenêtre 300x300)")
    _capture_options(p)
    p.add_argument("--size", type=int, default=300, help="côté de la fenêtre en pixels (défaut: 300)")
    p.add_argument("--conf", type=float, help="seuil de confiance (défaut: capture.min_confidence)")
    p.add_argument("--model", help="poids .pt à tester (défaut: modèle actif)")
    p.add_argument("--zone", type=int, default=200, metavar="PX",
                   help="analyse seulement le carré central de PX pixels (défaut: 200, 0 = image entière)")
    p.add_argument("--fps", type=float,
                   help="limite d'images/s (défaut: aucune pour écran/fenêtre/caméra, sinon capture.source.fps)")
    p.set_defaults(func=commands.cmd_preview)

    p = sub.add_parser("label", help="auto-labeling des captures (accepted/review/rejected)")
    p.add_argument("--folder", help="dossier à traiter (défaut: captures/)")
    p.add_argument("--limit", type=int)
    p.add_argument("--watch", action="store_true", help="continuer à surveiller le dossier")
    p.set_defaults(func=commands.cmd_label)

    p = sub.add_parser("retrain", help="ré-entraîner si assez de nouvelles images")
    p.add_argument("--force", action="store_true", help="ignorer training.min_new_images")
    p.add_argument("--epochs", type=int, help="défaut: training.retrain_epochs")
    p.add_argument("--model", action="append", metavar="POIDS",
                   help="entraîne cette architecture depuis ses poids pré-entraînés (répétable pour comparer); "
                        "sans cette option, le modèle actif est affiné")
    p.set_defaults(func=commands.cmd_retrain)

    p = sub.add_parser("benchmark", help="mesure la vitesse d'inférence des modèles enregistrés")
    p.add_argument("--version", type=int, action="append", metavar="N", help="version à mesurer (défaut: toutes)")
    p.set_defaults(func=commands.cmd_benchmark)

    p = sub.add_parser("stats", help="statistiques du projet")
    p.add_argument("--fast", action="store_true", help="ne pas calculer l'espace disque")
    p.set_defaults(func=commands.cmd_stats)

    p = sub.add_parser("version", help="versions des modèles et datasets")
    p.add_argument("--promote", type=int, metavar="N", help="activer le modèle vN (rollback)")
    p.set_defaults(func=commands.cmd_version)

    p = sub.add_parser("clean", help="purger rejetés/anciens runs, archiver les vieux datasets")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dedup", action="store_true", help="supprimer aussi les doublons du dataset")
    p.set_defaults(func=commands.cmd_clean)

    p = sub.add_parser("run", help="mode autonome: capture + labeling + ré-entraînement")
    _capture_options(p)
    p.set_defaults(func=commands.cmd_run)

    p = sub.add_parser("dataset", help="outils dataset")
    p.add_argument("action", choices=["report", "validate", "dedup", "merge", "import", "snapshot"])
    p.add_argument("sources", nargs="*", help="dossiers pour merge/import")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=commands.cmd_dataset)

    p = sub.add_parser("dashboard", help="tableau de bord web (FastAPI)")
    p.set_defaults(func=commands.cmd_dashboard)

    p = sub.add_parser("export", help="exporter le modèle actif en ONNX")
    p.add_argument("--dynamic", action="store_true", help="axes dynamiques")
    p.set_defaults(func=commands.cmd_export)
    return parser
