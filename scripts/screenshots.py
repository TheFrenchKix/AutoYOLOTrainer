"""Series of screenshots of one window or one screen, to compose the first dataset (Windows).

    python scripts/screenshots.py --window "Jeu" --count 150 --interval 2
    python scripts/screenshots.py --screen 1 --count 200 --interval 1.5 --delay 5

Images go to ``datasets/raw/`` by default, ready to be annotated in Label Studio
(``scripts/start_labelstudio.py``). ``--delay`` gives time to bring the window to the front.
A window that cannot be found at a given shot is skipped. Ctrl+C stops early; shots already
taken are kept. Single shots and window listing: ``scripts/capture.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from PIL import Image

from capture import unique_output_path
from src.capture.windows import (
    capture_window,
    enable_dpi_awareness,
    find_windows,
    grab,
    list_screens,
    list_windows,
)
from src.config.loader import PROJECT_ROOT

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "raw"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--window", metavar="TEXTE", help="capturer la fenêtre dont le titre contient TEXTE")
    target.add_argument("--screen", type=int, metavar="N", help="capturer l'écran N (0 = tous les écrans)")
    parser.add_argument("--count", type=int, default=100, help="nombre de captures (défaut : 100)")
    parser.add_argument("--interval", type=float, default=2.0, metavar="SECONDES", help="pause entre captures")
    parser.add_argument("--delay", type=float, default=3.0, metavar="SECONDES", help="attente avant la 1re capture")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="dossier de sortie")
    return parser


def take_shot(args: argparse.Namespace) -> Image.Image | None:
    """One picture of the target, or ``None`` when the window is no longer visible."""
    if args.window is not None:
        matches = find_windows(args.window, list_windows())
        return capture_window(matches[0]) if matches else None
    return grab(list_screens()[args.screen].bbox)


def check_target(args: argparse.Namespace) -> str | None:
    """Error message when the target does not exist, ``None`` otherwise."""
    if args.window is not None:
        if not find_windows(args.window, list_windows()):
            return f"Aucune fenêtre visible ne correspond à {args.window!r} (voir capture.py --list)"
        return None
    screens = list_screens()
    if not 0 <= args.screen < len(screens):
        return f"Écran {args.screen} inexistant (écrans 0 à {len(screens) - 1})"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform != "win32":
        print("Ce script fonctionne uniquement sous Windows.", file=sys.stderr)
        return 1
    enable_dpi_awareness()
    args = build_parser().parse_args(argv)
    if args.count < 1:
        print("Erreur : --count doit être supérieur à 0.", file=sys.stderr)
        return 1

    if args.delay > 0:
        print(f"Début dans {args.delay:g} s : placez la fenêtre au premier plan.")
        time.sleep(args.delay)
    error = check_target(args)
    if error is not None:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    try:
        for shot in range(1, args.count + 1):
            image = take_shot(args)
            if image is None:
                print(f"[{shot}/{args.count}] fenêtre introuvable, capture ignorée.")
            else:
                path = unique_output_path(folder, datetime.now())
                image.save(path)
                saved += 1
                print(f"[{shot}/{args.count}] {path.name} ({image.width}x{image.height})")
            if shot < args.count:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nArrêt demandé.")
    except OSError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    print(f"{saved} capture(s) enregistrée(s) dans {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
