"""Screenshot of one window or one screen (Windows).

    python scripts/capture.py --list
    python scripts/capture.py --screen 1
    python scripts/capture.py --window "Label Studio" --delay 3
    python scripts/capture.py --window Chrome -o captures/manuel/chrome.png

``--screen 0`` captures the whole virtual desktop (all monitors). Minimised windows
cannot be captured. A window is rendered by Windows (``PrintWindow``), so covering
windows do not appear in its picture; if that call fails, the screen pixels are used
instead and the window must be visible (``--delay`` helps).
Default output: ``captures/manuel/AAAAMMJJ_HHMMSS.png`` (outside the watched folder).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401 - makes ``src`` importable

from src.capture.windows import (
    WindowInfo,
    capture_window,
    enable_dpi_awareness,
    find_windows,
    grab,
    list_screens,
    list_windows,
)
from src.config.loader import PROJECT_ROOT

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "captures" / "manuel"


def unique_output_path(folder: Path, now: datetime) -> Path:
    """``AAAAMMJJ_HHMMSS.png``, with a numeric suffix if two shots land in the same second."""
    stem = now.strftime("%Y%m%d_%H%M%S")
    path = folder / f"{stem}.png"
    counter = 2
    while path.exists():
        path = folder / f"{stem}_{counter}.png"
        counter += 1
    return path


def print_listing() -> None:
    print("Écrans (utilisez --screen N) :")
    for screen in list_screens():
        width, height = screen.size
        if screen.index == 0:
            label = "tous les écrans"
        else:
            label = "principal" if screen.primary else "secondaire"
        print(f"  [{screen.index}] {width}x{height} à ({screen.bbox[0]}, {screen.bbox[1]}) - {label}")
    print("\nFenêtres visibles, premier plan en premier (utilisez --window TEXTE) :")
    for window in list_windows():
        width, height = window.size
        print(f"  {window.title}  [{width}x{height}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--list", action="store_true", help="lister les écrans et les fenêtres visibles")
    target.add_argument("--window", metavar="TEXTE", help="capturer la fenêtre dont le titre contient TEXTE")
    target.add_argument("--screen", type=int, metavar="N", help="capturer l'écran N (0 = tous les écrans)")
    parser.add_argument("-o", "--output", type=Path, help="fichier PNG de sortie")
    parser.add_argument("--delay", type=float, default=0.0, metavar="SECONDES", help="attendre avant la capture")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform != "win32":
        print("Ce script fonctionne uniquement sous Windows.", file=sys.stderr)
        return 1
    enable_dpi_awareness()
    sys.stdout.reconfigure(errors="replace")  # window titles may hold characters the console cannot encode
    args = build_parser().parse_args(argv)
    if args.list:
        print_listing()
        return 0

    window: WindowInfo | None = None
    try:
        if args.window is not None:
            matches = find_windows(args.window, list_windows())
            if not matches:
                raise LookupError(f"Aucune fenêtre visible ne correspond à {args.window!r} (voir --list)")
            if len(matches) > 1:
                print(f"{len(matches)} fenêtres correspondent, la première est utilisée : {matches[0].title}")
            window = matches[0]
        else:
            screens = list_screens()
            if not 0 <= args.screen < len(screens):
                raise LookupError(f"Écran {args.screen} inexistant (écrans 0 à {len(screens) - 1})")
            screen = screens[args.screen]

        if args.delay > 0:
            print(f"Capture dans {args.delay:g} s...")
            time.sleep(args.delay)

        image = capture_window(window) if window is not None else grab(screen.bbox)
        output = (args.output or unique_output_path(DEFAULT_OUTPUT_DIR, datetime.now())).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
    except (LookupError, OSError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    print(f"Capture enregistrée : {output} ({image.width}x{image.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
