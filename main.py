"""AutoYOLO Trainer - command-line entry point.

Examples::

    python main.py train
    python main.py capture --source camera
    python main.py label
    python main.py retrain
    python main.py stats
    python main.py version
    python main.py clean --dry-run
    python main.py run            # fully autonomous loop
    python main.py dashboard
"""

from __future__ import annotations

import sys

from src.cli.parser import build_parser
from src.context import AppContext
from src.utils.logging import get_logger


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ctx = AppContext.create(args.override)
    try:
        args.func(ctx, args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        get_logger("app").exception(f"Commande '{args.command}' échouée: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this module
    sys.exit(main())
