"""First training: YOLO11n, 100 epochs, auto batch, AMP and cache enabled.

    python scripts/train_initial.py
    python scripts/train_initial.py --epochs 50 --model yolo11s.pt

Result: ``models/v1/best.pt`` and ``models/v1/last.pt`` (active model).
Follow the training live with ``tensorboard --logdir runs``.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from src.cli.commands import print_outcome
from src.context import AppContext


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, help="défaut: training.epochs (100)")
    parser.add_argument("--model", help="défaut: training.base_model (yolo11n.pt)")
    parser.add_argument("-o", "--override", action="append", default=[])
    args = parser.parse_args()

    ctx = AppContext.create(args.override)
    print(f"TensorBoard: tensorboard --logdir \"{ctx.workspace.runs}\"")
    print_outcome(ctx.trainer.train_initial(epochs=args.epochs, base_model=args.model))
    return 0


if __name__ == "__main__":  # mandatory on Windows (DataLoader workers use spawn)
    sys.exit(main())
