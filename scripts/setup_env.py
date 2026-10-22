"""Create the ``.venv`` virtual environment and install the dependencies.

    py -3.12 scripts/setup_env.py              # PyTorch CUDA 12.6 (défaut)
    py -3.12 scripts/setup_env.py --cpu        # PyTorch CPU uniquement
    py -3.12 scripts/setup_env.py --cuda cu128 # autre version de CUDA

Only uses the standard library, so it runs before anything is installed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", action="store_true", help="installer PyTorch sans CUDA")
    parser.add_argument("--cuda", default="cu126", help="index PyTorch CUDA (cu118, cu126, cu128...)")
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 12):
        print(f"Attention: Python {sys.version.split()[0]} détecté, le projet cible Python 3.12.")

    if not VENV.exists():
        print(f"Création du venv {VENV}")
        venv.create(VENV, with_pip=True)
    python = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def pip(*arguments: str) -> None:
        subprocess.check_call([str(python), "-m", "pip", *arguments])

    pip("install", "--upgrade", "pip", "setuptools", "wheel")
    # PyTorch first, from the right index, so requirements.txt does not pull a CPU build.
    index = "cpu" if args.cpu else args.cuda
    pip("install", "torch", "torchvision", "--index-url", f"https://download.pytorch.org/whl/{index}")
    pip("install", "-r", str(ROOT / "requirements.txt"))
    subprocess.call([str(python), "-c",
                     "import torch; print('CUDA disponible:', torch.cuda.is_available())"])
    activate = r".venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
    print(f"\nTerminé. Activez l'environnement avec: {activate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
