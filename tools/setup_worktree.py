"""Check a Breeze worktree against an explicitly prepared ComfyUI runtime."""

import os
from pathlib import Path
import subprocess


python = os.environ.get("COMFYUI_PYTHON")
cutlery_root = os.environ.get("CUTLERY_NODES_ROOT")
if not python or not cutlery_root:
    raise SystemExit("Set COMFYUI_PYTHON to the prepared ComfyUI Python and CUTLERY_NODES_ROOT to the matching Cutlery-Nodes checkout.")
if not (Path(cutlery_root) / "nodes_transcription.py").is_file():
    raise SystemExit("CUTLERY_NODES_ROOT must contain Cutlery's nodes_transcription.py.")
subprocess.run(
    [python, "-c", "import sys, torch, torchaudio, numpy, faster_whisper; print(sys.version); print('PyTorch', torch.__version__); assert sys.version_info >= (3, 10), 'Python >=3.10 is required'"],
    check=True,
)
print("Prepared runtime and Cutlery source are available; no models loaded or services contacted.")
