from __future__ import annotations

import subprocess
import sys

# =========================================================================
# Offline install for StableCodec/finetune_lovif.py using YOUR self-exported
# tonyironman099/dpt099-stablecodec-wheels Kaggle dataset (built via
# kaggle_export_wheels_cell.py).
#
# Still using --no-deps: the export's `pip download` resolved a torch
# dependency chain transitively and pulled in CUDA-13 runtime wheels
# (nvidia_cublas, nvidia_cudnn_cu13, triton-3.7.1, cuda_toolkit, etc) even
# though the EXCLUDE filter correctly kept the actual torch/torchvision/numpy
# wheels OUT. Without --no-deps, a plain `pip install` could still try to
# upgrade those standalone CUDA library packages to versions that don't match
# Kaggle's pre-installed torch build, risking a broken GPU runtime. --no-deps
# installs only the 11 leaf packages below and leaves Kaggle's existing
# torch/CUDA stack untouched -- every transitive need (numpy, pillow,
# requests, tqdm, pyyaml, etc.) is already present on Kaggle's base image.
#
# Exact package list matches your original snippet (nothing renamed/dropped
# here -- gdown and open-clip-torch ARE included since your export grabbed
# them; they're simply unused by finetune_lovif.py's import chain, so their
# presence is harmless).
# =========================================================================

WHEELS_DIR = "/kaggle/input/datasets/tonyironman099/dpt099-stablecodec-wheels/wheels"

packages = [
    "gdown",
    "huggingface_hub",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "omegaconf",
    "einops",
    "timm",
    "compressai",
    "open-clip-torch",
    "lpips",
    "DISTS_pytorch",
    "pytorch_msssim",
    "matplotlib",
]


def run(cmd, check=True):
    print("\n$ " + " ".join(map(str, cmd)))
    result = subprocess.run(list(map(str, cmd)), text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")
    return result.returncode


run([
    sys.executable, "-m", "pip", "install", "-q",
    "--no-index", "--find-links", WHEELS_DIR, "--no-deps",
    *packages,
])

import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
