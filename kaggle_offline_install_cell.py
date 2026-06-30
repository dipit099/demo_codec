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
#
# NOTE on compressai: the dataset's `pip download` fell back to a source
# sdist for compressai==1.2.6 (no prebuilt wheel for this platform/Python),
# and somewhere in the export/upload pipeline it got auto-extracted into a
# loose `compressai-1.2.6/compressai-1.2.6/` source folder instead of staying
# a `.tar.gz`. `--find-links` only recognizes actual archive files
# (.whl/.tar.gz/.zip) sitting directly in the given directory -- it does NOT
# scan subfolders or treat an extracted tree as installable, so `pip install
# --find-links ... compressai` reports "no versions found" even though the
# source is right there. Fix: install compressai directly from that local
# source directory path (pip can build+install from any folder containing a
# setup.py/pyproject.toml) instead of going through --find-links for it.
# =========================================================================

WHEELS_DIR = "/kaggle/input/datasets/tonyironman099/dpt099-stablecodec-wheels/wheels"
COMPRESSAI_SRC_DIR = f"{WHEELS_DIR}/compressai-1.2.6/compressai-1.2.6"

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

# compressai installed separately from its extracted local source dir (see note above).
run([
    sys.executable, "-m", "pip", "install", "-q", "--no-deps", COMPRESSAI_SRC_DIR,
])

import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
