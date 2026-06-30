from __future__ import annotations

import subprocess
import sys

# =========================================================================
# Offline install for StableCodec/finetune_lovif.py using the
# ahnaftahmid24/lovif-aeic-offline Kaggle dataset's prebuilt wheels.
# No PyPI access required (--no-index --find-links).
#
# Dropped vs. the original snippet (neither is imported anywhere in
# finetune_lovif.py's call chain -- StableCodec.py / latent_codec.py /
# my_utils/training_utils.py / vision_aided_loss -- verified by grep):
#   - gdown            (not present in the dataset; only needed for manually
#                        fetching elic_official.pth / stablecodec_*.pkl from
#                        Drive, which you already have via
#                        mehedi052/stablecodec-checkpoints)
#   - open-clip-torch  (not present in the dataset; CLIPLoss uses HF
#                        transformers.CLIPVisionModelWithProjection, not
#                        open_clip)
#
# Versions: installed as whatever the dataset's wheels/ folder has, NOT
# the exact pins from the original snippet (accelerate 1.14.0 vs 1.9.0,
# timm 1.0.27 vs 1.0.22, compressai 1.2.8 vs 1.2.6) -- per your call to
# use the dataset's versions as-is.
# =========================================================================

WHEELS_DIR = "/kaggle/input/lovif-aeic-offline/results (3)/wheels"

packages = [
    "huggingface_hub",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "omegaconf",
    "einops",
    "timm",
    "compressai",
    "lpips",
    "DISTS_pytorch",
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
