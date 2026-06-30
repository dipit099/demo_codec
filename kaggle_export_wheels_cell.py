import subprocess, sys, zipfile
from pathlib import Path

# =========================================================================
# Run this ONCE on a Kaggle notebook WITH internet enabled (Settings ->
# Internet -> On). It downloads .whl/.tar.gz files (not a live install) for
# every package finetune_lovif.py needs, into /kaggle/working/wheels/, then
# zips them. Turn the resulting zip into a new Kaggle Dataset and you can
# `pip install --no-index --find-links <dataset_path> <packages>` offline
# from then on -- same pattern as ahnaftahmid24/lovif-aeic-offline.
#
# NOTE: pip download resolves the full dependency tree (unlike --no-deps
# installs), so this captures everything actually needed -- no more
# ModuleNotFoundError surprises like pytorch_msssim or torch_geometric.
# We exclude torch/torchvision/numpy themselves since Kaggle's runtime
# already provides CUDA-matched builds; redownloading those would bloat
# the dataset and risk a CUDA mismatch.
# =========================================================================

OUT_DIR = Path("/kaggle/working/wheels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

packages = [
    "gdown",
    "huggingface_hub==0.25.0",
    "diffusers==0.25.1",
    "transformers==4.46.3",
    "accelerate==1.9.0",
    "peft",
    "omegaconf",
    "einops>=0.6.1",
    "timm==1.0.22",
    "compressai==1.2.6",
    "open-clip-torch>=2.20.0",
    "lpips",
    "DISTS_pytorch",
    "pytorch-msssim",
    "matplotlib",
]

# Packages compressai/StableCodec pull in transitively that we don't want
# pip to also download (already on Kaggle's base image / provided by Kaggle's
# CUDA runtime -- redownloading risks a CUDA/driver mismatch).
EXCLUDE = ["torch", "torchvision", "torchaudio", "numpy", "torch-geometric"]


def run(cmd, check=True):
    print("\n$ " + " ".join(map(str, cmd)))
    result = subprocess.run(list(map(str, cmd)), text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")
    return result.returncode


# `pip download` resolves + fetches every transitive dependency as files,
# without installing anything into the current environment.
run([sys.executable, "-m", "pip", "download", "-d", str(OUT_DIR)] + packages)

# git+https URL deps (DISTS) aren't resolvable by `pip download` the same way
# as PyPI specs in older pip versions; if DISTS_pytorch above already pulled
# a real PyPI release this is a no-op safety net.
try:
    run([sys.executable, "-m", "pip", "download", "-d", str(OUT_DIR), "--no-deps",
         "git+https://github.com/dingkeyan93/DISTS.git"])
except Exception as e:
    print(f"[warn] DISTS git download skipped/failed (DISTS_pytorch PyPI wheel likely already covers it): {e}")

# Strip any torch/torchvision/numpy wheels that snuck in as transitive deps --
# keep the dataset small and avoid shipping a CUDA build that conflicts with
# whatever Kaggle's runtime provides later.
removed = []
for f in OUT_DIR.glob("*"):
    name = f.name.lower()
    if any(name.startswith(ex.replace("-", "_") + "-") or name.startswith(ex + "-") for ex in EXCLUDE):
        f.unlink()
        removed.append(f.name)
if removed:
    print("Removed (provided by Kaggle runtime instead):", removed)

print(f"\nDownloaded {len(list(OUT_DIR.glob('*')))} files to {OUT_DIR}")

zip_path = Path("/kaggle/working/lovif_stablecodec_wheels.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in OUT_DIR.glob("*"):
        zf.write(f, arcname=f.name)
print(f"Zipped -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
