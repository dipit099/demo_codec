#!/usr/bin/env python3
"""
LoViF 2026 – StableCodec inference + board-faithful scoring + submission ZIP.

Run on Kaggle T4 GPU:
    cd /kaggle/working/StableCodec-main
    python lovif_stablecodec_run.py

Or locally:
    python lovif_stablecodec_run.py --local
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ─────────────────────────── defaults ───────────────────────────
KAGGLE_CKPT_DIR = "/kaggle/input/datasets/mehedi052/stablecodec-checkpoints"
KAGGLE_VAL_DIR = "/kaggle/input/datasets/tonyironman099/lovif-2026-image-compression/dataset_val"
KAGGLE_WORK = "/kaggle/working"

BPP_BUDGET = 0.008
DEFAULT_SEED = 123


# ─────────────────────────── arg parser ─────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LoViF StableCodec full pipeline")

    p.add_argument("--local", action="store_true",
                   help="Run locally instead of on Kaggle (auto-detect paths)")

    # paths  (auto-filled for Kaggle; override for local)
    p.add_argument("--val_dir", type=str, default=None,
                   help="Ground-truth validation images folder")
    p.add_argument("--codec_path", type=str, default=None,
                   help="Path to stablecodec_ft16.pkl")
    p.add_argument("--elic_path", type=str, default=None,
                   help="Path to elic_official.pth")
    p.add_argument("--sd_path", type=str, default=None,
                   help="Path to sd-turbo dir (or HF repo id). If None, auto-downloads.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Working output directory")

    # pipeline knobs
    p.add_argument("--n_compress", type=int, default=None,
                   help="Number of images to compress+decompress (None=all)")
    p.add_argument("--n_score", type=int, default=None,
                   help="Number of images to score (None=all compressed). Must be <= n_compress.")
    p.add_argument("--color_fix", action="store_true",
                   help="Enable AdaIN color fix (recommended for high-res tiled images)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--skip_scoring", action="store_true",
                   help="Skip metric computation (just compress + package)")
    p.add_argument("--skip_zip", action="store_true",
                   help="Skip submission ZIP creation")

    # StableCodec model params (match defaults in testing_utils.py)
    p.add_argument("--lora_rank_unet", type=int, default=32)
    p.add_argument("--lora_rank_vae", type=int, default=16)
    p.add_argument("--vae_decoder_tiled_size", type=int, default=160)
    p.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    p.add_argument("--latent_tiled_size", type=int, default=96)
    p.add_argument("--latent_tiled_overlap", type=int, default=32)
    p.add_argument("--lambda_rate", type=float, default=0.5)
    p.add_argument("--pos_prompt", type=str,
                   default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")

    return p.parse_args()


def resolve_paths(args):
    """Fill in None paths based on --local flag and Kaggle conventions."""
    is_kaggle = Path("/kaggle/working").exists() and not args.local

    if args.val_dir is None:
        if is_kaggle:
            args.val_dir = KAGGLE_VAL_DIR
        else:
            args.val_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "dataset_val")

    if args.codec_path is None:
        if is_kaggle:
            args.codec_path = os.path.join(KAGGLE_CKPT_DIR, "stablecodec_ft16.pkl")
        else:
            args.codec_path = str(Path(__file__).resolve().parent.parent / "stablecodec_ft16.pkl")

    if args.elic_path is None:
        if is_kaggle:
            args.elic_path = os.path.join(KAGGLE_CKPT_DIR, "elic_official.pth")
        else:
            args.elic_path = str(Path(__file__).resolve().parent.parent / "elic_official.pth")

    if args.out_dir is None:
        if is_kaggle:
            args.out_dir = os.path.join(KAGGLE_WORK, "stablecodec_output")
        else:
            args.out_dir = str(Path(__file__).resolve().parent / "output")

    # validate
    val_dir = Path(args.val_dir)
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation dir not found: {args.val_dir}")
    pngs = sorted(val_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG images found in {args.val_dir}")

    if not Path(args.codec_path).is_file():
        raise FileNotFoundError(f"Codec checkpoint not found: {args.codec_path}")
    if not Path(args.elic_path).is_file():
        raise FileNotFoundError(f"ELIC checkpoint not found: {args.elic_path}")

    return args


# ─────────────────────────── helpers ────────────────────────────
def log(msg):
    print(msg, flush=True)


def count_pixels(path):
    with Image.open(path) as img:
        return img.size[0] * img.size[1]


# ─────────────── PHASE 1: compress + decompress ────────────────
def run_compression(args, all_images):
    """Run StableCodec compress.py on the selected images."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from accelerate.utils import set_seed

    src_dir = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(src_dir))

    from StableCodec import StableCodec
    from color_fix import adain_color_fix_quant
    from my_utils.compress_utils import write_body, read_body, filesize

    sd_path = args.sd_path
    if sd_path is None:
        from huggingface_hub import snapshot_download
        log("Downloading SD-Turbo from HuggingFace...")
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")

    if args.seed is not None:
        set_seed(args.seed)

    # build args namespace that StableCodec expects
    sc_args = argparse.Namespace(
        sd_path=sd_path,
        elic_path=args.elic_path,
        codec_path=args.codec_path,
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        vae_decoder_tiled_size=args.vae_decoder_tiled_size,
        vae_encoder_tiled_size=args.vae_encoder_tiled_size,
        latent_tiled_size=args.latent_tiled_size,
        latent_tiled_overlap=args.latent_tiled_overlap,
        lambda_rate=args.lambda_rate,
        pos_prompt=args.pos_prompt,
    )

    log("Loading StableCodec model...")
    net = StableCodec(sd_path=sd_path, args=sc_args)
    net.cuda().eval()
    net.codec.update(force=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    rec_dir = Path(args.out_dir) / "reconstructed"
    bin_dir = Path(args.out_dir) / "bitstream"
    rec_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    pos_tag_prompt = [1]
    bpp_list = []
    t_start = time.time()
    n = len(all_images)

    for i, img_path in enumerate(all_images, 1):
        fname = img_path.stem
        log(f"[{i:03d}/{n}] Processing {img_path.name}")

        img_pil = Image.open(img_path).convert("RGB")
        img = transform(img_pil).cuda().unsqueeze(0)
        ori_h, ori_w = img.shape[2:]

        pad_h = (math.ceil(ori_h / 256)) * 256 - ori_h
        pad_w = (math.ceil(ori_w / 256)) * 256 - ori_w
        img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            try:
                # compress
                output_dict = net.compress(img_padded)
                shape = output_dict["shape"]
                bin_file = bin_dir / f"{fname}.bin"
                with bin_file.open("wb") as f:
                    write_body(f, shape, output_dict["strings"])
                size = bin_file.stat().st_size
                bpp = float(size) * 8 / (ori_h * ori_w)

                # decompress
                with bin_file.open("rb") as f:
                    strings, shape2 = read_body(f)
                out_img = net.decompress(strings, shape2, pos_tag_prompt)
                out_img = out_img[:, :, :ori_h, :ori_w]
                out_img = (out_img * 0.5 + 0.5).float().cpu().detach()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    log(f"  CUDA OOM on {img_path.name}, skipping")
                    torch.cuda.empty_cache()
                    continue
                raise

        output_pil = transforms.ToPILImage()(out_img[0].clamp(0.0, 1.0))

        if args.color_fix:
            img_orig = (img * 0.5 + 0.5).float().cpu().detach()
            im_lr = transforms.ToPILImage()(img_orig[0].clamp(0.0, 1.0))
            output_pil = adain_color_fix_quant(output_pil, im_lr, 16)

        output_pil.save(rec_dir / f"{fname}.png")
        bpp_list.append(bpp)
        elapsed = time.time() - t_start
        log(f"  bpp={bpp:.5f}  elapsed={elapsed:.1f}s  ({elapsed/i:.1f}s/img)")

    # free GPU
    del net
    torch.cuda.empty_cache()

    avg_bpp = np.mean(bpp_list) if bpp_list else 999.0
    log(f"\nCompression done: {len(bpp_list)}/{n} images, avg_bpp={avg_bpp:.6f}")
    return avg_bpp, bpp_list


# ───────────────── PHASE 2: board-faithful scoring ──────────────
def run_scoring(args, images_to_score):
    """
    Compute board-faithful metrics: PSNR, MS-SSIM, LPIPS(alex), DISTS.
    Matches the Codabench leaderboard exactly.
    """
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF

    rec_dir = Path(args.out_dir) / "reconstructed"
    bin_dir = Path(args.out_dir) / "bitstream"

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ── hand-rolled PSNR + MS-SSIM (matches cod-lite / pyiqa) ──
    def _fspecial(size=11, sigma=1.5):
        mm = (size - 1) / 2.0
        y, x = np.ogrid[-mm:mm + 1, -mm:mm + 1]
        h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sm = h.sum()
        if sm != 0:
            h /= sm
        return torch.from_numpy(h)

    def _to_y255(x):
        w = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        y = (x * w).sum(1, keepdim=True) * 255.0
        return (y - y.detach() + y.round()).to(torch.float32)

    def _ssim(X, Y, win, dr=255.0):
        C1, C2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
        mu1, mu2 = F.conv2d(X, win, padding=0), F.conv2d(Y, win, padding=0)
        m1s, m2s, m12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1 = F.conv2d(X * X, win, padding=0) - m1s
        s2 = F.conv2d(Y * Y, win, padding=0) - m2s
        s12 = F.conv2d(X * Y, win, padding=0) - m12
        cs = F.relu((2 * s12 + C2) / (s1 + s2 + C2))
        ssim_map = ((2 * m12 + C1) / (m1s + m2s + C1)) * cs
        return ssim_map.mean([1, 2, 3]), cs.mean([1, 2, 3])

    def ms_ssim_score(rec01, ref01):
        X, Y = _to_y255(rec01), _to_y255(ref01)
        win = _fspecial().to(torch.float32).view(1, 1, 11, 11).to(X.device)
        wts = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float32, device=X.device)
        mcs, sv = [], None
        for _ in range(5):
            sv, cs = _ssim(X, Y, win)
            mcs.append(cs)
            pad = (X.shape[2] % 2, X.shape[3] % 2)
            X, Y = F.avg_pool2d(X, 2, padding=pad), F.avg_pool2d(Y, 2, padding=pad)
        mcs = torch.stack(mcs, 0)
        return (torch.prod(mcs[:-1] ** wts[:-1].unsqueeze(1), 0) * (sv ** wts[-1])).item()

    def psnr_score(rec01, ref01):
        return (10 * torch.log10(1.0 / torch.mean((rec01 - ref01) ** 2))).item()

    # ── LPIPS + DISTS (real pretrained weights, matches board) ──
    import lpips as _lpips_pkg
    import DISTS_pytorch
    from DISTS_pytorch import DISTS as _DISTS

    log("Loading LPIPS(alex) + DISTS scorers...")
    lp_model = _lpips_pkg.LPIPS(net="alex", verbose=False).to(dev).eval()
    ds_model = _DISTS(load_weights=False)
    wp = os.path.join(os.path.dirname(DISTS_pytorch.__file__), "weights.pt")
    w = torch.load(wp, map_location="cpu")
    ds_model.alpha.data, ds_model.beta.data = w["alpha"], w["beta"]
    ds_model = ds_model.to(dev).eval()

    acc = {"psnr": 0.0, "ms_ssim": 0.0, "lpips": 0.0, "dists": 0.0}
    tot_bytes = tot_px = 0
    scored = 0
    n = len(images_to_score)
    t0 = time.time()

    for i, gt_path in enumerate(images_to_score, 1):
        rp = rec_dir / gt_path.name
        bp = bin_dir / f"{gt_path.stem}.bin"
        if not rp.exists():
            log(f"  [{i:03d}/{n}] {gt_path.name}  MISSING rec -> skip")
            continue

        rec = TF.to_tensor(Image.open(rp).convert("RGB")).unsqueeze(0).to(dev)
        ref = TF.to_tensor(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(dev)
        if rec.shape != ref.shape:
            log(f"  [{i:03d}/{n}] {gt_path.name}  SHAPE MISMATCH rec={tuple(rec.shape[2:])} gt={tuple(ref.shape[2:])}")
            continue

        with torch.inference_mode():
            sc = {
                "psnr": psnr_score(rec, ref),
                "ms_ssim": ms_ssim_score(rec, ref),
                "lpips": lp_model(rec * 2 - 1, ref * 2 - 1).item(),
                "dists": ds_model(rec, ref).item(),
            }

        for k in acc:
            acc[k] += sc[k]

        with Image.open(gt_path) as oi:
            px = oi.size[0] * oi.size[1]
        bbytes = bp.stat().st_size if bp.exists() else 0
        tot_bytes += bbytes
        tot_px += px
        bpp = bbytes * 8 / px if px else 0
        scored += 1

        log(f"  [{i:03d}/{n}] {gt_path.name:<14} PSNR {sc['psnr']:6.3f} | MS-SSIM {sc['ms_ssim']:.4f} | "
            f"LPIPS {sc['lpips']:.4f} | DISTS {sc['dists']:.4f} | bpp {bpp:.5f}")

    if scored == 0:
        log("ERROR: no images scored")
        return None

    means = {k: acc[k] / scored for k in acc}
    final = means["psnr"] + 10.0 * means["ms_ssim"] + 20.0 * (1.0 - means["lpips"]) + 25.0 * (1.0 - means["dists"])
    avg_bpp = tot_bytes * 8 / tot_px if tot_px else 0.0

    log("\n" + "=" * 70)
    log(f"  BOARD-FAITHFUL SCORE  ({scored} images)")
    log("-" * 70)
    log(f"  PSNR        : {means['psnr']:.4f}")
    log(f"  MS-SSIM     : {means['ms_ssim']:.4f}")
    log(f"  LPIPS       : {means['lpips']:.4f}   (lower better)")
    log(f"  DISTS       : {means['dists']:.4f}   (lower better)")
    log(f"  avg_bpp     : {avg_bpp:.6f}  /  {BPP_BUDGET}   ({'OK' if avg_bpp <= BPP_BUDGET else 'OVER BUDGET!'})")
    log(f"  >>> FINAL_SCORE = {final:.4f} <<<")
    log("=" * 70)
    log(f"  scoring time: {time.time()-t0:.0f}s")

    del lp_model, ds_model
    torch.cuda.empty_cache()

    return {"means": means, "final": final, "avg_bpp": avg_bpp, "scored": scored}


# ──────────────── PHASE 3: package submission ZIP ───────────────
def package_submission(args, all_images, runtime_per_image):
    rec_dir = Path(args.out_dir) / "reconstructed"
    bin_dir = Path(args.out_dir) / "bitstream"
    sub_dir = Path(args.out_dir) / "submission"
    sub_rec = sub_dir / "reconstructed"
    sub_bin = sub_dir / "bitstream"

    if sub_dir.exists():
        shutil.rmtree(sub_dir)
    sub_rec.mkdir(parents=True)
    sub_bin.mkdir(parents=True)

    total_bytes = total_px = 0
    errors = []

    for img_path in all_images:
        rp = rec_dir / img_path.name
        bp = bin_dir / f"{img_path.stem}.bin"
        if not rp.exists():
            errors.append(f"missing rec: {img_path.name}")
            continue
        if not bp.exists():
            errors.append(f"missing bin: {img_path.stem}.bin")
            continue

        # validate resolution match
        with Image.open(img_path) as gt_img, Image.open(rp) as rec_img:
            if gt_img.size != rec_img.size:
                errors.append(f"size mismatch {img_path.name}: gt={gt_img.size} rec={rec_img.size}")
            px = gt_img.size[0] * gt_img.size[1]

        shutil.copy2(rp, sub_rec / img_path.name)
        shutil.copy2(bp, sub_bin / f"{img_path.stem}.bin")
        total_bytes += bp.stat().st_size
        total_px += px

    if errors:
        for e in errors[:20]:
            log(f"  ERROR: {e}")
        raise RuntimeError(f"{len(errors)} submission packaging errors")

    avg_bpp = 8.0 * total_bytes / total_px if total_px else 999.0
    if avg_bpp > BPP_BUDGET:
        log(f"WARNING: avg_bpp={avg_bpp:.6f} exceeds budget {BPP_BUDGET}")

    readme_text = f"""runtime per image [s] : {runtime_per_image:.4f}
CPU[1] / GPU[0] : 0
Extra Data [1] / No Extra Data [0] : 1
Other description: StableCodec (ft24.pkl) with SD-Turbo generative prior, LoRA-adapted VAE/UNet, ELIC auxiliary encoder. Color fix: {args.color_fix}. Average BPP: {avg_bpp:.6f}.
"""
    (sub_dir / "readme.txt").write_text(readme_text, encoding="utf-8")

    zip_path = Path(args.out_dir) / "submission.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(sub_rec.glob("*.png")):
            zf.write(p, f"reconstructed/{p.name}")
        for p in sorted(sub_bin.glob("*.bin")):
            zf.write(p, f"bitstream/{p.name}")
        zf.write(sub_dir / "readme.txt", "readme.txt")

    log(f"\nSubmission ZIP: {zip_path}")
    log(f"  images: {len(list(sub_rec.glob('*.png')))}")
    log(f"  avg_bpp: {avg_bpp:.6f}")
    log(f"  zip size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    return zip_path


# ─────────────────────────── main ───────────────────────────────
def main():
    args = parse_args()
    args = resolve_paths(args)

    val_dir = Path(args.val_dir)
    all_images = sorted(val_dir.glob("*.png"))
    total_count = len(all_images)
    log(f"Found {total_count} validation images in {val_dir}")

    # ── bound n_compress / n_score ──
    if args.n_compress is not None:
        args.n_compress = max(1, min(args.n_compress, total_count))
    else:
        args.n_compress = total_count

    if args.n_score is not None:
        args.n_score = max(1, min(args.n_score, args.n_compress))
    else:
        args.n_score = args.n_compress

    images_to_compress = all_images[:args.n_compress]
    images_to_score = all_images[:args.n_score]

    log(f"Pipeline: compress {args.n_compress} images, score {args.n_score} images")
    log(f"Color fix: {args.color_fix}")
    log(f"Codec: {args.codec_path}")
    log(f"Output: {args.out_dir}")

    # ── Phase 1: Compress + Decompress ──
    log("\n" + "=" * 70)
    log("PHASE 1: Compression + Decompression")
    log("=" * 70)
    t0 = time.time()
    avg_bpp, bpp_list = run_compression(args, images_to_compress)
    compress_time = time.time() - t0
    runtime_per_image = compress_time / len(images_to_compress)
    log(f"Phase 1 done in {compress_time:.0f}s ({runtime_per_image:.1f}s/img)")

    if avg_bpp > BPP_BUDGET:
        log(f"WARNING: avg_bpp {avg_bpp:.6f} > budget {BPP_BUDGET}")

    # ── Phase 2: Scoring ──
    if not args.skip_scoring:
        log("\n" + "=" * 70)
        log("PHASE 2: Board-Faithful Scoring")
        log("=" * 70)
        score_result = run_scoring(args, images_to_score)
    else:
        log("\nSkipping scoring (--skip_scoring)")
        score_result = None

    # ── Phase 3: Package submission ──
    if not args.skip_zip:
        log("\n" + "=" * 70)
        log("PHASE 3: Packaging Submission ZIP")
        log("=" * 70)
        zip_path = package_submission(args, images_to_compress, runtime_per_image)
    else:
        log("\nSkipping ZIP (--skip_zip)")

    # ── Summary ──
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Images compressed : {len(bpp_list)} / {args.n_compress}")
    log(f"  avg_bpp           : {avg_bpp:.6f}  (budget: {BPP_BUDGET})")
    log(f"  Runtime/image     : {runtime_per_image:.1f}s")
    if score_result:
        log(f"  FINAL_SCORE       : {score_result['final']:.4f}")
        log(f"  PSNR={score_result['means']['psnr']:.4f}  "
            f"MS-SSIM={score_result['means']['ms_ssim']:.4f}  "
            f"LPIPS={score_result['means']['lpips']:.4f}  "
            f"DISTS={score_result['means']['dists']:.4f}")
    log("=" * 70)


if __name__ == "__main__":
    main()
