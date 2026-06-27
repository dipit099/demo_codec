#!/usr/bin/env python3
"""
Board-faithful local scorer for a LoViF submission ZIP (or folder).

Scoring strategy is ported VERBATIM from cod-lite-stage1.ipynb, which is confirmed
to match the Codabench leaderboard:
  - LPIPS : the `lpips` package, LPIPS(net='alex'), REAL pretrained weights   (NOT pyiqa, NOT pretrained=False)
  - DISTS : DISTS_pytorch with its shipped weights.pt                         (REAL pretrained)
  - PSNR  : hand-written, matches pyiqa
  - MS-SSIM: hand-written on the Y channel, matches pyiqa to <1e-5
  Final = PSNR + 10*MS_SSIM + 20*(1-LPIPS) + 25*(1-DISTS)

It scores the reconstructed/*.png in the ZIP against dataset_val/*.png and ALSO
reports avg_bpp from bitstream/*.bin (budget 0.008).

USAGE — just edit the CONFIG block below, then:  python score_zip_boardfaithful.py
(No CLI args — paths with spaces broke argparse before.)
"""

import io
import os
import sys
import time
import zipfile
import tempfile
import subprocess
from pathlib import Path

# ============================== CONFIG (edit these) ==============================
ZIP_PATH   = "stablecodec_output.zip"          # zip OR a folder containing reconstructed/ + bitstream/
GT_DIR     = "../../dataset_val"          # ground-truth originals (100 pngs)
LIMIT      = 10                    # int = score only the first N images (e.g. 10 for a quick check); None = all
DEVICE     = "auto"                    # "auto" | "cpu" | "cuda" | "mps"
HISTORY    = "score_history_boardfaithful.txt"   # appends each run here
BUDGET_BPP = 0.008
# ================================================================================

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def log(msg):
    print(msg, flush=True)


def ensure_deps():
    need = []
    for mod, pip_name in [("torch", "torch"), ("torchvision", "torchvision"),
                          ("lpips", "lpips"), ("DISTS_pytorch", "DISTS_pytorch")]:
        try:
            __import__(mod)
        except Exception:
            need.append(pip_name)
    if need:
        log(f"installing missing deps: {need}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=False)


ensure_deps()
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def pick_device(want):
    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    # NOTE: MPS breaks pyiqa-style float64; our scorer is float32, but MS-SSIM avg_pool on
    # tiny tensors is fine on MPS. Still default to cpu for exact reproducibility.
    return "cpu"


DEV = pick_device(DEVICE)
log(f"device: {DEV}")


# ----------------------- metric implementations (cod-lite cell 13) -----------------------
def _fspecial(size=11, sigma=1.5):
    mm = nn_ = (size - 1) / 2.0
    y, x = np.ogrid[-mm:mm + 1, -nn_:nn_ + 1]
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


_LP = _DS = None
def _scorers():
    global _LP, _DS
    if _LP is None:
        import lpips as _l
        import DISTS_pytorch
        from DISTS_pytorch import DISTS as _D
        log("loading LPIPS(net='alex') + DISTS(weights.pt) ...")
        _LP = _l.LPIPS(net='alex', verbose=False).to(DEV).eval()
        ds = _D(load_weights=False)
        wp = os.path.join(os.path.dirname(DISTS_pytorch.__file__), 'weights.pt')
        w = torch.load(wp, map_location='cpu')
        ds.alpha.data, ds.beta.data = w['alpha'], w['beta']
        _DS = ds.to(DEV).eval()
        log("metric backbones ready.")
    return _LP, _DS


def score_pair(rec_path, ref_path):
    rec = TF.to_tensor(Image.open(rec_path).convert('RGB')).unsqueeze(0).to(DEV)
    ref = TF.to_tensor(Image.open(ref_path).convert('RGB')).unsqueeze(0).to(DEV)
    if rec.shape != ref.shape:
        raise ValueError(f"resolution mismatch rec{tuple(rec.shape[2:])} vs ref{tuple(ref.shape[2:])}")
    lp, ds = _scorers()
    with torch.inference_mode():
        return {'psnr': psnr_score(rec, ref),
                'ms_ssim': ms_ssim_score(rec, ref),
                'lpips': lp(rec * 2 - 1, ref * 2 - 1).item(),
                'dists': ds(rec, ref).item()}


# ----------------------------- submission handling -----------------------------
def prepare(path, tmp):
    p = Path(path)
    if p.is_dir():
        return p
    if p.is_file() and p.suffix.lower() == ".zip":
        out = Path(tmp) / "sub"
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p) as z:
            z.extractall(out)
        # the zip may wrap everything in a top folder; find the dir holding reconstructed/
        if (out / "reconstructed").is_dir():
            return out
        for sub in out.rglob("reconstructed"):
            if sub.is_dir():
                return sub.parent
        return out
    raise FileNotFoundError(f"not a zip or folder: {path}")


def main():
    t_start = time.time()
    gt = sorted(Path(GT_DIR).glob("*.png"))
    if not gt:
        log(f"ERROR: no PNGs in GT_DIR={GT_DIR}")
        return 1
    if LIMIT:
        gt = gt[:LIMIT]
    log(f"ground truth: {len(gt)} images from {GT_DIR}" + (f"  (LIMIT={LIMIT})" if LIMIT else ""))

    with tempfile.TemporaryDirectory() as tmp:
        sub = prepare(ZIP_PATH, tmp)
        rec_dir = sub / "reconstructed"
        bin_dir = sub / "bitstream"
        log(f"submission: {sub}")
        if not rec_dir.is_dir():
            log(f"ERROR: missing {rec_dir}")
            return 1

        rows = []
        tot_bytes = tot_px = 0
        acc = {k: 0.0 for k in ("psnr", "ms_ssim", "lpips", "dists")}
        n = len(gt)
        log("\nscoring:")
        for i, gp in enumerate(gt, 1):
            rp = rec_dir / gp.name
            if not rp.exists():
                log(f"  [{i:03d}/{n}] {gp.name}  MISSING reconstructed -> skipped")
                continue
            with Image.open(gp) as oi:
                w, h = oi.size
            px = w * h
            sc = score_pair(rp, gp)
            for k in acc:
                acc[k] += sc[k]
            bp = bin_dir / f"{gp.stem}.bin"
            bbytes = bp.stat().st_size if bp.exists() else 0
            tot_bytes += bbytes
            tot_px += px
            bpp = bbytes * 8 / px if px else 0
            rows.append((gp.name, sc, bpp))
            log(f"  [{i:03d}/{n}] {gp.name:<14} PSNR {sc['psnr']:6.3f} | MS-SSIM {sc['ms_ssim']:.4f} | "
                f"LPIPS {sc['lpips']:.4f} | DISTS {sc['dists']:.4f} | bpp {bpp:.5f} | "
                f"{(time.time()-t_start)/i:.2f}s/img")

        m = len(rows)
        if m == 0:
            log("ERROR: no scored images.")
            return 1
        means = {k: acc[k] / m for k in acc}
        final = (means['psnr'] + 10.0 * means['ms_ssim']
                 + 20.0 * (1.0 - means['lpips']) + 25.0 * (1.0 - means['dists']))
        avg_bpp = tot_bytes * 8 / tot_px if tot_px else 0.0
        max_bpp = max((r[2] for r in rows), default=0.0)
        within = avg_bpp <= BUDGET_BPP

        log("\n" + "=" * 70)
        log(f"  BOARD-FAITHFUL SCORE  ({m} images)   zip={Path(ZIP_PATH).name}")
        log("-" * 70)
        log(f"  PSNR        : {means['psnr']:.4f}")
        log(f"  MS-SSIM     : {means['ms_ssim']:.4f}")
        log(f"  LPIPS       : {means['lpips']:.4f}   (lower better)")
        log(f"  DISTS       : {means['dists']:.4f}   (lower better)")
        log(f"  avg_bpp     : {avg_bpp:.6f}  /  {BUDGET_BPP}   ({'OK' if within else 'OVER BUDGET!'})")
        log(f"  max_img_bpp : {max_bpp:.6f}")
        log(f"  >>> FINAL_SCORE = {final:.4f} <<<   (board best 66.5854 | rank#1 72.19)")
        log("=" * 70)
        log(f"  total time: {time.time()-t_start:.0f}s")

        # append to history
        try:
            with open(HISTORY, "a") as f:
                ts = time.strftime("%Y-%m-%d %H:%M")
                f.write(f"{ts}  {Path(ZIP_PATH).name:<40}  FINAL {final:8.4f}  "
                        f"PSNR {means['psnr']:.4f}  MSSSIM {means['ms_ssim']:.4f}  "
                        f"LPIPS {means['lpips']:.4f}  DISTS {means['dists']:.4f}  "
                        f"bpp {avg_bpp:.6f}  n={m}  {'OK' if within else 'OVER'}\n")
            log(f"  appended to {HISTORY}")
        except Exception as e:
            log(f"  (history write skipped: {e})")
    return 0 if within else 2


if __name__ == "__main__":
    raise SystemExit(main())
