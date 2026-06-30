#!/usr/bin/env python3
# =============================================================================
#  finetune_lovif.py — StableCodec finetune, RETUNED for the LoViF 2026 metrics
# =============================================================================
#  Same mechanism as src/finetune.py (codec + UNet-LoRA + conv_in + VAE-LoRA trained,
#  decoder base frozen; 3-step DINOv2 vision-aided GAN; lambda_rate = bpp knob) — but:
#    * NEW metric-aligned loss for  S = PSNR + 10·MS-SSIM + 20·(1-LPIPS) + 25·(1-DISTS):
#        loss = rate(λ_rate) + λ_dists·DISTS + λ_lpips·LPIPS + λ_msssim·(1-MS-SSIM)
#               + λ_l2·L2 (+ λ_gan·ADV + λ_clip·CLIP)
#      StableCodec's stock loss used L2(2.0)+LPIPS(1.0)+CLIP+GAN with NO DISTS and NO
#      MS-SSIM — i.e. it ignored the two highest-weighted competition metrics (DISTS ×25,
#      MS-SSIM ×10) and over-weighted PSNR (×1). This fixes that alignment.
#    * Trains on a PNG FOLDER (dataset_train + dataset_val), no .hdf5 build needed.
#    * Validates on dataset_val's first N images (default 25) with the BOARD metrics
#      (lpips-alex + DISTS weights.pt + MS-SSIM + PSNR) and logs the competition SCORE.
#    * CSV logs + loss_curves.png + eval_scores.png exported to OUTPUT_DIR.
#    * Resume across Kaggle sessions (accelerator.save_state/load_state).
#    * Kaggle T4 x2 friendly (bf16, grad-checkpointing, batch 1, GAN toggle).
#
#  NOTE: validating on dataset_val while ALSO training on it -> the eval numbers are
#  OPTIMISTIC (train-set metrics, not held-out). The true check is the hidden test set.
#  This is intentional per the run spec (use all data); just read the eval as a trend.
#
#  RUN (Kaggle, from repo root, dual T4), fully offline via the
#  ahnaftahmid24/lovif-aeic-offline + mehedi052/stablecodec-checkpoints datasets:
#    accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 \
#        src/finetune_lovif.py \
#        --sd_path /kaggle/input/lovif-aeic-offline/results/lovif_aeic_offline/sd-turbo \
#        --elic_path /kaggle/input/stablecodec-checkpoints/elic_official.pth \
#        --codec_path /kaggle/input/stablecodec-checkpoints/stablecodec_base.pkl \
#        --train_dirs /kaggle/input/.../dataset_train /kaggle/input/.../dataset_val \
#        --val_dir   /kaggle/input/.../dataset_val \
#        --lambda_rate 24 --output_dir /kaggle/working/sc_ft24
#
#  Offline asset coverage (see notebooks/offline-data-context-kaggle.txt for the full
#  dataset file listing):
#    - sd-turbo                     -> results/lovif_aeic_offline/sd-turbo/  (--sd_path)
#    - openai/clip-vit-base-patch32 -> train-assets/lovif_train_assets/clip-vit-base-patch32/
#    - dinov2 vitb14 (GAN disc)     -> train-assets/_hub_cache (torch.hub layout)
#    - lpips vgg16/alexnet backbone -> results (2)/torch_hub (torch.hub layout)
#  All of the above are picked up automatically via HF_HOME/TORCH_HOME below once the
#  dataset is attached to the notebook -- no internet access needed at runtime.
# =============================================================================

import os, gc, csv, sys, time, json, types, argparse, subprocess
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce T4 fragmentation (must precede torch CUDA init)


def _bootstrap_offline_caches():
    """Point HF/torch hub caches at the pre-downloaded ahnaftahmid24/lovif-aeic-offline
    Kaggle dataset (if present) so nothing falls back to a live internet download.
    Must run before transformers/diffusers/torch.hub/lpips/torchvision are imported.

    The dataset ships TWO separate torch.hub caches (results (2)/torch_hub has the
    lpips vgg16/alexnet checkpoints; train-assets/_hub_cache has dinov2_vitb14 + the
    facebookresearch/dinov2 repo snapshot) -- torch.hub only reads one TORCH_HOME, so
    we symlink-merge both `hub/checkpoints/*` and `hub/<repo>` entries into a single
    writable cache under /kaggle/working instead of picking one and losing the other.
    """
    base = "/kaggle/input/lovif-aeic-offline"
    if not os.path.isdir(base):
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    torch_home = "/kaggle/working/_torch_hub_cache"
    ckpt_dir = os.path.join(torch_home, "hub", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for src_root in (f"{base}/results (2)/torch_hub/hub", f"{base}/train-assets/_hub_cache/hub"):
        if not os.path.isdir(src_root):
            continue
        for name in os.listdir(src_root):
            src = os.path.join(src_root, name)
            if name == "checkpoints" and os.path.isdir(src):
                for fname in os.listdir(src):
                    dst = os.path.join(ckpt_dir, fname)
                    if not os.path.exists(dst):
                        os.symlink(os.path.join(src, fname), dst)
            else:
                dst = os.path.join(torch_home, "hub", name)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
    os.environ.setdefault("TORCH_HOME", torch_home)

    clip_local = f"{base}/train-assets/lovif_train_assets/clip-vit-base-patch32"
    if os.path.isdir(clip_local):
        os.environ.setdefault("LOVIF_CLIP_PATH", clip_local)


_bootstrap_offline_caches()  # MUST precede transformers/diffusers/torch imports
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _stub_torch_geometric():
    """compressai (pulled in by StableCodec -> latent_codec) imports torch_geometric on
    `import compressai`. It's heavy and absent on Kaggle; fabricate empty stub modules so
    the import chain succeeds (we never use point-cloud transforms)."""
    import importlib.util, importlib.abc, importlib.machinery
    if importlib.util.find_spec("torch_geometric") is not None:
        return
    class _SB:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return a[0] if (len(a) == 1 and not k) else None
        def __getattr__(self, n):
            if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
            return _SB()
    def _dec(*a, **k):
        if len(a) == 1 and not k and (isinstance(a[0], type) or callable(a[0])): return a[0]
        def wrap(o): return o
        return wrap
    class _SM(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"): raise AttributeError(name)
            return _dec if name[:1].islower() else _SB
    class _F(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch_geometric" or fullname.startswith("torch_geometric."):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None
        def create_module(self, spec):
            m = _SM(spec.name); m.__path__ = []; return m
        def exec_module(self, module): pass
    sys.meta_path.insert(0, _F())


_stub_torch_geometric()   # MUST precede the StableCodec/compressai import chain

import lpips
import transformers, diffusers
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available

from StableCodec import StableCodec               # repo src (on path when run as src/finetune_lovif.py)
from my_utils.training_utils import CLIPLoss


# --------------------------------------------------------------------------
#  args
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("StableCodec finetune — LoViF metric-aligned")
    # weights / models
    p.add_argument("--sd_path", default=None, help="sd-turbo dir; None -> HF download")
    p.add_argument("--elic_path", required=True, help="elic_official.pth")
    p.add_argument("--codec_path", required=True, help="stablecodec_base.pkl (Stage-1 base)")
    # data
    p.add_argument("--train_dirs", nargs="+", required=True, help="one or more PNG folders to train on")
    p.add_argument("--val_dir", required=True, help="folder to validate on (first --val_num imgs)")
    p.add_argument("--val_num", type=int, default=25)
    p.add_argument("--val_patch", type=int, default=384, help="center-crop for eval (multiple of 256, T4-safe)")
    # bitrate
    p.add_argument("--lambda_rate", type=float, default=24.0, help="bpp knob (=compression ratio; 24 -> ~0.008)")
    # ---- NEW metric-aligned loss weights (mirror score 25:20:10:1) ----
    p.add_argument("--lambda_dists",  type=float, default=2.5, help="DISTS (×25 metric) — lead term")
    p.add_argument("--lambda_lpips",  type=float, default=2.0, help="LPIPS (×20 metric)")
    p.add_argument("--lambda_msssim", type=float, default=1.0, help="(1-MS-SSIM) (×10 metric)")
    p.add_argument("--lambda_l2",     type=float, default=0.1, help="L2/PSNR (×1) — light anchor, avoid blur")
    p.add_argument("--lambda_gan",    type=float, default=0.0, help="adversarial (on-manifold). DEFAULT OFF for T4 "
                   "(dinov2 disc is heavy) AND because DISTS is now a DIRECT loss term. Enable later: --lambda_gan 0.1")
    p.add_argument("--lambda_clip",   type=float, default=0.0, help="CLIP semantic (0 -> OFF, saves mem/download)")
    p.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    p.add_argument("--lpips_net", default="vgg", choices=["vgg", "alex"], help="LPIPS net for the TRAINING loss")
    # model details (StableCodec defaults)
    p.add_argument("--lora_rank_unet", type=int, default=32)
    p.add_argument("--lora_rank_vae", type=int, default=16)
    p.add_argument("--vae_decoder_tiled_size", type=int, default=160)
    p.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    p.add_argument("--latent_tiled_size", type=int, default=96)
    p.add_argument("--latent_tiled_overlap", type=int, default=32)
    p.add_argument("--pos_prompt", type=str,
                   default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")
    # training
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--train_patch_size", type=int, default=386)   # MUST be a multiple of 256
                   # (codec y=patch/64 must be divisible by 4: 256->y4 ok, 384->y6 BREAKS, 512->y8 ok).
                   # 256 is valid AND fits the T4 (512 OOMs).
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=21000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--disc_lr", type=float, default=2e-5)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--enable_xformers", action="store_true", default=False)
    p.add_argument("--no_multi_gpu", action="store_true",
                   help="force single GPU. DEFAULT: auto-use ALL visible GPUs (DDP) by "
                        "re-launching under `accelerate launch --multi_gpu`.")
    # io
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--keep_last", type=int, default=3, help="keep only the N most-recent step checkpoints (best.pkl kept separately)")
    p.add_argument("--resume_steps", type=int, default=1000)
    p.add_argument("--eval_freq", type=int, default=500)
    p.add_argument("--plot_freq", type=int, default=500)
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--save_num", type=int, default=8, help="how many eval triptychs to save")
    return p.parse_args()


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------
#  data
# --------------------------------------------------------------------------
class PngFolders(Dataset):
    """Random-crop over a UNION of PNG/JPG folders -> [-1,1]."""
    def __init__(self, roots, patch):
        Image.MAX_IMAGE_PIXELS = None
        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        self.paths = []
        for r in roots:
            self.paths += [p for p in Path(r).rglob("*") if p.suffix.lower() in exts]
        assert self.paths, f"no images in {roots}"
        self.tf = transforms.Compose([
            transforms.RandomCrop(patch, pad_if_needed=True, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try: im = Image.open(self.paths[i]).convert("RGB")
        except Exception: im = Image.open(self.paths[(i+1) % len(self.paths)]).convert("RGB")
        return self.tf(im)


def load_val_crops(val_dir, n, patch, device):
    exts = (".png", ".jpg", ".jpeg")
    paths = sorted([p for p in Path(val_dir).rglob("*") if p.suffix.lower() in exts])[:n]
    P = max(256, (patch // 256) * 256)   # codec needs y=P/64 divisible by 4 -> P multiple of 256
    crops = []
    for p in paths:
        im = Image.open(p).convert("RGB"); W, H = im.size
        x = torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)
        x = (x.unsqueeze(0) * 2 - 1)
        ph, pw = max(0, P - H), max(0, P - W)
        if ph or pw: x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        H2, W2 = x.shape[2], x.shape[3]
        x = x[:, :, (H2-P)//2:(H2-P)//2+P, (W2-P)//2:(W2-P)//2+P]
        crops.append((p.stem, x.to(device)))
    return crops


# --------------------------------------------------------------------------
#  metrics (board-faithful: lpips-alex on [-1,1], DISTS weights.pt on [0,1])
# --------------------------------------------------------------------------
def build_metrics(device):
    m = {}
    m["lpips_alex"] = lpips.LPIPS(net="alex", verbose=False).to(device).eval(); m["lpips_alex"].requires_grad_(False)
    import DISTS_pytorch
    from DISTS_pytorch import DISTS
    ds = DISTS(load_weights=False)
    w = torch.load(os.path.join(os.path.dirname(DISTS_pytorch.__file__), "weights.pt"), map_location="cpu")
    ds.alpha.data, ds.beta.data = w["alpha"], w["beta"]
    m["dists"] = ds.to(device).eval(); m["dists"].requires_grad_(False)
    from pytorch_msssim import ms_ssim
    m["msssim"] = ms_ssim
    return m


def competition_score(psnr, msssim, lpips_v, dists_v):
    return psnr + 10*msssim + 20*(1 - lpips_v) + 25*(1 - dists_v)


@torch.no_grad()
def run_eval(net, metrics, crops, out_dir, step, save_num):
    net_u = net
    psnr = mss = lp = ds = bpp = 0.0; n = 0; imgs = {}
    for stem, x in crops:
        H, W = x.shape[2], x.shape[3]
        try:
            x_hat, RL = net_u(x, [1], H, W)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        x01 = (x*0.5+0.5).clamp(0,1).float(); xh01 = (x_hat*0.5+0.5).clamp(0,1).float()
        mse = torch.nn.functional.mse_loss(xh01, x01)
        psnr += float(10*(-torch.log10(mse)))
        mss += float(metrics["msssim"](xh01, x01, data_range=1.0))
        lp += float(metrics["lpips_alex"](xh01*2-1, x01*2-1).mean())
        ds += float(metrics["dists"](xh01, x01).mean())
        bpp += float(RL.quantized_total_bpp); n += 1
        if len(imgs) < save_num: imgs[stem] = (x01, xh01)
    if not n: return None
    res = {"psnr": psnr/n, "msssim": mss/n, "lpips": lp/n, "dists": ds/n, "bpp": bpp/n}
    res["score"] = competition_score(res["psnr"], res["msssim"], res["lpips"], res["dists"])
    # save triptychs (orig | recon)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sdir = Path(out_dir)/"eval"; sdir.mkdir(parents=True, exist_ok=True)
        k = len(imgs); fig, ax = plt.subplots(k, 2, figsize=(8, 4*k))
        if k == 1: ax = ax[None, :]
        for r, (stem, (x01, xh01)) in enumerate(imgs.items()):
            for c, (img, t) in enumerate([(x01, "original"), (xh01, "recon")]):
                ax[r, c].imshow(img.squeeze(0).cpu().numpy().transpose(1,2,0)); ax[r, c].axis("off")
                if r == 0: ax[r, c].set_title(t)
        fig.tight_layout(); fig.savefig(sdir/f"val_step{step:06d}.png", dpi=110); plt.close(fig)
    except Exception as e:
        log(f"  [eval png] {e}")
    el = Path(out_dir)/"eval_log.csv"; new = not el.exists()
    with el.open("a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["step","bpp","psnr","msssim","lpips","dists","score"])
        w.writerow([step, f"{res['bpp']:.6f}", f"{res['psnr']:.4f}", f"{res['msssim']:.4f}",
                    f"{res['lpips']:.4f}", f"{res['dists']:.4f}", f"{res['score']:.4f}"])
    return res


def plot_reports(out_dir):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        figs = Path(out_dir);
        tl = Path(out_dir)/"train_log.csv"
        if tl.exists():
            rows = list(csv.DictReader(tl.open()))
            if rows:
                steps = [int(r["step"]) for r in rows]
                keys = [k for k in rows[0] if k not in ("step","lr")]
                fig, ax = plt.subplots(figsize=(10,6))
                for k in keys: ax.plot(steps, [float(r[k]) for r in rows], label=k, alpha=.85)
                ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_yscale("log")
                ax.set_title("StableCodec LoViF finetune — training losses"); ax.legend(); ax.grid(alpha=.3)
                fig.tight_layout(); fig.savefig(figs/"loss_curves.png", dpi=130); plt.close(fig)
        el = Path(out_dir)/"eval_log.csv"
        if el.exists():
            rows = list(csv.DictReader(el.open()))
            if rows:
                steps = [int(r["step"]) for r in rows]
                fig, ax = plt.subplots(2, 3, figsize=(16, 8))
                for a, m in zip(ax.flat, ["bpp","psnr","msssim","lpips","dists","score"]):
                    a.plot(steps, [float(r[m]) for r in rows], "-o", ms=3)
                    a.set_title(m + (" ↓" if m in ("bpp","lpips","dists") else " ↑")); a.grid(alpha=.3)
                    if m == "bpp": a.axhline(0.008, ls="--", c="red", label="budget"); a.legend()
                fig.suptitle("Eval on dataset_val (first N) — competition metrics + SCORE")
                fig.tight_layout(); fig.savefig(figs/"eval_scores.png", dpi=130); plt.close(fig)
    except Exception as e:
        log(f"  [plot] {e}")


def prune_checkpoints(ckpt_dir, lam, keep_last):
    """Keep only the `keep_last` most-recent step checkpoints; never touch best.pkl/final."""
    cks = []
    for p in Path(ckpt_dir).glob(f"stablecodec_ft{int(lam)}_*.pkl"):
        s = p.stem.split("_")[-1]
        if s.isdigit(): cks.append((int(s), p))
    cks.sort()
    for _, p in cks[:-keep_last] if keep_last > 0 else cks:
        try: p.unlink()
        except Exception: pass


def lr_for_step(step, base):
    # StableCodec schedule shape (scaled): 5e-5 -> 2e-5 (5k) -> 1e-5 (10k) -> 1e-6 (15k)
    if step >= 15000: return base * 0.02
    if step >= 10000: return base * 0.2
    if step >= 5000:  return base * 0.4
    return base


# --------------------------------------------------------------------------
#  main
# --------------------------------------------------------------------------
def main(args):
    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        args.sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    # codec requires patch a multiple of 256 (y=patch/64 must be divisible by 4)
    if args.train_patch_size % 256 != 0:
        args.train_patch_size = max(256, (args.train_patch_size // 256) * 256)
        print(f"[fix] train_patch_size -> {args.train_patch_size} (must be multiple of 256)", flush=True)
    args.val_patch = max(256, (args.val_patch // 256) * 256)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision)
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    for sub in ("checkpoints", "eval", "resume"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # ---- model (StableCodec needs these arg attrs) ----
    net = StableCodec(sd_path=args.sd_path, args=args)
    net.set_train()
    if args.enable_xformers and is_xformers_available():
        net.unet.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        net.unet.enable_gradient_checkpointing()

    # ---- trainable params: codec + unet-LoRA + conv_in + vae-LoRA (StableCodec finetune) ----
    layers_to_opt = list(net.codec.parameters())
    for n_, p_ in net.unet.named_parameters():
        if "lora" in n_: layers_to_opt.append(p_)
    layers_to_opt += list(net.unet.conv_in.parameters())
    for n_, p_ in net.vae.named_parameters():
        if "lora" in n_: layers_to_opt.append(p_)
    log(f"generator trainable: {sum(p.numel() for p in layers_to_opt)/1e6:.2f}M")

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    # ---- discriminator (optional; AEIC/StableCodec use dinov2) ----
    net_disc = optimizer_disc = None
    if args.lambda_gan > 0:
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type="dinov2", output_type="conv_multi_level",
                                                   loss_type=args.gan_loss_type, device=str(accelerator.device))
        net_disc.cv_ensemble.requires_grad_(False); net_disc.train()
        optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.disc_lr, betas=(0.5, 0.9),
                                           weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
        log(f"GAN ON (dinov2): disc heads {sum(p.numel() for p in net_disc.parameters() if p.requires_grad)/1e6:.2f}M")
    else:
        log("GAN OFF (lambda_gan=0)")

    # ---- training loss nets ----
    net_lpips = lpips.LPIPS(net=args.lpips_net, verbose=False).to(accelerator.device).eval()
    net_lpips.requires_grad_(False)
    import DISTS_pytorch
    from DISTS_pytorch import DISTS as _D
    net_dists = _D(load_weights=False)
    _w = torch.load(os.path.join(os.path.dirname(DISTS_pytorch.__file__), "weights.pt"), map_location="cpu")
    net_dists.alpha.data, net_dists.beta.data = _w["alpha"], _w["beta"]
    net_dists = net_dists.to(accelerator.device).eval(); net_dists.requires_grad_(False)
    from pytorch_msssim import ms_ssim
    net_clip = CLIPLoss().to(accelerator.device).eval() if args.lambda_clip > 0 else None

    train_dl = DataLoader(PngFolders(args.train_dirs, args.train_patch_size),
                          batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers,
                          shuffle=True, pin_memory=True, drop_last=True)

    prep = [net, optimizer, train_dl]
    if net_disc is not None: prep += [net_disc, optimizer_disc]
    prepared = accelerator.prepare(*prep)
    net, optimizer, train_dl = prepared[0], prepared[1], prepared[2]
    if net_disc is not None: net_disc, optimizer_disc = prepared[3], prepared[4]

    metrics = build_metrics(accelerator.device) if accelerator.is_main_process else None
    crops = load_val_crops(args.val_dir, args.val_num, args.val_patch, accelerator.device) \
        if accelerator.is_main_process else []
    if accelerator.is_main_process:
        log(f"train imgs={len(train_dl.dataset)} | val crops={len(crops)} @ {args.val_patch} "
            f"| (note: val overlaps train -> optimistic eval)")

    # ---- resume ----
    global_step = 0
    rdir = out_dir/"resume"; sfile = rdir/"step.json"
    if any(rdir.iterdir()):
        try:
            accelerator.load_state(str(rdir))
            if sfile.exists(): global_step = json.loads(sfile.read_text())["global_step"]
            log(f"RESUMED @ {global_step}")
        except Exception as e: log(f"resume failed: {e}")

    mse_loss = torch.nn.MSELoss()
    log(f"=== finetune ft{int(args.lambda_rate)} | steps={args.max_train_steps} "
        f"eff_batch={accelerator.num_processes*args.train_batch_size*args.gradient_accumulation_steps} "
        f"prec={args.mixed_precision} GAN={'on' if net_disc is not None else 'off'} ===")
    t0 = time.time(); logs = {}; done = False; best_score = -1e9
    while not done:
        for batch in train_dl:
            lr = lr_for_step(global_step, args.lr)
            for g in optimizer.param_groups: g["lr"] = lr
            acc_mods = [net] + ([net_disc] if net_disc is not None else [])
            with accelerator.accumulate(*acc_mods):
                x = batch.to(accelerator.device)
                x_hat, RL = net(x, [1]*x.shape[0], args.train_patch_size, args.train_patch_size)
                x = x.float(); x_hat = x_hat.float()
                x01, xh01 = (x*0.5+0.5).clamp(0,1), (x_hat*0.5+0.5).clamp(0,1)
                # ---- NEW LoViF-aligned loss ----
                l_l2 = mse_loss(x_hat, x)
                l_lpips = net_lpips(x_hat, x).mean()
                l_dists = net_dists(xh01, x01).mean()
                l_msssim = 1.0 - ms_ssim(xh01, x01, data_range=1.0)
                loss_D = (args.lambda_l2*l_l2 + args.lambda_lpips*l_lpips
                          + args.lambda_dists*l_dists + args.lambda_msssim*l_msssim)
                logs = {"rate": float(RL.rate_loss.detach()), "l2": float(l_l2.detach()),
                        "lpips": float(l_lpips.detach()), "dists": float(l_dists.detach()),
                        "msssim": float(l_msssim.detach())}
                if net_clip is not None:
                    l_clip = net_clip(x_hat, x); loss_D = loss_D + args.lambda_clip*l_clip
                    logs["clip"] = float(l_clip.detach())
                if net_disc is not None:
                    l_adv = net_disc(x_hat, for_G=True).mean(); loss_D = loss_D + args.lambda_gan*l_adv
                    logs["adv"] = float(l_adv.detach())
                loss = RL.rate_loss + loss_D
                accelerator.backward(loss)
                if accelerator.sync_gradients: accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step(); optimizer.zero_grad()
                # ---- discriminator: D-real then D-fake (StableCodec order) ----
                if net_disc is not None:
                    l_real = net_disc(x.detach(), for_real=True).mean() * args.lambda_gan
                    accelerator.backward(l_real)
                    if accelerator.sync_gradients: accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step(); optimizer_disc.zero_grad()
                    l_fake = net_disc(x_hat.detach(), for_real=False).mean() * args.lambda_gan
                    accelerator.backward(l_fake)
                    if accelerator.sync_gradients: accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step(); optimizer_disc.zero_grad()
                    logs["d"] = float((l_real + l_fake).detach())

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % args.log_freq == 0:
                    sps = global_step / max(1e-6, time.time()-t0)
                    log(f"step {global_step}/{args.max_train_steps} lr={lr:.1e} | "
                        + " ".join(f"{k}={v:.4f}" for k, v in logs.items()) + f" | {sps:.2f} it/s")
                    tl = out_dir/"train_log.csv"; new = not tl.exists()
                    with tl.open("a", newline="") as f:
                        wc = csv.writer(f); cols = ["step","lr"]+list(logs.keys())
                        if new: wc.writerow(cols)
                        wc.writerow([global_step, f"{lr:.2e}"]+[f"{logs[k]:.6f}" for k in logs])
                if accelerator.is_main_process and global_step % args.resume_steps == 0:
                    accelerator.save_state(str(rdir)); sfile.write_text(json.dumps({"global_step": global_step}))
                if accelerator.is_main_process and global_step % args.eval_freq == 0:
                    r = run_eval(accelerator.unwrap_model(net), metrics, crops, out_dir, global_step, args.save_num)
                    if r:
                        log(f"  [eval @ {global_step}] bpp={r['bpp']:.5f} PSNR={r['psnr']:.3f} "
                            f"MS-SSIM={r['msssim']:.4f} LPIPS={r['lpips']:.4f} DISTS={r['dists']:.4f} "
                            f">>> SCORE={r['score']:.3f}")
                        if r["score"] > best_score:   # keep the single BEST checkpoint (by crop-eval score)
                            best_score = r["score"]
                            bestf = out_dir/"checkpoints"/f"stablecodec_ft{int(args.lambda_rate)}_best.pkl"
                            accelerator.unwrap_model(net).save_model(str(bestf))
                            log(f"  ** new best (score={best_score:.3f}) -> {bestf.name}")
                if accelerator.is_main_process and global_step % args.plot_freq == 0:
                    plot_reports(out_dir)
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    outf = out_dir/"checkpoints"/f"stablecodec_ft{int(args.lambda_rate)}_{global_step}.pkl"
                    accelerator.unwrap_model(net).save_model(str(outf))
                    prune_checkpoints(out_dir/"checkpoints", args.lambda_rate, args.keep_last)  # keep only last N + best
                    log(f"  saved {outf.name} (kept last {args.keep_last} + best)")
                if global_step >= args.max_train_steps:
                    done = True; break

    if accelerator.is_main_process:
        outf = out_dir/"checkpoints"/f"stablecodec_ft{int(args.lambda_rate)}_final.pkl"
        accelerator.unwrap_model(net).save_model(str(outf)); plot_reports(out_dir)
        log(f"DONE -> {outf}")


if __name__ == "__main__":
    _args = parse_args()
    # ---- AUTO DUAL-GPU: a plain `!python finetune_lovif.py ...` will re-launch itself on
    #      ALL visible GPUs via `accelerate launch --multi_gpu` (real DDP, both T4s).
    #      The _SC_RELAUNCHED guard stops infinite recursion (the spawned procs skip this). ----
    _ngpu = torch.cuda.device_count()
    if (not _args.no_multi_gpu) and os.environ.get("_SC_RELAUNCHED") != "1" and _ngpu > 1:
        os.environ["_SC_RELAUNCHED"] = "1"
        _cmd = ["accelerate", "launch", "--multi_gpu", f"--num_processes={_ngpu}",
                f"--mixed_precision={_args.mixed_precision}", os.path.abspath(__file__)] + sys.argv[1:]
        print(f"[multi-gpu] {_ngpu} GPUs detected -> re-launching via: {' '.join(_cmd)}", flush=True)
        raise SystemExit(subprocess.call(_cmd, env=os.environ))
    main(_args)
