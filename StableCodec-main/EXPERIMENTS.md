# StableCodec — Score-Boost Experiment Plan (LoViF 2026, team HackFleet)

**Goal:** raise the LoViF `Final` score by improving **LPIPS** and **DISTS** only,
with **no training**, using **inference-only** changes.

> ⚠️ **TEST-PHASE FIRST.** Everything we build must be valid for the *final test
> phase*: it must run from the submitted decoder + checkpoints alone (**≤ 4 GB**,
> **no ground-truth leak**, no test-time training). Any idea that only helps the
> dev leaderboard and **cannot** ship in the test decoder is marked **DEV-ONLY**
> and must be explicitly approved before use.

### Leakage rule (encoder vs decoder) — read this first
The codec has two sides, and they have **different** rules:
- **Encoder (`compress`)** — *is allowed to see the original image* (it is the
  input). Optimizing the bitstream per-image against the source — choosing the
  best rate/checkpoint, picking the encode that minimizes LPIPS/DISTS **to the
  source** — is **legitimate**, not leakage. The decoder reproduces it from the
  bitstream alone.
- **Decoder (`decompress`)** — *only* the bitstream + fixed models. Using the
  original image here (e.g. StableCodec `--color_fix`, which copies GT colour
  statistics) **is leakage** → forbidden.

So: per-image *encoder-side* choices = OK. Any *decoder-side* enhancer (StableSR,
knobs) must be **blind** (no GT), with its global strength tuned once on the val
set and then frozen.

```
Final = PSNR + 10·MS_SSIM + 20·(1 − LPIPS) + 25·(1 − DISTS)
```

## 1. Scoring math (what a change is "worth")

| metric | weight | a +0.01 better metric is worth |
|--------|-------:|-------------------------------:|
| DISTS ↓ | 25 | **+0.25 Final** |
| LPIPS ↓ | 20 | **+0.20 Final** |
| MS_SSIM ↑ | 10 | +0.10 Final |
| PSNR ↑ | 1 | +0.01 Final / +1 per dB |

**Implication:** perception (45 total weight) ≫ distortion (11). We can spend
several dB of PSNR to buy LPIPS/DISTS and still gain. Any candidate change is
judged by `ΔFinal = 20·(−ΔLPIPS) + 25·(−ΔDISTS) − ΔPSNR − 10·(−ΔMS_SSIM)`.

## 2. Where we actually stand (board-faithful local scoring, 100 imgs)

```
PSNR 21.12 | MS-SSIM 0.744 | LPIPS 0.244 | DISTS 0.112 | Final 65.88 | avg_bpp 0.0069
```
- **#1 (sunsean):** LPIPS 0.069, DISTS 0.033, Final 72.19.
- The gap is **systemic, not just a bad tail**: the best 85 images still average
  **LPIPS 0.222 / DISTS 0.106**. Fixing only the worst-15 moves aggregate LPIPS
  by just 0.244 → 0.222. So the bulk must improve.
- **Failure tail (worst 15 by `20·LPIPS+25·DISTS`)** = LPIPS 0.369 / DISTS 0.149,
  and these are also the **lowest-PSNR** images (`corr(PSNR,LPIPS)=−0.65`): the
  decode failed structurally, it is *not* merely soft.
  Names: `0054 0073 0017 0062 0022 0052 0089 0090 0014 0068 0020 0011 0038 0045 0003`.
- **Bitrate headroom:** avg 0.0069 vs budget 0.008; 41/100 images < 0.006 bpp.

## 3. Experiments already run

### EXP-0 — post-hoc sharpening (UnsharpMask / Sharpness) — ❌ REJECTED
Tested on worst-15, applied to the already-decoded PNGs (zero GPU). **Every**
variant made **LPIPS and DISTS worse** (best case `sharp×1.5`: dLPIPS +0.007,
dDISTS +0.005, dFinal −0.54). Adding high-freq detail onto wrong structure
amplifies the feature-space error. → **Deterministic sharpening / filtering is a
dead end. Do not pursue.** (Also implies a blind enhancer must be tuned, not
assumed.)

## 4. Why the "obvious" decoder knobs are low-confidence (math)

StableCodec was **end-to-end finetuned at one operating point**: residual weight
α=1, **no** classifier-free guidance, **1** denoise step at t=999, one fixed
prompt embedding (text-encoder is deleted after init). Each knob below is an
**out-of-distribution** perturbation of a trained system, so expected yield is
low — we still expose them as cheap sweeps, but with low priors:

- **CFG / `guidance_scale`** — defined (1.07≈1.0, i.e. effectively off) but never
  applied in `decompress`. SD-Turbo is *distilled to run without CFG*; adding it
  pushes the latent OOD. **Low.**
- **Multi-step decode** — model trained for 1-step; iterating risks OOD. **Low-med.**
- **Residual scaling `+α·res`** — α=1 is the trained optimum; α≠1 is OOD but acts
  in *latent* space (not RGB sharpening, so not refuted by EXP-0). **Low-med.**
- **Prompt swap / per-image caption** — 1-step text conditioning is weak; OOD. **Low.**

## 5. Prioritized experiment queue (all need the CUDA codec → Kaggle T4)

Ranked by expected `ΔFinal ÷ effort`. Each is scored board-faithfully on the
**worst-15** first (fast, biggest movement), then a 25-image mixed sample, and
only promising configs go to the full 100.

| # | Experiment | In-dist? | Fits 4 GB? | Prior | Rationale |
|---|------------|:--------:|:----------:|:-----:|-----------|
| **E1** | **`ft16` checkpoint** (more bits) vs `ft24` | ✅ yes | ✅ | **High** | Raises the whole rate-distortion curve; directly fixes the starved failure tail. **Must verify avg_bpp ≤ 0.008 over the full set.** If `ft16` alone busts budget, do **encoder-side per-image rate allocation** (`ft16` where it fits, `ft24` elsewhere) — legitimate, ships both checkpoints (both small, well under 4 GB). |
| **E2a** | **SD-Turbo SDEdit refinement** (blind img2img, no extra ckpt) — `--postproc sdturbo` | ⚠️ blind | ✅ | **Med** | Cheap test of "does diffusion refinement help LPIPS/DISTS?" using only SD-Turbo we already ship. Built. Sweep `--postproc_strength`. Only changes the PNG (bin/bpp unchanged). |
| **E2b** | **Full StableSR** (time-aware encoder + SFT) | ⚠️ blind | ⚠️ tight | **Med-High** | Pursue **only if E2a shows gains.** Size note: official StableSR needs its own SD UNet → ~4.2 GB stacked on our stack. Fits 4 GB only by **sharing one SD-Turbo base** (codec-LoRA + StableSR-SFT as adapters on the same frozen base, ELIC dropped at decode, text-encoder replaced by cached embeddings) → ~2.5 GB. Complex integration. |
| **E3** | Decoder knobs: `res_scale`, `num_steps`, `guidance_scale` sweeps | ❌ OOD | ✅ | Low-Med | Cheap to run once flags exist; keep only if board-faithful LPIPS+DISTS drop. |
| **E4 — DEV-ONLY ⚠️** | **DiffBIR / SUPIR** blind restoration | ⚠️ | ❌ (>4 GB) | Med | **Cannot ship in the ≤4 GB test decoder → DEV-ONLY.** Useful only to measure the achievable ceiling. **Requires explicit user approval before running.** |

**Realistic expectation:** inference-only changes can plausibly take LPIPS
0.244 → ~0.16–0.19 and DISTS 0.112 → ~0.08 (≈ +1.5–2.5 Final, rank 7 → ~4–5).
Reaching winner-level 0.069 LPIPS almost certainly needs their trained model
(out of scope here).

## 6. Harness & guardrails (implemented in `lovif_stablecodec_run.py`)

- `--worst_set` : run the pipeline+scoring on the 15 worst images only (fast loop).
- `--n_compress N --n_score M` : bounded subset runs (e.g. 25/25).
- Baseline compare: per-image baseline scores for the worst-15 are stored in
  `baselines/worst15_ft24.json`; the runner prints `ΔLPIPS/ΔDISTS/ΔFinal` vs it.
- **Kaggle disk safety:** after building `submission.zip`, the loose
  `reconstructed/` + `bitstream/` dirs are deleted (keep **only the zip**); before
  each image the runner checks free disk and **stops gracefully + cleans up** if
  below a threshold, so a 12 h run never dies on a full disk.
- All runs must finish in < 12 h: worst-15 ≈ minutes; 25 imgs ≈ <1 h; 100 ≈ 2–3 h.

## 7. Submittability rules (do not break)

- Decoder reads the **bitstream only** — never the original image at decode
  (so StableCodec's `--color_fix` stays **off**; it uses GT color stats = leak).
- No test-time training / fine-tuning. No selecting outputs by GT score.
- Test-phase decoder + checkpoints ≤ 4 GB (ship fp16). Dev phase has no size cap.
