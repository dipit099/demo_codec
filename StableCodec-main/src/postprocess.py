"""
Decoder-side BLIND refinement post-processors for StableCodec (LoViF 2026).

These run on the *decoded RGB image only* — no ground-truth, no original image —
so they are test-phase legal (the decoder reproduces them from the bitstream).
Global strength is meant to be tuned once on the val set and then frozen.

E2a backend: `SDTurboRefiner` — SDEdit-style img2img with SD-Turbo (the same
blind diffusion-prior mechanism StableSR builds on), reusing the SD-Turbo weights
we already download. No extra checkpoint, no 4 GB risk. This is the cheap test of
"does diffusion refinement help LPIPS/DISTS?" before investing in full StableSR.

Usage (from the runner, applied after decompress, before saving):
    refiner = SDTurboRefiner(sd_path, device="cuda")
    pil_out = refiner.enhance(pil_in, strength=0.3, steps=2)
"""

from __future__ import annotations
import math
from PIL import Image


class SDTurboRefiner:
    """Blind SDEdit refinement with SD-Turbo img2img.

    strength : SDEdit denoising strength in (0,1]. Low (0.2-0.4) = gentle texture
               refinement that preserves structure; high = more hallucination.
    steps    : target number of *effective* denoise steps (SD-Turbo works at 1-4).
    """

    def __init__(self, sd_path, device="cuda", dtype="fp16",
                 prompt="high quality, sharp, detailed, realistic texture, 8k"):
        import torch
        from diffusers import AutoPipelineForImage2Image

        td = torch.float16 if dtype == "fp16" else torch.float32
        self.device = device
        self.prompt = prompt
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            sd_path, torch_dtype=td, safety_checker=None, requires_safety_checker=False
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe = self.pipe.to(device)
        # memory: high-res images must not OOM the T4
        try:
            self.pipe.enable_vae_tiling()
        except Exception:
            pass
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

    def enhance(self, pil_in: Image.Image, strength=0.3, steps=2, guidance=0.0):
        import torch
        if strength <= 0:
            return pil_in
        # SD-Turbo img2img requires num_inference_steps * strength >= 1.
        nis = max(2, int(math.ceil(steps / max(strength, 1e-3))))
        w, h = pil_in.size
        try:
            with torch.no_grad():
                out = self.pipe(
                    prompt=self.prompt,
                    image=pil_in,
                    strength=float(strength),
                    num_inference_steps=nis,
                    guidance_scale=float(guidance),  # SD-Turbo: 0.0
                ).images[0]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                # Fall back to the un-refined image so we never drop an image
                # (a missing image would invalidate the submission).
                return pil_in
            raise
        if out.size != (w, h):
            out = out.resize((w, h), Image.LANCZOS)
        return out


def build_refiner(name, sd_path, device="cuda", prompt=None):
    """Factory so the runner can select a backend by string."""
    if name in (None, "none", ""):
        return None
    if name == "sdturbo":
        kw = {"prompt": prompt} if prompt else {}
        return SDTurboRefiner(sd_path, device=device, **kw)
    raise ValueError(f"unknown postproc backend: {name!r}")
