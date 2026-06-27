"""
L1 — Encoder-side latent optimization (perceptual RDO) for StableCodec.

Idea: the model is FROZEN. For one image, we treat the transmitted analysis
latent `y` as the only learnable tensor and run Adam to minimize
    LPIPS(decode(y), source) + lambda_dists * DISTS(decode(y), source)
    + lambda_rate * relu(bpp(y) - target_bpp)
through the *frozen, differentiable* decoder. The optimized y is then entropy
coded with `codec.compress_from_y`. The decoder is unchanged, so it reconstructs
the same y_hat the optimizer targeted.

This is encoder-side RDO (the encoder may use the source), NOT weight training:
no model parameter is updated, and the submitted decoder is untouched. Same
bitrate, output closer to this specific GT → better reference metrics.

Memory note: the loss needs the full diffusion+VAE decode graph for backprop,
which is heavy at high resolution. The runner caps this by pixel count and falls
back to plain compress above the cap. Gradient checkpointing is enabled to help.
"""

from __future__ import annotations
import torch


@torch.enable_grad()
def optimize_latent(net, x, pos_caption_enc, lpips_fn, dists_fn, *,
                    iters=40, lr=5e-3, lambda_dists=1.0, lambda_rate=0.0,
                    target_bpp=None, ori_h=None, ori_w=None, log=print):
    """Return an optimized analysis latent `y` (detached) for image `x`.

    net             : StableCodec (already .eval(), will be frozen here)
    x               : input image, normalized to [-1,1], padded, shape [1,3,H,W], cuda
    pos_caption_enc : the cached prompt embedding (net.pos_caption_enc)
    lpips_fn        : callable(img_a, img_b) in [-1,1]  (lpips alex)
    dists_fn        : callable(img_a, img_b) in [0,1]   (DISTS)
    """
    codec = net.codec

    # Freeze everything; only `y` will carry gradients.
    for p in net.parameters():
        p.requires_grad_(False)

    # Initial latents (no grad through the encoder analysis).
    with torch.no_grad():
        latent2 = net.aux_codec((x + 1) / 2).detach()
        lq_latent = net.vae.encode(x).latent_dist.mode() * net.vae.config.scaling_factor
        y0 = codec.g_a(lq_latent, latent2).detach()

    # Default rate target = the codec's own y0 bitrate, so optimization improves
    # distortion at (approximately) unchanged bits rather than buying quality
    # with bits and busting the budget.
    if lambda_rate > 0 and target_bpp is None:
        with torch.no_grad():
            _, r0, _ = codec.forward_from_y(y0, ori_h, ori_w)
            target_bpp = float(r0.quantized_total_bpp)
        if log:
            log(f"    [L1] rate target = y0 bpp = {target_bpp:.5f}")

    y = y0.clone().requires_grad_(True)
    opt = torch.optim.Adam([y], lr=lr)

    x01 = (x * 0.5 + 0.5).clamp(0, 1)
    best_y = y0.clone()
    best_loss = float("inf")

    for it in range(iters):
        opt.zero_grad(set_to_none=True)

        # Differentiable decode from y (frozen model).
        x_hat, rate_out, res = codec.forward_from_y(y, ori_h, ori_w)
        model_pred = net.unet(x_hat, net.timesteps,
                              encoder_hidden_states=pos_caption_enc).sample
        x_denoised = net.sched.step(model_pred, net.timesteps,
                                    x_hat[:, :4], return_dict=True).prev_sample \
            + net.res_scale * res
        img = net.vae.decode(x_denoised / net.vae.config.scaling_factor).sample.clamp(-1, 1)
        img01 = (img * 0.5 + 0.5).clamp(0, 1)

        loss_lpips = lpips_fn(img01 * 2 - 1, x01 * 2 - 1).mean()
        loss_dists = dists_fn(img01, x01).mean()
        loss = loss_lpips + lambda_dists * loss_dists

        bpp = rate_out.quantized_total_bpp
        if lambda_rate > 0 and target_bpp is not None:
            loss = loss + lambda_rate * torch.relu(bpp - target_bpp)

        loss.backward()
        opt.step()

        lv = float(loss.detach())
        if lv < best_loss:
            best_loss = lv
            best_y = y.detach().clone()

        if log and (it == 0 or (it + 1) % 10 == 0 or it == iters - 1):
            log(f"    [L1 it{it+1:03d}/{iters}] loss={lv:.4f} "
                f"lpips={float(loss_lpips):.4f} dists={float(loss_dists):.4f} "
                f"bpp={float(bpp):.5f}")

    return best_y
