"""Latent-space sampling for the stage-2 latent diffusion model (train_ldm.py).

Two samplers, both operating on VAE latents (latent_channels x latent_size x
latent_size) rather than pixels:

  * sample_fkc_latent     -- full Feynman-Kac-corrector sampling (n_particles > 1,
    every-step systematic resampling). Reuses the dimension-agnostic discrete-FKC
    core in fkc_sampling.py, just with a latent-shaped UNet and image_shape.
  * sample_normal_latent  -- plain DDPM ancestral sampling: the K=1 special case
    with the FKC score/weight/resample machinery stripped out. Roughly an order
    of magnitude cheaper per output sample.

Both take a single class-conditional UNet and one or more (condition, beta)
pairs; the per-step v-prediction is the beta-weighted sum over conditions,
matching the pixel-space FKC convention in fkc_sampling.py. A single condition
with beta=1 is ordinary conditional sampling.

After sampling, standardized latents are de-standardized (z = x * latent_std +
latent_mean) and decoded with the frozen VAE to pixels, then mapped to [0, 1]
using the checkpoint's image normalization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel

from room_experiments.fkc_sampling import DiffusersVPSchedule, ancestral_sample, fkc_sample_disc


@dataclass
class LDMBundle:
    """Everything needed to sample from and decode a latent-diffusion checkpoint."""

    model: UNet2DModel
    vae: AutoencoderKL
    scheduler: DDPMScheduler
    schedule: DiffusersVPSchedule
    classes: list[str]
    latent_shape: tuple[int, int, int]  # (C, H, W)
    latent_mean: torch.Tensor  # (1, C, 1, 1)
    latent_std: torch.Tensor   # (1, C, 1, 1)
    data_mean: list[float]
    data_std: list[float]
    device: torch.device


def load_ldm_checkpoint(checkpoint_dir: str | Path, device: str | torch.device = "cuda") -> LDMBundle:
    """Load the latent UNet (EMA weights), its frozen VAE, scheduler, class list
    and the latent normalization written by train_ldm.py."""
    device = torch.device(device)
    ckpt = Path(checkpoint_dir)
    weights_dir = ckpt / "ema_model"
    print(f"Loading UNet from {weights_dir}")
    model = UNet2DModel.from_pretrained(weights_dir).to(device)
    model.eval()

    with open(ckpt / "vae.json") as f:
        vae_info = json.load(f)
    print(f"Loading VAE {vae_info['vae']}")
    vae = AutoencoderKL.from_pretrained(vae_info["vae"]).to(device)
    vae.requires_grad_(False)
    vae.eval()

    C = int(vae_info["latent_channels"])
    sz = int(vae_info["latent_size"])
    latent_shape = (C, sz, sz)
    latent_mean = torch.tensor(vae_info["latent_mean"], dtype=torch.float32, device=device).view(1, C, 1, 1)
    latent_std = torch.tensor(vae_info["latent_std"], dtype=torch.float32, device=device).view(1, C, 1, 1)

    scheduler = DDPMScheduler.from_pretrained(ckpt / "scheduler")
    schedule = DiffusersVPSchedule(scheduler, device)

    with open(ckpt / "classes.json") as f:
        classes = json.load(f)["classes"]

    with open(ckpt / "normalize.json") as f:
        norm = json.load(f)
    data_mean, data_std = norm["mean"], norm["std"]

    print(
        f"  latent={C}x{sz}x{sz}  conditions={classes}\n"
        f"  latent_mean={vae_info['latent_mean']}  latent_std={vae_info['latent_std']}"
    )

    return LDMBundle(
        model=model, vae=vae, scheduler=scheduler, schedule=schedule, classes=classes,
        latent_shape=latent_shape, latent_mean=latent_mean, latent_std=latent_std,
        data_mean=data_mean, data_std=data_std, device=device,
    )


def _resolve_labels(
    classes_for_product: list[str], betas: list[float], all_classes: list[str],
) -> list[int]:
    if len(classes_for_product) != len(betas):
        raise ValueError(
            f"classes ({len(classes_for_product)}) and betas ({len(betas)}) must match"
        )
    unknown = [c for c in classes_for_product if c not in all_classes]
    if unknown:
        raise ValueError(f"Unknown conditions {unknown}; checkpoint has {all_classes}")
    return [all_classes.index(c) for c in classes_for_product]


def sample_normal_latent(
    bundle: LDMBundle,
    classes_for_product: list[str],
    betas: list[float],
    *,
    n_output: int,
    n_steps: int = 100,
) -> torch.Tensor:
    """Plain DDPM ancestral sampling in latent space (the K=1 shortcut).

    The per-step v-prediction is the beta-weighted sum over the requested
    conditions. Returns standardized latents of shape (n_output, C, H, W)."""
    labels = _resolve_labels(classes_for_product, betas, bundle.classes)
    return ancestral_sample(
        [(bundle.model, label) for label in labels], bundle.scheduler, betas,
        (n_output, *bundle.latent_shape), n_steps, bundle.device,
    )


def sample_fkc_latent(
    bundle: LDMBundle,
    classes_for_product: list[str],
    betas: list[float],
    *,
    n_output: int,
    n_particles: int = 4,
    n_steps: int = 100,
    fkc_max_step: int | None = None,
) -> torch.Tensor:
    """Full FKC sampling in latent space via the shared discrete-FKC core.

    Returns standardized latents of shape (n_output, C, H, W)."""
    x = fkc_sample_disc(
        bundle.model, bundle.scheduler, bundle.schedule,
        classes_for_product, betas, bundle.classes,
        image_shape=bundle.latent_shape,
        n_output=n_output,
        n_particles=n_particles,
        n_steps=n_steps,
        device=bundle.device,
        fkc_max_step=fkc_max_step,
        verbose=False,
    )
    return x.view(n_output, *bundle.latent_shape)


@torch.no_grad()
def decode_latents(bundle: LDMBundle, x_std: torch.Tensor) -> torch.Tensor:
    """De-standardize standardized latents and VAE-decode to pixel images in
    [0, 1]. x_std: (B, C, H, W) -> (B, 3, H*f, W*f)."""
    z = x_std * bundle.latent_std + bundle.latent_mean
    pixels = bundle.vae.decode(z).sample  # ~[-1, 1] (VAE was trained on [-1, 1])
    m = torch.tensor(bundle.data_mean, device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    s = torch.tensor(bundle.data_std, device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    return (pixels * s + m).clamp(0, 1)
