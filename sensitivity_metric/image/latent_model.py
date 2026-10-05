"""Load a latent diffusion checkpoint and its held-out source samples as latents.

A checkpoint is a train_ldm.py output directory (room_experiments). Its
conditions.yaml lists the base condition ("base") and the two conditions
being composed, in that order after the base: the first is source A, the
second is source B.

Held-out images are encoded with the checkpoint's frozen VAE using a sampled
posterior latent (the same encoding the diffusion model was trained on) and
standardized with the checkpoint's latent mean and std. One CPU generator,
seeded once, supplies the encoder noise for all three sources in the order
0, A, B.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from sensitivity_metric.image.vlb_ratio import DiscreteDDPM

SOURCES = ("0", "A", "B")


@dataclass
class LatentModel:
    unet: torch.nn.Module
    vae: torch.nn.Module
    schedule: DiscreteDDPM
    condition_names: list[str]
    class_weights: dict[str, dict[str, float]]
    classes: list[str]
    latent_mean: torch.Tensor  # (1, C, 1, 1)
    latent_std: torch.Tensor   # (1, C, 1, 1)
    data_mean: float
    data_std: float

    @property
    def role(self) -> dict[str, str]:
        """Map the source labels 0, A, B to this checkpoint's condition names."""
        others = [c for c in self.condition_names if c != "base"]
        return {"0": "base", "A": others[0], "B": others[1]}

    def eps_fns(self) -> dict:
        """{"0", "A", "B"} -> (t, x_t) -> raw UNet output for that condition."""
        def make(source: str):
            index = self.condition_names.index(self.role[source])

            def fn(t, x_t):
                labels = torch.full((x_t.shape[0],), index, dtype=torch.long,
                                    device=x_t.device)
                return self.unet(x_t, t, class_labels=labels).sample
            return fn
        return {s: make(s) for s in SOURCES}


def load_latent_model(ckpt: Path, *, device, lam_min: float, lam_max: float) -> LatentModel:
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel

    ckpt = Path(ckpt)
    unet = UNet2DModel.from_pretrained(str(ckpt / "ema_model")).to(device).eval()
    info = json.loads((ckpt / "vae.json").read_text())
    vae = AutoencoderKL.from_pretrained(info["vae"]).to(device).eval()
    vae.requires_grad_(False)

    scheduler = DDPMScheduler.from_pretrained(str(ckpt / "scheduler"))
    schedule = DiscreteDDPM(scheduler.alphas_cumprod.double().numpy(),
                            lam_min=lam_min, lam_max=lam_max)

    conditions = yaml.safe_load((ckpt / "conditions.yaml").read_text())["conditions"]
    channels = int(info["latent_channels"])
    norm = json.loads((ckpt / "normalize.json").read_text())
    view = (1, channels, 1, 1)
    return LatentModel(
        unet=unet, vae=vae, schedule=schedule,
        condition_names=[c["name"] for c in conditions],
        class_weights={c["name"]: dict(c["classes"]) for c in conditions},
        classes=sorted(conditions[0]["classes"]),
        latent_mean=torch.tensor(info["latent_mean"], dtype=torch.float32,
                                 device=device).view(view),
        latent_std=torch.tensor(info["latent_std"], dtype=torch.float32,
                                device=device).view(view),
        data_mean=float(norm["mean"][0]), data_std=float(norm["std"][0]),
    )


@torch.no_grad()
def _encode(lm: LatentModel, images: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Images in [-1, 1] (on the model device) -> standardized sampled latents (CPU)."""
    dist = lm.vae.encode(images).latent_dist
    eps = torch.randn(dist.mean.shape, generator=gen, dtype=torch.float32).to(images.device)
    z = dist.mean + dist.std * eps
    return ((z - lm.latent_mean) / lm.latent_std).float().cpu()


def room_samples(lm: LatentModel, image_root: Path, *, n: int, seed: int,
                 encode_seed: int = 0, batch: int = 32) -> dict[str, torch.Tensor]:
    """n held-out room images per source, drawn from the source's class mixture.

    For each image a class is drawn from the source's mixture weights, then a
    file uniformly from that class folder (with replacement).
    """
    image_root = Path(image_root)
    rng = np.random.default_rng(seed)
    paths = {c: sorted((image_root / c).glob("*.png")) for c in lm.classes}
    gen = torch.Generator(device="cpu").manual_seed(encode_seed)
    device = lm.latent_mean.device
    out = {}
    for source in SOURCES:
        weights = lm.class_weights[lm.role[source]]
        w = np.array([weights[c] for c in lm.classes], float)
        w = w / w.sum()
        cls = rng.choice(len(lm.classes), size=n, p=w)
        files = []
        for ci in cls:
            pool = paths[lm.classes[ci]]
            files.append(pool[rng.integers(len(pool))])
        latents = []
        for i in range(0, len(files), batch):
            arr = np.stack([np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
                            for p in files[i:i + batch]])
            img = torch.from_numpy(arr).permute(0, 3, 1, 2)
            img = ((img - lm.data_mean) / lm.data_std).to(device)
            latents.append(_encode(lm, img, gen))
        out[source] = torch.cat(latents)
    return out


def celeba_samples(lm: LatentModel, samples_path: Path, *, encode_seed: int = 0,
                   batch: int = 256) -> dict[str, torch.Tensor]:
    """Encode the held-out CelebA samples written by prepare_celeba_heldout.py."""
    z = np.load(samples_path, allow_pickle=False)
    gen = torch.Generator(device="cpu").manual_seed(encode_seed)
    device = lm.latent_mean.device
    out = {}
    for source in SOURCES:
        x = torch.from_numpy(z[f"x_{source}"]).to(torch.float32).div_(255.0).sub_(0.5).div_(0.5)
        out[source] = torch.cat([_encode(lm, x[i:i + batch].to(device), gen)
                                 for i in range(0, len(x), batch)])
    return out
