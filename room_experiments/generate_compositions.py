"""Generate composition images and save them as PNG files.

Samples from the weighted composition prod_i p(x | c_i)^{beta_i} and writes
one PNG per image to --out_dir. These are the images that judge_room_images.py
scores. Three kinds of checkpoints are supported:

  * one pixel-space conditional checkpoint from train.py,
  * one latent diffusion checkpoint from train_ldm.py (detected by its
    vae.json): sampling runs in latent space and the frozen VAE decodes,
  * several pixel-space checkpoints, one per condition (separate models), in
    the same order as --betas.

With --n_particles 1 the sampler is plain DDPM ancestral sampling of the
composed v-prediction (naive composition). With --n_particles > 1 it is FKC
with systematic resampling at every step. --fkc_max_step stops FKC weighting
and resampling after that many reverse steps.

Generation is resumable. Each batch is seeded by its first image index
(seed * 100003 + start), so a restarted run regenerates only the missing
images, and the result does not depend on how many restarts happened.

Example (FC + ID, naive composition, 500 images):

    python -m room_experiments.generate_compositions \\
      --checkpoint /path/to/FC_ID/checkpoint-epoch50 \\
      --classes couch framed_painting base --betas 1 1 -1 \\
      --n_particles 1 --n_output 500 --batch_size 50 --seed 0 --bf16 \\
      --out_dir /path/to/images/FC_ID/K1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from room_experiments.fkc_sampling import (
    DiffusersVPSchedule, ancestral_sample, denormalize, fkc_sample_disc,
    fkc_sample_disc_multimodel, load_checkpoint, scheduler_signature,
)
from room_experiments.ldm_sampling import (
    decode_latents, load_ldm_checkpoint, sample_fkc_latent, sample_normal_latent,
)

IMAGE_SHAPE = (3, 256, 256)


class _AutocastUNet(torch.nn.Module):
    """Run a UNet forward under autocast(dtype) but return an fp32 prediction.

    Only the convolution and matmul heavy forward uses reduced precision. The
    FKC score and weight computations downstream stay in fp32."""

    def __init__(self, model: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        with torch.autocast("cuda", dtype=self.dtype):
            out = self.model(*args, **kwargs)
        out.sample = out.sample.float()
        return out


def make_ldm_generator(checkpoint, classes, betas, device, bf16, fkc_max_step):
    bundle = load_ldm_checkpoint(checkpoint, device=device)
    if bf16:
        bundle.model = _AutocastUNet(bundle.model, torch.bfloat16)

    @torch.no_grad()
    def generate(n: int, n_particles: int, n_steps: int) -> torch.Tensor:
        if n_particles > 1:
            latents = sample_fkc_latent(
                bundle, classes, betas, n_output=n, n_particles=n_particles,
                n_steps=n_steps, fkc_max_step=fkc_max_step,
            )
        else:
            latents = sample_normal_latent(
                bundle, classes, betas, n_output=n, n_steps=n_steps,
            )
        return decode_latents(bundle, latents)

    return generate


def make_pixel_generator(checkpoint, classes, betas, device, bf16, fkc_max_step):
    model, scheduler, all_classes, mean, std = load_checkpoint(checkpoint, device)
    if bf16:
        model = _AutocastUNet(model, torch.bfloat16)
    unknown = [c for c in classes if c not in all_classes]
    if unknown:
        raise ValueError(f"Unknown conditions {unknown}; checkpoint has {all_classes}")
    schedule = DiffusersVPSchedule(scheduler, device)
    predictors = [(model, all_classes.index(c)) for c in classes]

    @torch.no_grad()
    def generate(n: int, n_particles: int, n_steps: int) -> torch.Tensor:
        if n_particles > 1:
            x = fkc_sample_disc(
                model, scheduler, schedule, classes, betas, all_classes,
                image_shape=IMAGE_SHAPE, n_output=n, n_particles=n_particles,
                n_steps=n_steps, device=device, fkc_max_step=fkc_max_step,
                verbose=False,
            )
            return denormalize(x.view(n, *IMAGE_SHAPE), mean, std)
        x = ancestral_sample(predictors, scheduler, betas, (n, *IMAGE_SHAPE), n_steps, device)
        return denormalize(x, mean, std)

    return generate


def make_multimodel_generator(checkpoints, betas, device, bf16, fkc_max_step):
    models, schedulers, means, stds = [], [], [], []
    for path in checkpoints:
        model, scheduler, _, mean, std = load_checkpoint(path, device)
        models.append(_AutocastUNet(model, torch.bfloat16) if bf16 else model)
        schedulers.append(scheduler)
        means.append(mean)
        stds.append(std)
    # All models must share one noise schedule for the FKC weight to be valid.
    if len({scheduler_signature(s) for s in schedulers}) != 1:
        raise ValueError("Separate-model checkpoints do not share one noise schedule.")
    scheduler, mean, std = schedulers[0], means[0], stds[0]
    schedule = DiffusersVPSchedule(scheduler, device)
    predictors = [(m, 0) for m in models]

    @torch.no_grad()
    def generate(n: int, n_particles: int, n_steps: int) -> torch.Tensor:
        if n_particles > 1:
            x = fkc_sample_disc_multimodel(
                models, scheduler, schedule, betas,
                image_shape=IMAGE_SHAPE, n_output=n, n_particles=n_particles,
                n_steps=n_steps, device=device, fkc_max_step=fkc_max_step,
                verbose=False,
            )
            return denormalize(x.view(n, *IMAGE_SHAPE), mean, std)
        x = ancestral_sample(predictors, scheduler, betas, (n, *IMAGE_SHAPE), n_steps, device)
        return denormalize(x, mean, std)

    return generate


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, nargs="+", required=True,
        help="One conditional checkpoint (pixel or latent), or one pixel "
        "checkpoint per condition (separate models, in --betas order).",
    )
    parser.add_argument(
        "--classes", type=str, nargs="+", default=None,
        help="Condition names for each product component (single checkpoint only).",
    )
    parser.add_argument("--betas", type=float, nargs="+", required=True)
    parser.add_argument("--n_particles", type=int, default=1, help="FKC particles K.")
    parser.add_argument("--n_output", type=int, required=True, help="Images to generate.")
    parser.add_argument("--batch_size", type=int, default=25,
                        help="Images per sampler call (the UNet batch is batch_size * K).")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fkc_max_step", type=int, default=None,
        help="Stop FKC weighting and resampling after this many reverse steps.",
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="Run the UNet forward under bf16 autocast. Predictions are cast "
        "back to fp32 before any FKC computation.",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.fkc_max_step is not None and args.n_particles <= 1:
        parser.error("--fkc_max_step only applies with --n_particles > 1.")

    if len(args.checkpoint) == 1:
        if args.classes is None or len(args.classes) != len(args.betas):
            parser.error("--classes must name one condition per --betas entry.")
        ckpt = args.checkpoint[0]
        if (Path(ckpt) / "vae.json").is_file():
            generate = make_ldm_generator(
                ckpt, args.classes, args.betas, device, args.bf16, args.fkc_max_step)
        else:
            generate = make_pixel_generator(
                ckpt, args.classes, args.betas, device, args.bf16, args.fkc_max_step)
    else:
        if len(args.checkpoint) != len(args.betas):
            parser.error("Pass one checkpoint per --betas entry for separate models.")
        generate = make_multimodel_generator(
            args.checkpoint, args.betas, device, args.bf16, args.fkc_max_step)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.png")))
    done = (existing // args.batch_size) * args.batch_size  # batch-aligned restart
    if done:
        print(f"Resuming from image index {done} ({existing} images present)")
    with tqdm(total=args.n_output, initial=min(done, args.n_output), desc="generate") as bar:
        start = done
        while start < args.n_output:
            n = min(args.batch_size, args.n_output - start)
            torch.manual_seed(args.seed * 100003 + start)
            images = generate(n, args.n_particles, args.n_steps)
            u8 = (images * 255).round().clamp(0, 255).to(torch.uint8)
            for j in range(n):
                Image.fromarray(u8[j].permute(1, 2, 0).cpu().numpy()).save(
                    out_dir / f"{start + j:04d}.png"
                )
            start += n
            bar.update(n)
    print(f"Saved {args.n_output} images to {out_dir}")


if __name__ == "__main__":
    main()
