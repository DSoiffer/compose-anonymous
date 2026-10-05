"""Train one three-condition CelebA DDPM (conditions 0, A, B).

The model is trained on the training side of the identity-disjoint split, on
the stratum mixtures of one regime (see data.REGIMES). The training stream
cycles evenly across the three conditions.

    python -m celeba_experiments.train --pair FC --regime id --output models/FC_ID.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from celeba_experiments.data import (
    PAIRS,
    REGIMES,
    MixtureTrainingDataset,
    extract_metadata,
    identity_split,
    load_celeba,
    stratum_indices,
)
from celeba_experiments.diffusion import DiffusionConfig, GaussianDiffusion
from celeba_experiments.model import EMA, ConditionalUNet, UNetConfig


def save_checkpoint(*, output, model, ema, optimizer, step, diffusion, experiment) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "model_config": model.config_dict(),
            "diffusion_config": diffusion.config_dict(),
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "experiment": experiment,
        },
        temporary,
    )
    temporary.replace(output)


def load_diffusion_checkpoint(
    path: Path, device: torch.device
) -> tuple[ConditionalUNet, GaussianDiffusion, dict[str, Any]]:
    """Load the EMA weights of a training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_config = dict(checkpoint["model_config"])
    model_config["channel_multipliers"] = tuple(model_config["channel_multipliers"])
    model = ConditionalUNet(UNetConfig(**model_config))
    model.load_state_dict(checkpoint["ema"]["state"])
    model.eval().to(device)
    diffusion = GaussianDiffusion(DiffusionConfig(**checkpoint["diffusion_config"]))
    return model, diffusion, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", choices=tuple(PAIRS), required=True)
    parser.add_argument("--regime", choices=tuple(REGIMES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=5_000)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--save_every", type=int, default=5_000,
                        help="Overwrite the output checkpoint every N steps.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    pair = PAIRS[args.pair]
    condition_mixtures = REGIMES[args.regime]
    dataset = load_celeba()
    metadata = extract_metadata(dataset, pair)
    train_indices, _ = identity_split(metadata.identities)
    strata = stratum_indices(metadata, pair, train_indices)

    training_data = MixtureTrainingDataset(
        dataset,
        strata,
        condition_mixtures,
        image_size=args.image_size,
        samples_per_epoch=args.steps * args.batch_size + 3,
        seed=args.seed,
    )
    loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    model = ConditionalUNet(
        UNetConfig(image_size=args.image_size, base_channels=args.base_channels)
    ).to(device)
    diffusion = GaussianDiffusion(DiffusionConfig(timesteps=args.diffusion_steps))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    ema = EMA(model, args.ema_decay)
    experiment = {
        "pair": list(pair),
        "regime": args.regime,
        "condition_mixtures_pi_00_10_01_11": {
            "P0": list(condition_mixtures[0]),
            "PA": list(condition_mixtures[1]),
            "PB": list(condition_mixtures[2]),
        },
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "ema_decay": args.ema_decay,
        "target_steps": args.steps,
        "train_stratum_sizes": [len(values) for values in strata],
    }

    def checkpoint(step: int) -> None:
        save_checkpoint(output=args.output, model=model, ema=ema, optimizer=optimizer,
                        step=step, diffusion=diffusion, experiment=experiment)

    start = time.monotonic()
    progress = tqdm(loader, total=args.steps, desc=f"{args.pair}/{args.regime}")
    for step, (clean, condition) in enumerate(progress, start=1):
        if step > args.steps:
            break
        clean = clean.to(device, non_blocking=True)
        condition = condition.to(device, non_blocking=True)
        timesteps = torch.randint(args.diffusion_steps, (len(clean),), device=device)
        noise = torch.randn_like(clean)
        noisy = diffusion.q_sample(clean, timesteps, noise)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(noisy, timesteps, condition)
            loss = F.mse_loss(prediction.float(), noise.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Linear warmup multiplied by a cosine decay over all steps.
        scale = min(1.0, step / max(args.warmup_steps, 1))
        cosine = 0.5 * (1 + math.cos(math.pi * step / args.steps))
        current_lr = args.learning_rate * scale * cosine
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()
        ema.update(model)
        if step % 50 == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
        if args.save_every > 0 and step % args.save_every == 0:
            checkpoint(step)
    experiment["elapsed_seconds"] = time.monotonic() - start
    checkpoint(args.steps)


if __name__ == "__main__":
    main()
