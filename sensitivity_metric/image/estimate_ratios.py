"""Estimate the composition importance weights for one latent diffusion cell.

Runs the four paired variational-bound passes (vlb_ratio.PASSES) on held-out
samples from P^0, P^A and P^B, and saves the per-sample log-ratio estimates
to an .npz file. summarize.py turns these into -log rESS values.

Windows (log-SNR bounds, mapped to DDPM timesteps by the schedule):
    trimmed         [-9.2, 2.15]  -> timesteps 100..900
    unconstrained   [-20, 20]     -> timesteps 1..997

Room cells (held-out room images, n samples per source):

    python -m sensitivity_metric.image.estimate_ratios --checkpoint /path/to/ldm_FC_ID/checkpoint-epoch50 \\
        --room_images /path/to/heldout_images --window trimmed --out ratios/room_FC_ID_trimmed.npz

CelebA cells (held-out samples from prepare_celeba_heldout.py):

    python -m sensitivity_metric.image.estimate_ratios --checkpoint /path/to/ldm_celeba_FC_ID/checkpoint-epoch50 \\
        --celeba_samples heldout/FC_ID.npz --window trimmed --batch 2048 \\
        --out ratios/celeba_FC_ID_trimmed.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sensitivity_metric.image.latent_model import celeba_samples, load_latent_model, room_samples
from sensitivity_metric.image.vlb_ratio import estimate_log_ratios

WINDOWS = {"trimmed": (-9.2, 2.15), "unconstrained": (-20.0, 20.0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--room_images", type=Path,
                       help="held-out room images, one folder per class")
    group.add_argument("--celeba_samples", type=Path,
                       help="held-out CelebA samples (.npz from prepare_celeba_heldout.py)")
    parser.add_argument("--window", choices=tuple(WINDOWS), required=True)
    parser.add_argument("--n_eval", type=int, default=1024,
                        help="room samples per source (CelebA sizes come from the .npz)")
    parser.add_argument("--n_mc", type=int, default=512,
                        help="Monte Carlo draws per sample (a multiple of 8)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds the room sample draw and the Monte Carlo draws")
    parser.add_argument("--batch", type=int, default=512,
                        help="(sample, draw) pairs per forward pass; affects speed only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam_min, lam_max = WINDOWS[args.window]
    lm = load_latent_model(args.checkpoint, device=device, lam_min=lam_min, lam_max=lam_max)
    print(f"window {args.window}: timesteps {lm.schedule.j_lo}..{lm.schedule.j_hi}")
    if args.room_images is not None:
        samples = room_samples(lm, args.room_images, n=args.n_eval, seed=args.seed)
    else:
        samples = celeba_samples(lm, args.celeba_samples)
    print("samples per source: " + ", ".join(f"{s}={len(x)}" for s, x in samples.items()))

    estimates = estimate_log_ratios(samples, lm.eps_fns(), lm.schedule, n_mc=args.n_mc,
                                    seed=args.seed, device=device, batch=args.batch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{f"d_{num}0_at_{src}": v for (num, src), v in estimates.items()},
             window=args.window, j_lo=lm.schedule.j_lo, j_hi=lm.schedule.j_hi,
             n_mc=args.n_mc)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
