"""Sample the composition PA * PB / P0 from a CelebA latent diffusion model.

Uses the latent samplers in room_experiments/ldm_sampling.py: naive
composition (plain DDPM ancestral sampling of the weighted v-prediction) or
FKC with systematic resampling at every step, followed by decoding with the
cell's frozen VAE. Writes the same file format as sample.py, so evaluate.py
scores both.

    python -m celeba_experiments.sample_ldm --checkpoint /path/to/ldm_FC_ID/checkpoint-epoch50 \\
        --method fkc --particles 16 --output samples/ldm_FC_ID_fkc16.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from room_experiments.ldm_sampling import (
    decode_latents, load_ldm_checkpoint, sample_fkc_latent, sample_normal_latent,
)

CONDITIONS = ["base", "PA", "PB"]
BETAS = [-1.0, 1.0, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="train_ldm.py checkpoint directory (EMA weights are used)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("naive", "fkc"), required=True)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--inference_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=250,
                        help="output images per sampler call (times K particles for FKC)")
    args = parser.parse_args()

    bundle = load_ldm_checkpoint(args.checkpoint, device="cuda")
    if bundle.classes != CONDITIONS:
        raise ValueError(f"{args.checkpoint}: conditions {bundle.classes}, "
                         f"expected {CONDITIONS} (label order must be base, PA, PB)")
    k = args.particles if args.method == "fkc" else 1

    torch.manual_seed(args.seed)
    chunks = []
    for start in range(0, args.num_samples, args.batch_size):
        n = min(args.batch_size, args.num_samples - start)
        if args.method == "fkc":
            z = sample_fkc_latent(bundle, CONDITIONS, BETAS, n_output=n, n_particles=k,
                                  n_steps=args.inference_steps)
        else:
            z = sample_normal_latent(bundle, CONDITIONS, BETAS, n_output=n,
                                     n_steps=args.inference_steps)
        images = decode_latents(bundle, z)  # [0, 1]
        chunks.append((images * 255.0).round().clamp(0, 255).to(torch.uint8).cpu())
        print(f"  {start + n}/{args.num_samples}", flush=True)

    payload = {
        "images": torch.cat(chunks),
        "metadata": {
            "checkpoint": str(args.checkpoint.resolve()),
            "method": args.method,
            "particles": k,
            "num_samples": args.num_samples,
            "inference_steps": args.inference_steps,
            "seed": args.seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved {args.num_samples} images to {args.output}")


if __name__ == "__main__":
    main()
