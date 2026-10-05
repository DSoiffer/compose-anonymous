"""Sample the composition PA * PB / P0 from a trained CelebA DDPM.

Writes a .pt file with {"images": uint8 (N, 3, 64, 64), "metadata": {...}},
which evaluate.py reads. Batch b is seeded with seed + b * 100000007, so the
output depends on --batch_size.

    python -m celeba_experiments.sample --checkpoint models/FC_ID.pt --method naive --output samples/FC_ID_naive.pt
    python -m celeba_experiments.sample --checkpoint models/FC_ID.pt --method fkc --particles 16 \\
        --output samples/FC_ID_fkc16.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from celeba_experiments.fkc import sample_composition_batch
from celeba_experiments.train import load_diffusion_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("naive", "fkc"), required=True)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--inference_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    model, diffusion, checkpoint = load_diffusion_checkpoint(args.checkpoint, device)
    image_size = model.config.image_size
    images = []
    for start in tqdm(range(0, args.num_samples, args.batch_size), desc=args.method):
        count = min(args.batch_size, args.num_samples - start)
        batch = sample_composition_batch(
            model,
            diffusion,
            batch_size=count,
            image_size=image_size,
            method=args.method,
            particles=args.particles,
            inference_steps=args.inference_steps,
            seed=args.seed + (start // args.batch_size) * 100_000_007,
            device=device,
        )
        images.append(((batch.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu())

    payload = {
        "images": torch.cat(images),
        "metadata": {
            "checkpoint": str(args.checkpoint.resolve()),
            "training_experiment": checkpoint["experiment"],
            "method": args.method,
            "particles": args.particles if args.method == "fkc" else 1,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "inference_steps": args.inference_steps,
            "seed": args.seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved {args.num_samples} images to {args.output}")


if __name__ == "__main__":
    main()
