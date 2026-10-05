"""Export the CelebA training split as stratum class folders for latent diffusion.

The latent diffusion trainers (room_experiments/train_vae.py and train_ldm.py)
read a folder per class and a conditions YAML of mixtures over those classes.
This script writes

    <out>/images64/<dataset index>.png        the training split at 64x64
    <out>/<pair>/{q00,q10,q01,q11}/           relative symlinks into images64/

so that ldm_configs/<cell>.yaml reproduces the pixel-space stratum mixtures.
Images are stored as uint8 after the same resize the pixel-space models use.
Both pairs share one image set, since the identity split is the same.

    python -m celeba_experiments.export_strata --out /path/to/celeba_strata
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from celeba_experiments.data import (
    PAIRS, STRATA, extract_metadata, get_image, identity_split, load_celeba, stratum_indices,
    uint8_transform,
)

SIZE = 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_celeba()
    attributes = sorted({a for pair in PAIRS.values() for a in pair})
    metadata = extract_metadata(dataset, attributes)
    train_idx, _ = identity_split(metadata.identities)

    image_dir = args.out / "images64"
    image_dir.mkdir(parents=True, exist_ok=True)
    transform = uint8_transform(SIZE)
    for i in tqdm(train_idx, desc="images64"):
        array = transform(get_image(dataset[int(i)])).permute(1, 2, 0).numpy()
        Image.fromarray(array).save(image_dir / f"{int(i):06d}.png", format="PNG")

    for name, pair in PAIRS.items():
        strata = stratum_indices(metadata, pair, train_idx)
        for stratum, idx in zip(STRATA, strata):
            folder = args.out / name / stratum
            folder.mkdir(parents=True, exist_ok=True)
            for i in idx:
                filename = f"{int(i):06d}.png"
                os.symlink(os.path.relpath(image_dir / filename, folder), folder / filename)
        print(f"{name} {pair}: " + ", ".join(f"{s}={len(v)}" for s, v in zip(STRATA, strata)))


if __name__ == "__main__":
    main()
