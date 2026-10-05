"""Mean-effect cosine heuristic for how factorized the room objects are.

Under exact Factorized Conditionals the mean-effect vectors are orthogonal:

    (mu_a1 - mu_0)^T (mu_a2 - mu_0) = 0.

For each object pair (couch with a framed painting, couch with a coffee
table) this prints

    |cos(mu(couch) - mu(empty), mu(X) - mu(empty))|,

where mu(c) is the mean feature of all images in class folder c, the empty
room is the `control` class, and X is the second object. This is the same
construction as celeba.py. The VAE row uses one autoencoder per pair.

With --mixtures, mu_a is instead the mean of condition a in each room training
config (room_experiments/train_configs/{FC,NFC}_{ID,OOD}.yaml), built from the
class means with that condition's class weights, and there is one column per
config.

    python -m factorization.rooms --data_dir /path/to/dataset_out \\
        --vae_fc /path/to/checkpoints/vae_FC_ID/checkpoint-epoch50/ema_model \\
        --vae_nfc /path/to/checkpoints/vae_NFC_ID/checkpoint-epoch50/ema_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from factorization.features import Encoder, abs_cosine

PAIRS = {"FC": "framed_painting", "NFC": "coffee_table"}  # pair -> second object
LABELS = {"FC": "Couch, painting", "NFC": "Couch, table"}
CONFIGS = ("FC_ID", "FC_OOD", "NFC_ID", "NFC_OOD")
BATCH_SIZE = {"CLIP": 64, "DINOv2": 8, "VAE": 8}
TRAIN_CONFIGS = Path(__file__).resolve().parents[1] / "room_experiments" / "train_configs"


def pair_classes(pair: str, mixtures: bool) -> tuple[str, ...]:
    second = PAIRS[pair]
    if mixtures:
        return ("control", "couch", second, f"couch+{second}")
    return ("control", "couch", second)


def load_uint8(paths) -> np.ndarray:
    arrays = []
    for path in paths:
        with Image.open(path) as image:
            arrays.append(np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1))
    return np.stack(arrays)


def class_mean(encode: Encoder, paths, batch_size: int, chunk: int = 512) -> np.ndarray:
    """Mean feature over all images in `paths`, loaded `chunk` images at a time."""
    total = 0.0
    for start in range(0, len(paths), chunk):
        features = encode(load_uint8(paths[start:start + chunk]), batch_size)
        total = total + features.astype(np.float64).sum(axis=0)
    return total / len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Room dataset root with one folder per class.")
    parser.add_argument("--vae_fc", required=True,
                        help="autoencoder for the couch-painting pair")
    parser.add_argument("--vae_nfc", required=True,
                        help="autoencoder for the couch-table pair")
    parser.add_argument("--mixtures", action="store_true",
                        help="use the condition mixtures of the training configs instead of "
                             "pure classes")
    args = parser.parse_args()
    device = torch.device("cuda")
    vaes = {"FC": args.vae_fc, "NFC": args.vae_nfc}
    classes = {c for pair in PAIRS for c in pair_classes(pair, args.mixtures)}
    paths = {c: sorted((args.data_dir / c).glob("*.png")) for c in classes}

    columns = list(CONFIGS) if args.mixtures else [LABELS[pair] for pair in PAIRS]
    print("| Encoder | " + " | ".join(columns) + " |")
    print("|---|" + "---:|" * len(columns))
    for name in ("CLIP", "DINOv2", "VAE"):
        # CLIP and DINOv2 serve both pairs; the VAE row uses each pair's autoencoder.
        vae_for = {pair: vaes[pair] if name == "VAE" else None for pair in PAIRS}
        means = {}  # (pair, class) -> mean feature
        for vae_path in dict.fromkeys(vae_for.values()):
            encode = Encoder(name, device=device, vae_path=vae_path)
            class_means = {}
            for pair in PAIRS:
                if vae_for[pair] != vae_path:
                    continue
                for c in pair_classes(pair, args.mixtures):
                    if c not in class_means:
                        class_means[c] = class_mean(encode, paths[c], BATCH_SIZE[name])
                    means[(pair, c)] = class_means[c]
            del encode
            torch.cuda.empty_cache()

        values = []
        if args.mixtures:
            for config in CONFIGS:
                pair = config.split("_")[0]
                conditions = yaml.safe_load((TRAIN_CONFIGS / f"{config}.yaml").read_text())
                weights = {c["name"]: c["classes"] for c in conditions["conditions"]}

                def mixture(condition):
                    return sum(float(weights[condition].get(c, 0.0)) * means[(pair, c)]
                               for c in pair_classes(pair, mixtures=True))

                values.append(abs_cosine(mixture("couch") - mixture("base"),
                                         mixture(PAIRS[pair]) - mixture("base")))
        else:
            for pair, second in PAIRS.items():
                empty = means[(pair, "control")]
                values.append(abs_cosine(means[(pair, "couch")] - empty,
                                         means[(pair, second)] - empty))
        print(f"| {name} | " + " | ".join(f"{v:.3f}" for v in values) + " |")


if __name__ == "__main__":
    main()
