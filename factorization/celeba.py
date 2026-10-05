"""Mean-effect cosine heuristic for how factorized the CelebA attribute pairs are.

For each pair (A, B) this prints

    |cos(mu(q10) - mu(q00), mu(q01) - mu(q00))|,

where mu(q) is the mean feature of stratum q (q00 neither, q10 A only, q01 B
only). Each stratum mean uses up to PER_STRATUM (20,000) images drawn
uniformly (seed 0) from all CelebA images, train and held-out, as the 64x64
uint8 images the models use; PER_STRATUM = -1 uses every image. The VAE row
uses one autoencoder per pair.

    python -m factorization.celeba \\
        --vae_fc /path/to/vae_celeba_FC_ID/checkpoint-epoch50/ema_model \\
        --vae_nfc /path/to/vae_celeba_NFC_ID/checkpoint-epoch50/ema_model
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from celeba_experiments.data import PAIRS, extract_metadata, get_image, load_celeba, uint8_transform
from factorization.features import Encoder, abs_cosine

SIZE = 64
# Strata codes: q00 = neither attribute, q10 = A only, q01 = B only
STRATA = {"q00": 0, "q10": 1, "q01": 2}
ATTRIBUTES = ("Bangs", "Mouth_Slightly_Open", "Brown_Hair", "Wavy_Hair")
PER_STRATUM = 20000  # images per stratum; -1 uses every image


class _Images(Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices
        self.transform = uint8_transform(SIZE)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.transform(get_image(self.dataset[int(self.indices[i])])).numpy()


def _stack(batch):
    return np.stack(batch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae_fc", required=True, help="autoencoder for the FC pair")
    parser.add_argument("--vae_nfc", required=True, help="autoencoder for the NFC pair")
    parser.add_argument("--workers", type=int, default=8, help="image loading processes")
    args = parser.parse_args()
    device = torch.device("cuda")
    vaes = {"FC": args.vae_fc, "NFC": args.vae_nfc}

    dataset = load_celeba()
    metadata = extract_metadata(dataset, ATTRIBUTES)
    rng = np.random.default_rng(0)
    selected = {}
    for pair, (a, b) in PAIRS.items():
        code = metadata.attributes[a].astype(int) + 2 * metadata.attributes[b].astype(int)
        for stratum, c in STRATA.items():
            members = np.flatnonzero(code == c)
            if PER_STRATUM == -1:
                selected[(pair, stratum)] = members
            else:
                selected[(pair, stratum)] = rng.choice(
                    members, size=min(PER_STRATUM, len(members)), replace=False)

    # Load every selected image once, in dataset order.
    index = np.unique(np.concatenate(list(selected.values())))
    loader = DataLoader(_Images(dataset, index), batch_size=256,
                        num_workers=args.workers, collate_fn=_stack)
    images = np.concatenate(list(loader))
    rows = {key: np.sort(np.searchsorted(index, idx)) for key, idx in selected.items()}
    print(f"{len(index)} images loaded")

    def pair_cosine(features, pair):
        means = {s: features[rows[(pair, s)]].astype(np.float64).mean(0) for s in STRATA}
        return abs_cosine(means["q10"] - means["q00"], means["q01"] - means["q00"])

    print("| Encoder | " + " | ".join(f"{p} ({' + '.join(PAIRS[p])})" for p in PAIRS) + " |")
    print("|---|" + "---:|" * len(PAIRS))
    for name in ("CLIP", "DINOv2"):
        features = Encoder(name, device=device)(images, 256)
        print(f"| {name} | " + " | ".join(f"{pair_cosine(features, p):.3f}" for p in PAIRS) + " |")
    values = []
    for p in PAIRS:
        features = Encoder("VAE", device=device, vae_path=vaes[p])(images, 256)
        values.append(f"{pair_cosine(features, p):.3f}")
    print("| VAE | " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
