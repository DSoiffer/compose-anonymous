"""CelebA loading, identity-disjoint split, attribute strata, and training mixtures.

The train, validation and test shards of flwrlabs/celeba are joined, and 10%
of the identities (seed 20250815) are held out. Models train on the remaining
images, and the held-out images give the sensitivity metric's held-out samples.

For an attribute pair (A, B), every image falls in one of four strata, coded
A + 2B and ordered (q00, q10, q01, q11) = (neither, A only, B only, both).
Each of the three conditions (0, A, B) is a mixture over these strata.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

DATASET = "flwrlabs/celeba"
DATASET_SPLIT = "train+valid+test"
SPLIT_SEED = 20250815
HELD_OUT_FRACTION = 0.1
STRATA = ("q00", "q10", "q01", "q11")

# The 40 CelebA attributes in their standard order (FaceXFormer output order).
ATTRIBUTES = (
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry",
    "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
    "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open",
    "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
)

# Attribute pairs used in the paper.
PAIRS = {
    "FC": ("Bangs", "Mouth_Slightly_Open"),
    "NFC": ("Brown_Hair", "Wavy_Hair"),
}

# Stratum probabilities (q00, q10, q01, q11) of the conditions P0, PA, PB.
REGIMES = {
    "id": (
        (0.99, 1 / 300, 1 / 300, 1 / 300),
        (0.0, 0.5, 0.0, 0.5),
        (0.0, 0.0, 0.5, 0.5),
    ),
    "ood": (
        (0.99, 1 / 300, 1 / 300, 1 / 300),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    ),
}


def load_celeba():
    """All 202,599 CelebA images (train, valid and test shards joined)."""
    from datasets import load_dataset

    return load_dataset(DATASET, split=DATASET_SPLIT)


@dataclass(frozen=True)
class CelebAMetadata:
    identities: np.ndarray
    attributes: Mapping[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.identities)


def extract_metadata(dataset: Any, attributes: Iterable[str]) -> CelebAMetadata:
    return CelebAMetadata(
        identities=np.asarray(dataset["celeb_id"], dtype=np.int64),
        attributes={name: np.asarray(dataset[name], dtype=bool) for name in attributes},
    )


def identity_split(identities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train and held-out image indices of the identity-disjoint split."""
    unique = np.unique(identities)
    rng = np.random.default_rng(SPLIT_SEED)
    held_out = rng.choice(unique, size=round(len(unique) * HELD_OUT_FRACTION), replace=False)
    is_test = np.isin(identities, held_out)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def stratum_indices(
    metadata: CelebAMetadata, pair: Sequence[str], allowed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a, b = pair
    codes = (
        metadata.attributes[a].astype(np.int8)
        + metadata.attributes[b].astype(np.int8) * 2
    )
    return tuple(allowed[codes[allowed] == code] for code in range(4))  # type: ignore[return-value]


def image_transform(size: int, *, train: bool) -> v2.Compose:
    """Resize the full 178x218 aligned frame to (size, size) and map to [-1, 1].

    The resize does not preserve aspect ratio. Training adds a random
    horizontal flip.
    """
    transforms: list[Any] = [v2.ToImage(), v2.Resize((size, size), antialias=True)]
    if train:
        transforms.append(v2.RandomHorizontalFlip())
    transforms.extend(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return v2.Compose(transforms)


def uint8_transform(size: int) -> v2.Compose:
    """image_transform(train=False) stopped before the float conversion."""
    return v2.Compose([v2.ToImage(), v2.Resize((size, size), antialias=True)])


def get_image(row: Mapping[str, Any]) -> Image.Image:
    return row["image"].convert("RGB")


class MixtureTrainingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Balanced 0/A/B samples drawn from explicit stratum mixtures.

    Sample i has condition i % 3. Its stratum is drawn from that condition's
    mixture and its image uniformly from the stratum, using a generator seeded
    by (seed, i), so the stream does not depend on the number of workers.
    """

    def __init__(
        self,
        dataset: Any,
        strata: Sequence[np.ndarray],
        condition_mixtures: Sequence[Sequence[float]],
        *,
        image_size: int,
        samples_per_epoch: int,
        seed: int,
    ) -> None:
        if any(len(values) == 0 for values in strata):
            raise ValueError("all four training strata must be nonempty")
        self.dataset = dataset
        self.strata = tuple(np.asarray(values) for values in strata)
        self.condition_mixtures = np.asarray(condition_mixtures, dtype=np.float64)
        self.samples_per_epoch = samples_per_epoch - samples_per_epoch % 3
        self.seed = seed
        self.transform = image_transform(image_size, train=True)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        condition = index % 3
        digest = hashlib.blake2b(
            f"{self.seed}:{index}".encode(), digest_size=8
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        row = self.condition_mixtures[condition]
        eligible = np.flatnonzero(row > 0)
        code = int(rng.choice(eligible, p=row[eligible]))
        source_index = int(rng.choice(self.strata[code]))
        image = self.transform(get_image(self.dataset[source_index]))
        return image, torch.tensor(condition, dtype=torch.long)
