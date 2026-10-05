"""Draw held-out CelebA samples from P0, PA and PB for the sensitivity metric.

Samples come from the held-out identities only, so no model was trained on
them. Each source gets --n samples (1024 in the paper). For each source, the
per-stratum counts are drawn from a multinomial with that source's stratum 
weights, and the images are then drawn without replacement inside each 
stratum. Images are stored as uint8 64x64 arrays after the same resize the 
models use.

The output .npz holds, for each source s in {0, A, B}: x_s (uint8 images),
cls_s (stratum codes), idx_s (dataset indices) and w_s (stratum weights).

    python -m sensitivity_metric.image.prepare_celeba_heldout --pair FC --regime id \\
        --out heldout/FC_ID.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from celeba_experiments.data import (
    PAIRS, REGIMES, STRATA, extract_metadata, get_image, identity_split, load_celeba,
    stratum_indices, uint8_transform,
)

SIZE = 64


def draw_indices(w: np.ndarray, strata: list[np.ndarray], n: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n (stratum code, dataset index) draws from one source, without replacement."""
    counts = rng.multinomial(n, w)
    cls, idx = [], []
    for code, k in enumerate(counts):
        if k == 0:
            continue
        pool = strata[code]
        if len(pool) < k:
            raise ValueError(f"stratum {STRATA[code]} has {len(pool)} held-out images "
                             f"but {k} were drawn")
        cls.append(np.full(k, code, np.int8))
        idx.append(rng.choice(pool, size=k, replace=False))
    order = rng.permutation(n)
    return np.concatenate(cls)[order], np.concatenate(idx)[order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", choices=tuple(PAIRS), required=True)
    parser.add_argument("--regime", choices=tuple(REGIMES), required=True)
    parser.add_argument("--n", type=int, default=1024, help="samples per source")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pair = PAIRS[args.pair]
    rows = REGIMES[args.regime]
    weights = {label: np.asarray(row, dtype=np.float64) for label, row in zip(("0", "A", "B"), rows)}

    dataset = load_celeba()
    metadata = extract_metadata(dataset, pair)
    _, test_idx = identity_split(metadata.identities)
    strata = list(stratum_indices(metadata, pair, test_idx))

    transform = uint8_transform(SIZE)
    rng = np.random.default_rng(args.seed)
    out: dict[str, np.ndarray] = {}
    for label in ("0", "A", "B"):
        cls, idx = draw_indices(weights[label], strata, args.n, rng)
        out[f"x_{label}"] = np.stack(
            [transform(get_image(dataset[int(i)])).numpy() for i in idx]
        ).astype(np.uint8)
        out[f"cls_{label}"] = cls
        out[f"idx_{label}"] = idx.astype(np.int64)
        out[f"w_{label}"] = weights[label]
        print(f"  P^{label}: {np.bincount(cls, minlength=4).tolist()} over {list(STRATA)}")

    meta = dict(pair=list(pair), regime=args.regime, n=args.n, seed=args.seed,
                strata=list(STRATA))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, meta=json.dumps(meta), **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
