"""Ratio between the L2 source error in the sensitivity bound and the TV source error.

The bound measures source error as ||u_i||_{L2(P^i)} = sqrt(chi^2(P^i_hat || P^i))
= sqrt(exp(D_2(P^i_hat || P^i)) - 1), while the error amplification A uses
TV(P^i, P^i_hat). For each configuration this reports the mean over refits of

    sum_i ||u_i||_{L2(P^i)} / sum_i TV(P^i, P^i_hat),

using the same refit and TV samples as run_families.py, with 500 refits.

    python -m sensitivity_metric.gaussian.check_l2_ratio
"""

from __future__ import annotations

import argparse

import numpy as np
from joblib import Parallel, delayed

from sensitivity_metric.gaussian.families import families
from sensitivity_metric.gaussian.gaussian_tools import FullGaussian, compose, refit_tv, renyi2
from sensitivity_metric.gaussian.run_families import N_FIT, N_TV

REFITS = 500


def ratio(config) -> float:
    Q = compose(config.PA, config.PB, config.P0)
    rng = np.random.default_rng([config.seed, 0, 2])
    tv_samples = {s: (lambda X: (X, P.log_prob(X)))(P.sample(N_TV, rng))
                  for s, P in {"Q": Q, **config.sources()}.items()}
    rep = refit_tv(config.sources(), N_FIT, REFITS, np.random.default_rng([7, N_FIT, 0]),
                   tv_samples=tv_samples, keep_fits=True)
    proper = rep["tv_proper"]
    fits, sources = rep["fits"], config.sources()
    l2 = np.array([[np.sqrt(np.expm1(renyi2(FullGaussian(fits[s][0][r], fits[s][1][r]),
                                            sources[s])))
                    for s in ("0", "A", "B")] for r in range(len(proper))])[proper]
    return float((l2.sum(1) / rep["tv_src"][proper].sum(1)).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    configs = [c for cs in families().values() for c in cs
               if compose(c.PA, c.PB, c.P0) is not None]
    values = Parallel(n_jobs=args.jobs)(delayed(ratio)(c) for c in configs)
    print(f"{len(values)} configurations: ratio ranges from {min(values):.2f} "
          f"to {max(values):.2f}")


if __name__ == "__main__":
    main()
