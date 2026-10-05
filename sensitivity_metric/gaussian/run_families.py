"""Error amplification and the rESS metric for every Gaussian family configuration.

For each configuration:

  * Error amplification A. Each source is refit by maximum likelihood on
    n = 20,000 fresh samples (sample covariance with divisor n), the fits are
    composed in closed form, and

        A_r = TV(P^w, P^w_hat_r) / sum_i TV(P^i, P^i_hat_r),

    averaged over R = 2,000 refits. Refits whose composed precision is not
    positive definite are left out. Each TV is estimated on 8,192 samples from
    the true distribution, drawn once per configuration.
  * The metric. -log rESS = max_i -log rESS_i from exact densities on
    N = 20,000 fresh samples per source, repeated for 12 independent draws
    (columns C0..C11).
  * C_pop = max_i D_2(P^w || P^i), infinite when E_{P^i}[w^2] is infinite.

Writes results/family_sweep.csv, which plot_ranking.py and plot_bound.py read.

    python -m sensitivity_metric.gaussian.run_families
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sensitivity_metric.gaussian.families import families
from sensitivity_metric.gaussian.gaussian_tools import (
    compose, composition_log_weights, refit_tv, renyi2,
)
from sensitivity_metric.ress import neg_log_ress

N_FIT, REFITS, N_TV, N_EVAL, DRAWS = 20000, 2000, 8192, 20000, 12
RESULTS = Path(__file__).resolve().parent / "results"


def run_config(config) -> dict:
    Q = compose(config.PA, config.PB, config.P0)
    seed = config.seed
    rng = np.random.default_rng([seed, 0, 2])
    tv_samples = {s: (lambda X: (X, P.log_prob(X)))(P.sample(N_TV, rng))
                  for s, P in {"Q": Q, **config.sources()}.items()}
    rep = refit_tv(config.sources(), N_FIT, REFITS, np.random.default_rng([7, N_FIT, 0]),
                   tv_samples=tv_samples)
    proper = rep["tv_proper"]
    amp_tv = float((rep["tv_q"][proper] / rep["tv_src"][proper].sum(1)).mean())

    draws = []
    for k in range(DRAWS):
        rng_k = np.random.default_rng([seed, 900 + k])
        held = {s: P.sample(N_EVAL, rng_k) for s, P in config.sources().items()}
        lw = composition_log_weights(config.P0, config.PA, config.PB,
                                     held["0"], held["A"], held["B"])
        draws.append(max(neg_log_ress(lw[s]) for s in ("0", "A", "B")))
    return dict(family=config.family, name=config.name, knob=config.knob, amp_tv=amp_tv,
                C_pop=max(renyi2(Q, P) for P in config.sources().values()),
                **{f"C{k}": draws[k] for k in range(DRAWS)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=3, help="parallel worker processes")
    parser.add_argument("--out", type=Path, default=RESULTS / "family_sweep.csv")
    args = parser.parse_args()

    configs = [c for cs in families().values() for c in cs
               if compose(c.PA, c.PB, c.P0) is not None]
    print(f"{len(configs)} configurations")
    rows = Parallel(n_jobs=args.jobs, verbose=1)(delayed(run_config)(c) for c in configs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
