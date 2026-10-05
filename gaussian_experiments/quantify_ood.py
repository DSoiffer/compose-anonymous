"""Measure how much of the target composition lies outside each source's 95% HDR.

For each source P^a this computes L_a, the fraction of target samples outside
the 95% highest-density region (HDR) of P^a, and L_all, the fraction outside
all three HDRs at once. This produces the support-mismatch table for the 2D
Gaussian and Gaussian mixture settings.

  * 2D Gaussian: the HDR of each source is the ellipsoid
    x^T Sigma_a^{-1} x <= chi2_{2, 0.95}. Target samples are exact draws from
    the closed-form target Gaussian.
  * GMM: the HDR threshold of each source is the 5th percentile of its log
    density over calibration samples from that source. Target samples come
    from rejection sampling (ID) or importance sampling (OOD).

Run from this directory:

    python -m gaussian_experiments.quantify_ood
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import chi2

from gaussian_experiments.config import CONFIGS_DIR, load_gaussian_2d_config, load_gmm_config
from gaussian_experiments.gmm_lib import (
    DiagonalGaussian,
    analytical_product_ratio_diagonal,
    sample_product_ratio_is,
    sample_product_ratio_rejection,
)
from gaussian_experiments.run_gmm_experiment import build_gmms


DEFAULT_SEED = 20250727

GAUSSIAN_SETTINGS = (
    ("FC + ID", CONFIGS_DIR / "gaussian_2d_fc_id.yaml"),
    ("NFC + ID", CONFIGS_DIR / "gaussian_2d_nfc_id.yaml"),
    ("FC + OOD", CONFIGS_DIR / "gaussian_2d_fc_ood.yaml"),
    ("NFC + OOD", CONFIGS_DIR / "gaussian_2d_nfc_ood.yaml"),
)
GMM_SETTINGS = (
    ("ID", CONFIGS_DIR / "gmm_id.yaml"),
    ("OOD", CONFIGS_DIR / "gmm_ood.yaml"),
)
SOURCE_NAMES = ("p0", "pa1", "pa2")


def escape_rates(masks: dict[str, np.ndarray]) -> dict[str, float]:
    """Fraction of target samples outside each HDR, and outside all three."""
    rates = {name: float(np.mean(masks[name])) for name in SOURCE_NAMES}
    rates["all"] = float(np.mean(np.logical_and.reduce([masks[n] for n in SOURCE_NAMES])))
    return rates


def gaussian_setting(config_path, *, n_target: int, seed: int) -> dict[str, float]:
    config = load_gaussian_2d_config(config_path)
    dist_a1 = DiagonalGaussian(np.zeros(2), config.variances_a1)
    dist_a2 = DiagonalGaussian(np.zeros(2), config.variances_a2)
    dist_p0 = DiagonalGaussian(np.zeros(2), config.variances_base)
    target = analytical_product_ratio_diagonal(dist_a1, dist_a2, denominators=[dist_p0])

    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(mean=target.mean, cov=np.diag(target.variances), size=n_target)
    threshold = float(chi2.ppf(0.95, df=2))
    masks = {}
    for name, source in (("p0", dist_p0), ("pa1", dist_a1), ("pa2", dist_a2)):
        precision = np.linalg.inv(np.diag(source.variances))
        masks[name] = np.einsum("ni,ij,nj->n", x, precision, x) > threshold
    return escape_rates(masks)


def gmm_setting(
    config_path, *, n_calibration: int, n_target: int, seed: int, oversample_factor: int
) -> dict[str, float]:
    config = load_gmm_config(config_path)
    gmm_a1, gmm_a2, gmm_p0, target = build_gmms(
        config.means, config.attr_a1, config.attr_a2, config.component_std, config.sampler
    )

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if target.sampler == "rejection":
        x, _ = sample_product_ratio_rejection(
            target.numerator_gmms, target.denominator_gmms, n_target, rng=rng
        )
    else:
        x = sample_product_ratio_is(
            target.numerator_gmms, target.denominator_gmms, n_target,
            oversample_factor=oversample_factor, rng=rng,
        )

    masks = {}
    for index, (name, source) in enumerate((("p0", gmm_p0), ("pa1", gmm_a1), ("pa2", gmm_a2))):
        # The 5th percentile of the source's own log density marks its 95% HDR.
        np.random.seed(seed + 10 + 2 * index)
        calibration = source.sample(n_calibration)
        threshold = float(np.quantile(source.log_prob(calibration), 0.05))
        masks[name] = source.log_prob(x) < threshold
    return escape_rates(masks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian_samples", type=int, default=1_000_000)
    parser.add_argument("--gmm_calibration_samples", type=int, default=100_000)
    parser.add_argument("--gmm_target_samples", type=int, default=5_000)
    parser.add_argument("--gmm_oversample_factor", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    rows = []
    for index, (setting, path) in enumerate(GAUSSIAN_SETTINGS):
        rates = gaussian_setting(path, n_target=args.gaussian_samples, seed=args.seed + 100 * index)
        rows.append(("Gaussian", setting, rates))
    for index, (setting, path) in enumerate(GMM_SETTINGS):
        rates = gmm_setting(
            path,
            n_calibration=args.gmm_calibration_samples,
            n_target=args.gmm_target_samples,
            seed=args.seed + 1000 + 100 * index,
            oversample_factor=args.gmm_oversample_factor,
        )
        rows.append(("GMM", setting, rates))

    print("| Dataset | Setting | Outside H_0 | Outside H_a1 | Outside H_a2 | Outside all three |")
    print("|---|---|---:|---:|---:|---:|")
    for dataset, setting, r in rows:
        cells = " | ".join(f"{100 * r[k]:.2f}%" for k in (*SOURCE_NAMES, "all"))
        print(f"| {dataset} | {setting} | {cells} |")


if __name__ == "__main__":
    main()
