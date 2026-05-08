"""Diagonal Gaussian distribution and analytical product/ratio."""

import numpy as np


class DiagonalGaussian:
    """Single Gaussian with diagonal covariance: N(mu, diag(variances))."""

    def __init__(self, mean, variances):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.variances = np.asarray(variances, dtype=np.float64)
        self.D = len(self.mean)
        assert self.variances.shape == (self.D,)

    @property
    def precisions(self):
        return 1.0 / self.variances

    def log_prob(self, x):
        x = np.atleast_2d(x)
        diff = x - self.mean
        mahal = -0.5 * np.sum(diff**2 / self.variances, axis=-1)
        log_norm = -0.5 * np.sum(np.log(2 * np.pi * self.variances))
        return log_norm + mahal

    def score(self, x):
        x = np.atleast_2d(x)
        return -(x - self.mean) / self.variances

    def sample(self, n):
        return np.random.randn(n, self.D) * np.sqrt(self.variances) + self.mean


def analytical_product_ratio_diagonal(*numerators, denominators=None):
    """Exact product/ratio of zero-mean DiagonalGaussians.

    Returns the DiagonalGaussian proportional to
        prod_i N(0, Sigma_i) / prod_j N(0, Sigma_j)
    """
    if denominators is None:
        denominators = []
    D = numerators[0].D
    prec = np.zeros(D)
    for g in numerators:
        prec += g.precisions
    for g in denominators:
        prec -= g.precisions
    if np.any(prec <= 0):
        raise ValueError(f"Product/ratio precision has non-positive entries: {prec}. ")
    return DiagonalGaussian(np.zeros(D), 1.0 / prec)
