"""The relative effective sample size (rESS) used by the sensitivity metric."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def neg_log_ress(log_w) -> float:
    """-log of the relative ESS of self-normalized weights exp(log_w)."""
    lw = np.asarray(log_w, np.float64).ravel()
    log_n = np.log(lw.size)
    return float(logsumexp(2.0 * lw) - log_n - 2.0 * (logsumexp(lw) - log_n))
