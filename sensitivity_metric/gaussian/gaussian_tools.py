"""Full-covariance Gaussians, maximum-likelihood refits, and composition.

The base composition Q proportional to P^A P^B / P^0 of Gaussians is exact in
natural parameters (precision Lam and eta = Lam mu):

    Lam_Q = Lam_A + Lam_B - Lam_0,    eta_Q = eta_A + eta_B - eta_0,

and is not a valid distribution when Lam_Q is not positive definite.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

LOG_2PI = float(np.log(2.0 * np.pi))
SOURCES = ("0", "A", "B")


def _sym(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + np.swapaxes(m, -1, -2))


class FullGaussian:
    """N(mean, cov) with full covariance."""

    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, np.float64).ravel()
        self.d = len(self.mean)
        self.cov = _sym(np.asarray(cov, np.float64).reshape(self.d, self.d))
        self.chol = np.linalg.cholesky(self.cov)

    @classmethod
    def diagonal(cls, mean, variances, d: int = 2) -> "FullGaussian":
        """N(mean, diag(variances)), with scalars broadcast to d dimensions."""
        mean = np.broadcast_to(np.asarray(mean, float), (d,)).copy()
        variances = np.broadcast_to(np.asarray(variances, float), (d,)).copy()
        return cls(mean, np.diag(variances))

    @property
    def precision(self) -> np.ndarray:
        return _sym(np.linalg.inv(self.cov))

    @property
    def eta(self) -> np.ndarray:
        return self.precision @ self.mean

    def log_prob(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, np.float64))
        z = solve_triangular(self.chol, (x - self.mean).T, lower=True).T
        return (-0.5 * np.sum(z * z, axis=1)
                - np.sum(np.log(np.diag(self.chol))) - 0.5 * self.d * LOG_2PI)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.mean + rng.standard_normal((n, self.d)) @ self.chol.T


def compose(PA: FullGaussian, PB: FullGaussian, P0: FullGaussian) -> FullGaussian | None:
    """Q proportional to P^A P^B / P^0, or None if Lam_Q is not positive definite."""
    lam = _sym(PA.precision + PB.precision - P0.precision)
    if np.linalg.eigvalsh(lam).min() <= 0:
        return None
    cov = _sym(np.linalg.inv(lam))
    return FullGaussian(cov @ (PA.eta + PB.eta - P0.eta), cov)


def renyi2(q: FullGaussian, p: FullGaussian) -> float:
    """D_2(q || p); +inf when 2 Sigma_p - Sigma_q is not positive definite."""
    s2 = _sym(2.0 * p.cov - q.cov)
    if np.linalg.eigvalsh(s2).min() <= 0:
        return float("inf")
    dm = q.mean - p.mean
    quad = float(dm @ np.linalg.solve(s2, dm))
    ld = (np.linalg.slogdet(s2)[1] + np.linalg.slogdet(q.cov)[1]
          - 2.0 * np.linalg.slogdet(p.cov)[1])
    return quad - 0.5 * ld


def composition_log_weights(P0, PA, PB, x0, xa, xb) -> dict[str, np.ndarray]:
    """log w_i = log Q - log P^i (up to a constant) on samples from each source."""
    return {
        "0": PA.log_prob(x0) + PB.log_prob(x0) - 2.0 * P0.log_prob(x0),
        "A": PB.log_prob(xa) - P0.log_prob(xa),
        "B": PA.log_prob(xb) - P0.log_prob(xb),
    }


def fit_mle_batch(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, n, d) samples -> MLE means (R, d) and covariances (R, d, d), divisor n."""
    m = X.mean(axis=1)
    Xc = X - m[:, None, :]
    S = np.einsum("rni,rnj->rij", Xc, Xc) / X.shape[1]
    return m, _sym(S)


def compose_batch(fits: dict[str, tuple[np.ndarray, np.ndarray]]):
    """Batched composition of fitted sources -> (mean_Q, cov_Q, proper)."""
    prec = {s: _sym(np.linalg.inv(S)) for s, (_, S) in fits.items()}
    lam = _sym(prec["A"] + prec["B"] - prec["0"])
    h = sum(sign * np.einsum("rij,rj->ri", prec[s], fits[s][0])
            for s, sign in (("A", 1.0), ("B", 1.0), ("0", -1.0)))
    proper = np.linalg.eigvalsh(lam)[:, 0] > 0
    R, d = h.shape
    mQ = np.full((R, d), np.nan)
    SQ = np.full((R, d, d), np.nan)
    if proper.any():
        SQ[proper] = _sym(np.linalg.inv(lam[proper]))
        mQ[proper] = np.einsum("rij,rj->ri", SQ[proper], h[proper])
    return mQ, SQ, proper


def log_prob_batch(means: np.ndarray, covs: np.ndarray, X: np.ndarray) -> np.ndarray:
    """log N(X; means[r], covs[r]) for R Gaussians at N points -> (R, N)."""
    X = np.atleast_2d(np.asarray(X, np.float64))
    d = X.shape[1]
    P = np.linalg.inv(covs)
    diff = X[None, :, :] - means[:, None, :]
    quad = np.einsum("rni,rij,rnj->rn", diff, P, diff)
    return -0.5 * quad - 0.5 * np.linalg.slogdet(covs)[1][:, None] - 0.5 * d * LOG_2PI


def tv_mc(log_p_true: np.ndarray, log_p_hat: np.ndarray) -> np.ndarray:
    """TV(P, P_hat) = E_P[max(0, 1 - p_hat/p)] from samples of P; (N,), (R, N) -> (R,)."""
    lr = np.minimum(log_p_hat - log_p_true[None, :], 50.0)
    return np.mean(np.clip(1.0 - np.exp(lr), 0.0, None), axis=1)


def refit_tv(sources: dict[str, FullGaussian], n: int, reps: int, rng: np.random.Generator,
             tv_samples: dict[str, tuple[np.ndarray, np.ndarray]],
             max_elems: int = 4_000_000, keep_fits: bool = False) -> dict:
    """Refit every source by MLE `reps` times on n fresh samples, and recompose.

    tv_samples maps "Q", "0", "A", "B" to (X, log p_true(X)) for a fixed sample
    of the true distribution, shared across replicates. Returns per-replicate
    TV(Q, Q_hat) (tv_q), TV(P^i, P^i_hat) (tv_src, shape (R, 3)) and whether
    Q_hat is a valid distribution (tv_proper). With keep_fits, also returns the
    fitted parameters per source.
    """
    chunk = max(1, max_elems // n)
    parts = {k: [] for k in ("tv_q", "tv_src", "tv_proper")}
    kept = {s: ([], []) for s in SOURCES}
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        fits = {}
        for s in SOURCES:
            P = sources[s]
            X = P.mean + rng.standard_normal((r, n, P.d)) @ P.chol.T
            fits[s] = fit_mle_batch(X)
        mQ, SQ, proper = compose_batch(fits)
        tvs = []
        for s in SOURCES:
            X, lp = tv_samples[s]
            tvs.append(tv_mc(lp, log_prob_batch(fits[s][0], fits[s][1], X)))
        tvq = np.ones(r)
        if proper.any():
            X, lp = tv_samples["Q"]
            tvq[proper] = tv_mc(lp, log_prob_batch(mQ[proper], SQ[proper], X))
        parts["tv_q"].append(tvq)
        parts["tv_proper"].append(proper)
        parts["tv_src"].append(np.stack(tvs, 1))
        if keep_fits:
            for s in SOURCES:
                kept[s][0].append(fits[s][0])
                kept[s][1].append(fits[s][1])
        done += r
    out = {k: np.concatenate(v, 0) for k, v in parts.items()}
    if keep_fits:
        out["fits"] = {s: (np.concatenate(kept[s][0]), np.concatenate(kept[s][1]))
                       for s in SOURCES}
    return out
