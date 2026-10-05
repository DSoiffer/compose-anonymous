"""Paired variational-bound estimates of log P^a(x) - log P^0(x) for a conditional DDPM.

For a condition c the finite-step DDPM variational bound over a set of
timesteps T is

    L_c(x) = sum_{j in T} beta_j / (2 (1 - beta_j) (1 - abar_{j-1})) E_eps || eps - eps_c(x_j, j) ||^2,
    x_j = sqrt(abar_j) x + sqrt(1 - abar_j) eps,

and log P^a(x) - log P^0(x) is approximated by L_0(x) - L_a(x). Both
conditions are evaluated on the same (j, eps, x_j), and the two squared errors
are subtracted before averaging.

Timesteps are drawn uniformly in log-SNR over the window, with randomly
shifted van der Corput points, and each draw is weighted by
span * w_j / g_j (w_j the coefficient above, g_j the log-SNR width owned by
step j), so the Monte Carlo mean is unbiased for the discrete sum over T.
The window is given in log-SNR; usable_range maps it to timesteps.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# The four (numerator condition, sample source) pairs the metric needs, each
# estimated in its own Monte Carlo pass with seed offset `offset`:
#   at P^0 samples: d_A0 and d_B0, so log w_0 = d_A0 + d_B0
#   at P^A samples: d_B0 = log w_A
#   at P^B samples: d_A0 = log w_B
PASSES = (
    ("A", "0", 0),
    ("B", "0", 1),
    ("B", "A", 2),
    ("A", "B", 3),
)


def shifted_van_der_corput(n: int, m: int, rng: np.random.Generator,
                           n_replicates: int) -> np.ndarray:
    """(n, m) points in [0, 1): base-2 radical inverse, randomly shifted per row.

    The m points are split into n_replicates independently shifted groups,
    interleaved so that draw j belongs to group j % n_replicates.
    """
    if m % n_replicates:
        raise ValueError(f"m={m} must be divisible by n_replicates={n_replicates}")
    per = m // n_replicates
    k = max(int(np.ceil(np.log2(max(per, 2)))), 1)
    j = np.arange(per, dtype=np.int64)
    rev = np.zeros(per, dtype=np.int64)
    for i in range(k):
        rev = (rev << 1) | ((j >> i) & 1)
    base = rev / float(1 << k)
    shift = rng.random((n, n_replicates))
    u = np.remainder(base[None, None, :] + shift[:, :, None], 1.0)  # (n, R, per)
    return u.transpose(0, 2, 1).reshape(n, m)


class DiscreteDDPM:
    """Timestep draws and weights for a 1000-step v-prediction DDPM (alphas_cumprod).

    The Monte Carlo draws per sample are split into 8 independently shifted
    replicates, so the number of draws must be a multiple of 8.
    """

    n_replicates = 8

    def __init__(self, alphas_cumprod: np.ndarray, *, lam_min: float, lam_max: float):
        self.alphas_cumprod = np.asarray(alphas_cumprod, np.float64)
        self.j_lo, self.j_hi = self.usable_range(lam_min, lam_max)

    @property
    def T(self) -> int:
        return len(self.alphas_cumprod)

    @property
    def alpha_tab(self) -> np.ndarray:
        return np.sqrt(np.clip(self.alphas_cumprod, 0.0, 1.0))

    @property
    def sigma_tab(self) -> np.ndarray:
        return np.sqrt(np.clip(1.0 - self.alphas_cumprod, 0.0, 1.0))

    @property
    def log_snr(self) -> np.ndarray:
        a = np.clip(self.alphas_cumprod, 0.0, 1.0)
        with np.errstate(divide="ignore"):
            return np.log(a) - np.log1p(-a)

    @property
    def betas(self) -> np.ndarray:
        ab = self.alphas_cumprod
        return 1.0 - ab / np.concatenate([[1.0], ab[:-1]])

    @property
    def forward_gap(self) -> np.ndarray:
        """g[j] = lam[j] - lam[j+1], the log-SNR width owned by step j."""
        lam = self.log_snr
        with np.errstate(invalid="ignore"):
            return lam - np.concatenate([lam[1:], [-np.inf]])

    @property
    def vlb_weight(self) -> np.ndarray:
        """beta_j / (2 (1 - beta_j) (1 - abar_{j-1})), the DDPM bound's coefficient."""
        ab_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
        beta = self.betas
        with np.errstate(divide="ignore", invalid="ignore"):
            return beta / (2.0 * (1.0 - beta) * (1.0 - ab_prev))

    def usable_range(self, lam_min: float, lam_max: float) -> tuple[int, int]:
        """First and last timestep whose log-SNR lies in [lam_min, lam_max].

        Timestep 0 (the reconstruction term) is always excluded, and so is any
        step whose log-SNR cell has an infinite edge (the zero-SNR terminal steps).
        """
        lam = self.log_snr
        ok = np.where(np.isfinite(lam) & (lam <= lam_max) & (lam >= lam_min))[0]
        ok = ok[ok > 0]
        if len(ok) < 2:
            raise ValueError(f"log-SNR window [{lam_min}, {lam_max}] has <2 usable steps")
        j_lo, j_hi = int(ok.min()), int(ok.max())
        while j_hi > j_lo and not (j_hi + 1 < self.T and np.isfinite(lam[j_hi + 1])):
            j_hi -= 1
        return j_lo, j_hi

    def span(self) -> float:
        lam = self.log_snr
        return float(lam[self.j_lo] - lam[self.j_hi + 1])

    def draw(self, n: int, m: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """(n, m) timestep indices and their importance weights."""
        lam = self.log_snr
        rng = np.random.default_rng(seed)
        u = shifted_van_der_corput(n, m, rng, self.n_replicates)
        target = lam[self.j_hi + 1] + (lam[self.j_lo] - lam[self.j_hi + 1]) * u
        seg = lam[self.j_lo:self.j_hi + 1][::-1]  # increasing
        idx = np.clip(np.searchsorted(seg, target, side="left"), 0, len(seg) - 1)
        j = (self.j_hi - idx).astype(np.int64)
        weight = self.span() * self.vlb_weight[j] / self.forward_gap[j]
        return j, weight

    def alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        j = t.detach().cpu().numpy()
        a = torch.as_tensor(self.alpha_tab[j], dtype=torch.float32, device=t.device)
        s = torch.as_tensor(self.sigma_tab[j], dtype=torch.float32, device=t.device)
        return a.view(-1, 1, 1, 1), s.view(-1, 1, 1, 1)

    def to_eps(self, out: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """v-prediction to eps: with v = alpha eps - sigma x_0, eps = alpha v + sigma x_t."""
        a, s = self.alpha_sigma(t)
        return a * out + s * x_t


@torch.no_grad()
def paired_squared_errors(x0: torch.Tensor, eps_fns: dict, schedule: DiscreteDDPM, *,
                          n_mc: int, seed: int, device, batch: int) -> dict[str, np.ndarray]:
    """Weighted ||eps - eps_c(x_t, t)||^2 for each condition c, on shared draws.

    Returns {condition: (N, n_mc)}. The noise is drawn in chunks from one CPU
    generator; every chunk except the last draws a multiple of 16 scalars, so
    the stream does not depend on `batch`.
    """
    x0 = x0.float()
    n = x0.shape[0]
    m = n_mc
    j, weight = schedule.draw(n, m, seed=seed)
    event = tuple(x0.shape[1:])
    per = int(np.prod(event))
    eff = max(1, min(int(batch), n * m))
    step = 16 // math.gcd(16, per)
    eff = max(step, (eff // step) * step)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = {c: np.empty((n, m), dtype=np.float64) for c in eps_fns}
    for start in range(0, n * m, eff):
        sel = np.arange(start, min(start + eff, n * m))
        i_s, i_d = sel // m, sel % m
        eps = torch.randn((len(sel),) + event, generator=gen).to(device)
        t = torch.as_tensor(j[i_s, i_d], dtype=torch.long, device=device)
        a, s = schedule.alpha_sigma(t)
        x_t = a * x0[i_s].to(device) + s * eps
        w = torch.as_tensor(weight[i_s, i_d], dtype=torch.float64, device=device)
        for c, fn in eps_fns.items():
            e = schedule.to_eps(fn(t, x_t), x_t, t)
            sq = ((eps - e) ** 2).flatten(1).sum(1).double()
            out[c][i_s, i_d] = (w * sq).cpu().numpy()
    return out


def estimate_log_ratios(samples: dict[str, torch.Tensor], eps_fns: dict,
                        schedule: DiscreteDDPM, *, n_mc: int, seed: int, device,
                        batch: int) -> dict[tuple[str, str], np.ndarray]:
    """{(numerator, source): per-sample estimate of log P^num - log P^0}.

    eps_fns maps "0", "A", "B" to callables (t, x_t) -> raw model output, and
    samples maps "0", "A", "B" to that source's held-out samples.
    """
    estimates = {}
    for numerator, source, offset in PASSES:
        sq = paired_squared_errors(
            samples[source], {"0": eps_fns["0"], numerator: eps_fns[numerator]}, schedule,
            n_mc=n_mc, seed=seed + 1000 * (offset + 1), device=device, batch=batch)
        estimates[(numerator, source)] = (sq["0"] - sq[numerator]).mean(axis=1)
        print(f"  pass {offset}: d_{numerator}0 at P^{source} done", flush=True)
    return estimates
