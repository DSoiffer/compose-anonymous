"""Within-family ranking figure (within_family_ranking.pdf).

For each family, the Spearman correlation between rESS^{-1/2} and the measured
error amplification A across the family's 20 configurations, computed
separately for each of the 12 held-out draws. The dot is the median over
draws and the bar spans the full range.

Colors are the family classes in figstyle.FAMILY_COLOR.

    python -m sensitivity_metric.gaussian.plot_ranking
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sensitivity_metric.gaussian import figstyle as fs
from sensitivity_metric.gaussian.families import DISPLAY_NAMES
from sensitivity_metric.gaussian.figstyle import BLUE, FAMILY_COLOR, GREY, GRID, ORANGE, RULE

HERE = Path(__file__).resolve().parent
K = 12


def main(csv: Path, out: Path) -> None:
    d = pd.read_csv(csv)
    rows = []
    for fam, g in d.groupby("family"):
        rho = np.array([spearmanr(g.amp_tv, g[f"C{k}"]).statistic for k in range(K)])
        rows.append(dict(fam=fam, n=len(g), med=np.median(rho),
                         lo=rho.min(), hi=rho.max(), col=FAMILY_COLOR[fam]))
    t = pd.DataFrame(rows).sort_values("med").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.axvline(0.0, color=RULE, lw=1.2, zorder=2)
    for i, r in t.iterrows():
        ax.plot([r.lo, r.hi], [i, i], color=r.col, lw=2.2, alpha=0.35,
                solid_capstyle="round", zorder=3)
        ax.plot(r.med, i, "o", ms=7, color=r.col, markeredgecolor="white",
                markeredgewidth=0.9, zorder=4)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([DISPLAY_NAMES[f] for f in t.fam], fontsize=13)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-0.7, len(t) - 0.3)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel(rf"Spearman $\rho$ between {fs.metric_label()} and error amplification $A$")
    ax.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    fs.despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    h = [plt.Line2D([], [], color=c, lw=2.4, marker="o", ms=6) for c in (BLUE, ORANGE, GREY)]
    ax.legend(h, [r"$\mathbf{P}^w$ broader or displaced", r"$\mathbf{P}^w$ narrower than sources",
                  r"$\mathbb{E}_{P^i}[w^2]=\infty$ throughout"],
              loc="upper left", frameon=False, fontsize=10.5)
    fig.tight_layout()
    fs.save(fig, out)
    print(t[["fam", "n", "med", "lo", "hi"]].to_string(
        index=False, float_format=lambda x: f"{x:7.3f}"))


if __name__ == "__main__":
    fs.setup()
    (HERE / "figures").mkdir(exist_ok=True)
    main(HERE / "results" / "family_sweep.csv", HERE / "figures" / "within_family_ranking.png")
