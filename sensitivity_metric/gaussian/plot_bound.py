"""Bound versus amplification figure (bound_vs_amplification_by_family.pdf).

x = measured error amplification A (mean over refits), y = rESS^{-1/2}, with
-log rESS the median over the 12 held-out draws. The dashed diagonal is the
first-order bound A <= rESS^{-1/2}. Each family has its own marker, and its
color is the family's class (figstyle.FAMILY_COLOR) as in the ranking figure.

    python -m sensitivity_metric.gaussian.plot_bound
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sensitivity_metric.gaussian import figstyle as fs
from sensitivity_metric.gaussian.families import DISPLAY_NAMES
from sensitivity_metric.gaussian.figstyle import FAMILY_COLOR, GRID, INK2, RULE

HERE = Path(__file__).resolve().parent
K = 12
# Axis limits: data spans A 0.29-26.8 and the metric 1.00-141.4, padded about 1.15x.
XLIM, YLIM = (0.25, 33.0), (0.87, 175.0)

MARKERS = {  # family key -> marker, in legend order
    "shared_precision_p_lt_1": "s",
    "anisotropic": "^",
    "opposed_anisotropic": "o",
    "orthogonal_shifts": "D",
    "opposing_shifts": "v",
    "anisotropic_r_gt_2": "X",
    "pw_far_from_base": "P",
    "shared_precision_p_gt_1": "*",
}


def main(csv: Path, out: Path) -> None:
    d = pd.read_csv(csv)
    C = d[[f"C{k}" for k in range(K)]].to_numpy()
    d["x"] = np.exp(np.median(C, axis=1) / 2.0)

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.plot(XLIM, XLIM, color=RULE, lw=1.5, ls=(0, (5, 3)), alpha=0.6, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.tick_params(colors=INK2, labelsize=11.5)
    fs.despine(ax)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    handles = []
    for key, mk in MARKERS.items():
        g = d[d.family == key]
        lab, col = DISPLAY_NAMES[key], FAMILY_COLOR[key]
        sz = 95 if mk == "*" else (46 if mk in "DP" else 40)
        ax.scatter(g.amp_tv, g.x, marker=mk, s=sz, zorder=4,
                   color=col, linewidths=0.6, edgecolors="white")
        handles.append(plt.Line2D([], [], ls="", marker=mk, color=col,
                                  ms=7 if mk != "*" else 10, markeredgecolor="white",
                                  markeredgewidth=0.6, label=lab))
    ax.set_xlabel(r"Error amplification $A$")
    ax.set_ylabel(fs.metric_label())
    leg = ax.legend(handles=handles, loc="lower right", fontsize=9.5,
                    labelcolor=INK2, handletextpad=0.3, borderpad=0.4, labelspacing=0.3,
                    frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    leg.set_zorder(6)  # above the markers (4) and the diagonal (3)
    fig.tight_layout()
    fs.save(fig, out)
    below = d[d.amp_tv > d.x]
    print(f"below the diagonal: {len(below)}/{len(d)}")


if __name__ == "__main__":
    fs.setup()
    (HERE / "figures").mkdir(exist_ok=True)
    main(HERE / "results" / "family_sweep.csv",
         HERE / "figures" / "bound_vs_amplification_by_family.png")
