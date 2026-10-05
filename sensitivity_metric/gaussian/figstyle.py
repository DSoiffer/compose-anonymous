"""Shared style for the two metric figures.

When pdflatex is installed, the figures are typeset with LaTeX through
matplotlib's pgf backend, with mathptmx for Times text and math (this also
needs the pgf LaTeX package). Otherwise they fall back to matplotlib's own math
rendering with Times-like fonts, and some labels render slightly differently.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

# palette
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"
INK2, GRID = "#52514e", "#e4e3de"
RULE = "#9a9892"   # reference lines: recessive, distinct from the GREY data class

# Color class of each family: blue = P^w broader than or displaced from its
# sources, orange = P^w narrower than every source, grey = E_{P^i}[w^2]
# infinite at every configuration.
FAMILY_COLOR = {
    "anisotropic": BLUE,
    "orthogonal_shifts": BLUE,
    "opposing_shifts": BLUE,
    "shared_precision_p_lt_1": BLUE,
    "opposed_anisotropic": BLUE,
    "shared_precision_p_gt_1": ORANGE,
    "pw_far_from_base": ORANGE,
    "anisotropic_r_gt_2": GREY,
}



def uses_tex() -> bool:
    """True when the figures are typeset with LaTeX (the pgf backend)."""
    return matplotlib.get_backend().lower() == "pgf"


def metric_label() -> str:
    """Axis label for rESS^{-1/2}."""
    if uses_tex():
        return r"$\widehat{\operatorname{rESS}}^{-1/2}$"
    return r"$\mathrm{rESS}^{-1/2}$"


def setup() -> None:
    """Select the backend and set the figure style. Call before plotting."""
    if shutil.which("pdflatex") is not None:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage{mathptmx}\usepackage{amsmath}\usepackage{amssymb}",
        })
    else:
        print("pdflatex not found; drawing the figures without LaTeX")
        matplotlib.use("Agg")
        matplotlib.rcParams.update({
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        })
    matplotlib.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 13, "axes.labelsize": 14,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "figure.dpi": 200, "savefig.bbox": "tight",
        # embed TrueType rather than Type 3: most venues reject Type 3 outlines
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, out) -> None:
    """Write the figure as both PNG (for viewing) and PDF (vector, for the paper)."""
    out = Path(out)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.pdf').name}")


def despine(ax, keep=("left", "bottom")) -> None:
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
        if s in keep:
            ax.spines[s].set_color(GRID)
