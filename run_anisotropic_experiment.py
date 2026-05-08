"""Anisotropic Gaussian experiment: composing distributions with different
covariance structures via Feynman-Kac correctors.

Default distributions (zero-mean, diagonal covariance):
  A = N(0, diag(10, 1))   stretched along x
  B = N(0, diag(1, 10))   stretched along y
  C = N(0, diag(1, 1))    isotropic baseline

Analytical product A * B / C has precision diag(1/10, 1/10), so the target
is N(0, diag(10, 10)).

Two entry points:
  python run_anisotropic_experiment.py             single-shot pipeline + plots
  python run_anisotropic_experiment.py --sweep     (training_size x K) sweep
"""

import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from gmm_lib import DiagonalGaussian, analytical_product_ratio_diagonal
from noise_schedule import VPSchedule
from score_model import (
    train_score_model,
    train_conditional_score_model,
    BoundConditionalModel,
    sample_eps_model,
)
from feynman_kac import feynman_kac_sample
from evaluation import compute_distribution_metrics, mmd_rbf
from sweep import sweep_training_and_particles


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_ROOT = os.path.join(SCRIPT_DIR, "figures")

SEED = 1
D = 2
GRID_RANGE = 12.0
N_SAMPLES = 5000
K_PARTICLES = 16
FIG_EXT = ".pdf"

DEFAULT_VARIANCES_A = (10.0, 1.0)
DEFAULT_VARIANCES_B = (1.0, 10.0)
DEFAULT_VARIANCES_C = (1.0, 1.0)


def _fmt_var(v: float) -> str:
    return f"{float(v):g}".replace(".", "p")


def _variances_id(va, vb, vc) -> str:
    def part(name, vs):
        return name + "-".join(_fmt_var(v) for v in vs)
    return f"{part('a', va)}_{part('b', vb)}_{part('c', vc)}"


def _diag_text(vs) -> str:
    vs = [float(v) for v in vs]
    if all(v == vs[0] for v in vs):
        if vs[0] == 1.0:
            return "I"
        return f"{vs[0]:g}I"
    return "diag(" + ", ".join(f"{v:g}" for v in vs) + ")"


def figures_dir_for(variances_a, variances_b, variances_c, mode: str) -> str:
    vid = _variances_id(variances_a, variances_b, variances_c)
    suffix = "" if mode == "separate" else f"_{mode}"
    return os.path.join(FIGURES_ROOT, f"anisotropic{suffix}_{vid}")


def build_distributions(
    variances_a=DEFAULT_VARIANCES_A,
    variances_b=DEFAULT_VARIANCES_B,
    variances_c=DEFAULT_VARIANCES_C,
):
    dist_a = DiagonalGaussian(mean=[0, 0], variances=list(variances_a))
    dist_b = DiagonalGaussian(mean=[0, 0], variances=list(variances_b))
    dist_c = DiagonalGaussian(mean=[0, 0], variances=list(variances_c))
    dist_gt = analytical_product_ratio_diagonal(dist_a, dist_b, denominators=[dist_c])
    return dist_a, dist_b, dist_c, dist_gt


def dist_data_fn(dist: DiagonalGaussian):
    def sample(bs):
        return torch.from_numpy(dist.sample(bs)).float()
    return sample


def eps_model_to_score_fn(model, schedule):
    def score_fn(t, x):
        return -model(t, x) / schedule.sigma(t)
    return score_fn


def analytical_score_fn(dist: DiagonalGaussian, schedule):
    """Exact score of the noised distribution at reverse time t.

    At reverse time t, x_t ~ N(alpha(t)*mu, alpha(t)^2 * diag(var) + sigma(t)^2 * I).
    """
    mean_cpu = torch.from_numpy(dist.mean).float()
    var_cpu = torch.from_numpy(dist.variances).float()
    cache: dict = {}

    def score_fn(t, x):
        device = x.device
        if device not in cache:
            cache[device] = (mean_cpu.to(device), var_cpu.to(device))
        mean_d, var_d = cache[device]
        alpha = schedule.alpha(t)
        sigma = schedule.sigma(t)
        mu_t = alpha * mean_d
        noised_var = alpha**2 * var_d + sigma**2
        return -(x - mu_t) / noised_var

    return score_fn


def _annotate_metric(ax, text: str) -> None:
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3),
    )


def _fmt_metrics(metrics: dict) -> str:
    return f"MMD$^2$={metrics['mmd2']:.4f}"


def _print_metrics(prefix: str, metrics: dict) -> None:
    print(f"{prefix}SW2={metrics['sw2']:.4f}  MMD^2={metrics['mmd2']:.5f}")


def sample_from_model(model, schedule, device, n_output=N_SAMPLES, n_steps=500):
    x_final = sample_eps_model(
        model, schedule, ndim=D,
        n_output=n_output, n_steps=n_steps,
        device=device, verbose=False,
    )
    return x_final.cpu().numpy()


def run_fk(score_fns, schedule, device, n_output=N_SAMPLES,
           n_particles=K_PARTICLES, n_steps=500):
    x_final = feynman_kac_sample(
        score_fns,
        betas=[1.0, 1.0, -1.0],
        schedule=schedule,
        ndim=D,
        n_output=n_output,
        n_particles=n_particles,
        n_steps=n_steps,
        device=device,
        verbose=False,
    )
    return x_final.cpu().numpy()


def plot_distributions_overview(dist_a, dist_b, dist_c, learned_samples, n_samples=N_SAMPLES):
    """Top row: ground-truth samples from A, B, C. Bottom row: learned samples
    from the trained models with MMD vs ground truth overlaid."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), squeeze=False)
    dists = [dist_a, dist_b, dist_c]
    labels = [
        f"A = N(0, {_diag_text(dist_a.variances)})",
        f"B = N(0, {_diag_text(dist_b.variances)})",
        f"C = N(0, {_diag_text(dist_c.variances)})",
    ]
    colors = ["tab:blue", "tab:orange", "tab:gray"]

    true_samples = [d.sample(n_samples) for d in dists]
    for ax, s, label, color in zip(axes[0], true_samples, labels, colors):
        ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.3, color=color)
        ax.set_title(label)

    learned_labels = ["Learned A", "Learned B", "Learned C"]
    for ax, key, label, color, gt_s in zip(
        axes[1], ["A", "B", "C"], learned_labels, colors, true_samples,
    ):
        s = learned_samples[key]
        ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.3, color=color)
        ax.set_title(label)
        mmd = mmd_rbf(s, gt_s, seed=SEED)
        _annotate_metric(ax, f"MMD$^2$={mmd:.4f}")

    for row in axes:
        for ax in row:
            ax.set_xlim(-GRID_RANGE, GRID_RANGE)
            ax.set_ylim(-GRID_RANGE, GRID_RANGE)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Anisotropic Gaussian Composition")
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.4)
    return fig


def plot_comparison_summary(
    analytical_samples, learned_samples, dist_gt,
    analytical_metrics, learned_metrics,
):
    """Three panels: ground truth vs FK with analytical scores vs FK with learned scores."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    n = min(N_SAMPLES, len(learned_samples))

    gt_s = dist_gt.sample(n)
    axes[0].scatter(gt_s[:, 0], gt_s[:, 1], s=2, alpha=0.3, color="green")
    axes[0].set_title("Ground truth")

    axes[1].scatter(analytical_samples[:n, 0], analytical_samples[:n, 1],
                    s=2, alpha=0.3, color="teal")
    axes[1].set_title(f"FKC (analytical scores, K={K_PARTICLES})")
    _annotate_metric(axes[1], _fmt_metrics(analytical_metrics))

    axes[2].scatter(learned_samples[:n, 0], learned_samples[:n, 1],
                    s=2, alpha=0.3, color="red")
    axes[2].set_title(f"FKC (learned scores, K={K_PARTICLES})")
    _annotate_metric(axes[2], _fmt_metrics(learned_metrics))

    for ax in axes:
        ax.set_xlim(-GRID_RANGE, GRID_RANGE)
        ax.set_ylim(-GRID_RANGE, GRID_RANGE)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.4)
    return fig


def train_separate_models(schedule, dist_a, dist_b, dist_c, device, num_iterations=10_000):
    print("=== Training separate models for A, B, C ===")
    return [
        train_score_model(dist_data_fn(d), schedule, D, device, num_iterations=num_iterations)
        for d in (dist_a, dist_b, dist_c)
    ]


def train_conditional_models(schedule, dist_a, dist_b, dist_c, device, num_iterations=10_000):
    print("=== Training single conditional model for A, B, C ===")
    cond_model = train_conditional_score_model(
        data_fns=[dist_data_fn(dist_a), dist_data_fn(dist_b), dist_data_fn(dist_c)],
        schedule=schedule, x_dim=D, device=device,
        num_iterations=num_iterations,
    )
    return [BoundConditionalModel(cond_model, idx) for idx in range(3)]


def main(
    variances_a=DEFAULT_VARIANCES_A,
    variances_b=DEFAULT_VARIANCES_B,
    variances_c=DEFAULT_VARIANCES_C,
    *,
    mode: str = "conditional",
    train_iters: int = 10_000,
):
    """Train models, run FKC with analytical and learned scores, save plots."""
    assert mode in ("separate", "conditional")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = VPSchedule(beta_min=0.1, beta_max=20.0)
    dist_a, dist_b, dist_c, dist_gt = build_distributions(variances_a, variances_b, variances_c)

    figures_dir = figures_dir_for(variances_a, variances_b, variances_c, mode)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Device: {device}, seed: {SEED}, samples: {N_SAMPLES}")
    print(f"A: N(0, {_diag_text(dist_a.variances)}),  "
          f"B: N(0, {_diag_text(dist_b.variances)}),  "
          f"C: N(0, {_diag_text(dist_c.variances)})")
    print(f"Target A*B/C: N(0, {_diag_text(dist_gt.variances)})")
    print(f"Output dir: {figures_dir}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if mode == "separate":
        model_a, model_b, model_c = train_separate_models(
            schedule, dist_a, dist_b, dist_c, device, num_iterations=train_iters,
        )
    else:
        model_a, model_b, model_c = train_conditional_models(
            schedule, dist_a, dist_b, dist_c, device, num_iterations=train_iters,
        )

    print("\n=== FKC with analytical scores ===")
    analytical_score_fns = [analytical_score_fn(d, schedule) for d in (dist_a, dist_b, dist_c)]
    analytical_samples = run_fk(analytical_score_fns, schedule, device)
    analytical_metrics = compute_distribution_metrics(
        analytical_samples, dist_gt, n_gt_samples=N_SAMPLES, seed=SEED,
    )
    _print_metrics("  ", analytical_metrics)

    print("\n=== FKC with learned scores ===")
    learned_score_fns = [eps_model_to_score_fn(m, schedule) for m in (model_a, model_b, model_c)]
    learned_samples = run_fk(learned_score_fns, schedule, device)
    learned_metrics = compute_distribution_metrics(
        learned_samples, dist_gt, n_gt_samples=N_SAMPLES, seed=SEED,
    )
    _print_metrics("  ", learned_metrics)

    print("\n=== Sampling from trained models for overview plot ===")
    learned_marginals = {
        "A": sample_from_model(model_a, schedule, device),
        "B": sample_from_model(model_b, schedule, device),
        "C": sample_from_model(model_c, schedule, device),
    }

    fig_overview = plot_distributions_overview(dist_a, dist_b, dist_c, learned_marginals)
    overview_out = os.path.join(figures_dir, f"distributions_overview{FIG_EXT}")
    fig_overview.savefig(overview_out, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_overview)
    print(f"Saved {overview_out}")

    fig_summary = plot_comparison_summary(
        analytical_samples, learned_samples, dist_gt,
        analytical_metrics, learned_metrics,
    )
    summary_out = os.path.join(figures_dir, f"comparison_summary{FIG_EXT}")
    fig_summary.savefig(summary_out, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_summary)
    print(f"Saved {summary_out}")

    print("\nMETRICS SUMMARY")
    print(f"  Analytical scores: SW2={analytical_metrics['sw2']:.4f}  "
          f"MMD^2={analytical_metrics['mmd2']:.5f}")
    print(f"  Learned scores:    SW2={learned_metrics['sw2']:.4f}  "
          f"MMD^2={learned_metrics['mmd2']:.5f}")


def run_anisotropic_sweep(
    *,
    variances_a=DEFAULT_VARIANCES_A,
    variances_b=DEFAULT_VARIANCES_B,
    variances_c=DEFAULT_VARIANCES_C,
    training_sizes=(100, 1000, 10_000, "analytical"),
    particle_counts=("naive", 1, 4, 16, 64, 256),
    n_output: int = N_SAMPLES,
    n_runs: int = 5,
    train_iters: int = 20_000,
    separate_models: bool = False,
    save_plot: bool = True,
):
    """Sweep over (training_size, K) for the anisotropic experiment.

    training_size is either the string "analytical" (closed-form scores) or
    an integer giving the number of samples drawn once per distribution into
    a fixed pool that the score model trains on.

    If separate_models=False (default): a single ConditionalMLPScoreModel is
    trained on all three distributions. If True: three independent models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = VPSchedule(beta_min=0.1, beta_max=20.0)
    dist_a, dist_b, dist_c, dist_gt = build_distributions(
        variances_a, variances_b, variances_c,
    )

    mode_label = "separate" if separate_models else "conditional"
    figures_dir = figures_dir_for(variances_a, variances_b, variances_c, mode_label)

    print(f"=== Anisotropic {mode_label}-model sweep ===")
    print(f"  A: N(0, {_diag_text(dist_a.variances)}),  "
          f"B: N(0, {_diag_text(dist_b.variances)}),  "
          f"C: N(0, {_diag_text(dist_c.variances)})")
    print(f"  Target A*B/C: N(0, {_diag_text(dist_gt.variances)})")
    print(f"  training_sizes: {list(training_sizes)}")
    print(f"  particle_counts: {list(particle_counts)}")
    print(f"  n_runs: {n_runs}, train_iters: {train_iters}")
    print(f"  output dir: {figures_dir}")

    def _fixed_pool_data_fn(dist, n_pool):
        pool = torch.from_numpy(dist.sample(n_pool)).float()

        def fn(bs):
            idx = torch.randint(0, n_pool, (bs,))
            return pool[idx]

        return fn

    def train_and_get_score_fns(ts):
        if ts == "analytical":
            return [analytical_score_fn(d, schedule) for d in (dist_a, dist_b, dist_c)]
        ts_int = int(ts)
        dists = (dist_a, dist_b, dist_c)
        if separate_models:
            models = [
                train_score_model(
                    _fixed_pool_data_fn(d, ts_int),
                    schedule, D, device,
                    num_iterations=train_iters, verbose=False,
                )
                for d in dists
            ]
        else:
            data_fns = [_fixed_pool_data_fn(d, ts_int) for d in dists]
            cond_model = train_conditional_score_model(
                data_fns=data_fns,
                schedule=schedule, x_dim=D, device=device,
                num_iterations=train_iters, verbose=False,
            )
            models = [BoundConditionalModel(cond_model, idx) for idx in range(3)]
        return [eps_model_to_score_fn(m, schedule) for m in models]

    def ts_label(ts):
        return "analytical" if ts == "analytical" else f"N={ts}"

    def pc_label(pc):
        return "naive" if pc == "naive" else f"K={pc}"

    out_path = None
    out_tex_path = None
    if save_plot:
        os.makedirs(figures_dir, exist_ok=True)
        base = f"sweep_training_particles_{mode_label}_n{n_runs}"
        out_path = os.path.join(figures_dir, f"{base}{FIG_EXT}")
        out_tex_path = os.path.join(figures_dir, f"{base}.tex")

    return sweep_training_and_particles(
        train_and_get_score_fns,
        betas=[1.0, 1.0, -1.0],
        schedule=schedule,
        gt_dist=dist_gt,
        device=device,
        training_sizes=list(training_sizes),
        particle_counts=list(particle_counts),
        n_output=n_output,
        n_runs=n_runs,
        out_path=out_path,
        out_tex_path=out_tex_path,
        tex_caption_prefix=(
            f"Anisotropic FKC sweep ({mode_label} models, "
            f"A=N(0,{_diag_text(dist_a.variances)}), "
            f"B=N(0,{_diag_text(dist_b.variances)}), "
            f"C=N(0,{_diag_text(dist_c.variances)}), "
            f"target N(0,{_diag_text(dist_gt.variances)}))"
        ),
        grid_range=GRID_RANGE,
        ndim=D,
        seed=SEED,
        n_steps=500,
        plot_color="red",
        ts_label=ts_label,
        pc_label=pc_label,
        title=(
            f"Anisotropic FKC ({mode_label}): "
            f"A=N(0,{_diag_text(dist_a.variances)}), "
            f"B=N(0,{_diag_text(dist_b.variances)}), "
            f"C=N(0,{_diag_text(dist_c.variances)}) -- training samples x K"
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run the (training_size x particle_count) sweep instead of the main pipeline.",
    )
    parser.add_argument(
        "--sweep-runs", type=int, default=5,
        help="Number of independent runs per sweep cell.",
    )
    parser.add_argument(
        "--separate", action="store_true",
        help="Train three independent score models instead of one conditional model.",
    )
    args = parser.parse_args()

    if args.sweep:
        run_anisotropic_sweep(
            n_runs=args.sweep_runs,
            separate_models=args.separate,
        )
    else:
        main(mode="separate" if args.separate else "conditional")
