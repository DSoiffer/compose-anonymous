# GMM and 2D Gaussian Experiments

Two experiments that compose simple distributions via Feynman-Kac correctors
(FKC), corresponding to the 2D Gaussian and Gaussian mixture experiments in the paper.

- The **Gaussian mixture (GMM)** experiment composes three conditions
  (`a1`, `a2`, `base`) built from a shared 2-D isotropic GMM, with target `a1 * a2 / base`. (The target is not itself a GMM in general, so ground-truth samples come from either rejection sampling or importance sampling on the exact ratio density.)
- The **2D Gaussian** experiment composes three diagonal Gaussians (`a1`,
  `a2`, `base`) with different covariance structures, with target `a1 * a2 / base`.(The analytical target is a Gaussian, so ground-truth samples come from closed-form sampling.)


## Running
Before running, ensure you have installed all dependencies with `uv sync` from the root directory. `uv` can be installed [here](https://docs.astral.sh/uv/getting-started/installation/).

Both experiments can be run in one of two modes. Single-shot mode trains and runs a model a single time, producing plots using samples from its learned distributions compared against the ground truth. Sweep mode performs a sweep over different training sizes and number of FKC particles, writing the mean and std over runs as a LaTeX file (corresponding to the tables in the paper), plus a plot of the same results. The `--sweep` argument enables sweep mode. All results are stored under `gaussian_experiments/figures` (created if it does not exist).

To run the experiments, run either of these from the repository root (the parent of this directory):

`python -m gaussian_experiments.run_gaussian_2d_experiment --config gaussian_experiments/configs/gaussian_2d_fc_ood.yaml`

or

`python -m gaussian_experiments.run_gmm_experiment --config gaussian_experiments/configs/gmm_id.yaml`


### Configs

Each experiment reads its parameters from a YAML config:

- `configs/gaussian_2d_fc_ood.yaml`: diagonal-covariance variances for `a1`,
  `a2`, `base`, plus additional hyperparameters (described in the config file). This is the factorized conditionals, out-of-distribution (FC + OOD) setting. The other three settings in the paper only change the base variances:

  | Setting | Config | Base variances |
  |---|---|---|
  | FC + ID | `configs/gaussian_2d_fc_id.yaml` | `[10, 10]` |
  | NFC + ID | `configs/gaussian_2d_nfc_id.yaml` | `[20, 20]` |
  | FC + OOD | `configs/gaussian_2d_fc_ood.yaml` | `[1, 1]` |
  | NFC + OOD | `configs/gaussian_2d_nfc_ood.yaml` | `[1.1, 1.1]` |
- `configs/gmm_id.yaml`: component means, condition weights, component std, plus additional hyperparameters. This is the in-distribution case. `configs/gmm_ood.yaml` is the out-of-distribution case.

To override either config, copy the YAML, edit, and pass it as `--config`.

Within the config, the `single_shot:` block holds parameters for the per-experiment training + plotting pipeline, the recommended use is for producing plots. To alter settings for the sweep over different training sizes and numbers of particles, change parameters under the `sweep:` block.

To reproduce the separate-model tables, set `separate_models: true` under the `sweep:` block. This trains one model per condition instead of one conditional model.

Further config details are given in the config files.

Note that the results presented in the paper use a value of 15.0 for a `g_clip` parameter that we implement in FKC sampling, which clips the magnitude of the updates to the incremental `g` correction weight term. In practice, this tends to lower the score estimation error in OOD settings for the learned models by cutting off erroneously high weights. However, this also prevents the analytical score sampling's error from reducing as quickly as it should (using pure vanilla FKC) at high values of K. To disable this feature, set the default value to `None` in `feynman_kac.py`'s FKC sampling.


## Out-of-distribution support table

`quantify_ood.py` computes the fraction of target samples that fall outside the 95% highest-density region of each source distribution, and outside all three at once, using the ground truth distributions. Run it from the repository root:

```
python -m gaussian_experiments.quantify_ood
```

It prints a Markdown table with one row per 2D Gaussian and GMM setting.
