# Anisotropic Gaussian FKC Experiment

Composing diagonal-Gaussian distributions with different covariance structures
via Feynman-Kac correctors (FKC).

By default the three input distributions are
```
A = N(0, diag(10, 1))     stretched along x
B = N(0, diag(1, 10))     stretched along y
C = N(0, diag(1,  1))     isotropic baseline
```
and the analytical target is `A * B / C = N(0, diag(10, 10))`.

## Files

- `gmm_lib.py` -- `DiagonalGaussian` and analytical product/ratio.
- `noise_schedule.py` -- VP noise schedule (PyTorch).
- `score_model.py` -- MLP eps-prediction models and training loops
  (independent, conditional, and a binding wrapper).
- `feynman_kac.py` -- FKC sampler 
- `evaluation.py` -- Sliced-W2 and MMD^2 metrics.
- `sweep.py` -- generic `(training_size x particle_count)` sweep that drives
  the FKC sampler, aggregates metrics, and writes a grid plot plus a LaTeX table.
- `run_anisotropic_experiment.py` -- entry point. Builds the anisotropic
  distributions and runs either the single-shot pipeline (with overview and
  comparison plots) or the full sweep.

## Setup

Dependencies are managed with `uv`. Run `uv sync` to install all dependencies.

A working CUDA install is optional, the code falls back to CPU.


### Single-shot pipeline (overview and comparison plots)

```
python run_anisotropic_experiment.py
```
This trains a single conditional score model on A, B, C, runs FKC twice (once
with closed-form analytical scores, once with the learned scores), and writes
two figures into `figures/anisotropic_conditional_<variances>/`:

- `distributions_overview.pdf` -- the three input distributions on the top
  row and samples from the trained models on the bottom row, with MMD^2 vs
  the corresponding ground truth overlaid.
- `comparison_summary.pdf` -- ground truth vs FKC with analytical scores vs
  FKC with learned scores.

To train three independent models instead of one conditional model:
```
python run_anisotropic_experiment.py --separate
```

### Sweep over training size and particle count

```
python run_anisotropic_experiment.py --sweep
```
This sweeps over a grid of `training_size` (number of training samples drawn
once into a fixed pool, plus an `"analytical"` row that uses closed-form
scores) and `particle_count` K (per-swarm SMC ensemble size, plus a `"naive"`
column that drops FKC and just adds the scores). Each cell runs `--sweep-runs`
independent trials (default 5) and aggregates Sliced-W2 and MMD^2 to mean and
std.

Outputs in `figures/anisotropic_conditional_<variances>/`:

- `sweep_training_particles_conditional_n<runs>.pdf` -- grid of scatter plots
  with metrics annotated per cell.
- `sweep_training_particles_conditional_n<runs>.tex` -- LaTeX tables for both
  metrics.

Add `--separate` to train three independent models per cell instead of a
single conditional model.

### Customizing the experiment

The default training sizes, particle counts, run count, training iterations,
and variance choices for A, B, C are all keyword arguments of the `main()`
and `run_anisotropic_sweep()` functions in `run_anisotropic_experiment.py`.
Edit the `__main__` block at the bottom of the file (or import the functions
from a notebook) to override them.
