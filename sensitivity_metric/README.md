# Sensitivity Metric

Code for the rESS sensitivity metric. `gaussian/` evaluates the metric on 2D Gaussian compositions, where all densities are exact. `image/` estimates it for the room and CelebA latent diffusion models.

Run all commands from the repository root (the parent of this directory).


## Prerequisites

**`gaussian/`** has no prerequisites. It runs on its own. Its two figures use LaTeX (`pdflatex` and the `pgf` package) when available, and matplotlib's own math rendering otherwise.

**`image/`** cannot run until you have the following:

1. **Trained latent diffusion models**, one per setting (`FC_ID`, `FC_OOD`, `NFC_ID`, `NFC_OOD`).
   - Rooms: train them with `room_experiments/train_vae.py` and then `room_experiments/train_ldm.py`. See "Latent diffusion models" in `room_experiments/README.md`.
   - CelebA: train them with the same two scripts on the exported CelebA images. See "Latent diffusion models" in `celeba_experiments/README.md`.

   `estimate_ratios.py` takes the latent diffusion checkpoint directory (for example `/path/to/checkpoints/ldm_FC_ID/checkpoint-epoch50`). It finds the autoencoder through the path saved in that directory's `vae.json`, so the autoencoder checkpoint must still be at the path where it was trained.

2. **Held-out samples** that no model was trained on.
   - Rooms: a directory of held-out room images with one folder per class, like the training set. Create it with the dataset steps in `room_experiments/README.md` ("Creating the Dataset"), using a different `--seed` and a different output directory than the training set.
   - CelebA: one `.npz` file per setting, written by `image/prepare_celeba_heldout.py` (step 1 under "Image models" below). This needs only the CelebA dataset, which is downloaded automatically.


## Gaussian families

Run the sweep first. The plot scripts read its output, `gaussian/results/family_sweep.csv`.

```
python -m sensitivity_metric.gaussian.run_families --jobs 16
python -m sensitivity_metric.gaussian.plot_ranking     # within_family_ranking figure
python -m sensitivity_metric.gaussian.plot_bound       # bound_vs_amplification_by_family figure
python -m sensitivity_metric.gaussian.check_l2_ratio   # L2 to TV source-error factor
```

The figures are written to `gaussian/figures/` as PNG and PDF. The eight families are defined in `gaussian/families.py`, with names that match the figure labels.


## Image models

1. CelebA only: draw the held-out samples (1,024 per source) for each setting.

   ```
   python -m sensitivity_metric.image.prepare_celeba_heldout --pair FC --regime id \
     --out heldout/FC_ID.npz
   ```

2. Estimate the density ratios once per setting and log-SNR window (`trimmed` or `unconstrained`).

   ```
   python -m sensitivity_metric.image.estimate_ratios \
     --checkpoint /path/to/checkpoints/ldm_FC_ID/checkpoint-epoch50 \
     --room_images /path/to/heldout_room_images --window trimmed \
     --out ratios/room_FC_ID_trimmed.npz

   python -m sensitivity_metric.image.estimate_ratios \
     --checkpoint /path/to/checkpoints/ldm_celeba_FC_ID/checkpoint-epoch50 \
     --celeba_samples heldout/FC_ID.npz --window trimmed --batch 2048 \
     --out ratios/celeba_FC_ID_trimmed.npz
   ```

   (Each source has 1,024 held-out samples. `--batch` only sets how many (sample, draw) pairs are evaluated in one forward pass.)

3. Print the table for one model family and window, with one file per setting.

   ```
   python -m sensitivity_metric.image.summarize \
     FC_ID=ratios/room_FC_ID_trimmed.npz FC_OOD=ratios/room_FC_OOD_trimmed.npz \
     NFC_ID=ratios/room_NFC_ID_trimmed.npz NFC_OOD=ratios/room_NFC_OOD_trimmed.npz
   ```
