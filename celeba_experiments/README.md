# CelebA Experiments

Code for the CelebA experiments: pixel-space and latent diffusion models of three conditions, naive and FKC composition, and FaceXFormer evaluation.

Run all commands from the repository root (the parent of this directory).


## Prerequisites

- Install the dependencies with `uv sync` (see the root `README.md`).
- Nothing else needs to run first. The CelebA images (Hugging Face dataset `flwrlabs/celeba`, subject to the CelebA dataset agreement) and the FaceXFormer checkpoint are downloaded automatically on first use.

Within this directory, each step needs the output of the step before it:

| Step | Needs |
|---|---|
| Train a pixel-space model (`train.py`) | Nothing |
| Sample (`sample.py`) | A trained pixel-space model |
| Export the strata (`export_strata.py`) | Nothing |
| Train the autoencoder (`room_experiments/train_vae.py`) | The exported strata |
| Train the latent diffusion model (`room_experiments/train_ldm.py`) | The exported strata and the trained autoencoder |
| Sample (`sample_ldm.py`) | A trained latent diffusion model |
| Evaluate (`evaluate.py`) | Sample files from `sample.py` or `sample_ldm.py` |


## Settings

For an attribute pair (A, B), every image falls in one of four strata, ordered (q00, q10, q01, q11) = (neither, A only, B only, both). Each condition is a mixture over the strata:

| Regime | P0 | PA | PB |
|---|---|---|---|
| `id` | (0.99, 1/300, 1/300, 1/300) | (0, 0.5, 0, 0.5) | (0, 0, 0.5, 0.5) |
| `ood` | (0.99, 1/300, 1/300, 1/300) | (0, 1, 0, 0) | (0, 0, 1, 0) |

The pairs are `FC` = Bangs + Mouth_Slightly_Open and `NFC` = Brown_Hair + Wavy_Hair. The composition is PA * PB / P0. Models train on 90% of the identities, the other 10% are held out for the sensitivity metric.


## Pixel-space models

1. Train one model per setting. Repeat with `--pair FC|NFC` and `--regime id|ood`.

   ```
   python -m celeba_experiments.train --pair FC --regime id --output models/FC_ID.pt
   ```

2. Sample 1,000 compositions with naive composition and with FKC (16 particles).

   ```
   python -m celeba_experiments.sample --checkpoint models/FC_ID.pt --method naive \
     --output samples/FC_ID_naive.pt
   python -m celeba_experiments.sample --checkpoint models/FC_ID.pt --method fkc --particles 16 \
     --output samples/FC_ID_fkc16.pt
   ```

3. Evaluate. This prints the joint success rate (both attributes present) and the two marginal rates for each file.

   ```
   python -m celeba_experiments.evaluate --pair FC samples/FC_ID_naive.pt samples/FC_ID_fkc16.pt
   ```


## Latent diffusion models

1. Export the training images as one folder per stratum, which the latent diffusion trainers read.

   ```
   python -m celeba_experiments.export_strata --out /path/to/celeba_strata
   ```

2. Train the autoencoder and then the latent diffusion model for each setting (here FC + ID). `ldm_configs/` holds the stratum mixtures of the four settings.

   ```
   accelerate launch --mixed_precision bf16 --num_processes 1 -m room_experiments.train_vae \
     --data_dir /path/to/celeba_strata/FC \
     --output_dir /path/to/checkpoints/vae_celeba_FC_ID \
     --conditions celeba_experiments/ldm_configs/FC_ID.yaml \
     --num_epochs 50 --checkpoint_interval 10 --batch_size 16 --compile \
     --latent_channels 3 --down_blocks 3 --image_size 64

   accelerate launch --mixed_precision bf16 --num_processes 1 -m room_experiments.train_ldm \
     --data_dir /path/to/celeba_strata/FC \
     --vae /path/to/checkpoints/vae_celeba_FC_ID/checkpoint-epoch50 \
     --output_dir /path/to/checkpoints/ldm_celeba_FC_ID \
     --conditions celeba_experiments/ldm_configs/FC_ID.yaml \
     --image_size 64 --num_epochs 50 --checkpoint_interval 10 --batch_size 16
   ```

3. Sample, then evaluate the samples with `evaluate.py` as above.

   ```
   python -m celeba_experiments.sample_ldm \
     --checkpoint /path/to/checkpoints/ldm_celeba_FC_ID/checkpoint-epoch50 \
     --method fkc --particles 16 --output samples/ldm_FC_ID_fkc16.pt
   ```


## FaceXFormer

`evaluate.py` uses FaceXFormer ([Narayan et al., 2024](https://arxiv.org/abs/2403.12960)). Its network code is copied into `facexformer/` from the [official repository](https://github.com/Kartik-3004/facexformer) under its MIT License. `facexformer/README.md` lists the changes and the citation.
