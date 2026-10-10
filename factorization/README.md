# Factorization Heuristic

These scripts compute the factorization heuristic tables for the rooms and for CelebA: the unsigned cosine between the two mean-effect vectors, `|cos(mu(A only) - mu(neither), mu(B only) - mu(neither))|`, in CLIP, DINOv2 and autoencoder feature spaces. Each mean is taken directly over the images of one class: for the rooms, the empty room, couch only, and second object only; for CelebA, the corresponding attributes.

Run all commands from the repository root (the parent of this directory).


## Prerequisites

**`rooms.py`** cannot run until you have:

- The room dataset, with one folder of images per class. Build it with "Creating the Dataset" in `room_experiments/README.md`.
- The room autoencoders for the `FC_ID` and `NFC_ID` settings, trained with `room_experiments/train_vae.py`. See "Latent diffusion models" in `room_experiments/README.md`.

**`celeba.py`** cannot run until you have:

- The CelebA autoencoders for the `FC_ID` and `NFC_ID` settings, trained with `room_experiments/train_vae.py`. See "Latent diffusion models" in `celeba_experiments/README.md`. The CelebA images are downloaded automatically.

Both scripts take the autoencoder's `ema_model` directory inside the checkpoint (for example `/path/to/checkpoints/vae_FC_ID/checkpoint-epoch50/ema_model`). CLIP and DINOv2 are downloaded automatically.


## Rooms

Prints one column per object pair (couch with a framed painting, couch with a coffee table):

```
python -m factorization.rooms --data_dir /path/to/dataset_out \
  --vae_fc /path/to/checkpoints/vae_FC_ID/checkpoint-epoch50/ema_model \
  --vae_nfc /path/to/checkpoints/vae_NFC_ID/checkpoint-epoch50/ema_model
```

With `--mixtures`, the means are instead the condition means of each training config in `room_experiments/train_configs` (mixtures of the class means), with one column per config.


## CelebA

Prints one column per attribute pair (`FC`, `NFC`):

```
python -m factorization.celeba \
  --vae_fc /path/to/checkpoints/vae_celeba_FC_ID/checkpoint-epoch50/ema_model \
  --vae_nfc /path/to/checkpoints/vae_celeba_NFC_ID/checkpoint-epoch50/ema_model
```

Each mean uses at most 20,000 randomly chosen images per attribute group by default for quick evaluation. To use a different amount, set `--per_stratum -n` where `n` is the desired number of images. Set `n` to `-1` to use all images.
