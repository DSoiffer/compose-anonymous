"""Stage-1 autoencoder training for latent diffusion.

Trains a KL-regularized autoencoder (diffusers ``AutoencoderKL``) on the
image dataset. This is the first of the two latent diffusion stages: once the
autoencoder is trained and frozen, ``train_ldm.py`` trains the diffusion model
on its latents instead of on raw pixels.

Loss = L1 reconstruction + LPIPS perceptual + (small) KL regularizer.

Training images are drawn from the same distribution that a
``train.py --conditions`` run uses (conditions as mixtures over real classes),
after a fixed 2% of the images is held out for reconstruction validation.

The checkpoint is saved with ``AutoencoderKL.save_pretrained`` so the diffusion
stage can call ``AutoencoderKL.from_pretrained(...)`` directly. A
``latent_stats.json`` (per-channel latent mean/std over the validation
images) is written alongside it.
"""

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import lpips
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Dataset/condition machinery and the [-1, 1] normalization (what the
# AutoencoderKL decoder and LPIPS both expect) are shared with the pixel-space
# trainer.
from room_experiments.train import (
    DATA_MEAN,
    DATA_STD,
    ClassSubsetDataset,
    condition_real_classes,
    default_transform,
    make_condition_dataloader,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Dataset root with one subdirectory per class.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write checkpoints into.",
    )
    parser.add_argument(
        "--conditions", type=str, required=True,
        help="train.py-style training-config YAML (conditions as mixtures over "
        "real classes). Training images are drawn from the same distribution "
        "that train.py --conditions uses.",
    )
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--compile", action="store_true",
        help="torch.compile the training forward (encode/sample/decode). The "
        "compiled wrapper is used only inside the train loop, so saved state "
        "dicts are unaffected.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=200)

    # Autoencoder shape. Four down blocks give 8x spatial downsampling (f=8),
    # i.e. 256x256x3 -> 32x32x{latent_channels}.
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument(
        "--down_blocks", type=int, default=4, choices=[3, 4],
        help="Encoder/decoder blocks: 4 gives f=8 (default), 3 gives f=4 (used "
        "for 64x64 images). train_ldm.py reads the factor from the saved config.",
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Training image size. Only recorded in the VAE config "
        "(sample_size). Images are not resized and must already be this size.",
    )

    # Loss weights.
    parser.add_argument("--lpips_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=1e-6)

    parser.add_argument(
        "--val_fraction", type=float, default=0.02,
        help="Fraction of images held out for reconstruction validation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class VAETrainWrapper(nn.Module):
    """Wraps the AutoencoderKL so the full encode->sample->decode path runs in a
    single ``forward``. This is what gets DDP-wrapped: under multi-GPU, calling
    ``vae.encode``/``vae.decode`` on the DistributedDataParallel object fails
    (only ``forward`` is exposed) and would also skip gradient sync."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample()
        recon = self.vae.decode(z).sample
        return recon, posterior.kl().mean()


@torch.no_grad()
def compute_latent_stats(vae_module, dataloader, device):
    """Per-channel mean/std of *sampled* posterior latents over one full pass of
    `dataloader`, used to standardize latents in the diffusion stage.

    Sampled, not posterior means: the diffusion stage encodes with
    ``latent_dist.sample()``, and by the law of total variance
    Var(sample) = Var(mean) + E[sigma^2]. A collapsed channel (mean ~const,
    sigma ~1) has near-zero mean spread but ~unit sample spread, so
    standardizing it by the mean spread would amplify pure noise."""
    vae_module.eval()
    n, mean, msq = 0, None, None
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        z = vae_module.encode(imgs).latent_dist.sample()  # (B, C, H, W)
        flat = z.permute(1, 0, 2, 3).reshape(z.shape[1], -1).float()
        if mean is None:
            mean = torch.zeros(z.shape[1], device=z.device)
            msq = torch.zeros(z.shape[1], device=z.device)
        cnt = flat.shape[1]
        mean += flat.sum(dim=1)
        msq += (flat ** 2).sum(dim=1)
        n += cnt
    mean = mean / n
    std = (msq / n - mean ** 2).clamp_min(1e-12).sqrt()
    return mean.cpu().tolist(), std.cpu().tolist()


def save_checkpoint(args, accelerator, vae_module, ema_vae, save_path, latent_stats,
                    conditions_list=None):
    save_path.mkdir(parents=True, exist_ok=True)
    unwrapped = vae_module
    mean, std = latent_stats
    # Store the latent normalization stats in the model config so any diffusers
    # consumer can read them straight from from_pretrained.
    unwrapped.register_to_config(latents_mean=mean, latents_std=std)
    unwrapped.save_pretrained(save_path)

    ema_path = save_path / "ema_model"
    ema_path.mkdir(exist_ok=True)
    ema_vae.store(unwrapped.parameters())
    ema_vae.copy_to(unwrapped.parameters())
    unwrapped.save_pretrained(ema_path)
    ema_vae.restore(unwrapped.parameters())

    with open(save_path / "normalize.json", "w") as f:
        json.dump({"mean": list(DATA_MEAN), "std": list(DATA_STD)}, f)

    with open(save_path / "latent_stats.json", "w") as f:
        json.dump({"latent_mean": mean, "latent_std": std}, f)

    with open(save_path / "conditions.yaml", "w") as f:
        yaml.safe_dump({"conditions": conditions_list}, f, sort_keys=False)

    accelerator.print(f"Saved checkpoint to {save_path}")

    all_ckpts = sorted(
        Path(args.output_dir).glob("checkpoint-epoch*"),
        key=lambda p: int(p.name.split("epoch")[1]),
    )
    for old in all_ckpts[:-2]:
        shutil.rmtree(old)
        accelerator.print(f"Deleted old checkpoint {old}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    accelerator = Accelerator(mixed_precision="bf16")

    with open(args.conditions) as f:
        conditions_list = yaml.safe_load(f)["conditions"]

    # Validation is a deterministic holdout from the flat image set. The train
    # loader then samples the remaining images according to the condition
    # mixtures (train.py's distribution).
    flat_classes = condition_real_classes(conditions_list)
    full_dataset = ClassSubsetDataset(args.data_dir, flat_classes, default_transform())
    n_val = max(1, int(len(full_dataset) * args.val_fraction))
    n_train = len(full_dataset) - n_val
    _, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_paths = {full_dataset.samples[i][0] for i in val_set.indices}
    train_loader, _ = make_condition_dataloader(
        args.data_dir, conditions_list, args.batch_size, args.num_workers,
        exclude_paths=val_paths,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    if accelerator.is_main_process:
        accelerator.print("Training distribution from conditions:")
        for cond in conditions_list:
            accelerator.print(f"  {cond['name']}: {dict(cond['classes'])}")
        accelerator.print(f"Train images: {n_train}  Val images: {n_val}")

    # KL autoencoder with the same topology as the Stable Diffusion VAE (f=8);
    # --down_blocks 3 drops the last block for f=4.
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * args.down_blocks,
        up_block_types=("UpDecoderBlock2D",) * args.down_blocks,
        block_out_channels=(128, 256, 512, 512)[:args.down_blocks],
        layers_per_block=2,
        latent_channels=args.latent_channels,
        sample_size=args.image_size,
    )
    ema_vae = EMAModel(vae.parameters(), decay=0.9999)
    model = VAETrainWrapper(vae)

    # LPIPS expects inputs in [-1, 1], which our images already are. It is a
    # fixed, frozen feature network used only to score reconstructions.
    lpips_fn = lpips.LPIPS(net="vgg")
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.num_epochs * len(train_loader),
    )

    prepared = accelerator.prepare(model, optimizer, train_loader, val_loader, lr_scheduler)
    model, optimizer, train_loader, val_loader, lr_scheduler = prepared
    # Compiled alias for the training forward only. `model` (same parameters)
    # stays the handle for EMA, validation and checkpointing.
    train_model = torch.compile(model) if args.compile else model
    lpips_fn = lpips_fn.to(accelerator.device)
    ema_vae.to(accelerator.device)

    global_step = 0
    window, wcount = defaultdict(float), defaultdict(int)

    def wadd(key, value):
        window[key] += value
        wcount[key] += 1

    for epoch in tqdm(range(args.num_epochs), disable=not accelerator.is_main_process):
        model.train()
        t0 = time.time()
        for imgs, _ in train_loader:
            recon, kl_loss = train_model(imgs)

            rec_loss = F.l1_loss(recon, imgs)
            lpips_loss = lpips_fn(recon, imgs).mean()
            loss = rec_loss + args.lpips_weight * lpips_loss + args.kl_weight * kl_loss

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            ema_vae.step(accelerator.unwrap_model(model).vae.parameters())

            wadd("rec", rec_loss.item())
            wadd("lpips", lpips_loss.item())
            wadd("kl", kl_loss.item())
            global_step += 1

            if accelerator.is_main_process and global_step % args.log_every == 0:
                avg = {k: window[k] / max(1, wcount[k]) for k in window}
                lr = lr_scheduler.get_last_lr()[0]
                accelerator.print(
                    f"  step {global_step}  rec={avg.get('rec', 0):.4f}  "
                    f"lpips={avg.get('lpips', 0):.4f}  kl={avg.get('kl', 0):.1f}  "
                    f"lr={lr:.2e}"
                )
                window.clear()
                wcount.clear()

        # Validation reconstruction quality (EMA weights).
        model.eval()
        vae_module = accelerator.unwrap_model(model).vae
        ema_vae.store(vae_module.parameters())
        ema_vae.copy_to(vae_module.parameters())
        val_rec, val_lpips, n_val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                z = vae_module.encode(imgs).latent_dist.mode()
                recon = vae_module.decode(z).sample
                val_rec += F.l1_loss(recon, imgs).item()
                val_lpips += lpips_fn(recon, imgs).mean().item()
                n_val_batches += 1
        ema_vae.restore(vae_module.parameters())

        if accelerator.is_main_process:
            accelerator.print(
                f"Epoch {epoch+1}/{args.num_epochs}  "
                f"val_rec={val_rec/max(1,n_val_batches):.4f}  "
                f"val_lpips={val_lpips/max(1,n_val_batches):.4f}  "
                f"time={time.time()-t0:.1f}s"
            )

        if accelerator.is_main_process and (epoch + 1) % args.checkpoint_interval == 0:
            vae_module = accelerator.unwrap_model(model).vae
            latent_stats = compute_latent_stats(vae_module, val_loader, accelerator.device)
            save_path = Path(args.output_dir) / f"checkpoint-epoch{epoch+1}"
            save_checkpoint(args, accelerator, vae_module, ema_vae, save_path,
                            latent_stats, conditions_list)

    accelerator.print("Stage-1 autoencoder training complete.")


if __name__ == "__main__":
    main()
