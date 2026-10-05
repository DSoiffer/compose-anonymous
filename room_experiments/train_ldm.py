"""Latent diffusion (stage-2) training.

Trains the same class-conditional, v-prediction + zero-terminal-SNR UNet as
train.py, but in the latent space of a *frozen* autoencoder trained with
train_vae.py. Images are encoded to latents on the fly, so the diffusion model
never sees pixels.

Differences from train.py:
  * A frozen VAE (its EMA weights) encodes each batch to latents, which are
    standardized per channel. The mean and std are computed once at startup
    from *sampled* posteriors on the training data, which are the statistics of
    what the UNet actually sees.
  * The UNet operates on latent_channels x (size/f) x (size/f) tensors.
  * The UNet uses 3 down/up blocks instead of 4. The VAE already did the
    spatial compression, so on a 32x32 latent a 3-block UNet (8x8 bottleneck)
    is the right depth.

Everything else (v_prediction, rescale_betas_zero_snr, EMA, cosine LR,
conditions handling, checkpointing) matches train.py.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import broadcast
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from tqdm import tqdm

# Reuse the dataset / condition machinery from the pixel-space trainer.
from room_experiments.train import DATA_MEAN, DATA_STD, make_condition_dataloader
# Shared sampled-posterior latent statistics (see its docstring for why the
# stats must come from .sample(), not the posterior means).
from room_experiments.train_vae import compute_latent_stats


# Latent UNet: on a 32x32 latent the bottleneck is 8x8.
UNET_ARCH = dict(
    block_out_channels=(128, 256, 512),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--vae", type=str, required=True,
        help="Frozen autoencoder: a train_vae.py checkpoint dir (its ema_model/ "
        "subdirectory is used).",
    )
    parser.add_argument(
        "--conditions", type=str, required=True,
        help="YAML defining training conditions (mixtures over real classes). "
        "The model is conditioned on the condition index.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--checkpoint_interval", type=int, default=10,
        help="Save every N epochs. The final epoch is always saved.",
    )
    parser.add_argument("--log_every", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")

    with open(args.conditions) as f:
        conditions_list = yaml.safe_load(f)["conditions"]

    train_dataloader, cond_dataset = make_condition_dataloader(
        args.data_dir, conditions_list, args.batch_size, args.num_workers,
    )
    label_names = [c["name"] for c in conditions_list]
    num_label_classes = len(label_names)
    if accelerator.is_main_process:
        accelerator.print("Conditions:")
        for cond in conditions_list:
            accelerator.print(f"  {cond['name']}: {dict(cond['classes'])}")
        accelerator.print(
            f"Total: {len(cond_dataset)} images, {num_label_classes} conditions"
        )

    # Frozen autoencoder (its EMA weights), kept in fp32. Only the UNet runs
    # under bf16 autocast.
    vae_load_path = str(Path(args.vae) / "ema_model")
    vae = AutoencoderKL.from_pretrained(vae_load_path)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device)

    latent_channels = vae.config.latent_channels
    downsample = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_size = args.image_size // downsample

    # Standardization stats from sampled posteriors over one full pass of the
    # training loader (as many draws from the condition mixtures as there are
    # training images), computed on the main process and broadcast so every
    # rank standardizes identically.
    stats = torch.zeros(2, latent_channels, device=accelerator.device)
    if accelerator.is_main_process:
        m, s = compute_latent_stats(vae, train_dataloader, accelerator.device)
        stats[0] = torch.tensor(m, device=accelerator.device)
        stats[1] = torch.tensor(s, device=accelerator.device)
    stats = broadcast(stats, from_process=0)
    latent_mean = stats[0].view(1, -1, 1, 1)
    latent_std = stats[1].view(1, -1, 1, 1)
    norm_source = "sampled posteriors, recomputed at train start"
    if accelerator.is_main_process:
        accelerator.print(
            f"VAE: {vae_load_path}  latent={latent_channels}x{latent_size}x{latent_size} "
            f"(f={downsample})"
        )
        accelerator.print(f"  latent_mean={latent_mean.flatten().tolist()}")
        accelerator.print(f"  latent_std ={latent_std.flatten().tolist()}")

    model = UNet2DModel(
        sample_size=latent_size, in_channels=latent_channels, out_channels=latent_channels,
        layers_per_block=2,
        num_class_embeds=num_label_classes,
        **UNET_ARCH,
    )

    ema_model = EMAModel(model.parameters(), decay=0.9999)

    # clip_sample only affects the reverse step() at sampling time. The
    # diffusers default (clip_sample=True, range=1.0) clamps the predicted x0 to
    # [-1, 1], which is right for pixels but wrong for unit-variance
    # standardized latents, so it is disabled in the saved scheduler.
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        rescale_betas_zero_snr=True,
        prediction_type="v_prediction",
        clip_sample=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_update_steps_per_epoch = len(train_dataloader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.num_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler,
    )
    ema_model.to(accelerator.device)

    global_step = 0
    window_loss = 0.0
    for epoch in tqdm(range(args.num_epochs), disable=not accelerator.is_main_process):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_dataloader:
            clean_images, class_labels = batch
            with torch.no_grad():
                z = vae.encode(clean_images).latent_dist.sample()
            latents = (z - latent_mean) / latent_std

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device, dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = model(noisy_latents, timesteps, class_labels=class_labels).sample
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
            loss = F.mse_loss(model_pred, target)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            ema_model.step(model.parameters())

            epoch_loss += loss.item()
            window_loss += loss.item()
            global_step += 1

            if accelerator.is_main_process and global_step % args.log_every == 0:
                accelerator.print(
                    f"  step {global_step}  loss={window_loss / args.log_every:.6f}  "
                    f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                )
                window_loss = 0.0

        if accelerator.is_main_process:
            avg_loss = epoch_loss / len(train_dataloader)
            accelerator.print(
                f"Epoch {epoch+1}/{args.num_epochs}  loss={avg_loss:.6f}  "
                f"lr={lr_scheduler.get_last_lr()[0]:.2e}  time={time.time()-t0:.1f}s"
            )

        is_last_epoch = epoch + 1 == args.num_epochs
        if accelerator.is_main_process and (
            (epoch + 1) % args.checkpoint_interval == 0 or is_last_epoch
        ):
            save_path = Path(args.output_dir) / f"checkpoint-epoch{epoch+1}"
            save_path.mkdir(parents=True, exist_ok=True)

            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(save_path)

            ema_path = save_path / "ema_model"
            ema_path.mkdir(exist_ok=True)
            ema_model.store(unwrapped.parameters())
            ema_model.copy_to(unwrapped.parameters())
            unwrapped.save_pretrained(ema_path)
            ema_model.restore(unwrapped.parameters())

            accelerator.save_state(str(save_path / "training_state"))
            noise_scheduler.save_pretrained(save_path / "scheduler")

            with open(save_path / "normalize.json", "w") as f:
                json.dump({"mean": list(DATA_MEAN), "std": list(DATA_STD)}, f)

            with open(save_path / "classes.json", "w") as f:
                json.dump({"classes": label_names}, f)

            # Record the VAE and the exact latent normalization used, so
            # sampling can encode, decode and de-standardize consistently.
            # "vae" is the resolved load path (the ema_model subdir).
            with open(save_path / "vae.json", "w") as f:
                json.dump({
                    "vae": vae_load_path,
                    "vae_arg": args.vae,
                    "unet_arch": "3block",
                    "latent_channels": latent_channels,
                    "latent_size": latent_size,
                    "downsample_factor": downsample,
                    "latent_mean": latent_mean.flatten().tolist(),
                    "latent_std": latent_std.flatten().tolist(),
                    "latent_stats_source": norm_source,
                }, f)

            with open(save_path / "conditions.yaml", "w") as f:
                yaml.safe_dump({"conditions": conditions_list}, f, sort_keys=False)

            accelerator.print(f"Saved checkpoint to {save_path}")

            # Keep only the newest checkpoint.
            all_ckpts = sorted(
                Path(args.output_dir).glob("checkpoint-epoch*"),
                key=lambda p: int(p.name.split("epoch")[1]),
            )
            for old in all_ckpts[:-1]:
                shutil.rmtree(old)
                accelerator.print(f"Deleted old checkpoint {old}")

    accelerator.print("Latent-diffusion training complete.")


if __name__ == "__main__":
    main()
