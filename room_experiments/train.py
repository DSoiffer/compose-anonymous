"""Class-conditional DDPM training for the furniture dataset.

Trains a v-prediction UNet with zero-terminal-SNR rescaling on condition
indices, where each condition is a mixture distribution over the underlying
real classes (see the conditions YAMLs in train_configs/).
"""

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel


DATA_MEAN = (0.5, 0.5, 0.5)
DATA_STD = (0.5, 0.5, 0.5)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Dataset root; must contain one subdirectory per real class "
        "referenced by --conditions.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write checkpoints into.",
    )
    parser.add_argument(
        "--conditions", type=str, required=True,
        help="Path to a YAML file defining training conditions (mixtures over real "
        "classes). The model is conditioned on the condition index.",
    )
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument(
        "--log_every", type=int, default=200,
        help="Print step-level loss every N steps.",
    )
    return parser.parse_args()


def default_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(list(DATA_MEAN), list(DATA_STD)),
    ])


class ClassSubsetDataset(Dataset):
    """Loads images from data_dir/{class_name}/ for each class in classes."""

    def __init__(self, data_dir, classes, transform=None):
        self.transform = transform if transform is not None else default_transform()
        self.samples = []
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Class directory not found: {cls_dir}")
            paths = [
                os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith(".png")
            ]
            if not paths:
                raise ValueError(f"No images found for class '{cls}' in {cls_dir}")
            self.samples.extend((p, idx) for p in paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def condition_real_classes(conditions):
    """Sorted union of the real classes referenced by a conditions list."""
    return sorted({c for cond in conditions for c in cond["classes"]})


class ConditionDataset(Dataset):
    """Each "condition" is a mixture distribution over the underlying real classes.
    __getitem__ ignores the index: at every call we draw a condition uniformly,
    pick a real class according to that condition's mixture, then return a random
    image from that class. The label returned is the condition index.
    exclude_paths removes specific images (e.g. a validation holdout) from the
    sampling pool."""

    def __init__(self, data_dir, conditions, transform=None, exclude_paths=None):
        self.transform = transform if transform is not None else default_transform()
        exclude = set(exclude_paths) if exclude_paths else ()

        real_classes = condition_real_classes(conditions)
        cls_to_idx = {c: i for i, c in enumerate(real_classes)}
        self.real_classes = real_classes

        self.paths_by_class = []
        for cls in real_classes:
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Class directory not found: {cls_dir}")
            paths = sorted(
                os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith(".png")
            )
            paths = [p for p in paths if p not in exclude]
            if not paths:
                raise ValueError(f"No images found for class '{cls}' in {cls_dir}")
            self.paths_by_class.append(paths)

        self.condition_names = [c["name"] for c in conditions]
        self.condition_choices = []
        for cond in conditions:
            idxs, weights = [], []
            for name, p in cond["classes"].items():
                if name not in cls_to_idx:
                    raise KeyError(
                        f"Condition '{cond['name']}' references unknown class '{name}'"
                    )
                if p > 0:
                    idxs.append(cls_to_idx[name])
                    weights.append(float(p))
            total = sum(weights)
            if not idxs or abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"Condition '{cond['name']}' probabilities sum to {total:.4f} (need 1.0)"
                )
            self.condition_choices.append((idxs, weights))

        self.total = sum(len(p) for p in self.paths_by_class)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        cond_idx = random.randrange(len(self.condition_names))
        idxs, weights = self.condition_choices[cond_idx]
        cls_idx = random.choices(idxs, weights=weights, k=1)[0]
        path = random.choice(self.paths_by_class[cls_idx])
        image = Image.open(path).convert("RGB")
        return self.transform(image), cond_idx


def _worker_init_fn(worker_id):
    # Fold in the distributed rank: torch.initial_seed() is identical across
    # ranks when the trainer seeds torch globally, and ConditionDataset draws
    # samples from `random` (ignoring the index), so without this every rank
    # would see the same batches.
    rank = int(os.environ.get("RANK", 0))
    seed = (torch.initial_seed() + worker_id + 100_003 * rank) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_condition_dataloader(data_dir, conditions, batch_size, num_workers,
                              **dataset_kwargs):
    dataset = ConditionDataset(data_dir, conditions, transform=default_transform(),
                               **dataset_kwargs)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader, dataset


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")
    batch_size = args.batch_size

    with open(args.conditions) as f:
        conditions_list = yaml.safe_load(f)["conditions"]

    train_dataloader, cond_dataset = make_condition_dataloader(
        args.data_dir, conditions_list, batch_size, args.num_workers,
    )
    label_names = [c["name"] for c in conditions_list]
    num_label_classes = len(label_names)
    if accelerator.is_main_process:
        accelerator.print("Conditions:")
        for cond in conditions_list:
            accelerator.print(f"  {cond['name']}: {dict(cond['classes'])}")
        for cls, paths in zip(cond_dataset.real_classes, cond_dataset.paths_by_class):
            accelerator.print(f"  underlying {cls}: {len(paths)} images")
        accelerator.print(
            f"Total: {len(cond_dataset)} images, {num_label_classes} conditions"
        )

    model = UNet2DModel(
        sample_size=256, in_channels=3, out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        num_class_embeds=num_label_classes,
    )

    ema_model = EMAModel(model.parameters(), decay=0.9999)

    # v-prediction + zero terminal SNR. Zero-SNR makes
    # alpha_bar[T]=0 exactly, which is well-defined for v-prediction.
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        rescale_betas_zero_snr=True,
        prediction_type="v_prediction",
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

    # Train
    global_step = 0
    window_loss = 0.0
    for epoch in tqdm(range(args.num_epochs), disable=not accelerator.is_main_process):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_dataloader:
            clean_images, class_labels = batch
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=clean_images.device, dtype=torch.long,
            )
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            model_pred = model(noisy_images, timesteps, class_labels=class_labels).sample
            target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
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
                f"lr={lr_scheduler.get_last_lr()[0]:.2e}  "
                f"time={time.time()-t0:.1f}s"
            )

        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % args.checkpoint_interval == 0:
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

            with open(save_path / "conditions.yaml", "w") as f:
                yaml.safe_dump({"conditions": conditions_list}, f, sort_keys=False)

            accelerator.print(f"Saved checkpoint to {save_path}")

            # Keep only the 2 most recent checkpoints
            all_ckpts = sorted(
                Path(args.output_dir).glob("checkpoint-epoch*"),
                key=lambda p: int(p.name.split("epoch")[1]),
            )
            for old in all_ckpts[:-2]:
                shutil.rmtree(old)
                accelerator.print(f"Deleted old checkpoint {old}")

    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
