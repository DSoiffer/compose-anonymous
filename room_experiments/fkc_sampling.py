from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from safetensors.torch import load_file
from tqdm import trange


# Schedule

class DiffusersVPSchedule:
    """Closed-form continuous VP schedule matching a DDPMScheduler with
    rescale_betas_zero_snr=True and a linear beta schedule.

    The continuous limit of the discrete linear-beta DDPM (rectangle rule) is
        beta_orig(tau)     = T*beta_start + tau * T*(beta_end - beta_start)
        bar_alpha_orig(tau) = exp(-(T*beta_start*tau + (1/2)*T*(beta_end-beta_start)*tau^2))
    The zero-SNR rescaling acts on sqrt(bar_alpha) as an affine transform:
        sqrt(bar_alpha_new)(tau) = D * (sqrt(bar_alpha_orig)(tau) - C_T)
    with C_T = sqrt(bar_alpha_orig)(tau=1) and
    D = sqrt(bar_alpha_orig)(0) / (sqrt(bar_alpha_orig)(0) - C_T).

    Time convention: t in [0, 1] with t=0 noise and t=1 data; tau = 1 - t.
    Note: this convention is opposite from the paper.
    """

    def __init__(self, scheduler: DDPMScheduler, device: torch.device):
        cfg = scheduler.config
        if cfg.beta_schedule != "linear":
            raise NotImplementedError(
                f"Only beta_schedule='linear' is supported; got {cfg.beta_schedule!r}."
            )
        if not getattr(cfg, "rescale_betas_zero_snr", False):
            raise NotImplementedError(
                "This schedule wrapper requires rescale_betas_zero_snr=True."
            )
        if cfg.prediction_type != "v_prediction":
            raise NotImplementedError(
                f"Only prediction_type='v_prediction' is supported; got {cfg.prediction_type!r}."
            )

        self.T = int(cfg.num_train_timesteps)
        self.device = device
        self._b0 = float(self.T * cfg.beta_start)
        self._b1 = float(self.T * cfg.beta_end)
        sa0 = self._sqrt_bar_alpha_orig_scalar(0.0)
        sa1 = self._sqrt_bar_alpha_orig_scalar(1.0)
        self._C_T = sa1
        self._D = sa0 / (sa0 - sa1)

    def _log_bar_alpha_orig(self, tau: torch.Tensor) -> torch.Tensor:
        return -(self._b0 * tau + 0.5 * (self._b1 - self._b0) * tau * tau)

    def _sqrt_bar_alpha_orig(self, tau: torch.Tensor) -> torch.Tensor:
        return torch.exp(0.5 * self._log_bar_alpha_orig(tau))

    def _sqrt_bar_alpha_orig_scalar(self, tau: float) -> float:
        return float(np.exp(-0.5 * (self._b0 * tau + 0.5 * (self._b1 - self._b0) * tau * tau)))

    def _beta_orig(self, tau: torch.Tensor) -> torch.Tensor:
        return self._b0 + tau * (self._b1 - self._b0)

    def _bar_alpha(self, t: torch.Tensor) -> torch.Tensor:
        tau = 1.0 - t
        sa = self._D * (self._sqrt_bar_alpha_orig(tau) - self._C_T)
        return sa * sa

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        tau = 1.0 - t
        beta_orig = self._beta_orig(tau)
        sa = self._sqrt_bar_alpha_orig(tau)
        return beta_orig * sa / (sa - self._C_T)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self._bar_alpha(t).sqrt()

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - self._bar_alpha(t)).sqrt()

    def drift_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return self._beta(t) / 2.0

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return self._beta(t).sqrt()


# Resampling

def _batched_systematic_resample(log_weights: torch.Tensor) -> torch.Tensor:
    """Batched systematic resample. (N, K) log-weights -> (N, K) indices, with
    slot order shuffled per row so any single slot is an unbiased stratified
    draw."""
    N, K = log_weights.shape
    w = torch.softmax(log_weights, dim=1)
    bins = torch.cumsum(w, dim=1)
    u0 = torch.rand(N, 1, device=log_weights.device) / K
    u = u0 + torch.arange(K, device=log_weights.device, dtype=torch.float32) / K
    idx = torch.searchsorted(bins, u).clamp(max=K - 1)
    perm = torch.argsort(torch.rand(N, K, device=log_weights.device), dim=1)
    return idx.gather(1, perm)


# Discrete FKC core

def _fkc_disc_core(
    model_preds_fn,
    scheduler: DDPMScheduler,
    schedule: DiffusersVPSchedule,
    betas: list[float],
    *,
    image_shape: tuple[int, int, int],
    n_output: int,
    n_particles: int,
    n_steps: int,
    device: torch.device,
    fkc_max_step: int | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Integer-k DDPM ancestral with FKC weighting and every-step resampling.

    model_preds_fn(x_img, k_int) -> list[Tensor] returns one v-prediction per
    product component (in the same order as betas).

    fkc_max_step: if set, weight and resample only on the first `fkc_max_step`
    steps. The remaining steps run the plain reverse process, with the particles
    evolving independently and no reweighting. Because log_w is zeroed after
    every resample, the final pick is then a uniform draw from the particles
    that survived step `fkc_max_step - 1`. None (the default) weights at every
    step. This counts reverse-process steps, not the continuous time t.
    """
    if fkc_max_step is not None and fkc_max_step < 0:
        raise ValueError(f"fkc_max_step must be >= 0, got {fkc_max_step}")
    K, N = n_particles, n_output
    B = N * K
    C, H, W = image_shape
    ndim = C * H * W
    T = scheduler.config.num_train_timesteps
    beta_sum = float(sum(betas))

    scheduler.set_timesteps(n_steps, device=device)
    timesteps = scheduler.timesteps
    t_continuous = (1.0 - timesteps.to(torch.float32) / (T - 1)).clamp(1e-5, 1.0 - 1e-5).tolist()
    t_continuous.append(1.0 - 1e-5)

    def preds_to_scores_flat(
        preds: list[torch.Tensor], x_img: torch.Tensor, t_val: float
    ) -> list[torch.Tensor]:
        t_t = torch.full((B, 1), float(t_val), device=device)
        alpha_4d = schedule.alpha(t_t).view(B, 1, 1, 1)
        sigma_4d = schedule.sigma(t_t).view(B, 1, 1, 1)
        return [
            (-(sigma_4d * x_img + alpha_4d * p) / sigma_4d).reshape(B, -1)
            for p in preds
        ]

    x = torch.randn(B, C, H, W, device=device)
    log_w = torch.zeros(N, K, device=device)

    cached_preds: list[torch.Tensor] | None = None
    iterator = trange(n_steps) if verbose else range(n_steps)
    with torch.no_grad():
        for step in iterator:
            k_curr = int(timesteps[step].item())
            t_next = float(t_continuous[step + 1])
            dt = t_next - float(t_continuous[step])

            preds = cached_preds if cached_preds is not None else model_preds_fn(x, k_curr)

            pred_combined = torch.zeros_like(x)
            for b, p in zip(betas, preds):
                pred_combined = pred_combined + b * p
            x = scheduler.step(pred_combined, k_curr, x).prev_sample

            k_next = int(timesteps[step + 1].item()) if step < n_steps - 1 else 0
            preds_new = model_preds_fn(x, k_next)

            if fkc_max_step is not None and step >= fkc_max_step:
                # Gated off: no weighting and no resampling.
                cached_preds = preds_new
                continue

            scores_new = preds_to_scores_flat(preds_new, x, t_next)
            ws_new = sum(b * s for b, s in zip(betas, scores_new))

            t_n = torch.full((B, 1), t_next, device=device)
            sigma_t = schedule.diffusion(t_n)
            s2 = (sigma_t ** 2).squeeze(1)
            norm_ws_sq = (ws_new ** 2).sum(dim=1)
            weighted_norm_sq = sum(b * (s ** 2).sum(dim=1) for b, s in zip(betas, scores_new))
            div_f = ndim * schedule.drift_coeff(t_n).squeeze(1)
            g = (beta_sum - 1.0) * div_f + 0.5 * s2 * (norm_ws_sq - weighted_norm_sq)

            log_w = log_w + (g * dt).view(N, K)
            cached_preds = preds_new

            idx_local = _batched_systematic_resample(log_w)
            offset = (torch.arange(N, device=device) * K)[:, None]
            flat_idx = (idx_local + offset).view(-1)
            x = x[flat_idx].clone()
            log_w = torch.zeros_like(log_w)
            cached_preds = [c[flat_idx] for c in cached_preds]

    final_idx_local = _batched_systematic_resample(log_w)[:, 0]
    pick_global = final_idx_local + torch.arange(N, device=device) * K
    return x[pick_global]


def fkc_sample_disc(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    schedule: DiffusersVPSchedule,
    classes_for_product: list[str],
    betas: list[float],
    all_classes: list[str],
    *,
    image_shape: tuple[int, int, int],
    n_output: int,
    n_particles: int,
    n_steps: int,
    device: torch.device,
    fkc_max_step: int | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """FKC sampling with a single class-conditional UNet, one class label per
    product component."""
    if len(classes_for_product) != len(betas):
        raise ValueError(
            f"classes ({len(classes_for_product)}) and betas ({len(betas)}) must match"
        )
    unknown = [c for c in classes_for_product if c not in all_classes]
    if unknown:
        raise ValueError(f"Unknown classes {unknown}; checkpoint has {all_classes}")

    B = n_output * n_particles
    label_idx = [all_classes.index(c) for c in classes_for_product]
    labels_per_class = [
        torch.full((B,), li, device=device, dtype=torch.long) for li in label_idx
    ]

    def model_preds_fn(x_img: torch.Tensor, k_int: int) -> list[torch.Tensor]:
        k_in = torch.full((B,), int(k_int), device=device, dtype=torch.long)
        return [model(x_img, k_in, class_labels=lab).sample for lab in labels_per_class]

    return _fkc_disc_core(
        model_preds_fn,
        scheduler,
        schedule,
        betas,
        image_shape=image_shape,
        n_output=n_output,
        n_particles=n_particles,
        n_steps=n_steps,
        device=device,
        fkc_max_step=fkc_max_step,
        verbose=verbose,
    )


def fkc_sample_disc_multimodel(
    models: list[UNet2DModel],
    scheduler: DDPMScheduler,
    schedule: DiffusersVPSchedule,
    betas: list[float],
    *,
    image_shape: tuple[int, int, int],
    n_output: int,
    n_particles: int,
    n_steps: int,
    device: torch.device,
    fkc_max_step: int | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Multi-model FKC variant: one separate UNet per product component, each
    called with class label 0. All models must share the same noise schedule
    and prediction type."""
    if len(models) != len(betas):
        raise ValueError(f"Got {len(models)} models but {len(betas)} betas")
    B = n_output * n_particles
    zero_labels = torch.zeros((B,), device=device, dtype=torch.long)

    def model_preds_fn(x_img: torch.Tensor, k_int: int) -> list[torch.Tensor]:
        k_in = torch.full((B,), int(k_int), device=device, dtype=torch.long)
        return [m(x_img, k_in, class_labels=zero_labels).sample for m in models]

    return _fkc_disc_core(
        model_preds_fn,
        scheduler,
        schedule,
        betas,
        image_shape=image_shape,
        n_output=n_output,
        n_particles=n_particles,
        n_steps=n_steps,
        device=device,
        fkc_max_step=fkc_max_step,
        verbose=verbose,
    )


@torch.no_grad()
def ancestral_sample(
    predictors: list[tuple[torch.nn.Module, int]],
    scheduler: DDPMScheduler,
    betas: list[float],
    shape: tuple[int, ...],
    n_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Plain DDPM ancestral sampling of the beta-weighted v-prediction (naive
    composition, the K=1 case without FKC weights).

    predictors: one (model, class label) pair per product component.
    shape: (n, C, H, W) of the returned samples.
    """
    n = shape[0]
    scheduler.set_timesteps(n_steps, device=device)
    x = torch.randn(*shape, device=device)
    for k in scheduler.timesteps:
        k_int = int(k.item())
        k_in = torch.full((n,), k_int, device=device, dtype=torch.long)
        pred = torch.zeros_like(x)
        for b, (model, label) in zip(betas, predictors):
            labels = torch.full((n,), label, device=device, dtype=torch.long)
            pred = pred + b * model(x, k_in, class_labels=labels).sample
        x = scheduler.step(pred, k_int, x).prev_sample
    return x


# Pixel-space checkpoints written by train.py

def load_checkpoint(checkpoint_dir: str | Path, device: torch.device):
    """Load the UNet (EMA weights), scheduler, normalization stats, and condition
    names from a training checkpoint produced by train.py."""
    ckpt = Path(checkpoint_dir)
    weights = ckpt / "ema_model" / "diffusion_pytorch_model.safetensors"
    print(f"Loading weights from {weights}")
    state_dict = load_file(str(weights))
    with open(ckpt / "classes.json") as f:
        classes = json.load(f)["classes"]

    model = UNet2DModel(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        num_class_embeds=len(classes),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    scheduler = DDPMScheduler.from_pretrained(ckpt / "scheduler")
    with open(ckpt / "normalize.json") as f:
        norm = json.load(f)
    data_mean, data_std = norm["mean"], norm["std"]
    print(f"  classes={classes}")
    print(f"  normalize: mean={data_mean}, std={data_std}")

    return model, scheduler, classes, data_mean, data_std


def denormalize(image: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=image.device, dtype=image.dtype).view(3, 1, 1)
    s = torch.tensor(std, device=image.device, dtype=image.dtype).view(3, 1, 1)
    return (image * s + m).clamp(0, 1)


def scheduler_signature(scheduler):
    """Identifying tuple of a DDPMScheduler config. Used to verify that
    multimodel checkpoints share the same noise schedule; mismatch would
    invalidate the FKC weight."""
    cfg = scheduler.config
    return (
        cfg.prediction_type,
        bool(getattr(cfg, "rescale_betas_zero_snr", False)),
        int(cfg.num_train_timesteps),
        float(cfg.beta_start),
        float(cfg.beta_end),
        cfg.beta_schedule,
    )
