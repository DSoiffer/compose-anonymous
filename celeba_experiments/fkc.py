"""Naive composition and Feynman-Kac corrected (FKC) sampling for the CelebA DDPM.

The target is P(x) proportional to PA(x) PB(x) / P0(x), i.e. composed scores
with weights (-1, +1, +1) on conditions (0, A, B). Naive sampling runs the
reverse process with the composed epsilon prediction. FKC additionally keeps
K weighted particles per output, adds the product-potential increment to the
log weights at every step, and resamples systematically at every step.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from celeba_experiments.diffusion import GaussianDiffusion, _extract

# Composition weights on the conditions (0, A, B).
CONDITIONS = (0, 1, 2)
WEIGHTS = (-1.0, 1.0, 1.0)


def weighted_product_potential(
    scores: Sequence[torch.Tensor],
    weights: Sequence[float],
    sigma_squared_dt: float | torch.Tensor,
) -> torch.Tensor:
    """FKC log-weight increment for composition weights that sum to one."""
    composed = sum(weight * score for weight, score in zip(weights, scores))
    composed_norm = composed.square().flatten(1).sum(1)
    expert_norm = sum(
        weight * score.square().flatten(1).sum(1)
        for weight, score in zip(weights, scores)
    )
    return 0.5 * sigma_squared_dt * (composed_norm - expert_norm)


def _systematic_resample(
    particles: torch.Tensor, log_weights: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    batch, count = log_weights.shape
    probabilities = torch.softmax(log_weights, dim=1)
    cumulative = probabilities.cumsum(1)
    offsets = (
        torch.rand((batch, 1), generator=generator, device=particles.device) / count
    )
    positions = offsets + torch.arange(count, device=particles.device)[None] / count
    ancestors = torch.searchsorted(
        cumulative.contiguous(), positions.contiguous()
    ).clamp(max=count - 1)
    gather = ancestors[:, :, None, None, None].expand_as(particles)
    return torch.gather(particles, 1, gather)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    result = torch.Generator(device=device)
    result.manual_seed(seed)
    return result


@torch.inference_mode()
def sample_composition_batch(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    *,
    batch_size: int,
    image_size: int,
    method: str,
    particles: int,
    inference_steps: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """One batch of composed samples in [-1, 1]; method is "naive" or "fkc"."""
    if method == "fkc" and particles < 2:
        raise ValueError("FKC requires at least two particles")
    count = particles if method == "fkc" else 1
    root = torch.randn(
        (batch_size, 3, image_size, image_size),
        generator=_generator(device, seed * 3_000_017 + 17),
        device=device,
    )
    state = root[:, None].repeat(1, count, 1, 1, 1)
    log_weights = torch.zeros((batch_size, count), device=device)
    schedule = diffusion.inference_schedule(inference_steps)
    for index, timestep in enumerate(schedule):
        previous_timestep = schedule[index + 1] if index + 1 < len(schedule) else -1
        flat = state.flatten(0, 1)
        times = torch.full((len(flat),), timestep, dtype=torch.long, device=device)
        expert_x = flat.repeat(len(CONDITIONS), 1, 1, 1)
        expert_times = times.repeat(len(CONDITIONS))
        expert_conditions = torch.cat(
            [
                torch.full((len(flat),), condition, dtype=torch.long, device=device)
                for condition in CONDITIONS
            ]
        )
        expert_eps = model(expert_x, expert_times, expert_conditions).reshape(
            len(CONDITIONS), len(flat), *flat.shape[1:]
        )
        eps = list(expert_eps)
        composed_eps = sum(weight * value for weight, value in zip(WEIGHTS, eps))
        mean, variance = diffusion.p_mean_variance_to(
            flat, times, previous_timestep, composed_eps
        )
        noise = torch.stack(
            [
                torch.randn(
                    (batch_size, 3, image_size, image_size),
                    generator=_generator(
                        device, seed * 1_000_003 + timestep * 10_007 + slot
                    ),
                    device=device,
                )
                for slot in range(count)
            ],
            dim=1,
        ).flatten(0, 1)
        state = (
            mean + (previous_timestep >= 0) * variance.sqrt() * noise
        ).reshape_as(state)
        if method == "naive":
            continue
        score_scale = _extract(diffusion.sqrt_one_minus_alpha_bars, times, flat.shape)
        scores = [-value / score_scale for value in eps]
        increment = weighted_product_potential(
            scores,
            WEIGHTS,
            diffusion.effective_beta(timestep, previous_timestep),
        ).reshape(batch_size, count)
        increment = torch.nan_to_num(increment, nan=0.0, posinf=80.0, neginf=-80.0)
        log_weights += increment
        log_weights -= log_weights.max(1, keepdim=True).values
        state = _systematic_resample(
            state,
            log_weights,
            _generator(device, seed * 2_000_003 + timestep * 20_011),
        )
        log_weights.zero_()
    if method == "naive":
        return state[:, 0]
    probabilities = torch.softmax(log_weights, dim=1)
    selected = torch.multinomial(
        probabilities, 1, generator=_generator(device, seed * 4_000_037 + 29)
    )
    gather = selected[:, :, None, None, None].expand(
        batch_size, 1, 3, image_size, image_size
    )
    return torch.gather(state, 1, gather)[:, 0]
