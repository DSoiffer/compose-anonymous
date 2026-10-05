"""Linear-beta DDPM forward process and respaced reverse transitions (epsilon prediction)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


def _extract(
    values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size
) -> torch.Tensor:
    result = values.to(timesteps.device)[timesteps]
    return result.reshape((len(timesteps),) + (1,) * (len(shape) - 1))


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class GaussianDiffusion:
    def __init__(self, config: DiffusionConfig) -> None:
        self.config = config
        self.betas = torch.linspace(
            config.beta_start, config.beta_end, config.timesteps
        )
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_previous = torch.cat((torch.ones(1), self.alpha_bars[:-1]))
        self.sqrt_alpha_bars = self.alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1 - self.alpha_bars).sqrt()

    def q_sample(
        self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            _extract(self.sqrt_alpha_bars, timesteps, clean.shape) * clean
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, clean.shape) * noise
        )

    def inference_schedule(self, steps: int) -> list[int]:
        """Evenly respaced training timesteps in descending inference order."""
        if not 1 <= steps <= self.config.timesteps:
            raise ValueError(
                f"inference steps must be in [1, {self.config.timesteps}], got {steps}"
            )
        ascending = torch.linspace(
            0,
            self.config.timesteps - 1,
            steps,
            dtype=torch.float64,
        ).round().to(torch.long)
        if len(torch.unique(ascending)) != steps:
            raise RuntimeError("respaced inference schedule contains duplicates")
        return ascending.flip(0).tolist()

    def p_mean_variance_to(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        previous_timestep: int,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior transition from a training timestep to an earlier retained one."""
        if previous_timestep >= int(timesteps.min()):
            raise ValueError("previous_timestep must be earlier than timesteps")
        alpha_bar = _extract(self.alpha_bars, timesteps, noisy.shape)
        if previous_timestep < 0:
            alpha_bar_previous = torch.ones_like(alpha_bar)
        else:
            value = self.alpha_bars[previous_timestep].to(
                device=noisy.device, dtype=noisy.dtype
            )
            alpha_bar_previous = value.reshape((1,) + (1,) * (noisy.ndim - 1))
        effective_alpha = (alpha_bar / alpha_bar_previous).clamp(max=1)
        effective_beta = 1 - effective_alpha
        predicted_clean = (
            (noisy - (1 - alpha_bar).sqrt() * epsilon) / alpha_bar.sqrt()
        ).clamp(-1, 1)
        denominator = (1 - alpha_bar).clamp(min=1e-20)
        mean = (
            effective_beta * alpha_bar_previous.sqrt() / denominator * predicted_clean
            + (1 - alpha_bar_previous)
            * effective_alpha.sqrt()
            / denominator
            * noisy
        )
        variance = effective_beta * (1 - alpha_bar_previous) / denominator
        return mean, variance.clamp(min=0)

    def effective_beta(self, timestep: int, previous_timestep: int) -> float:
        """Forward beta represented by one retained reverse transition."""
        alpha_bar_previous = (
            1.0
            if previous_timestep < 0
            else float(self.alpha_bars[previous_timestep])
        )
        return 1 - float(self.alpha_bars[timestep]) / alpha_bar_previous

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)
