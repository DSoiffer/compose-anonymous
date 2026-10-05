"""Conditional U-Net denoiser and an exponential moving average of its weights."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = -math.log(10_000) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32) * scale
        )
        angles = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        return F.pad(embedding, (0, self.dimension - embedding.shape[1]))


class ResBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, embedding_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.embedding = nn.Linear(embedding_dim, 2 * out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.embedding(F.silu(embedding)).chunk(2, dim=1)
        hidden = (
            self.norm2(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        )
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return self.skip(x) + hidden


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        sequence = self.norm(x).flatten(2).transpose(1, 2)
        attended = self.attention(sequence, sequence, sequence, need_weights=False)[0]
        return x + attended.transpose(1, 2).reshape(batch, channels, height, width)


class DownLevel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        dropout: float,
        down: bool,
    ) -> None:
        super().__init__()
        self.block1 = ResBlock(in_channels, out_channels, embedding_dim, dropout)
        self.block2 = ResBlock(out_channels, out_channels, embedding_dim, dropout)
        self.down = (
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            if down
            else nn.Identity()
        )


class UpLevel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        embedding_dim: int,
        dropout: float,
        up: bool,
    ) -> None:
        super().__init__()
        self.block1 = ResBlock(
            in_channels + skip_channels, out_channels, embedding_dim, dropout
        )
        self.block2 = ResBlock(
            out_channels + skip_channels, out_channels, embedding_dim, dropout
        )
        self.up = (
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            )
            if up
            else nn.Identity()
        )


@dataclass(frozen=True)
class UNetConfig:
    image_size: int = 64
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 2, 4)
    dropout: float = 0.1
    num_conditions: int = 3


class ConditionalUNet(nn.Module):
    """A compact DDPM-style U-Net conditioned only on labels 0/A/B."""

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config
        embedding_dim = config.base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(config.base_channels),
            nn.Linear(config.base_channels, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.condition_embedding = nn.Embedding(config.num_conditions, embedding_dim)
        self.input = nn.Conv2d(3, config.base_channels, 3, padding=1)
        channels = [
            config.base_channels * multiplier
            for multiplier in config.channel_multipliers
        ]
        downs = []
        current = config.base_channels
        for index, out_channels in enumerate(channels):
            downs.append(
                DownLevel(
                    current,
                    out_channels,
                    embedding_dim,
                    config.dropout,
                    down=index < len(channels) - 1,
                )
            )
            current = out_channels
        self.downs = nn.ModuleList(downs)
        self.middle1 = ResBlock(current, current, embedding_dim, config.dropout)
        self.middle_attention = AttentionBlock(current)
        self.middle2 = ResBlock(current, current, embedding_dim, config.dropout)
        ups = []
        for reverse_index, skip_channels in enumerate(reversed(channels)):
            ups.append(
                UpLevel(
                    current,
                    skip_channels,
                    skip_channels,
                    embedding_dim,
                    config.dropout,
                    up=reverse_index < len(channels) - 1,
                )
            )
            current = skip_channels
        self.ups = nn.ModuleList(ups)
        self.output_norm = nn.GroupNorm(_groups(current), current)
        self.output = nn.Conv2d(current, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        embedding = self.time_embedding(timesteps) + self.condition_embedding(condition)
        hidden = self.input(x)
        skips: list[torch.Tensor] = []
        for level in self.downs:
            hidden = level.block1(hidden, embedding)
            skips.append(hidden)
            hidden = level.block2(hidden, embedding)
            skips.append(hidden)
            hidden = level.down(hidden)
        hidden = self.middle1(hidden, embedding)
        hidden = self.middle_attention(hidden)
        hidden = self.middle2(hidden, embedding)
        for level in self.ups:
            hidden = level.block1(torch.cat((hidden, skips.pop()), dim=1), embedding)
            hidden = level.block2(torch.cat((hidden, skips.pop()), dim=1), embedding)
            hidden = level.up(hidden)
        return self.output(F.silu(self.output_norm(hidden)))

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.state[name].lerp_(value.detach(), 1 - self.decay)
            else:
                self.state[name].copy_(value)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "state": self.state}

