"""A true volumetric conditional diffusion baseline for UrbanRadio3D.

This is an independent RadioDiff-3D-style adaptation, not the authors'
official implementation.  The denoiser uses 3-D convolutions over the four
receiver-height slices while down/up-sampling only the spatial axes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4:
            raise ValueError("Time embedding dimension must be at least 4")
        self.dimension = int(dimension)

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.dimension // 2
        denominator = max(half - 1, 1)
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / denominator
        )
        arguments = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat((arguments.sin(), arguments.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return embedding


class TimeResidualBlock3D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_dimension: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(time_dimension, out_channels)
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1)
        )
        # The residual branch starts near zero, a standard diffusion U-Net
        # stabilization that is especially useful with only four z slices.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, features: Tensor, time_embedding: Tensor) -> Tensor:
        residual = self.skip(features)
        features = self.conv1(F.silu(self.norm1(features)))
        features = features + self.time_projection(time_embedding)[:, :, None, None, None]
        features = self.conv2(self.dropout(F.silu(self.norm2(features))))
        return features + residual


class HeightAttention(nn.Module):
    """Attend across receiver heights independently at each spatial site."""

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        heads = min(int(heads), channels)
        while channels % heads:
            heads -= 1
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, features: Tensor) -> Tensor:
        batch, channels, depth, height, width = features.shape
        normalized = self.norm(features)
        sequence = normalized.permute(0, 3, 4, 2, 1).reshape(
            batch * height * width, depth, channels
        )
        attended, _ = self.attention(
            sequence, sequence, sequence, need_weights=False
        )
        attended = attended.reshape(batch, height, width, depth, channels).permute(
            0, 4, 3, 1, 2
        )
        return features + attended


class ConditionalVolumeUNet(nn.Module):
    """Diffusion target predictor conditioned on volumetric environment maps."""

    def __init__(
        self,
        *,
        condition_channels: int = 3,
        target_channels: int = 1,
        base_channels: int = 32,
        channel_multipliers: Sequence[int] = (1, 2, 4, 6),
        time_dimension: int = 128,
        dropout: float = 0.0,
        expected_depth: int = 4,
        multiscale_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if not channel_multipliers:
            raise ValueError("channel_multipliers cannot be empty")
        if min(channel_multipliers) <= 0:
            raise ValueError("channel_multipliers must be positive")
        self.condition_channels = int(condition_channels)
        self.target_channels = int(target_channels)
        self.expected_depth = int(expected_depth)
        self.multiscale_conditioning = bool(multiscale_conditioning)
        widths = [int(base_channels) * int(value) for value in channel_multipliers]
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dimension),
            nn.Linear(time_dimension, time_dimension * 4),
            nn.SiLU(),
            nn.Linear(time_dimension * 4, time_dimension),
        )
        self.input_projection = nn.Conv3d(
            self.target_channels + self.condition_channels, widths[0], 3, padding=1
        )

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = widths[0]
        for level, width in enumerate(widths):
            self.encoder_blocks.append(
                nn.ModuleList(
                    (
                        TimeResidualBlock3D(current, width, time_dimension, dropout),
                        TimeResidualBlock3D(width, width, time_dimension, dropout),
                    )
                )
            )
            current = width
            if level < len(widths) - 1:
                self.downsamples.append(
                    nn.Conv3d(
                        width,
                        widths[level + 1],
                        kernel_size=(1, 4, 4),
                        stride=(1, 2, 2),
                        padding=(0, 1, 1),
                    )
                )
                current = widths[level + 1]

        self.middle1 = TimeResidualBlock3D(current, current, time_dimension, dropout)
        self.height_attention = HeightAttention(current)
        self.middle2 = TimeResidualBlock3D(current, current, time_dimension, dropout)

        self.encoder_condition_projections = nn.ModuleList()
        self.decoder_condition_projections = nn.ModuleList()
        self.middle_condition_projection: nn.Module | None = None
        if self.multiscale_conditioning:
            self.encoder_condition_projections.extend(
                nn.Conv3d(self.condition_channels, width, 1) for width in widths
            )
            self.middle_condition_projection = nn.Conv3d(
                self.condition_channels, widths[-1], 1
            )

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.upsamples.append(
                nn.ConvTranspose3d(
                    current,
                    width,
                    kernel_size=(1, 4, 4),
                    stride=(1, 2, 2),
                    padding=(0, 1, 1),
                )
            )
            self.decoder_blocks.append(
                nn.ModuleList(
                    (
                        TimeResidualBlock3D(width * 2, width, time_dimension, dropout),
                        TimeResidualBlock3D(width, width, time_dimension, dropout),
                    )
                )
            )
            if self.multiscale_conditioning:
                self.decoder_condition_projections.append(
                    nn.Conv3d(self.condition_channels, width, 1)
                )
            current = width
        self.output_norm = nn.GroupNorm(_group_count(current), current)
        self.output_projection = nn.Conv3d(current, self.target_channels, 3, padding=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _project_condition(
        condition: Tensor, reference: Tensor, projection: nn.Module
    ) -> Tensor:
        if condition.shape[2:] != reference.shape[2:]:
            condition = F.interpolate(
                condition,
                size=reference.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        return projection(condition)

    def forward(self, noisy_target: Tensor, condition: Tensor, timesteps: Tensor) -> Tensor:
        if noisy_target.ndim != 5 or condition.ndim != 5:
            raise ValueError("noisy_target and condition must be [B,C,D,H,W]")
        if noisy_target.shape[0] != condition.shape[0] or noisy_target.shape[2:] != condition.shape[2:]:
            raise ValueError("noisy_target and condition batch/volume shapes must match")
        if noisy_target.shape[1] != self.target_channels:
            raise ValueError(f"Expected {self.target_channels} target channels")
        if condition.shape[1] != self.condition_channels:
            raise ValueError(f"Expected {self.condition_channels} condition channels")
        if noisy_target.shape[2] != self.expected_depth:
            raise ValueError(
                f"Expected a depth-{self.expected_depth} h1--h4 volume, got depth={noisy_target.shape[2]}"
            )
        if timesteps.ndim != 1 or timesteps.shape[0] != noisy_target.shape[0]:
            raise ValueError("timesteps must have shape [B]")

        time_embedding = self.time_embedding(timesteps)
        features = self.input_projection(torch.cat((noisy_target, condition), dim=1))
        skips: list[Tensor] = []
        for level, blocks in enumerate(self.encoder_blocks):
            if self.multiscale_conditioning:
                features = features + self._project_condition(
                    condition, features, self.encoder_condition_projections[level]
                )
            features = blocks[0](features, time_embedding)
            features = blocks[1](features, time_embedding)
            if level < len(self.downsamples):
                skips.append(features)
                features = self.downsamples[level](features)
        if self.multiscale_conditioning:
            assert self.middle_condition_projection is not None
            features = features + self._project_condition(
                condition, features, self.middle_condition_projection
            )
        features = self.middle1(features, time_embedding)
        features = self.height_attention(features)
        features = self.middle2(features, time_embedding)
        for decoder_index, (upsample, blocks, skip) in enumerate(
            zip(self.upsamples, self.decoder_blocks, reversed(skips))
        ):
            features = upsample(features)
            if features.shape[2:] != skip.shape[2:]:
                raise ValueError(
                    "Spatial dimensions must be divisible by the U-Net downsampling factor"
                )
            features = torch.cat((features, skip), dim=1)
            features = blocks[0](features, time_embedding)
            if self.multiscale_conditioning:
                features = features + self._project_condition(
                    condition,
                    features,
                    self.decoder_condition_projections[decoder_index],
                )
            features = blocks[1](features, time_embedding)
        return self.output_projection(F.silu(self.output_norm(features)))


def _cosine_betas(timesteps: int, offset: float = 0.008) -> Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    cumulative = torch.cos(
        ((x / timesteps) + offset) / (1.0 + offset) * math.pi * 0.5
    ).square()
    cumulative = cumulative / cumulative[0]
    betas = 1.0 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-5, 0.999).float()


class RadioDiff3DAdapted(nn.Module):
    """Conditional diffusion training with fixed-seed DDIM/DDPM inference."""

    def __init__(
        self,
        denoiser: ConditionalVolumeUNet,
        *,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        parameterization: str = "epsilon",
        loss_type: str = "l2",
        loss_support: str = "valid_rx",
    ) -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.parameterization = str(parameterization).lower()
        self.loss_type = str(loss_type).lower()
        self.loss_support = str(loss_support).lower()
        if self.parameterization not in {"epsilon", "x0"}:
            raise ValueError("parameterization must be 'epsilon' or 'x0'")
        if self.loss_type not in {"l1", "l2"}:
            raise ValueError("loss_type must be 'l1' or 'l2'")
        if self.loss_support not in {"full", "valid_rx"}:
            raise ValueError("loss_support must be 'full' or 'valid_rx'")
        if beta_schedule == "cosine":
            betas = _cosine_betas(self.timesteps)
        elif beta_schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, self.timesteps)
        else:
            raise ValueError("beta_schedule must be 'cosine' or 'linear'")
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", cumulative)
        self.register_buffer("sqrt_alphas_cumprod", cumulative.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - cumulative).sqrt())

    @staticmethod
    def _extract(values: Tensor, timesteps: Tensor, reference: Tensor) -> Tensor:
        selected = values.gather(0, timesteps)
        return selected.reshape(timesteps.shape[0], *((1,) * (reference.ndim - 1)))

    def q_sample(self, clean: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        return (
            self._extract(self.sqrt_alphas_cumprod, timesteps, clean) * clean
            + self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean) * noise
        )

    def training_loss(
        self,
        condition: Tensor,
        target_zero_to_one: Tensor,
        valid_mask: Tensor,
        *,
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        clean = target_zero_to_one * 2.0 - 1.0
        batch = clean.shape[0]
        if timesteps is None:
            timesteps = torch.randint(0, self.timesteps, (batch,), device=clean.device)
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = self.q_sample(clean, timesteps, noise)
        model_output = self.denoiser(noisy, condition, timesteps)
        objective = noise if self.parameterization == "epsilon" else clean
        if self.loss_support == "full":
            support = torch.ones_like(model_output, dtype=torch.bool)
        else:
            support = valid_mask.bool().expand_as(model_output)
        if not support.any():
            raise ValueError("Diffusion batch has no valid receiver voxels")
        error = model_output[support].float() - objective[support].float()
        loss = error.abs().mean() if self.loss_type == "l1" else error.square().mean()
        return loss, timesteps

    def predict_clean(self, noisy: Tensor, predicted_noise: Tensor, timestep: int) -> Tensor:
        alpha = self.alphas_cumprod[int(timestep)]
        return (noisy - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()

    def model_predictions(
        self, noisy: Tensor, model_output: Tensor, timestep: int
    ) -> tuple[Tensor, Tensor]:
        """Convert the configured training target to `(x0, epsilon)`."""

        alpha = self.alphas_cumprod[int(timestep)]
        if self.parameterization == "epsilon":
            predicted_noise = model_output
            predicted_clean = self.predict_clean(noisy, predicted_noise, timestep)
        else:
            predicted_clean = model_output
            predicted_noise = (
                noisy - alpha.sqrt() * predicted_clean
            ) / (1.0 - alpha).sqrt().clamp_min(1e-12)
        return predicted_clean, predicted_noise

    @torch.inference_mode()
    def ddim_sample(
        self,
        condition: Tensor,
        *,
        sampling_steps: int,
        generator: torch.Generator,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> Tensor:
        if not 1 <= sampling_steps <= self.timesteps:
            raise ValueError("sampling_steps must be in [1, timesteps]")
        if eta < 0:
            raise ValueError("eta must be non-negative")
        shape = (
            condition.shape[0],
            self.denoiser.target_channels,
            *condition.shape[2:],
        )
        sample = torch.randn(
            shape, device=condition.device, dtype=condition.dtype, generator=generator
        )
        sequence = torch.linspace(
            self.timesteps - 1,
            0,
            sampling_steps,
            device=condition.device,
            dtype=torch.float64,
        ).round().long()
        # Consecutive uniqueness matters only when a caller uses very small T.
        sequence = torch.unique_consecutive(sequence)
        previous = torch.cat((sequence[1:], sequence.new_tensor([-1])))
        for timestep, previous_timestep in zip(sequence.tolist(), previous.tolist()):
            batch_timesteps = torch.full(
                (condition.shape[0],), timestep, device=condition.device, dtype=torch.long
            )
            model_output = self.denoiser(sample, condition, batch_timesteps)
            alpha = self.alphas_cumprod[timestep]
            predicted_clean, predicted_noise = self.model_predictions(
                sample, model_output, timestep
            )
            if clip_denoised:
                predicted_clean = predicted_clean.clamp(-1.0, 1.0)
            if previous_timestep < 0:
                sample = predicted_clean
                continue
            alpha_previous = self.alphas_cumprod[previous_timestep]
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_previous) / (1.0 - alpha))
                * (1.0 - alpha / alpha_previous).clamp_min(0.0)
            )
            direction = torch.sqrt((1.0 - alpha_previous - sigma.square()).clamp_min(0.0))
            random_noise = (
                torch.randn(
                    sample.shape,
                    device=sample.device,
                    dtype=sample.dtype,
                    generator=generator,
                )
                if eta > 0
                else 0.0
            )
            sample = (
                alpha_previous.sqrt() * predicted_clean
                + direction * predicted_noise
                + sigma * random_noise
            )
        return sample

    @torch.inference_mode()
    def ddpm_sample(
        self,
        condition: Tensor,
        *,
        generator: torch.Generator,
        clip_denoised: bool = True,
    ) -> Tensor:
        """Ancestral DDPM sampling over every configured diffusion step."""

        shape = (
            condition.shape[0],
            self.denoiser.target_channels,
            *condition.shape[2:],
        )
        sample = torch.randn(
            shape, device=condition.device, dtype=condition.dtype, generator=generator
        )
        for timestep in range(self.timesteps - 1, -1, -1):
            batch_timesteps = torch.full(
                (condition.shape[0],),
                timestep,
                device=condition.device,
                dtype=torch.long,
            )
            model_output = self.denoiser(sample, condition, batch_timesteps)
            predicted_clean, _ = self.model_predictions(
                sample, model_output, timestep
            )
            if clip_denoised:
                predicted_clean = predicted_clean.clamp(-1.0, 1.0)
            if timestep == 0:
                sample = predicted_clean
                continue
            alpha_bar = self.alphas_cumprod[timestep]
            alpha_bar_previous = self.alphas_cumprod[timestep - 1]
            beta = self.betas[timestep]
            alpha = 1.0 - beta
            denominator = (1.0 - alpha_bar).clamp_min(1e-12)
            mean = (
                beta * alpha_bar_previous.sqrt() / denominator * predicted_clean
                + alpha.sqrt()
                * (1.0 - alpha_bar_previous)
                / denominator
                * sample
            )
            variance = beta * (1.0 - alpha_bar_previous) / denominator
            noise = torch.randn(
                sample.shape,
                device=sample.device,
                dtype=sample.dtype,
                generator=generator,
            )
            sample = mean + variance.clamp_min(0.0).sqrt() * noise
        return sample


def build_radiodiff3d_adapted(
    *,
    condition_channels: int = 3,
    target_channels: int = 1,
    base_channels: int = 32,
    channel_multipliers: Sequence[int] = (1, 2, 4, 6),
    time_dimension: int = 128,
    dropout: float = 0.0,
    expected_depth: int = 4,
    multiscale_conditioning: bool = False,
    timesteps: int = 1000,
    beta_schedule: str = "cosine",
    parameterization: str = "epsilon",
    loss_type: str = "l2",
    loss_support: str = "valid_rx",
) -> RadioDiff3DAdapted:
    denoiser = ConditionalVolumeUNet(
        condition_channels=condition_channels,
        target_channels=target_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        time_dimension=time_dimension,
        dropout=dropout,
        expected_depth=expected_depth,
        multiscale_conditioning=multiscale_conditioning,
    )
    return RadioDiff3DAdapted(
        denoiser,
        timesteps=timesteps,
        beta_schedule=beta_schedule,
        parameterization=parameterization,
        loss_type=loss_type,
        loss_support=loss_support,
    )
