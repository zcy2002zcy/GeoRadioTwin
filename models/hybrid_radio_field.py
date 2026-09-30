"""Lightweight raster-context residual over the continuous geometry field."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.continuous_radio_field import ContinuousRadioField


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )


class EnvironmentEncoder(nn.Module):
    """Compact multiscale encoder with a full-map receptive field."""

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.stem = ConvBlock(3, 32)
        self.down1 = ConvBlock(32, 64, stride=2)
        self.down2 = ConvBlock(64, 96, stride=2)
        self.down3 = ConvBlock(96, 128, stride=2)
        self.context = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, output_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.context(self.down3(self.down2(self.down1(self.stem(x)))))


class HybridGeoRadioField(nn.Module):
    """Pretrained continuous field plus a ray-aware raster-context correction."""

    def __init__(
        self,
        base: ContinuousRadioField,
        environment_dim: int = 64,
        ray_samples: int = 9,
        context_mode: str = "ray",
    ) -> None:
        super().__init__()
        if ray_samples < 2:
            raise ValueError("ray_samples must be at least 2")
        if context_mode not in {"ray", "endpoints"}:
            raise ValueError("context_mode must be 'ray' or 'endpoints'")
        self.base = base
        self.ray_samples = ray_samples
        self.context_mode = context_mode
        self.environment_encoder = EnvironmentEncoder(environment_dim)
        # Rx-local, Tx-local, ray mean/max + coordinates + analytic geometry + base.
        context_dim = environment_dim * (4 if context_mode == "ray" else 2)
        self.residual_head = nn.Sequential(
            nn.Linear(context_dim + 10 + base.geometry_input_dim + 1, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def encode_environment(self, environment_map: Tensor) -> Tensor:
        return self.environment_encoder(environment_map)

    def forward_with_encoded(
        self,
        coordinate_features: Tensor,
        geometry_features: Tensor | None,
        feature_map: Tensor,
        raster_shape: tuple[int, int],
    ) -> Tensor:
        base_prediction = self.base(coordinate_features, geometry_features)
        height, width = raster_shape
        tx_x = coordinate_features[..., 0]
        tx_y = coordinate_features[..., 1]
        rx_x = coordinate_features[..., 3]
        rx_y = coordinate_features[..., 4]

        def normalized_grid(x: Tensor, y: Tensor) -> Tensor:
            row = (height - 1) - x
            grid_x = 2.0 * y / max(width - 1, 1) - 1.0
            grid_y = 2.0 * row / max(height - 1, 1) - 1.0
            return torch.stack((grid_x, grid_y), dim=-1)

        rx_grid = normalized_grid(rx_x, rx_y).unsqueeze(2)
        tx_grid = normalized_grid(tx_x, tx_y).unsqueeze(2)
        endpoints = torch.stack((tx_grid.squeeze(2), rx_grid.squeeze(2)), dim=-2)
        fractions = torch.linspace(
            0.0, 1.0, self.ray_samples, device=feature_map.device, dtype=feature_map.dtype
        ).view(1, 1, self.ray_samples, 1)
        ray_grid = endpoints[..., :1, :] * (1.0 - fractions) + endpoints[..., 1:, :] * fractions

        def sample(grid: Tensor) -> Tensor:
            return F.grid_sample(
                feature_map, grid, mode="bilinear", padding_mode="border", align_corners=True
            ).permute(0, 2, 3, 1)

        rx_local = sample(rx_grid).squeeze(2)
        tx_local = sample(tx_grid).squeeze(2)
        normalized_coordinates = coordinate_features / self.base.coordinate_scales
        parts = [rx_local, tx_local]
        if self.context_mode == "ray":
            ray_features = sample(ray_grid)
            parts.extend((ray_features.mean(dim=2), ray_features.amax(dim=2)))
        parts.append(normalized_coordinates)
        if geometry_features is not None:
            parts.append(geometry_features)
        parts.append(base_prediction)
        residual_input = torch.cat(parts, dim=-1)
        return base_prediction + self.residual_head(residual_input)

    def forward(
        self,
        coordinate_features: Tensor,
        geometry_features: Tensor | None,
        environment_map: Tensor,
    ) -> Tensor:
        return self.forward_with_encoded(
            coordinate_features,
            geometry_features,
            self.encode_environment(environment_map),
            environment_map.shape[-2:],
        )
