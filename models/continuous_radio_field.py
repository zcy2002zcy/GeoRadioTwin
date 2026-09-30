"""Geometry-aware continuous PathLoss field."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.geometry_encoder import GeometryEncoder
from models.positional_encoding import FourierPositionalEncoding


class ContinuousRadioField(nn.Module):
    """Fourier coordinate MLP with an optional geometry branch."""

    def __init__(
        self,
        *,
        coordinate_dim: int = 10,
        num_frequencies: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 6,
        geometry_input_dim: int = 0,
        geometry_hidden_dim: int = 32,
        geometry_output_dim: int = 32,
        use_layer_norm: bool = False,
        xy_scale: float = 255.0,
        z_scale: float = 20.0,
        distance_scale: float = 362.0,
        calibration_input_dim: int = 0,
        calibration_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if coordinate_dim != 10:
            raise ValueError("Version 1 expects the documented 10 coordinate features")
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be positive")
        scales = torch.tensor(
            [xy_scale, xy_scale, z_scale, xy_scale, xy_scale, z_scale,
             xy_scale, xy_scale, z_scale, distance_scale],
            dtype=torch.float32,
        )
        if torch.any(scales <= 0):
            raise ValueError("All coordinate scales must be positive")
        self.register_buffer("coordinate_scales", scales)
        self.position_encoder = FourierPositionalEncoding(
            coordinate_dim, num_frequencies, include_input=True
        )
        self.geometry_input_dim = geometry_input_dim
        self.geometry_encoder = (
            GeometryEncoder(geometry_input_dim, geometry_hidden_dim, geometry_output_dim)
            if geometry_input_dim > 0
            else None
        )
        input_dim = self.position_encoder.output_dim
        if self.geometry_encoder is not None:
            input_dim += self.geometry_encoder.output_dim

        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.field = nn.Sequential(*layers)
        self.calibration_input_dim = int(calibration_input_dim)
        if self.calibration_input_dim > 0:
            self.calibration_encoder = GeometryEncoder(
                self.calibration_input_dim, calibration_hidden_dim, calibration_hidden_dim
            )
            self.calibration_head = nn.Sequential(
                nn.Linear(input_dim + calibration_hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            nn.init.zeros_(self.calibration_head[-1].weight)
            nn.init.zeros_(self.calibration_head[-1].bias)
        else:
            self.calibration_encoder = None
            self.calibration_head = None

    def forward(
        self,
        coordinate_features: Tensor,
        geometry_features: Tensor | None = None,
        sparse_context_features: Tensor | None = None,
    ) -> Tensor:
        normalized = coordinate_features / self.coordinate_scales
        encoded = self.position_encoder(normalized)
        if self.geometry_encoder is None:
            if geometry_features is not None and geometry_features.shape[-1] > 0:
                raise ValueError("Coordinate-only model received geometry features")
            combined = encoded
        else:
            if geometry_features is None:
                raise ValueError("Geometry-aware model requires geometry_features")
            combined = torch.cat((encoded, self.geometry_encoder(geometry_features)), dim=-1)
        prediction = self.field(combined)
        if self.calibration_head is not None:
            if sparse_context_features is None:
                raise ValueError("Calibration model requires sparse_context_features")
            assert self.calibration_encoder is not None
            context = self.calibration_encoder(sparse_context_features)
            prediction = prediction + self.calibration_head(torch.cat((combined, context), dim=-1))
        elif sparse_context_features is not None and sparse_context_features.shape[-1] > 0:
            raise ValueError("Non-calibration model received sparse context features")
        return prediction
