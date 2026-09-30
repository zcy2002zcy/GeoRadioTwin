"""Small MLP encoder for LoS and/or geometry-depth features."""

from __future__ import annotations

from torch import Tensor, nn


class GeometryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 32) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("Geometry encoder dimensions must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} geometry features, got {features.shape[-1]}"
            )
        return self.network(features)

