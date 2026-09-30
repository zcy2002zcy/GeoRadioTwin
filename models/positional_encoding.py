"""Fourier positional encoding for continuous coordinate features."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FourierPositionalEncoding(nn.Module):
    """Encode each scalar with sin/cos bands 2^k pi x."""

    def __init__(self, input_dim: int, num_frequencies: int = 6, include_input: bool = True) -> None:
        super().__init__()
        if input_dim <= 0 or num_frequencies <= 0:
            raise ValueError("input_dim and num_frequencies must be positive")
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        frequencies = (2.0 ** torch.arange(num_frequencies)) * math.pi
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        multiplier = 2 * self.num_frequencies + int(self.include_input)
        return self.input_dim * multiplier

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got {values.shape[-1]}"
            )
        angles = values.unsqueeze(-1) * self.frequencies
        encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        encoded = encoded.flatten(start_dim=-2)
        return torch.cat((values, encoded), dim=-1) if self.include_input else encoded

