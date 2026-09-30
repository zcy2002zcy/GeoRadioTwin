"""Thin, documented adapter around the pinned official PMNet implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from torch import nn


def build_pmnet_adapted(in_channels: int = 3) -> nn.Module:
    """Load official PMNet-v1 and minimally adapt its two input convolutions."""
    source = Path(__file__).resolve().parents[1] / "external" / "PMNet" / "models" / "pmnet_v1.py"
    if not source.is_file():
        raise FileNotFoundError(f"Pinned PMNet source missing: {source}")
    spec = importlib.util.spec_from_file_location("pinned_official_pmnet_v1", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import PMNet from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.PMNet(
        n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=16,
    )
    # Official model assumes [city, Tx]. We add only the Rx-height plane.
    model.layer1.conv1.conv = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
    model.conv_up00[0] = nn.Conv2d(128 + in_channels, 64, 3, padding=1)
    return model
