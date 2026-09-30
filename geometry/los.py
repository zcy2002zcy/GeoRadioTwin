"""Line-of-sight query API."""

from __future__ import annotations

from numpy.typing import ArrayLike, NDArray
import numpy as np

from geometry.ray_query import query_propagation_geometry


def query_los(
    building_height: ArrayLike,
    tx_position: ArrayLike,
    rx_positions: ArrayLike,
    *,
    samples_per_unit: float = 2.0,
) -> NDArray[np.bool_]:
    """Return True for rays without a sampled building obstruction."""

    return query_propagation_geometry(
        building_height,
        tx_position,
        rx_positions,
        samples_per_unit=samples_per_unit,
    ).los

