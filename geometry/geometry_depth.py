"""Geometry-depth query API."""

from __future__ import annotations

from numpy.typing import ArrayLike, NDArray
import numpy as np

from geometry.ray_query import query_propagation_geometry


def query_geometry_depth(
    building_height: ArrayLike,
    tx_position: ArrayLike,
    rx_positions: ArrayLike,
    *,
    samples_per_unit: float = 2.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return absolute and normalized first-obstruction depths."""

    result = query_propagation_geometry(
        building_height,
        tx_position,
        rx_positions,
        samples_per_unit=samples_per_unit,
    )
    return result.depth, result.normalized_depth

