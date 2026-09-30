"""Vectorized 3D line queries through a building-height raster."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from geometry.distance import euclidean_distance_3d


@dataclass(frozen=True)
class RayQueryResult:
    """Propagation features returned for each Tx-Rx ray."""

    los: NDArray[np.bool_]
    depth: NDArray[np.float32]
    normalized_depth: NDArray[np.float32]
    distance: NDArray[np.float32]
    first_obstruction_ratio: NDArray[np.float32]
    num_obstruction_samples: NDArray[np.int32]


def query_propagation_geometry(
    building_height: ArrayLike,
    tx_position: ArrayLike,
    rx_positions: ArrayLike,
    *,
    samples_per_unit: float = 2.0,
    obstruction_epsilon: float = 1e-6,
    chunk_size: int = 1024,
) -> RayQueryResult:
    """Query LoS and first-obstruction depth for arbitrary Rx positions.

    Physical X points upward from the raster's lower-left corner and Y points
    right. Continuous ray samples use nearest-neighbour geometry lookup because
    the source building raster has no information above its native resolution.
    Tx and Rx endpoints are excluded from obstruction testing.
    """

    buildings = np.asarray(building_height, dtype=np.float32)
    tx = np.asarray(tx_position, dtype=np.float32)
    rx = np.asarray(rx_positions, dtype=np.float32)
    if buildings.ndim != 2:
        raise ValueError(f"building_height must be 2D, got {buildings.shape}")
    if tx.shape != (3,):
        raise ValueError(f"tx_position must have shape (3,), got {tx.shape}")
    if rx.ndim == 1:
        rx = rx.reshape(1, -1)
    if rx.ndim != 2 or rx.shape[1] != 3:
        raise ValueError(f"rx_positions must have shape (N, 3), got {rx.shape}")
    if samples_per_unit <= 0:
        raise ValueError("samples_per_unit must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    height, width = buildings.shape
    _validate_xy(tx[:2], height, width, "Tx")
    if len(rx):
        _validate_xy_batch(rx[:, :2], height, width)

    count = len(rx)
    los = np.empty(count, dtype=np.bool_)
    depth = np.empty(count, dtype=np.float32)
    normalized_depth = np.empty(count, dtype=np.float32)
    first_ratio = np.empty(count, dtype=np.float32)
    obstruction_count = np.empty(count, dtype=np.int32)
    distance = euclidean_distance_3d(tx, rx)

    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        block_rx = rx[start:end]
        delta = block_rx - tx
        horizontal = np.linalg.norm(delta[:, :2], axis=1)
        segments = np.maximum(
            np.ceil(horizontal * samples_per_unit).astype(np.int32), 1
        )
        max_segments = int(segments.max(initial=1))
        if max_segments <= 1:
            block_distance = distance[start:end]
            los[start:end] = True
            first_ratio[start:end] = 1.0
            normalized_depth[start:end] = 1.0
            depth[start:end] = block_distance
            obstruction_count[start:end] = 0
            continue
        steps = np.arange(1, max_segments, dtype=np.float32)[None, :]
        ratios = steps / segments[:, None]
        interior = steps < segments[:, None]

        sample_x = tx[0] + ratios * delta[:, 0:1]
        sample_y = tx[1] + ratios * delta[:, 1:2]
        sample_z = tx[2] + ratios * delta[:, 2:3]
        x_index = np.clip(np.rint(sample_x).astype(np.int32), 0, height - 1)
        y_index = np.clip(np.rint(sample_y).astype(np.int32), 0, width - 1)
        rows = (height - 1) - x_index
        sampled_buildings = buildings[rows, y_index]
        obstructed = interior & (
            sampled_buildings + obstruction_epsilon >= sample_z
        )
        has_obstruction = obstructed.any(axis=1)
        first_indices = np.argmax(obstructed, axis=1)
        block_first_ratio = np.ones(end - start, dtype=np.float32)
        obstructed_rows = np.flatnonzero(has_obstruction)
        if len(obstructed_rows):
            block_first_ratio[obstructed_rows] = ratios[
                obstructed_rows, first_indices[obstructed_rows]
            ]

        block_distance = distance[start:end]
        los[start:end] = ~has_obstruction
        first_ratio[start:end] = block_first_ratio
        normalized_depth[start:end] = block_first_ratio
        depth[start:end] = block_distance * block_first_ratio
        obstruction_count[start:end] = obstructed.sum(axis=1, dtype=np.int32)

    return RayQueryResult(
        los=los,
        depth=depth,
        normalized_depth=normalized_depth,
        distance=distance,
        first_obstruction_ratio=first_ratio,
        num_obstruction_samples=obstruction_count,
    )


def _validate_xy(xy: NDArray[np.float32], height: int, width: int, name: str) -> None:
    if not (0 <= xy[0] <= height - 1 and 0 <= xy[1] <= width - 1):
        raise ValueError(
            f"{name} physical XY={tuple(xy.tolist())} outside raster "
            f"X=[0,{height - 1}], Y=[0,{width - 1}]"
        )


def _validate_xy_batch(
    xy: NDArray[np.float32], height: int, width: int
) -> None:
    valid = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] <= height - 1)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] <= width - 1)
    )
    if not valid.all():
        index = int(np.flatnonzero(~valid)[0])
        _validate_xy(xy[index], height, width, f"Rx[{index}]")
