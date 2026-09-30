"""Distance utilities for Tx-Rx propagation queries."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def euclidean_distance_3d(
    tx_position: ArrayLike, rx_positions: ArrayLike
) -> NDArray[np.float32]:
    """Return 3D Euclidean distances from one Tx to one or more Rx points."""

    tx = np.asarray(tx_position, dtype=np.float32)
    rx = np.asarray(rx_positions, dtype=np.float32)
    if tx.shape != (3,):
        raise ValueError(f"tx_position must have shape (3,), got {tx.shape}")
    if rx.shape[-1:] != (3,):
        raise ValueError(f"rx_positions must end in dimension 3, got {rx.shape}")
    return np.linalg.norm(rx - tx, axis=-1).astype(np.float32, copy=False)

