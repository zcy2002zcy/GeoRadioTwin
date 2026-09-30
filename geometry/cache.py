"""On-disk cache helpers for full native-grid geometry features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Any

import numpy as np
from numpy.typing import NDArray

from geometry.coordinates import row_col_to_physical_xy
from geometry.ray_query import RayQueryResult, query_propagation_geometry


def geometry_cache_path(
    cache_root: Path,
    scene_id: int,
    tx_x: float,
    tx_y: float,
    rx_z: float,
) -> Path:
    """Return a stable cache key containing scene, Tx XY, and Rx height."""

    def number(value: float) -> str:
        return f"{value:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")

    return (
        cache_root
        / f"scene_{scene_id:04d}"
        / f"tx_x{number(tx_x)}_y{number(tx_y)}_rxz{number(rx_z)}.npz"
    )


def build_full_grid_geometry(
    building_height: NDArray[np.generic],
    tx_position: tuple[float, float, float],
    rx_z: float,
    *,
    samples_per_unit: float = 2.0,
    chunk_size: int = 1024,
) -> RayQueryResult:
    """Query every native raster cell in row-major order."""

    height, width = building_height.shape
    rows, cols = np.indices((height, width))
    rx_x, rx_y = row_col_to_physical_xy(rows.ravel(), cols.ravel(), height)
    rx_positions = np.stack(
        (
            np.asarray(rx_x, dtype=np.float32),
            np.asarray(rx_y, dtype=np.float32),
            np.full(height * width, rx_z, dtype=np.float32),
        ),
        axis=1,
    )
    return query_propagation_geometry(
        building_height,
        tx_position,
        rx_positions,
        samples_per_unit=samples_per_unit,
        chunk_size=chunk_size,
    )


def save_geometry_cache(
    path: Path,
    result: RayQueryResult,
    shape: tuple[int, int],
    metadata: Mapping[str, object],
) -> None:
    """Atomically save compressed feature maps and provenance metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    arrays = {
        "los": result.los.reshape(shape).astype(np.uint8),
        "depth": result.depth.reshape(shape).astype(np.float32),
        "normalized_depth": result.normalized_depth.reshape(shape).astype(np.float32),
        "distance": result.distance.reshape(shape).astype(np.float32),
        "first_obstruction_ratio": result.first_obstruction_ratio.reshape(shape).astype(np.float32),
        "num_obstruction_samples": result.num_obstruction_samples.reshape(shape).astype(np.int32),
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    }
    with temp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temp_path.replace(path)


def load_geometry_cache(path: Path) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
    """Load cached arrays and their provenance metadata."""

    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: archive[name].copy()
            for name in archive.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(archive["metadata_json"].item()))
    return arrays, metadata
