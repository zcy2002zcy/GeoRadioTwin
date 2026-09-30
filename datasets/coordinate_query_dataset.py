"""Lazy continuous Tx-Rx query sampling from PathLoss maps."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from geometry.coordinates import row_col_to_physical_xy
from geometry.cache import geometry_cache_path, load_geometry_cache


class QueryMode(str, Enum):
    """Supported query policies for maps with known ground truth."""

    RANDOM = "random"
    FIXED_RANDOM = "fixed_random"
    FULL_GRID = "full_grid"


@dataclass(frozen=True)
class MapRecord:
    """Minimal manifest fields needed to sample one PathLoss map."""

    scene_id: int
    tx_x: float
    tx_y: float
    tx_z: float
    rx_height: float
    pathloss_path: Path
    building_height_path: Path | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "MapRecord":
        return cls(
            scene_id=int(row["scene_id"]),
            tx_x=float(row["tx_x"]),
            tx_y=float(row["tx_y"]),
            tx_z=float(row["tx_z"]),
            rx_height=float(row["rx_height"]),
            pathloss_path=Path(str(row["pathloss_path"])),
            building_height_path=(
                Path(str(row["building_height_path"]))
                if row.get("building_height_path")
                else None
            ),
        )


def read_manifest(
    path: str | Path,
    max_records: int | None = None,
    *,
    selection_strategy: str = "sequential",
    seed: int = 42,
) -> list[MapRecord]:
    """Read map-level metadata only; raster pixels remain lazy on disk."""

    manifest_path = Path(path)
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise ValueError(f"Manifest is missing or empty: {manifest_path}")
    if selection_strategy not in {"sequential", "balanced_scene"}:
        raise ValueError("selection_strategy must be sequential or balanced_scene")
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        records = []
        for row in csv.DictReader(handle):
            records.append(MapRecord.from_mapping(row))
            if selection_strategy == "sequential" and max_records is not None and len(records) >= max_records:
                break
    if selection_strategy == "balanced_scene" and max_records is not None:
        records = _balanced_scene_records(records, max_records, seed)
    if not records:
        raise ValueError(f"Manifest contains no map rows: {manifest_path}")
    return records


def _balanced_scene_records(
    records: Sequence[MapRecord], max_records: int, seed: int
) -> list[MapRecord]:
    """Round-robin shuffled scenes so validation is not one-scene dominated."""

    if max_records <= 0:
        raise ValueError("max_records must be positive")
    groups: dict[int, list[MapRecord]] = {}
    for record in records:
        groups.setdefault(record.scene_id, []).append(record)
    rng = random.Random(seed)
    scene_ids = sorted(groups)
    rng.shuffle(scene_ids)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[MapRecord] = []
    round_index = 0
    while len(selected) < min(max_records, len(records)):
        added = False
        for scene_id in scene_ids:
            group = groups[scene_id]
            if round_index < len(group):
                selected.append(group[round_index])
                added = True
                if len(selected) >= max_records:
                    break
        if not added:
            break
        round_index += 1
    return selected


def load_raw_pathloss(path: Path, channel: int = 0) -> np.ndarray:
    """Load an unscaled PathLoss channel without asserting a dB mapping."""

    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 2:
        if channel != 0:
            raise ValueError(f"Single-channel image only supports channel 0: {path}")
        selected = array
    elif array.ndim == 3:
        if not 0 <= channel < array.shape[2]:
            raise ValueError(
                f"Channel {channel} outside image with {array.shape[2]} channels: {path}"
            )
        selected = array[..., channel]
    else:
        raise ValueError(f"Unsupported PathLoss raster shape {array.shape}: {path}")
    # Copy because PIL may own a read-only backing buffer.
    return np.asarray(selected, dtype=np.float32).copy()


class CoordinateQueryDataset(Dataset[dict[str, Tensor]]):
    """Return one lazily sampled Tx-Rx query set per map.

    Coordinate features are kept in physical grid units. Normalization belongs
    to the model/data transform and must be fitted only on the training split.
    """

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        records: Sequence[Mapping[str, object] | MapRecord] | None = None,
        points_per_map: int = 4096,
        query_mode: str | QueryMode = QueryMode.RANDOM,
        seed: int = 42,
        selection_seed: int | None = None,
        pathloss_channel: int = 0,
        max_records: int | None = None,
        geometry_cache_root: str | Path | None = None,
        use_los: bool = False,
        use_depth: bool = False,
        depth_feature: str = "normalized_depth",
        exclude_building_occupied: bool = False,
        exclude_target_values: Sequence[float] = (),
        target_exclusion_tolerance: float = 0.0,
        selection_strategy: str = "sequential",
        sparse_sample_rate: float = 0.0,
        sparse_context_radii: Sequence[int] = (4, 16, 64),
        include_environment_map: bool = False,
    ) -> None:
        if (manifest_path is None) == (records is None):
            raise ValueError("Provide exactly one of manifest_path or records")
        if points_per_map <= 0:
            raise ValueError("points_per_map must be positive")
        if manifest_path is not None:
            self.records = read_manifest(
                manifest_path,
                max_records=max_records,
                selection_strategy=selection_strategy,
                seed=seed if selection_seed is None else selection_seed,
            )
        else:
            assert records is not None
            self.records = [
                value if isinstance(value, MapRecord) else MapRecord.from_mapping(value)
                for value in records
            ]
        if not self.records:
            raise ValueError("At least one map record is required")
        self.points_per_map = int(points_per_map)
        self.query_mode = QueryMode(query_mode)
        self.seed = int(seed)
        self.pathloss_channel = int(pathloss_channel)
        self.geometry_cache_root = (
            Path(geometry_cache_root) if geometry_cache_root is not None else None
        )
        self.use_los = bool(use_los)
        self.use_depth = bool(use_depth)
        if depth_feature not in {"depth", "normalized_depth"}:
            raise ValueError("depth_feature must be 'depth' or 'normalized_depth'")
        self.depth_feature = depth_feature
        self.exclude_building_occupied = bool(exclude_building_occupied)
        self.exclude_target_values = tuple(float(value) for value in exclude_target_values)
        self.target_exclusion_tolerance = float(target_exclusion_tolerance)
        self.sparse_sample_rate = float(sparse_sample_rate)
        self.sparse_context_radii = tuple(int(value) for value in sparse_context_radii)
        self.include_environment_map = bool(include_environment_map)
        if not 0.0 <= self.sparse_sample_rate < 1.0:
            raise ValueError("sparse_sample_rate must be in [0, 1)")
        if any(radius < 0 for radius in self.sparse_context_radii):
            raise ValueError("sparse_context_radii must be non-negative")
        if self.target_exclusion_tolerance < 0:
            raise ValueError("target_exclusion_tolerance must be non-negative")
        if (self.use_los or self.use_depth) and self.geometry_cache_root is None:
            raise ValueError("Geometry features require geometry_cache_root")
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic training samples while preserving reproducibility."""

        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        target_map = load_raw_pathloss(record.pathloss_path, self.pathloss_channel)
        height, width = target_map.shape
        valid_map = self._valid_query_mask(record, target_map)
        rows, cols = self._query_pixels(index, valid_map)
        rx_x, rx_y = row_col_to_physical_xy(rows, cols, height)
        rx_x = np.asarray(rx_x, dtype=np.float32)
        rx_y = np.asarray(rx_y, dtype=np.float32)
        rx_z = np.full_like(rx_x, record.rx_height, dtype=np.float32)
        tx_x = np.full_like(rx_x, record.tx_x, dtype=np.float32)
        tx_y = np.full_like(rx_x, record.tx_y, dtype=np.float32)
        tx_z = np.full_like(rx_x, record.tx_z, dtype=np.float32)
        dx = rx_x - tx_x
        dy = rx_y - tx_y
        dz = rx_z - tx_z
        distance = np.sqrt(dx * dx + dy * dy + dz * dz)
        features = np.stack(
            (tx_x, tx_y, tx_z, rx_x, rx_y, rx_z, dx, dy, dz, distance),
            axis=-1,
        ).astype(np.float32, copy=False)
        target = target_map[rows, cols].reshape(-1, 1)
        rx_position = np.stack((rx_x, rx_y, rx_z), axis=-1)
        sample = {
            "coordinate_features": torch.from_numpy(features),
            "target": torch.from_numpy(target),
            "rx_position": torch.from_numpy(rx_position),
            "pixel_row_col": torch.from_numpy(
                np.stack((rows, cols), axis=-1).astype(np.int64, copy=False)
            ),
            "scene_id": torch.tensor(record.scene_id, dtype=torch.int64),
            "map_index": torch.tensor(index, dtype=torch.int64),
            "valid_mask": torch.from_numpy(valid_map[rows, cols].reshape(-1, 1)),
        }
        if self.include_environment_map:
            if record.building_height_path is None:
                raise ValueError("Environment map requires building heights")
            with Image.open(record.building_height_path) as image:
                building = np.asarray(image, dtype=np.float32)
            source = np.zeros_like(target_map, dtype=np.float32)
            tx_row = int(round((height - 1) - record.tx_x))
            tx_col = int(round(record.tx_y))
            source[np.clip(tx_row, 0, height - 1), np.clip(tx_col, 0, width - 1)] = 1.0
            environment = np.stack(
                (
                    np.clip(building / 255.0, 0.0, 1.0),
                    source,
                    np.full_like(target_map, record.rx_height / 20.0, dtype=np.float32),
                ),
                axis=0,
            ).astype(np.float32, copy=False)
            sample["environment_map"] = torch.from_numpy(environment)
        if self.sparse_sample_rate > 0.0:
            observation_mask, context_map = self._sparse_context(index, target_map, valid_map)
            sample["sparse_context_features"] = torch.from_numpy(
                context_map[rows, cols].astype(np.float32, copy=False)
            )
            sample["observation_mask"] = torch.from_numpy(
                observation_mask[rows, cols].reshape(-1, 1)
            )
        if self.use_los or self.use_depth:
            assert self.geometry_cache_root is not None
            cache_path = geometry_cache_path(
                self.geometry_cache_root,
                record.scene_id,
                record.tx_x,
                record.tx_y,
                record.rx_height,
            )
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"Geometry cache missing: {cache_path}. Run build_geometry_cache.py first."
                )
            cache, _ = load_geometry_cache(cache_path)
            parts = []
            if self.use_los:
                parts.append(cache["los"][rows, cols].astype(np.float32))
            if self.use_depth:
                parts.append(cache[self.depth_feature][rows, cols].astype(np.float32))
            geometry = np.stack(parts, axis=-1)
            sample["geometry_features"] = torch.from_numpy(geometry)
        return sample

    def _sparse_context(
        self, index: int, target_map: np.ndarray, valid_map: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build deterministic random observations and leakage-safe local summaries."""
        eligible = np.flatnonzero(valid_map.ravel())
        count = max(1, int(round(len(eligible) * self.sparse_sample_rate)))
        epoch = self.epoch if self.query_mode is QueryMode.RANDOM else 0
        rng = np.random.default_rng(
            self.seed + 17_171 + epoch * 1_000_003 + index * 97_409
        )
        selected = rng.choice(eligible, count, replace=False)
        observation_mask = np.zeros(target_map.shape, dtype=np.bool_)
        observation_mask.ravel()[selected] = True
        observed_values = np.where(observation_mask, target_map / 255.0, 0.0)
        features: list[np.ndarray] = []
        for radius in self.sparse_context_radii:
            value_sum = self._box_sum(observed_values, radius)
            observation_count = self._box_sum(observation_mask.astype(np.float32), radius)
            mean = np.divide(
                value_sum,
                observation_count,
                out=np.zeros_like(value_sum, dtype=np.float32),
                where=observation_count > 0,
            )
            area = float((2 * radius + 1) ** 2)
            features.extend((mean.astype(np.float32), (observation_count / area).astype(np.float32)))
        global_mean = float(target_map.ravel()[selected].mean() / 255.0)
        features.append(np.full(target_map.shape, global_mean, dtype=np.float32))
        return observation_mask, np.stack(features, axis=-1)

    @staticmethod
    def _box_sum(array: np.ndarray, radius: int) -> np.ndarray:
        padded = np.pad(array.astype(np.float64), ((1, 0), (1, 0)))
        integral = padded.cumsum(axis=0).cumsum(axis=1)
        height, width = array.shape
        rows = np.arange(height)
        cols = np.arange(width)
        top = np.maximum(rows - radius, 0)
        bottom = np.minimum(rows + radius + 1, height)
        left = np.maximum(cols - radius, 0)
        right = np.minimum(cols + radius + 1, width)
        return (
            integral[bottom[:, None], right[None, :]]
            - integral[top[:, None], right[None, :]]
            - integral[bottom[:, None], left[None, :]]
            + integral[top[:, None], left[None, :]]
        ).astype(np.float32)

    def _query_pixels(
        self, index: int, valid_map: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = valid_map.shape
        pixel_count = height * width
        if self.query_mode is QueryMode.FULL_GRID:
            flat = np.arange(pixel_count, dtype=np.int64)
        else:
            epoch = self.epoch if self.query_mode is QueryMode.RANDOM else 0
            # Large coprime multipliers make each (seed, epoch, map) stream stable.
            stream_seed = self.seed + epoch * 1_000_003 + index * 97_409
            rng = np.random.default_rng(stream_seed)
            eligible = np.flatnonzero(valid_map.ravel())
            if len(eligible) == 0:
                record = self.records[index]
                raise ValueError(
                    f"No valid Rx pixels for scene={record.scene_id}, map={record.pathloss_path}"
                )
            replace = self.points_per_map > len(eligible)
            flat = rng.choice(eligible, self.points_per_map, replace=replace)
        rows, cols = np.divmod(flat, width)
        return rows.astype(np.int64, copy=False), cols.astype(np.int64, copy=False)

    def _valid_query_mask(
        self, record: MapRecord, target_map: np.ndarray
    ) -> np.ndarray:
        valid = np.ones(target_map.shape, dtype=np.bool_)
        if self.exclude_building_occupied:
            if record.building_height_path is None:
                raise ValueError(
                    f"Building path required for valid-Rx masking: {record.pathloss_path}"
                )
            with Image.open(record.building_height_path) as image:
                building = np.asarray(image, dtype=np.float32)
            if building.shape != target_map.shape:
                raise ValueError(
                    f"Building/PathLoss shape mismatch: {building.shape} vs {target_map.shape}"
                )
            valid &= building < record.rx_height
        for excluded in self.exclude_target_values:
            valid &= ~np.isclose(
                target_map,
                excluded,
                atol=self.target_exclusion_tolerance,
                rtol=0.0,
            )
        return valid
