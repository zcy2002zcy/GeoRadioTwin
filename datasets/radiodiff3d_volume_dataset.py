"""Strict h1--h4 volumes for the RadioDiff-3D-style adapted baseline.

The common project manifests are map-level.  This dataset first audits the
entire split and then groups maps that share the same scene and transmitter.
It deliberately fails on missing or duplicate receiver-height slices instead
of silently training a nominally 3-D model on incomplete volumes.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from datasets.coordinate_query_dataset import MapRecord, load_raw_pathloss, read_manifest


VolumeKey = tuple[int, float, float, float]


def _number(value: object) -> float:
    """Canonicalize manifest coordinates without changing their grid value."""

    return round(float(value), 8)


def _key(row: Mapping[str, object]) -> VolumeKey:
    return (
        int(row["scene_id"]),
        _number(row["tx_x"]),
        _number(row["tx_y"]),
        _number(row["tx_z"]),
    )


@dataclass(frozen=True)
class RadioVolumeRecord:
    """One transmitter with a strictly ordered receiver-height stack."""

    key: VolumeKey
    maps: tuple[MapRecord, ...]
    anchor_rx_height: float | None = None

    @property
    def scene_id(self) -> int:
        return self.key[0]


def _record_key(record: MapRecord) -> VolumeKey:
    return (
        int(record.scene_id),
        _number(record.tx_x),
        _number(record.tx_y),
        _number(record.tx_z),
    )


def _balanced_keys(
    keys: Sequence[VolumeKey], maximum: int, seed: int
) -> list[VolumeKey]:
    if maximum <= 0:
        raise ValueError("max_volumes must be positive")
    groups: dict[int, list[VolumeKey]] = {}
    for key in keys:
        groups.setdefault(key[0], []).append(key)
    rng = random.Random(seed)
    scene_ids = sorted(groups)
    rng.shuffle(scene_ids)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[VolumeKey] = []
    round_index = 0
    target = min(maximum, len(keys))
    while len(selected) < target:
        added = False
        for scene_id in scene_ids:
            group = groups[scene_id]
            if round_index < len(group):
                selected.append(group[round_index])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        round_index += 1
    return selected


def read_complete_volume_records(
    manifest_path: str | Path,
    *,
    expected_rx_heights: Iterable[float] = (1.0, 2.0, 3.0, 4.0),
    max_volumes: int | None = None,
    selection_strategy: str = "sequential",
    seed: int = 42,
) -> list[RadioVolumeRecord]:
    """Audit a map manifest and return complete receiver-height volumes.

    Two streaming passes avoid retaining hundreds of thousands of raster paths
    in memory.  Pass one validates every transmitter group in the split; pass
    two materializes only the selected volumes.
    """

    path = Path(manifest_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Manifest is missing or empty: {path}")
    if selection_strategy not in {"sequential", "balanced_scene"}:
        raise ValueError("selection_strategy must be sequential or balanced_scene")
    heights = tuple(_number(value) for value in expected_rx_heights)
    if not heights or len(set(heights)) != len(heights):
        raise ValueError("expected_rx_heights must contain unique values")
    expected = set(heights)
    masks: dict[VolumeKey, set[float]] = {}
    duplicates: list[tuple[VolumeKey, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = _key(row)
            height = _number(row["rx_height"])
            if height not in expected:
                raise ValueError(
                    f"Unexpected rx_height={height:g} for volume {key}; expected {heights}"
                )
            seen = masks.setdefault(key, set())
            if height in seen:
                duplicates.append((key, height))
            seen.add(height)
    if not masks:
        raise ValueError(f"Manifest contains no map rows: {path}")
    incomplete = [(key, tuple(sorted(expected - found))) for key, found in masks.items() if found != expected]
    if duplicates or incomplete:
        details: list[str] = []
        if duplicates:
            details.append(f"duplicate slices (first 5): {duplicates[:5]}")
        if incomplete:
            details.append(f"missing slices (first 5): {incomplete[:5]}")
        raise ValueError(
            "RadioDiff-3D requires exactly one map at every configured receiver "
            f"height for each (scene, Tx) group; {'; '.join(details)}"
        )

    all_keys = list(masks)
    if max_volumes is None:
        selected_keys = all_keys
    elif selection_strategy == "balanced_scene":
        selected_keys = _balanced_keys(all_keys, int(max_volumes), int(seed))
    else:
        if max_volumes <= 0:
            raise ValueError("max_volumes must be positive")
        selected_keys = all_keys[: int(max_volumes)]
    selected = set(selected_keys)
    materialized: dict[VolumeKey, dict[float, MapRecord]] = {
        key: {} for key in selected_keys
    }
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = _key(row)
            if key in selected:
                materialized[key][_number(row["rx_height"])] = MapRecord.from_mapping(row)

    records: list[RadioVolumeRecord] = []
    for key in selected_keys:
        by_height = materialized[key]
        ordered = tuple(by_height[height] for height in heights)
        building_paths = {record.building_height_path for record in ordered}
        if None in building_paths or len(building_paths) != 1:
            raise ValueError(
                f"Volume {key} must reference one shared building-height raster; "
                f"found {sorted(str(value) for value in building_paths)}"
            )
        records.append(RadioVolumeRecord(key=key, maps=ordered))
    return records


def read_anchor_volume_records(
    manifest_path: str | Path,
    *,
    max_anchor_maps: int,
    expected_rx_heights: Iterable[float] = (1.0, 2.0, 3.0, 4.0),
    selection_strategy: str = "balanced_scene",
    seed: int = 42,
) -> list[RadioVolumeRecord]:
    """Complete volumes around the exact map identities selected by the common protocol.

    The anchor maps are selected by the project's canonical ``read_manifest``
    function.  Each anchor's three companion heights are loaded solely to
    condition a 3-D prediction; only the anchor height is later scored.
    """

    path = Path(manifest_path)
    heights = tuple(_number(value) for value in expected_rx_heights)
    if not heights or len(set(heights)) != len(heights):
        raise ValueError("expected_rx_heights must contain unique values")
    expected = set(heights)
    anchors = read_manifest(
        path,
        max_records=int(max_anchor_maps),
        selection_strategy=selection_strategy,
        seed=int(seed),
    )
    anchor_keys = [_record_key(record) for record in anchors]
    if len(set(anchor_keys)) != len(anchor_keys):
        raise ValueError(
            "Common anchor selection contains multiple heights from the same "
            "(scene, Tx) volume; unique volume keys are required for one-sample-per-anchor evaluation"
        )
    selected = set(anchor_keys)
    materialized: dict[VolumeKey, dict[float, MapRecord]] = {
        key: {} for key in anchor_keys
    }
    duplicates: list[tuple[VolumeKey, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = _key(row)
            if key not in selected:
                continue
            height = _number(row["rx_height"])
            if height not in expected:
                raise ValueError(
                    f"Unexpected rx_height={height:g} for anchor volume {key}; expected {heights}"
                )
            if height in materialized[key]:
                duplicates.append((key, height))
            materialized[key][height] = MapRecord.from_mapping(row)

    records: list[RadioVolumeRecord] = []
    incomplete: list[tuple[VolumeKey, tuple[float, ...]]] = []
    for anchor, key in zip(anchors, anchor_keys):
        by_height = materialized[key]
        if set(by_height) != expected:
            incomplete.append((key, tuple(sorted(expected - set(by_height)))))
            continue
        ordered = tuple(by_height[height] for height in heights)
        building_paths = {record.building_height_path for record in ordered}
        if None in building_paths or len(building_paths) != 1:
            raise ValueError(
                f"Anchor volume {key} must reference one shared building-height raster; "
                f"found {sorted(str(value) for value in building_paths)}"
            )
        records.append(
            RadioVolumeRecord(
                key=key,
                maps=ordered,
                anchor_rx_height=_number(anchor.rx_height),
            )
        )
    if duplicates or incomplete:
        details: list[str] = []
        if duplicates:
            details.append(f"duplicate slices (first 5): {duplicates[:5]}")
        if incomplete:
            details.append(f"missing slices (first 5): {incomplete[:5]}")
        raise ValueError(
            "Every common-protocol anchor map must have exactly one h1--h4 "
            f"conditioning volume; {'; '.join(details)}"
        )
    return records


class RadioDiff3DVolumeDataset(Dataset[dict[str, Tensor]]):
    """Return `[C,D,H,W]` conditions and `[1,D,H,W]` path-loss volumes.

    ``paper_v2`` follows the three no-sampling inputs stated by the
    RadioDiff-3D paper: building segmentation, building height, and
    transmitter location.  ``legacy_height`` preserves the first project
    adaptation so its already-produced checkpoints remain reproducible.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        max_volumes: int,
        selection_strategy: str,
        seed: int,
        expected_rx_heights: Sequence[float] = (1.0, 2.0, 3.0, 4.0),
        exclude_target_values: Sequence[float] = (0.0,),
        building_scale: float = 255.0,
        rx_height_scale: float = 20.0,
        condition_layout: str = "paper_v2",
    ) -> None:
        self.expected_rx_heights = tuple(float(value) for value in expected_rx_heights)
        self.records = read_complete_volume_records(
            manifest_path,
            expected_rx_heights=self.expected_rx_heights,
            max_volumes=max_volumes,
            selection_strategy=selection_strategy,
            seed=seed,
        )
        self.exclude_target_values = tuple(float(value) for value in exclude_target_values)
        self.building_scale = float(building_scale)
        self.rx_height_scale = float(rx_height_scale)
        self.condition_layout = str(condition_layout)
        if self.building_scale <= 0 or self.rx_height_scale <= 0:
            raise ValueError("building_scale and rx_height_scale must be positive")
        if self.condition_layout not in {"paper_v2", "legacy_height"}:
            raise ValueError(
                "condition_layout must be 'paper_v2' or 'legacy_height'"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        volume = self.records[index]
        building_path = volume.maps[0].building_height_path
        if building_path is None:  # guarded during manifest audit
            raise ValueError(f"Volume {volume.key} has no building-height raster")
        with Image.open(building_path) as image:
            building = np.asarray(image, dtype=np.float32)
        if building.ndim != 2:
            raise ValueError(f"Expected a 2-D building raster, got {building.shape}: {building_path}")

        targets: list[np.ndarray] = []
        valid_masks: list[np.ndarray] = []
        for record in volume.maps:
            target = load_raw_pathloss(record.pathloss_path)
            if target.shape != building.shape:
                raise ValueError(
                    f"Raster shape mismatch for volume {volume.key}: "
                    f"building={building.shape}, target={target.shape}"
                )
            valid = building < record.rx_height
            for value in self.exclude_target_values:
                valid &= ~np.isclose(target, value, atol=0.0, rtol=0.0)
            if not valid.any():
                raise ValueError(
                    f"Volume {volume.key}, rx_height={record.rx_height:g} has no valid receiver pixels"
                )
            targets.append(target)
            valid_masks.append(valid)

        target_volume = np.stack(targets, axis=0).astype(np.float32)
        valid_volume = np.stack(valid_masks, axis=0)
        depth, height, width = target_volume.shape
        source_2d = np.zeros((height, width), dtype=np.float32)
        first = volume.maps[0]
        tx_row = int(round((height - 1) - first.tx_x))
        tx_col = int(round(first.tx_y))
        source_2d[np.clip(tx_row, 0, height - 1), np.clip(tx_col, 0, width - 1)] = 1.0

        building_height_volume = np.broadcast_to(
            np.clip(building / self.building_scale, 0.0, 1.0)[None],
            (depth, height, width),
        )
        building_segmentation_volume = np.broadcast_to(
            (building > 0.0).astype(np.float32)[None],
            (depth, height, width),
        )
        source_volume = np.broadcast_to(source_2d[None], (depth, height, width))
        height_values = np.asarray(self.expected_rx_heights, dtype=np.float32)
        height_volume = np.broadcast_to(
            (height_values / self.rx_height_scale)[:, None, None],
            (depth, height, width),
        )
        if self.condition_layout == "paper_v2":
            condition = np.stack(
                (
                    building_segmentation_volume,
                    building_height_volume,
                    source_volume,
                ),
                axis=0,
            ).astype(np.float32)
        else:
            condition = np.stack(
                (building_height_volume, source_volume, height_volume), axis=0
            ).astype(np.float32)
        sample = {
            "condition": torch.from_numpy(condition.copy()),
            "target": torch.from_numpy((target_volume / 255.0)[None]),
            "target_raw": torch.from_numpy(target_volume[None]),
            "valid_mask": torch.from_numpy(valid_volume[None]),
            "scene_id": torch.tensor(volume.scene_id, dtype=torch.long),
            "tx_x": torch.tensor(first.tx_x, dtype=torch.float32),
            "tx_y": torch.tensor(first.tx_y, dtype=torch.float32),
            "tx_z": torch.tensor(first.tx_z, dtype=torch.float32),
            "rx_heights": torch.tensor(self.expected_rx_heights, dtype=torch.float32),
        }
        if volume.anchor_rx_height is not None:
            anchor_height = _number(volume.anchor_rx_height)
            try:
                depth_index = tuple(_number(value) for value in self.expected_rx_heights).index(anchor_height)
            except ValueError as error:
                raise ValueError(
                    f"Anchor height {anchor_height:g} is outside configured heights {self.expected_rx_heights}"
                ) from error
            sample["score_depth_index"] = torch.tensor(depth_index, dtype=torch.long)
            sample["score_rx_height"] = torch.tensor(anchor_height, dtype=torch.float32)
        return sample


class RadioDiff3DAnchorEvaluationDataset(RadioDiff3DVolumeDataset):
    """Generate full h1--h4 volumes but score canonical map-level anchors only."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        max_anchor_maps: int,
        selection_strategy: str,
        seed: int,
        expected_rx_heights: Sequence[float] = (1.0, 2.0, 3.0, 4.0),
        exclude_target_values: Sequence[float] = (0.0,),
        building_scale: float = 255.0,
        rx_height_scale: float = 20.0,
        condition_layout: str = "paper_v2",
    ) -> None:
        self.expected_rx_heights = tuple(float(value) for value in expected_rx_heights)
        self.records = read_anchor_volume_records(
            manifest_path,
            max_anchor_maps=max_anchor_maps,
            expected_rx_heights=self.expected_rx_heights,
            selection_strategy=selection_strategy,
            seed=seed,
        )
        self.exclude_target_values = tuple(float(value) for value in exclude_target_values)
        self.building_scale = float(building_scale)
        self.rx_height_scale = float(rx_height_scale)
        self.condition_layout = str(condition_layout)
        if self.building_scale <= 0 or self.rx_height_scale <= 0:
            raise ValueError("building_scale and rx_height_scale must be positive")
        if self.condition_layout not in {"paper_v2", "legacy_height"}:
            raise ValueError(
                "condition_layout must be 'paper_v2' or 'legacy_height'"
            )
