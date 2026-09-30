"""Shared, auditable helpers for digital-twin-facing validation experiments."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.coordinate_query_dataset import CoordinateQueryDataset, MapRecord, read_manifest
from geometry.cache import build_full_grid_geometry, geometry_cache_path, save_geometry_cache
from models.hybrid_radio_field import HybridGeoRadioField
from scripts.evaluate import build_model, metrics, normalized_metrics, resolve_path
from utils.config import load_config


@dataclass(frozen=True)
class LoadedHybrid:
    config: dict
    config_path: Path
    checkpoint_path: Path
    checkpoint_epoch: int
    device: torch.device
    model: HybridGeoRadioField
    cache_root: Path
    target_scale: float
    query_batch_size: int


def output_name(config: dict) -> str:
    return f"{config['experiment']['name']}_{config['training']['run_tag']}"


def load_hybrid(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    *,
    device_name: str = "auto",
) -> LoadedHybrid:
    config, resolved_config = load_config(config_path)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    output = resolve_path(config["output_root"], resolved_config) / output_name(config)
    checkpoint = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path is not None
        else output / "checkpoints" / "best.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Hybrid checkpoint not found: {checkpoint}")
    base = build_model(config)
    model = HybridGeoRadioField(
        base,
        int(config["model"]["environment_dim"]),
        int(config["model"]["ray_samples"]),
        str(config["model"].get("context_mode", "ray")),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return LoadedHybrid(
        config=config,
        config_path=resolved_config,
        checkpoint_path=checkpoint,
        checkpoint_epoch=int(state["epoch"]),
        device=device,
        model=model,
        cache_root=resolve_path(config["geometry"]["cache_root"], resolved_config),
        target_scale=float(config["training"]["target_scale"]),
        query_batch_size=int(config["evaluation"]["query_batch_size"]),
    )


def test_manifest(loaded: LoadedHybrid) -> Path:
    return (
        resolve_path(loaded.config["data"]["output_root"], loaded.config_path)
        / "splits"
        / "test.csv"
    )


def make_dataset(
    loaded: LoadedHybrid,
    *,
    records: Sequence[MapRecord] | None = None,
    max_maps: int | None = None,
    selection_strategy: str = "balanced_scene",
) -> CoordinateQueryDataset:
    config = loaded.config
    experiment = config["experiment"]
    kwargs = dict(
        points_per_map=1,
        query_mode="full_grid",
        seed=int(config["project"]["seed"]),
        selection_seed=int(config["training"].get("data_seed", config["project"]["seed"])),
        pathloss_channel=int(config["data"]["pathloss_channel"]),
        geometry_cache_root=loaded.cache_root,
        use_los=bool(experiment["use_los"]),
        use_depth=bool(experiment["use_depth"]),
        depth_feature=str(experiment["depth_feature"]),
        exclude_building_occupied=bool(config["dataset"]["exclude_building_occupied"]),
        exclude_target_values=config["dataset"]["exclude_target_values"],
        target_exclusion_tolerance=float(config["dataset"]["target_exclusion_tolerance"]),
        include_environment_map=True,
    )
    if records is None:
        return CoordinateQueryDataset(
            manifest_path=test_manifest(loaded),
            max_records=max_maps,
            selection_strategy=selection_strategy,
            **kwargs,
        )
    return CoordinateQueryDataset(records=records, **kwargs)


@torch.inference_mode()
def predict_sample(loaded: LoadedHybrid, sample: dict[str, torch.Tensor]) -> np.ndarray:
    coordinates = sample["coordinate_features"]
    geometry = sample.get("geometry_features")
    environment = sample["environment_map"].unsqueeze(0).to(loaded.device)
    encoded = loaded.model.encode_environment(environment)
    predictions: list[torch.Tensor] = []
    for start in range(0, len(coordinates), loaded.query_batch_size):
        end = min(start + loaded.query_batch_size, len(coordinates))
        block_geometry = (
            geometry[start:end].unsqueeze(0).to(loaded.device)
            if geometry is not None
            else None
        )
        output = loaded.model.forward_with_encoded(
            coordinates[start:end].unsqueeze(0).to(loaded.device),
            block_geometry,
            encoded,
            environment.shape[-2:],
        )
        predictions.append((output[0, :, 0] * loaded.target_scale).cpu())
    return torch.cat(predictions).numpy()


def build_missing_geometry(loaded: LoadedHybrid, records: Iterable[MapRecord]) -> tuple[int, int]:
    geometry = loaded.config["geometry"]
    samples_per_unit = float(geometry["samples_per_unit"])
    chunk_size = int(geometry["ray_chunk_size"])
    built = skipped = 0
    seen: set[tuple[int, float, float, float]] = set()
    for record in records:
        key = (record.scene_id, record.tx_x, record.tx_y, record.rx_height)
        if key in seen:
            continue
        seen.add(key)
        destination = geometry_cache_path(loaded.cache_root, *key)
        if destination.is_file():
            skipped += 1
            continue
        if record.building_height_path is None:
            raise ValueError(f"Building map missing for {record.pathloss_path}")
        with Image.open(record.building_height_path) as image:
            building = np.asarray(image, dtype=np.float32)
        result = build_full_grid_geometry(
            building,
            (record.tx_x, record.tx_y, record.tx_z),
            record.rx_height,
            samples_per_unit=samples_per_unit,
            chunk_size=chunk_size,
        )
        save_geometry_cache(
            destination,
            result,
            building.shape,
            {
                "scene_id": record.scene_id,
                "tx_position": (record.tx_x, record.tx_y, record.tx_z),
                "rx_z": record.rx_height,
                "building_height_path": str(record.building_height_path.resolve()),
                "samples_per_unit": samples_per_unit,
                "geometry_sampling": "nearest_native_raster",
                "coordinate_convention": "row=(H-1)-X,col=Y",
                "purpose": "twin_state_transition_validation",
            },
        )
        built += 1
        print(f"built transition cache {destination}")
    return built, skipped


def require_geometry(loaded: LoadedHybrid, records: Iterable[MapRecord]) -> None:
    missing = []
    for record in records:
        path = geometry_cache_path(
            loaded.cache_root,
            record.scene_id,
            record.tx_x,
            record.tx_y,
            record.rx_height,
        )
        if not path.is_file():
            missing.append(path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} geometry caches. Re-run with --build-missing-caches.\n{preview}"
        )


def _record_key(record: MapRecord) -> tuple[int, float, float, float]:
    return record.scene_id, record.tx_x, record.tx_y, record.rx_height


def select_state_records(
    manifest: Path,
    *,
    seed: int,
    transition_scenes: int,
    planning_scenes: int,
    candidates: int,
) -> tuple[list[tuple[str, MapRecord, MapRecord]], dict[tuple[int, float], list[MapRecord]]]:
    """Select deterministic, scene-balanced state changes from the untouched test split."""
    records = read_manifest(manifest)
    rng = random.Random(seed)
    by_height_key: dict[tuple[int, float, float], list[MapRecord]] = {}
    by_scene_height: dict[tuple[int, float], list[MapRecord]] = {}
    for record in records:
        by_height_key.setdefault((record.scene_id, record.tx_x, record.tx_y), []).append(record)
        by_scene_height.setdefault((record.scene_id, record.rx_height), []).append(record)

    scene_ids = sorted({record.scene_id for record in records})
    rng.shuffle(scene_ids)
    transitions: list[tuple[str, MapRecord, MapRecord]] = []
    used_height_scenes: set[int] = set()
    for scene_id in scene_ids:
        candidate_keys = [key for key in by_height_key if key[0] == scene_id]
        rng.shuffle(candidate_keys)
        for key in candidate_keys:
            group = sorted(by_height_key[key], key=lambda value: value.rx_height)
            if len({item.rx_height for item in group}) < 2:
                continue
            transitions.append(("rx_height", group[0], group[-1]))
            used_height_scenes.add(scene_id)
            break
        if len(used_height_scenes) >= transition_scenes:
            break

    used_tx_scenes: set[int] = set()
    for scene_id in scene_ids:
        heights = sorted(key[1] for key in by_scene_height if key[0] == scene_id)
        if not heights:
            continue
        group = list(by_scene_height[(scene_id, heights[0])])
        rng.shuffle(group)
        unique: dict[tuple[float, float], MapRecord] = {}
        for record in group:
            unique.setdefault((record.tx_x, record.tx_y), record)
        chosen = list(unique.values())
        if len(chosen) < 2:
            continue
        transitions.append(("tx_position", chosen[0], chosen[1]))
        used_tx_scenes.add(scene_id)
        if len(used_tx_scenes) >= transition_scenes:
            break

    planning: dict[tuple[int, float], list[MapRecord]] = {}
    for scene_id in scene_ids:
        heights = sorted(key[1] for key in by_scene_height if key[0] == scene_id)
        if not heights:
            continue
        key = (scene_id, heights[0])
        group = list(by_scene_height[key])
        rng.shuffle(group)
        unique = {}
        for record in group:
            unique.setdefault((record.tx_x, record.tx_y), record)
        chosen = list(unique.values())[:candidates]
        if len(chosen) >= 2:
            planning[key] = chosen
        if len(planning) >= planning_scenes:
            break
    return transitions, planning


def unique_records(
    transitions: Sequence[tuple[str, MapRecord, MapRecord]],
    planning: dict[tuple[int, float], list[MapRecord]],
) -> list[MapRecord]:
    values: dict[tuple[int, float, float, float], MapRecord] = {}
    for _, before, after in transitions:
        values[_record_key(before)] = before
        values[_record_key(after)] = after
    for group in planning.values():
        for record in group:
            values[_record_key(record)] = record
    return list(values.values())


def delta_scores(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted_n = predicted.astype(np.float64) / 127.5
    target_n = target.astype(np.float64) / 127.5
    error = predicted_n - target_n
    mse = float(np.mean(np.square(error)))
    power = max(float(np.mean(np.square(target_n))), 1e-12)
    if len(target_n) > 1 and np.std(predicted_n) > 0 and np.std(target_n) > 0:
        correlation = float(np.corrcoef(predicted_n, target_n)[0, 1])
    else:
        correlation = math.nan
    active = np.abs(target_n) > 1e-6
    sign_agreement = (
        float(np.mean(np.sign(predicted_n[active]) == np.sign(target_n[active])))
        if active.any()
        else math.nan
    )
    return {
        "delta_rmse": math.sqrt(mse),
        "delta_nmse": mse / power,
        "delta_mae": float(np.mean(np.abs(error))),
        "delta_correlation": correlation,
        "delta_sign_agreement": sign_agreement,
    }


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def map_scores(prediction: np.ndarray, target: np.ndarray, support: np.ndarray) -> dict[str, float]:
    score = normalized_metrics(prediction[support], target[support])
    raw = metrics(prediction[support], target[support], 255.0)
    return {**score, **{f"raw_{key}": value for key, value in raw.items()}}
