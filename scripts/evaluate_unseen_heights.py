"""Evaluate HybridGeo on receiver heights excluded from model training.

The experiment discovers raw UrbanRadio3D maps at the requested heights, keeps
only the original scene-disjoint test scenes, and selects the same
(scene, transmitter) anchors at every height.  No target pixels from these
heights are used for fitting, checkpoint selection, or calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.coordinate_query_dataset import MapRecord
from datasets.raw_urbanradio3d import ManifestBuilder, RawDataConfig
from scripts.twin_validation_common import (
    build_missing_geometry,
    load_hybrid,
    make_dataset,
    map_scores,
    predict_sample,
    require_geometry,
    write_csv,
    write_json,
)
from utils.config import load_config


DEFAULT_CONFIGS = (
    Path("configs/hybrid_geo.yaml"),
    Path("configs/hybrid_geo_seed43.yaml"),
    Path("configs/hybrid_geo_seed44.yaml"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, default=list(DEFAULT_CONFIGS))
    parser.add_argument("--heights", nargs="+", type=int, default=[5, 6, 7, 8])
    parser.add_argument("--anchors", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/unseen_height_generalization"))
    parser.add_argument("--build-missing-caches", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _test_scenes(config: dict, config_path: Path) -> set[int]:
    data_root = Path(str(config["data"]["output_root"]))
    if not data_root.is_absolute():
        data_root = (config_path.parent / data_root).resolve()
    path = data_root / "splits" / "test_scenes.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Scene split missing: {path}")
    return {int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def discover_unseen_records(
    config_path: Path,
    heights: Iterable[int],
    *,
    anchors: int,
    seed: int,
) -> list[MapRecord]:
    config, resolved = load_config(config_path)
    requested = tuple(sorted(set(int(value) for value in heights)))
    trained = set(int(value) for value in config["data"]["rx_heights"])
    overlap = trained.intersection(requested)
    if overlap:
        raise ValueError(f"Requested unseen heights overlap training heights: {sorted(overlap)}")
    raw_mapping = dict(config["data"])
    raw_mapping["rx_heights"] = list(requested)
    raw = RawDataConfig.from_mapping(raw_mapping, resolved.parent)
    builder = ManifestBuilder(raw)
    builder.validate_roots()
    buildings = builder.index_buildings()
    held_out_scenes = _test_scenes(config, resolved)

    grouped: dict[tuple[int, float, float], dict[int, MapRecord]] = {}
    for item in builder.iter_maps():
        if item.scene_id not in held_out_scenes:
            continue
        building = buildings.get(item.scene_id)
        if building is None:
            raise ValueError(f"Building raster missing for scene {item.scene_id}")
        key = (item.scene_id, float(item.tx_x), float(item.tx_y))
        grouped.setdefault(key, {})[item.rx_height] = MapRecord(
            scene_id=item.scene_id,
            tx_x=float(item.tx_x),
            tx_y=float(item.tx_y),
            tx_z=float(item.tx_z),
            rx_height=float(item.rx_height),
            pathloss_path=item.pathloss_path,
            building_height_path=building,
        )
    complete = {key: values for key, values in grouped.items() if set(values) == set(requested)}
    if not complete:
        raise ValueError(f"No common test anchors contain every requested height {requested}")

    by_scene: dict[int, list[tuple[int, float, float]]] = {}
    for key in complete:
        by_scene.setdefault(key[0], []).append(key)
    rng = random.Random(seed)
    scene_ids = sorted(by_scene)
    rng.shuffle(scene_ids)
    for keys in by_scene.values():
        rng.shuffle(keys)
    selected: list[tuple[int, float, float]] = []
    round_index = 0
    while len(selected) < min(anchors, len(complete)):
        added = False
        for scene_id in scene_ids:
            values = by_scene[scene_id]
            if round_index < len(values):
                selected.append(values[round_index])
                added = True
                if len(selected) >= anchors:
                    break
        if not added:
            break
        round_index += 1
    records = [complete[key][height] for key in selected for height in requested]
    if len(records) != len(selected) * len(requested):
        raise AssertionError("Paired unseen-height selection is incomplete")
    return records


def _manifest_rows(records: list[MapRecord]) -> list[dict[str, object]]:
    return [
        {
            "scene_id": record.scene_id,
            "tx_x": record.tx_x,
            "tx_y": record.tx_y,
            "tx_z": record.tx_z,
            "rx_height": record.rx_height,
            "pathloss_path": str(record.pathloss_path),
            "building_height_path": str(record.building_height_path),
        }
        for record in records
    ]


def _seed_from_loaded(loaded) -> int:
    return int(loaded.config["project"]["seed"])


def main() -> None:
    args = parse_args()
    if args.anchors <= 0:
        raise ValueError("--anchors must be positive")
    if args.smoke_test:
        args.anchors = 1
        args.heights = args.heights[:1]
        args.configs = args.configs[:1]
        args.output = args.output.with_name(args.output.name + "_smoke")

    records = discover_unseen_records(
        args.configs[0], args.heights, anchors=args.anchors, seed=args.selection_seed
    )
    output = args.output.expanduser().resolve()
    write_csv(output / "selected_manifest.csv", _manifest_rows(records))

    per_map: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    for config_path in args.configs:
        loaded = load_hybrid(config_path, device_name=args.device)
        if args.build_missing_caches:
            built, reused = build_missing_geometry(loaded, records)
        else:
            require_geometry(loaded, records)
            built, reused = 0, len(records)
        dataset = make_dataset(loaded, records=records)
        seed = _seed_from_loaded(loaded)
        checkpoints.append({
            "seed": seed,
            "checkpoint": str(loaded.checkpoint_path),
            "checkpoint_epoch": loaded.checkpoint_epoch,
            "geometry_caches_built": built,
            "geometry_caches_reused": reused,
        })
        for index in range(len(dataset)):
            sample = dataset[index]
            record = dataset.records[index]
            prediction = predict_sample(loaded, sample)
            target = sample["target"][:, 0].numpy()
            support = sample["valid_mask"][:, 0].numpy().astype(bool)
            per_map.append({
                "seed": seed,
                "map_index": index,
                "scene_id": record.scene_id,
                "tx_x": record.tx_x,
                "tx_y": record.tx_y,
                "rx_height": record.rx_height,
                "points": int(support.sum()),
                **map_scores(prediction, target, support),
            })
            print(
                f"unseen-height seed={seed} [{index + 1}/{len(dataset)}] "
                f"h={record.rx_height:g} rmse={per_map[-1]['rmse']:.5f}"
            )

    metric_names = ("rmse", "nmse", "mae", "psnr", "raw_rmse", "raw_nmse", "raw_mae", "raw_psnr")
    per_seed_height: list[dict[str, object]] = []
    for seed in sorted({int(row["seed"]) for row in per_map}):
        for height in sorted({float(row["rx_height"]) for row in per_map}):
            group = [row for row in per_map if int(row["seed"]) == seed and float(row["rx_height"]) == height]
            if not group:
                continue
            result: dict[str, object] = {"seed": seed, "rx_height": height, "maps": len(group)}
            for metric in metric_names:
                values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
                result[f"{metric}_mean"] = float(values.mean())
                result[f"{metric}_map_std"] = float(values.std(ddof=0))
            per_seed_height.append(result)

    aggregate: list[dict[str, object]] = []
    for height in sorted({float(row["rx_height"]) for row in per_seed_height}):
        group = [row for row in per_seed_height if float(row["rx_height"]) == height]
        result = {"rx_height": height, "seeds": len(group), "maps_per_seed": int(group[0]["maps"])}
        for metric in metric_names:
            values = np.asarray([float(row[f"{metric}_mean"]) for row in group], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_seed_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(result)

    write_csv(output / "metrics_per_map.csv", per_map)
    write_csv(output / "metrics_per_seed_height.csv", per_seed_height)
    write_csv(output / "metrics_by_height.csv", aggregate)
    write_json(output / "summary.json", {
        "protocol": "receiver_heights_disjoint_from_training_heights",
        "training_heights": list(load_config(args.configs[0])[0]["data"]["rx_heights"]),
        "test_heights": sorted(set(args.heights)),
        "anchors": len(records) // len(set(args.heights)),
        "maps_per_seed": len(records),
        "selection_seed": args.selection_seed,
        "scene_split": "original held-out test scenes only",
        "paired_anchor_policy": "same scene and transmitter at every requested height",
        "checkpoints": checkpoints,
        "metrics_by_height": aggregate,
        "physical_unit_verified": False,
    })
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
