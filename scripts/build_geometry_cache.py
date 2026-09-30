"""Build native-grid LoS/depth caches for selected manifest maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry.cache import build_full_grid_geometry, geometry_cache_path, save_geometry_cache
from datasets.coordinate_query_dataset import read_manifest
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument(
        "--max-maps",
        type=int,
        help="Maximum unique map keys; use 0 for the complete split",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--selection-strategy",
        choices=("sequential", "balanced_scene"),
        help="Map selection policy before geometry generation",
    )
    return parser.parse_args()


def resolve_path(value: object, config_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    geometry = config["geometry"]
    split_name = args.split or str(geometry["build_split"])
    configured_limit = int(geometry["build_max_maps"])
    max_maps = configured_limit if args.max_maps is None else args.max_maps
    selection_strategy = args.selection_strategy or str(
        geometry["build_selection_strategy"]
    )
    manifest = resolve_path(config["data"]["output_root"], config_path) / "splits" / f"{split_name}.csv"
    if not manifest.is_file() or manifest.stat().st_size == 0:
        raise ValueError(f"Split manifest is missing or empty: {manifest}")
    cache_root = resolve_path(geometry["cache_root"], config_path)
    samples_per_unit = float(geometry["samples_per_unit"])
    chunk_size = int(geometry["ray_chunk_size"])

    records = read_manifest(
        manifest,
        max_records=None if max_maps == 0 else max_maps,
        selection_strategy=selection_strategy,
        seed=int(config["training"].get("data_seed", config["project"]["seed"])),
    )
    built = skipped = 0
    seen: set[tuple[int, float, float, float]] = set()
    for record in records:
        key = (record.scene_id, record.tx_x, record.tx_y, record.rx_height)
        if key in seen:
            continue
        seen.add(key)
        destination = geometry_cache_path(cache_root, *key)
        if destination.exists() and not args.overwrite:
            skipped += 1
            print(f"skip {destination}")
            continue
        if record.building_height_path is None:
            raise ValueError(f"Building path missing for {record.pathloss_path}")
        building_path = record.building_height_path
        with Image.open(building_path) as image:
            building = np.asarray(image, dtype=np.float32)
        if building.ndim != 2:
            raise ValueError(f"Building raster must be single-channel: {building_path}")
        tx = (key[1], key[2], record.tx_z)
        result = build_full_grid_geometry(
            building,
            tx,
            key[3],
            samples_per_unit=samples_per_unit,
            chunk_size=chunk_size,
        )
        metadata = {
            "scene_id": key[0],
            "tx_position": tx,
            "rx_z": key[3],
            "building_height_path": str(building_path.resolve()),
            "samples_per_unit": samples_per_unit,
            "geometry_sampling": "nearest_native_raster",
            "coordinate_convention": "row=(H-1)-X,col=Y",
        }
        save_geometry_cache(destination, result, building.shape, metadata)
        built += 1
        print(
            f"built {destination} los_ratio={result.los.mean():.4f} "
            f"depth=[{result.depth.min():.3f},{result.depth.max():.3f}]"
        )
    print(json.dumps({"built": built, "skipped": skipped, "cache_root": str(cache_root)}))


if __name__ == "__main__":
    main()
