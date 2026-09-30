"""Prepare held-out building interventions and external-simulator job manifests.

This script edits building rasters only.  It never creates PathLoss labels.
Every edited PathLoss file named in the generated manifest must be produced by
the same external ray tracer or measurement pipeline used for the base state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.coordinate_query_dataset import MapRecord, read_manifest
from scripts.evaluate import resolve_path
from utils.config import load_config


MANIFEST_FIELDS = (
    "pair_id", "scene_id", "intervention", "tx_x", "tx_y", "tx_z", "rx_height",
    "base_building_path", "base_pathloss_path", "edited_building_path", "edited_pathloss_path",
    "simulator_name", "simulator_version", "simulator_config_sha256", "intervention_metadata_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_geo.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/counterfactual/generated"))
    parser.add_argument("--manifest", type=Path, default=Path("data/counterfactual/manifest.csv"))
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--tx-per-scene", type=int, default=1)
    parser.add_argument("--rx-heights", nargs="+", type=float, default=[1.0, 4.0])
    parser.add_argument(
        "--interventions", nargs="+",
        choices=("building_removal", "height_increase", "building_addition"),
        default=["building_removal", "height_increase", "building_addition"],
    )
    parser.add_argument("--height-delta", type=int, default=5)
    parser.add_argument("--addition-size", type=int, default=12)
    parser.add_argument("--addition-height", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros(mask.shape, dtype=np.bool_)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for start_row, start_col in np.argwhere(mask):
        if visited[start_row, start_col]:
            continue
        queue = deque([(int(start_row), int(start_col))])
        visited[start_row, start_col] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            row, col = queue.popleft()
            pixels.append((row, col))
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + drow, col + dcol
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        components.append(np.asarray(pixels, dtype=np.int64))
    return components


def _largest_building(building: np.ndarray) -> np.ndarray:
    components = _components(building > 0)
    if not components:
        raise ValueError("Building raster contains no occupied component")
    return max(components, key=len)


def _empty_patch(building: np.ndarray, size: int, forbidden: set[tuple[int, int]]) -> tuple[int, int]:
    if size <= 0 or size > min(building.shape):
        raise ValueError("Invalid --addition-size")
    occupied = (building > 0).astype(np.int64)
    integral = np.pad(occupied, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    candidates: list[tuple[float, int, int]] = []
    center = np.asarray(building.shape, dtype=np.float64) / 2.0
    for row in range(2, building.shape[0] - size - 2):
        for col in range(2, building.shape[1] - size - 2):
            total = integral[row + size, col + size] - integral[row, col + size] - integral[row + size, col] + integral[row, col]
            if total != 0:
                continue
            if any(row <= fr < row + size and col <= fc < col + size for fr, fc in forbidden):
                continue
            distance = float(np.linalg.norm(np.asarray((row + size / 2, col + size / 2)) - center))
            candidates.append((distance, row, col))
    if not candidates:
        raise ValueError(f"No empty {size}x{size} patch available for building addition")
    _, row, col = min(candidates)
    return row, col


def _edit_building(
    building: np.ndarray,
    intervention: str,
    *,
    height_delta: int,
    addition_size: int,
    addition_height: int,
    forbidden: set[tuple[int, int]],
) -> tuple[np.ndarray, dict[str, object]]:
    edited = building.copy()
    if intervention in {"building_removal", "height_increase"}:
        pixels = _largest_building(building)
        rows, cols = pixels[:, 0], pixels[:, 1]
        before = edited[rows, cols].astype(np.int64)
        if intervention == "building_removal":
            edited[rows, cols] = 0
        else:
            edited[rows, cols] = np.clip(before + height_delta, 0, 255).astype(np.uint8)
        metadata = {
            "component_pixels": int(len(pixels)),
            "bbox_row_col": [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())],
            "height_before_min_max": [int(before.min()), int(before.max())],
            "height_after_min_max": [int(edited[rows, cols].min()), int(edited[rows, cols].max())],
        }
    else:
        row, col = _empty_patch(building, addition_size, forbidden)
        edited[row:row + addition_size, col:col + addition_size] = np.uint8(addition_height)
        metadata = {
            "component_pixels": int(addition_size * addition_size),
            "bbox_row_col": [row, col, row + addition_size - 1, col + addition_size - 1],
            "height_before_min_max": [0, 0],
            "height_after_min_max": [addition_height, addition_height],
        }
    changed = edited != building
    if not changed.any():
        raise ValueError(f"Intervention {intervention} did not change the building raster")
    metadata.update({
        "intervention": intervention,
        "changed_pixels": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
    })
    return edited, metadata


def _select_records(records: list[MapRecord], scenes: int, tx_per_scene: int, heights: set[float], seed: int):
    by_scene_tx: dict[tuple[int, float, float], dict[float, MapRecord]] = {}
    for record in records:
        if record.rx_height in heights:
            by_scene_tx.setdefault((record.scene_id, record.tx_x, record.tx_y), {})[record.rx_height] = record
    complete = {key: value for key, value in by_scene_tx.items() if set(value) == heights}
    by_scene: dict[int, list[tuple[int, float, float]]] = {}
    for key in complete:
        by_scene.setdefault(key[0], []).append(key)
    rng = random.Random(seed)
    scene_ids = sorted(by_scene)
    rng.shuffle(scene_ids)
    selected = {}
    for scene_id in scene_ids:
        choices = by_scene[scene_id]
        rng.shuffle(choices)
        if len(choices) >= tx_per_scene:
            selected[scene_id] = choices[:tx_per_scene]
        if len(selected) >= scenes:
            break
    if len(selected) < scenes:
        raise ValueError(f"Only {len(selected)} scenes support the requested paired heights")
    return selected, complete


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.scenes = 1
        args.tx_per_scene = 1
        args.rx_heights = args.rx_heights[:1]
        args.interventions = args.interventions[:1]
        args.output_root = args.output_root.with_name(args.output_root.name + "_smoke")
        args.manifest = args.manifest.with_name(args.manifest.stem + "_smoke.csv")
    if args.scenes <= 0 or args.tx_per_scene <= 0:
        raise ValueError("--scenes and --tx-per-scene must be positive")

    config, resolved = load_config(args.config)
    data_root = resolve_path(config["data"]["output_root"], resolved)
    records = read_manifest(data_root / "splits" / "test.csv")
    selected, complete = _select_records(
        records, args.scenes, args.tx_per_scene, set(args.rx_heights), args.seed
    )
    output_root = args.output_root.expanduser().resolve()
    building_root = output_root / "buildings"
    label_root = output_root / "pathloss_labels"
    metadata_root = output_root / "metadata"
    for path in (building_root, label_root, metadata_root):
        path.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    jobs: list[dict[str, object]] = []
    for scene_id, anchor_keys in selected.items():
        representative = complete[anchor_keys[0]][args.rx_heights[0]]
        assert representative.building_height_path is not None
        with Image.open(representative.building_height_path) as image:
            building = np.asarray(image.convert("L"), dtype=np.uint8)
        forbidden = {
            (int(round((building.shape[0] - 1) - key[1])), int(round(key[2])))
            for key in anchor_keys
        }
        for intervention in args.interventions:
            edited, metadata = _edit_building(
                building,
                intervention,
                height_delta=args.height_delta,
                addition_size=args.addition_size,
                addition_height=args.addition_height,
                forbidden=forbidden,
            )
            edited_path = building_root / f"scene_{scene_id:04d}_{intervention}.png"
            metadata_path = metadata_root / f"scene_{scene_id:04d}_{intervention}.json"
            if (edited_path.exists() or metadata_path.exists()) and not args.force:
                raise FileExistsError(f"Refusing to overwrite {edited_path}; use --force")
            Image.fromarray(edited).save(edited_path)
            metadata.update({
                "scene_id": scene_id,
                "base_building_path": str(representative.building_height_path.resolve()),
                "edited_building_path": str(edited_path),
                "base_building_sha256": _sha256(representative.building_height_path),
                "edited_building_sha256": _sha256(edited_path),
                "generation_seed": args.seed,
            })
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            for key in anchor_keys:
                for rx_height in args.rx_heights:
                    record = complete[key][rx_height]
                    pair_id = f"s{scene_id:04d}_x{record.tx_x:g}_y{record.tx_y:g}_h{rx_height:g}_{intervention}"
                    edited_label = label_root / f"{pair_id}.png"
                    row = {
                        "pair_id": pair_id,
                        "scene_id": scene_id,
                        "intervention": intervention,
                        "tx_x": record.tx_x,
                        "tx_y": record.tx_y,
                        "tx_z": record.tx_z,
                        "rx_height": rx_height,
                        "base_building_path": str(record.building_height_path.resolve()),
                        "base_pathloss_path": str(record.pathloss_path.resolve()),
                        "edited_building_path": str(edited_path),
                        "edited_pathloss_path": str(edited_label),
                        "simulator_name": "FILL_REQUIRED",
                        "simulator_version": "FILL_REQUIRED",
                        "simulator_config_sha256": "FILL_REQUIRED",
                        "intervention_metadata_path": str(metadata_path),
                    }
                    manifest_rows.append(row)
                    jobs.append({
                        "pair_id": pair_id,
                        "scene_id": scene_id,
                        "tx_x": record.tx_x,
                        "tx_y": record.tx_y,
                        "tx_z": record.tx_z,
                        "rx_height": rx_height,
                        "building_path": str(edited_path),
                        "output_pathloss_path": str(edited_label),
                    })

    manifest = args.manifest.expanduser().resolve()
    if manifest.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {manifest}; use --force")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (output_root / "simulator_jobs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=jobs[0].keys())
        writer.writeheader()
        writer.writerows(jobs)
    (output_root / "protocol.json").write_text(json.dumps({
        "scene_split": "original held-out test scenes only",
        "interventions": args.interventions,
        "scenes": len(selected),
        "transmitters_per_scene": args.tx_per_scene,
        "receiver_heights": args.rx_heights,
        "pairs": len(manifest_rows),
        "label_policy": "edited labels must be independently generated with identical simulator settings",
        "manifest": str(manifest),
    }, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest_rows)} simulator jobs. Fill labels and provenance in {manifest}.")


if __name__ == "__main__":
    main()
