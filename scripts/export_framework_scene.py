#!/usr/bin/env python
"""Export one protocol-selected real scene for the framework figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.coordinate_query_dataset import CoordinateQueryDataset
from models.hybrid_radio_field import HybridGeoRadioField
from scripts.evaluate import build_model, resolve_path
from utils.config import load_config


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_geo.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selected-index", type=int, default=4,
                        help="Index within the canonical 128-map balanced test selection")
    parser.add_argument("--output", type=Path,
                        default=Path("paper_artifacts/source_data/framework_scene71.npz"))
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    training = config["training"]
    output_root = resolve_path(config["output_root"], config_path)
    run_name = config["experiment"]["name"] + "_" + training["run_tag"]
    checkpoint_path = (args.checkpoint or output_root / run_name / "checkpoints" / "best.pt").resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = build_model(config)
    model = HybridGeoRadioField(
        base,
        int(config["model"]["environment_dim"]),
        int(config["model"]["ray_samples"]),
        str(config["model"].get("context_mode", "ray")),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data_root = resolve_path(config["data"]["output_root"], config_path)
    cache_root = resolve_path(config["geometry"]["cache_root"], config_path)
    dataset = CoordinateQueryDataset(
        manifest_path=data_root / "splits" / "test.csv",
        points_per_map=1,
        query_mode="full_grid",
        seed=int(config["project"]["seed"]),
        selection_seed=int(training.get("data_seed", config["project"]["seed"])),
        pathloss_channel=int(config["data"]["pathloss_channel"]),
        max_records=int(config["evaluation"]["test_max_maps"]),
        selection_strategy=str(config["evaluation"]["selection_strategy"]),
        geometry_cache_root=cache_root,
        use_los=True,
        use_depth=True,
        depth_feature=str(config["experiment"]["depth_feature"]),
        exclude_building_occupied=bool(config["dataset"]["exclude_building_occupied"]),
        exclude_target_values=config["dataset"]["exclude_target_values"],
        target_exclusion_tolerance=float(config["dataset"]["target_exclusion_tolerance"]),
        include_environment_map=True,
    )
    if not 0 <= args.selected_index < len(dataset):
        raise IndexError(f"selected-index must be in [0, {len(dataset)-1}]")
    sample = dataset[args.selected_index]
    record = dataset.records[args.selected_index]
    height, width = (int(value) for value in sample["environment_map"].shape[-2:])

    environment = sample["environment_map"].unsqueeze(0).to(device)
    feature_map = model.encode_environment(environment)
    coordinates = sample["coordinate_features"]
    geometry = sample["geometry_features"]
    query_batch = int(config["evaluation"]["query_batch_size"])
    predicted = []
    scale = float(training["target_scale"])
    for start in range(0, len(coordinates), query_batch):
        end = min(start + query_batch, len(coordinates))
        values = model.forward_with_encoded(
            coordinates[start:end].unsqueeze(0).to(device),
            geometry[start:end].unsqueeze(0).to(device),
            feature_map,
            (height, width),
        )
        predicted.append((values[0, :, 0] * scale).cpu().numpy())

    rows_cols = sample["pixel_row_col"].numpy()
    target_map = np.full((height, width), np.nan, dtype=np.float32)
    prediction_map = np.full((height, width), np.nan, dtype=np.float32)
    valid_map = np.zeros((height, width), dtype=bool)
    los_map = np.full((height, width), np.nan, dtype=np.float32)
    depth_map = np.full((height, width), np.nan, dtype=np.float32)
    rr, cc = rows_cols[:, 0], rows_cols[:, 1]
    target_map[rr, cc] = sample["target"][:, 0].numpy()
    prediction_map[rr, cc] = np.concatenate(predicted)
    valid_map[rr, cc] = sample["valid_mask"][:, 0].numpy().astype(bool)
    los_map[rr, cc] = geometry[:, 0].numpy()
    depth_map[rr, cc] = geometry[:, 1].numpy()

    with Image.open(record.building_height_path) as image:
        building = np.asarray(image, dtype=np.float32)

    valid_rows, valid_cols = np.nonzero(valid_map)
    target_row, target_col = int(0.18 * height), int(0.82 * width)
    nearest = np.argmin((valid_rows-target_row)**2 + (valid_cols-target_col)**2)
    rx_row, rx_col = int(valid_rows[nearest]), int(valid_cols[nearest])
    rx_x, rx_y = float((height - 1) - rx_row), float(rx_col)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "scene_id": int(record.scene_id),
        "selected_index": int(args.selected_index),
        "selection_protocol": "balanced_scene test selection, seed 42, 128 maps",
        "tx_x": float(record.tx_x),
        "tx_y": float(record.tx_y),
        "tx_z": float(record.tx_z),
        "rx_height": float(record.rx_height),
        "illustrative_rx_x": rx_x,
        "illustrative_rx_y": rx_y,
        "illustrative_rx_row": rx_row,
        "illustrative_rx_col": rx_col,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "physical_unit_verified": False,
    }
    np.savez_compressed(
        args.output,
        building=building,
        target=target_map,
        prediction=prediction_map,
        valid_mask=valid_map,
        los=los_map,
        depth=depth_map,
        metadata=np.asarray(json.dumps(metadata)),
    )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
