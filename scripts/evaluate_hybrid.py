"""Evaluate the hybrid raster-context continuous GeoRadioTwin."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.coordinate_query_dataset import CoordinateQueryDataset
from models.hybrid_radio_field import HybridGeoRadioField
from scripts.evaluate import METRIC_NAMES, build_model, metrics, normalized_metrics, resolve_path
from utils.config import load_config


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_geo.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-maps", type=int)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    training = config["training"]
    name = config["experiment"]["name"] + "_" + training["run_tag"]
    output = resolve_path(config["output_root"], config_path) / name
    checkpoint_path = args.checkpoint or output / "checkpoints" / "best.pt"
    checkpoint_path = checkpoint_path.resolve()
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
    evaluation = config["evaluation"]
    dataset = CoordinateQueryDataset(
        manifest_path=data_root / "splits" / "test.csv",
        points_per_map=1, query_mode="full_grid", seed=int(config["project"]["seed"]),
        selection_seed=int(training.get("data_seed", config["project"]["seed"])),
        pathloss_channel=int(config["data"]["pathloss_channel"]),
        max_records=args.max_maps or int(evaluation["test_max_maps"]),
        selection_strategy=str(evaluation["selection_strategy"]),
        geometry_cache_root=cache_root,
        use_los=bool(config["experiment"]["use_los"]),
        use_depth=bool(config["experiment"]["use_depth"]),
        depth_feature=str(config["experiment"]["depth_feature"]),
        exclude_building_occupied=bool(config["dataset"]["exclude_building_occupied"]),
        exclude_target_values=config["dataset"]["exclude_target_values"],
        target_exclusion_tolerance=float(config["dataset"]["target_exclusion_tolerance"]),
        include_environment_map=True,
    )
    rows = []
    query_batch = int(evaluation["query_batch_size"])
    scale = float(training["target_scale"])
    for index in range(len(dataset)):
        sample = dataset[index]
        coordinates = sample["coordinate_features"]
        geometry = sample.get("geometry_features")
        environment = sample["environment_map"].unsqueeze(0).to(device)
        encoded = model.encode_environment(environment)
        predictions = []
        for start in range(0, len(coordinates), query_batch):
            end = min(start + query_batch, len(coordinates))
            prediction = model.forward_with_encoded(
                coordinates[start:end].unsqueeze(0).to(device),
                geometry[start:end].unsqueeze(0).to(device) if geometry is not None else None,
                encoded,
                environment.shape[-2:],
            )
            predictions.append((prediction[0, :, 0] * scale).cpu())
        prediction = torch.cat(predictions).numpy()
        target = sample["target"][:, 0].numpy()
        valid = sample["valid_mask"][:, 0].numpy().astype(bool)
        score = normalized_metrics(prediction[valid], target[valid])
        raw = metrics(prediction[valid], target[valid], 255.0)
        record = dataset.records[index]
        rows.append({"map_index": index, "scene_id": record.scene_id, "valid_points": int(valid.sum()), **score, **{f"raw_{key}": value for key, value in raw.items()}})
        print(f"[{index + 1}/{len(dataset)}] rmse={score['rmse']:.4f} nmse={score['nmse']:.6f}")
    result = output / "evaluation" / "test"
    result.mkdir(parents=True, exist_ok=True)
    with (result / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for metric_name in (*METRIC_NAMES, *(f"raw_{name}" for name in METRIC_NAMES)):
        values = np.asarray([row[metric_name] for row in rows], dtype=np.float64)
        summary[metric_name] = {"mean": float(values.mean()), "std": float(values.std())}
    summary.update({"maps": len(rows), "checkpoint_epoch": checkpoint["epoch"], "checkpoint": str(checkpoint_path), "support": "valid_rx_only", "parameter_count": sum(parameter.numel() for parameter in model.parameters())})
    (result / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
