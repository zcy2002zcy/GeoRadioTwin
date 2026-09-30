"""Evaluate the height-adapted RMTransformer-style baseline consistently."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dense_baseline_dataset import DenseBaselineDataset
from models.rmtransformer_baseline import build_rmtransformer_adapted
from scripts.evaluate import metrics, normalized_metrics
from utils.config import load_config


METHOD_LABEL = "RMTransformer-style (height-adapted)"
SOURCE_DOI = "10.1109/VTC2025-SPRING65109.2025.11174709"
METRIC_NAMES = (
    "rmse", "nmse", "mae", "psnr",
    "raw_rmse", "raw_nmse", "raw_mae", "raw_psnr",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rmtransformer_adapted.yaml")
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-maps", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{name}' must be a mapping")
    return value


def main() -> None:
    args = arguments()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config, _ = load_config(config_path)
    data = _section(config, "data")
    evaluation = _section(config, "evaluation")
    data_seed = int(args.data_seed if args.data_seed is not None else data.get("data_seed", 42))
    name = f"rmtransformer_adapted_random0_seed{args.seed}"
    if args.smoke_test:
        name += "_smoke"
    output = PROJECT_ROOT / "outputs" / name
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else output / "checkpoints" / "best.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_data = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if "model_config" not in checkpoint_data:
        raise KeyError(
            "RMTransformer checkpoint lacks model_config; refusing to guess its architecture"
        )
    model = build_rmtransformer_adapted(**checkpoint_data["model_config"])
    model.load_state_dict(checkpoint_data["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    configured_maximum = int(
        evaluation.get("test_max_maps", data.get("test_maps", 128))
    )
    maximum = int(args.max_maps) if args.max_maps is not None else configured_maximum
    if args.smoke_test:
        maximum = min(maximum, 1)
    dataset = DenseBaselineDataset(
        PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv",
        max_records=maximum,
        selection_strategy=(
            "sequential"
            if args.smoke_test
            else str(evaluation.get("selection_strategy", "balanced_scene"))
        ),
        seed=data_seed,
        sample_rate=0.0,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0 if args.smoke_test else args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    rows: list[dict[str, object]] = []
    amp_enabled = bool(_section(config, "training").get("amp", True)) and device.type == "cuda"
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                normalized_prediction = model(
                    batch["input"].to(device, non_blocking=True)
                )
            prediction = normalized_prediction.float().cpu().numpy()[0, 0] * 255.0
            target = batch["target_raw"].numpy()[0, 0]
            valid = batch["valid_mask"].numpy()[0, 0].astype(bool)
            observed = batch["observation_mask"].numpy()[0, 0].astype(bool)
            support = valid & ~observed
            if not support.any():
                raise ValueError(f"Test map {index} has no valid receiver pixels")
            encoded = normalized_metrics(prediction[support], target[support])
            raw = metrics(prediction[support], target[support], data_range=255.0)
            row: dict[str, object] = {
                "map_index": index,
                "scene_id": int(batch["scene_id"].item()),
                "valid_points": int(valid.sum()),
                "observed_points": int(observed.sum()),
                "evaluated_points": int(support.sum()),
                **encoded,
                **{f"raw_{key}": value for key, value in raw.items()},
            }
            rows.append(row)
            print(
                f"[{index + 1}/{len(dataset)}] scene={row['scene_id']} "
                f"rmse={encoded['rmse']:.4f} nmse={encoded['nmse']:.6f}"
            )
    if not rows:
        raise ValueError("Test dataset is empty")

    result = output / "evaluation" / "test"
    result.mkdir(parents=True, exist_ok=True)
    with (result / "metrics_per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {}
    for metric_name in METRIC_NAMES:
        values = np.asarray(
            [float(row[metric_name]) for row in rows], dtype=np.float64
        )
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    summary.update(
        {
            "maps": len(rows),
            "unique_scenes": len({int(row["scene_id"]) for row in rows}),
            "seed": args.seed,
            "data_seed": data_seed,
            "sample_rate": 0.0,
            "support": "valid_rx_only",
            "primary_metric_domain": "normalized_minus1_to_1",
            "normalization": "prediction and target mapped from PNG code x to x / 127.5 - 1.0",
            "raw_domain": "UrbanRadio3D PNG code 0..255; not asserted to be dB",
            "per_map_macro_average": True,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint_data["epoch"]),
            "method_label": METHOD_LABEL,
            "implementation_status": "independent_paper_guided_adaptation",
            "faithful_reproduction": False,
            "external_pretraining": False,
            "source": {"doi": SOURCE_DOI, "arxiv": "2501.05190"},
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
    )
    (result / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (result / "metrics_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "mean", "std"))
        writer.writeheader()
        for metric_name in METRIC_NAMES:
            writer.writerow({"metric": metric_name, **summary[metric_name]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
