"""Fit a training-only log-distance regression and evaluate it on the common test set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.coordinate_query_dataset import CoordinateQueryDataset
from scripts.evaluate import metrics, normalized_metrics


def dataset(split: str, query_mode: str, points: int, max_records: int, seed: int):
    return CoordinateQueryDataset(
        manifest_path=PROJECT_ROOT / "data" / "processed" / "splits" / f"{split}.csv",
        points_per_map=points,
        query_mode=query_mode,
        seed=seed,
        selection_seed=42,
        max_records=max_records,
        selection_strategy="balanced_scene",
        exclude_building_occupied=True,
        exclude_target_values=(0,),
    )


def fit_log_distance(train_set) -> tuple[float, float, int]:
    count = 0
    sum_x = sum_y = sum_xx = sum_xy = 0.0
    for index in range(len(train_set)):
        sample = train_set[index]
        x = np.log10(np.maximum(sample["coordinate_features"][:, 9].numpy(), 1e-3)).astype(np.float64)
        y = sample["target"][:, 0].numpy().astype(np.float64)
        count += len(x)
        sum_x += float(x.sum())
        sum_y += float(y.sum())
        sum_xx += float(np.dot(x, x))
        sum_xy += float(np.dot(x, y))
    denominator = count * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-12:
        raise RuntimeError("Degenerate log-distance design matrix")
    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / count
    return intercept, slope, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-maps", type=int, default=4096)
    parser.add_argument("--fit-points-per-map", type=int, default=4096)
    parser.add_argument("--test-maps", type=int, default=128)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    train_set = dataset(
        "train", "fixed_random",
        128 if args.smoke_test else args.fit_points_per_map,
        4 if args.smoke_test else args.train_maps,
        args.seed,
    )
    intercept, slope, fit_points = fit_log_distance(train_set)
    test_set = dataset(
        "test", "full_grid", 1,
        2 if args.smoke_test else args.test_maps,
        args.seed,
    )
    rows = []
    for index in range(len(test_set)):
        sample = test_set[index]
        distance = sample["coordinate_features"][:, 9].numpy()
        prediction = intercept + slope * np.log10(np.maximum(distance, 1e-3))
        target = sample["target"][:, 0].numpy()
        valid = sample["valid_mask"][:, 0].numpy().astype(bool)
        score = normalized_metrics(prediction[valid], target[valid])
        raw = metrics(prediction[valid], target[valid], 255.0)
        rows.append({
            "map_index": index, "scene_id": test_set.records[index].scene_id,
            "valid_points": int(valid.sum()), **score,
            **{f"raw_{key}": value for key, value in raw.items()},
        })
        print(f"[{index + 1}/{len(test_set)}] rmse={score['rmse']:.4f}")
    suffix = "_smoke" if args.smoke_test else ""
    output = PROJECT_ROOT / "outputs" / f"log_distance_fitted_seed{args.seed}{suffix}" / "evaluation" / "test"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for name in ("rmse", "nmse", "mae", "psnr", "raw_rmse", "raw_nmse", "raw_mae", "raw_psnr"):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summary[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    summary.update({
        "method_label": "Fitted log-distance regression",
        "fit_split": "scene-disjoint train only",
        "fit_points": fit_points,
        "intercept": intercept,
        "log10_distance_slope": slope,
        "seed": args.seed,
        "data_seed": 42,
        "maps": len(rows),
        "support": "valid_rx_only",
        "primary_metric_domain": "normalized_minus1_to_1",
        "normalization": "x / 127.5 - 1.0",
        "parameter_count": 2,
    })
    (output / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

