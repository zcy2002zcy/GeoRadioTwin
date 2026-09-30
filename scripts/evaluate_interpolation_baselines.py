"""Evaluate public-standard interpolation baselines with identical sparse masks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dense_baseline_dataset import DenseBaselineDataset
from scripts.evaluate import metrics, normalized_metrics


def interpolate(target: np.ndarray, observed: np.ndarray, query: np.ndarray, method: str) -> np.ndarray:
    points = np.column_stack(np.nonzero(observed)).astype(np.float64)
    values = target[observed].astype(np.float64)
    query_points = np.column_stack(np.nonzero(query)).astype(np.float64)
    if method == "nearest":
        _, indices = cKDTree(points).query(query_points, k=1)
        result = values[indices]
    elif method == "idw":
        distances, indices = cKDTree(points).query(query_points, k=min(8, len(points)))
        if distances.ndim == 1:
            distances = distances[:, None]; indices = indices[:, None]
        weights = 1.0 / np.maximum(distances, 1e-6) ** 2
        result = (weights * values[indices]).sum(axis=1) / weights.sum(axis=1)
    elif method == "linear":
        result = griddata(points, values, query_points, method="linear")
        missing = ~np.isfinite(result)
        if missing.any():
            _, indices = cKDTree(points).query(query_points[missing], k=1)
            result[missing] = values[indices]
    else:
        raise ValueError(method)
    return result.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=float, choices=(0.01, 0.05, 0.10), required=True)
    parser.add_argument("--methods", nargs="+", choices=("nearest", "idw", "linear"), default=("nearest", "idw", "linear"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-maps", type=int, default=128)
    args = parser.parse_args()
    dataset = DenseBaselineDataset(
        PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv",
        max_records=args.max_maps, selection_strategy="balanced_scene", seed=args.seed,
        sample_rate=args.sample_rate, training=False,
    )
    rate_label = str(int(round(args.sample_rate * 100)))
    for method in args.methods:
        rows = []
        for index in range(len(dataset)):
            sample = dataset[index]
            target = sample["target_raw"].numpy()[0]
            valid = sample["valid_mask"].numpy()[0].astype(bool)
            observed = sample["observation_mask"].numpy()[0].astype(bool)
            support = valid & ~observed
            prediction = interpolate(target, observed, support, method)
            truth = target[support]
            score = normalized_metrics(prediction, truth)
            raw = metrics(prediction, truth, 255.0)
            rows.append({"map_index": index, "scene_id": int(sample["scene_id"]), "valid_points": int(valid.sum()), "observed_points": int(observed.sum()), **score, **{f"raw_{k}": v for k, v in raw.items()}})
            print(f"{method} [{index + 1}/{len(dataset)}] rmse={score['rmse']:.4f}")
        output = PROJECT_ROOT / "outputs" / f"interpolation_{method}_random{rate_label}_seed{args.seed}" / "evaluation" / "test"
        output.mkdir(parents=True, exist_ok=True)
        with (output / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        summary = {}
        for name in ("rmse", "nmse", "mae", "psnr", "raw_rmse", "raw_nmse", "raw_mae", "raw_psnr"):
            values = np.asarray([row[name] for row in rows])
            summary[name] = {"mean": float(values.mean()), "std": float(values.std())}
        summary.update({
            "method": method, "sample_rate": args.sample_rate, "seed": args.seed,
            "maps": len(rows), "support": "unobserved_valid_rx_only",
            "mask_protocol": "GeoRadioTwin seeded random valid-Rx mask",
            "idw_k": 8 if method == "idw" else None,
            "idw_power": 2 if method == "idw" else None,
        })
        (output / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
