"""Evaluate three-seed HybridGeo uncertainty with validation-only calibration.

The predictive mean and standard deviation come from independently trained
model seeds.  Split-conformal scale factors are fitted on the scene-disjoint
validation split and frozen before test evaluation; test targets never tune the
intervals.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.coordinate_query_dataset import read_manifest
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
from scripts.evaluate import resolve_path


DEFAULT_CONFIGS = (
    Path("configs/hybrid_geo.yaml"),
    Path("configs/hybrid_geo_seed43.yaml"),
    Path("configs/hybrid_geo_seed44.yaml"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, default=list(DEFAULT_CONFIGS))
    parser.add_argument("--calibration-maps", type=int, default=128)
    parser.add_argument(
        "--calibration-offset", type=int, default=128,
        help="Skip the first validation anchors used by checkpoint selection",
    )
    parser.add_argument("--test-maps", type=int, default=128)
    parser.add_argument("--max-calibration-pixels", type=int, default=1_000_000)
    parser.add_argument("--max-diagnostic-pixels", type=int, default=1_000_000)
    parser.add_argument("--uncertainty-floor", type=float, default=1e-4)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/ensemble_uncertainty"))
    parser.add_argument("--build-missing-caches", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _split_records(loaded, split: str, count: int, seed: int, *, offset: int = 0):
    data_root = resolve_path(loaded.config["data"]["output_root"], loaded.config_path)
    records = read_manifest(
        data_root / "splits" / f"{split}.csv",
        max_records=count + offset,
        selection_strategy="balanced_scene",
        seed=seed,
    )
    if len(records) < count + offset:
        raise ValueError(
            f"Split {split} has only {len(records)} selectable maps; "
            f"need calibration/test offset {offset} plus count {count}"
        )
    return records[offset:offset + count]


def _subsample(values: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or len(values) <= maximum:
        return values
    rng = np.random.default_rng(seed)
    return values[rng.choice(len(values), size=maximum, replace=False)]


def _paired_subsample(arrays: tuple[np.ndarray, ...], maximum: int, seed: int) -> tuple[np.ndarray, ...]:
    length = len(arrays[0])
    if any(len(value) != length for value in arrays):
        raise ValueError("Diagnostic arrays must have the same length")
    if maximum <= 0 or length <= maximum:
        return arrays
    rng = np.random.default_rng(seed)
    indices = rng.choice(length, size=maximum, replace=False)
    return tuple(value[indices] for value in arrays)


def _conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    finite = np.sort(scores[np.isfinite(scores)])
    if len(finite) == 0:
        raise ValueError("No finite validation conformity scores")
    rank = int(math.ceil((len(finite) + 1) * coverage)) - 1
    return float(finite[min(max(rank, 0), len(finite) - 1)])


def _ensemble_prediction(loaded_models, sample) -> np.ndarray:
    return np.stack([predict_sample(loaded, sample) for loaded in loaded_models], axis=0)


def _validate_models(loaded_models) -> None:
    seeds = [int(loaded.config["project"]["seed"]) for loaded in loaded_models]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate ensemble seeds: {seeds}")
    data_seeds = {
        int(loaded.config["training"].get("data_seed", loaded.config["project"]["seed"]))
        for loaded in loaded_models
    }
    if len(data_seeds) != 1:
        raise ValueError(f"Ensemble checkpoints use different data selections: {sorted(data_seeds)}")


def main() -> None:
    args = parse_args()
    if len(args.configs) < 2:
        raise ValueError("Uncertainty evaluation requires at least two independent model seeds")
    if args.uncertainty_floor <= 0:
        raise ValueError("--uncertainty-floor must be positive")
    if args.calibration_offset < 0:
        raise ValueError("--calibration-offset must be non-negative")
    if args.smoke_test:
        args.calibration_maps = 2
        args.calibration_offset = 2
        args.test_maps = 2
        args.max_calibration_pixels = 50_000
        args.max_diagnostic_pixels = 50_000
        args.output = args.output.with_name(args.output.name + "_smoke")

    loaded_models = [load_hybrid(path, device_name=args.device) for path in args.configs]
    _validate_models(loaded_models)
    reference = loaded_models[0]
    calibration_records = _split_records(
        reference, "val", args.calibration_maps, args.selection_seed,
        offset=args.calibration_offset,
    )
    test_records = _split_records(reference, "test", args.test_maps, args.selection_seed)
    required = list(calibration_records) + list(test_records)
    if args.build_missing_caches:
        built, reused = build_missing_geometry(reference, required)
    else:
        require_geometry(reference, required)
        built, reused = 0, len(required)

    calibration_dataset = make_dataset(reference, records=calibration_records)
    calibration_errors: list[np.ndarray] = []
    calibration_scales: list[np.ndarray] = []
    for index in range(len(calibration_dataset)):
        sample = calibration_dataset[index]
        ensemble = _ensemble_prediction(loaded_models, sample)
        target = sample["target"][:, 0].numpy()
        valid = sample["valid_mask"][:, 0].numpy().astype(bool)
        mean = ensemble.mean(axis=0)
        std = ensemble.std(axis=0, ddof=1)
        calibration_errors.append(np.abs(mean[valid] - target[valid]).astype(np.float32) / 127.5)
        calibration_scales.append(std[valid].astype(np.float32) / 127.5 + args.uncertainty_floor)
        print(f"uncertainty calibration [{index + 1}/{len(calibration_dataset)}]")
    cal_error = np.concatenate(calibration_errors)
    cal_scale = np.concatenate(calibration_scales)
    cal_error, cal_scale = _paired_subsample(
        (cal_error, cal_scale), args.max_calibration_pixels, args.selection_seed + 101
    )
    conformity = cal_error / cal_scale
    q90 = _conformal_quantile(conformity, 0.90)
    q95 = _conformal_quantile(conformity, 0.95)

    test_dataset = make_dataset(reference, records=test_records)
    per_map: list[dict[str, object]] = []
    diagnostic_error: list[np.ndarray] = []
    diagnostic_sigma: list[np.ndarray] = []
    for index in range(len(test_dataset)):
        sample = test_dataset[index]
        record = test_dataset.records[index]
        ensemble = _ensemble_prediction(loaded_models, sample)
        target = sample["target"][:, 0].numpy()
        valid = sample["valid_mask"][:, 0].numpy().astype(bool)
        mean = ensemble.mean(axis=0)
        sigma = ensemble.std(axis=0, ddof=1) / 127.5 + args.uncertainty_floor
        error = np.abs(mean - target) / 127.5
        valid_error = error[valid]
        valid_sigma = sigma[valid]
        coverage90 = float(np.mean(valid_error <= q90 * valid_sigma))
        coverage95 = float(np.mean(valid_error <= q95 * valid_sigma))
        correlation = (
            float(np.corrcoef(valid_sigma, valid_error)[0, 1])
            if len(valid_error) > 1 and np.std(valid_sigma) > 0 and np.std(valid_error) > 0
            else math.nan
        )
        variance = np.square(valid_sigma)
        gaussian_nll = float(np.mean(0.5 * np.log(2.0 * np.pi * variance) + 0.5 * np.square(valid_error) / variance))
        per_map.append({
            "map_index": index,
            "scene_id": record.scene_id,
            "rx_height": record.rx_height,
            "points": int(valid.sum()),
            **map_scores(mean, target, valid),
            "mean_ensemble_std": float(np.mean(valid_sigma)),
            "error_uncertainty_correlation": correlation,
            "coverage_90": coverage90,
            "coverage_95": coverage95,
            "mean_interval_width_90": float(np.mean(2.0 * q90 * valid_sigma)),
            "mean_interval_width_95": float(np.mean(2.0 * q95 * valid_sigma)),
            "gaussian_nll": gaussian_nll,
        })
        diagnostic_error.append(valid_error.astype(np.float32))
        diagnostic_sigma.append(valid_sigma.astype(np.float32))
        print(
            f"uncertainty test [{index + 1}/{len(test_dataset)}] "
            f"rmse={per_map[-1]['rmse']:.5f} cov95={coverage95:.4f}"
        )

    all_error, all_sigma = _paired_subsample(
        (np.concatenate(diagnostic_error), np.concatenate(diagnostic_sigma)),
        args.max_diagnostic_pixels,
        args.selection_seed + 202,
    )
    uncertainty_correlation = (
        float(np.corrcoef(all_sigma, all_error)[0, 1])
        if np.std(all_sigma) > 0 and np.std(all_error) > 0
        else math.nan
    )
    risk_rows: list[dict[str, object]] = []
    uncertainty_order = np.argsort(all_sigma)
    oracle_order = np.argsort(all_error)
    for retained in (1.0, 0.9, 0.75, 0.5, 0.25):
        count = max(1, int(round(len(all_error) * retained)))
        predicted_risk = float(np.sqrt(np.mean(np.square(all_error[uncertainty_order[:count]]))))
        oracle_risk = float(np.sqrt(np.mean(np.square(all_error[oracle_order[:count]]))))
        risk_rows.append({
            "retained_fraction": retained,
            "pixels": count,
            "uncertainty_ranked_rmse": predicted_risk,
            "oracle_ranked_rmse": oracle_risk,
            "random_reference_rmse": float(np.sqrt(np.mean(np.square(all_error)))),
        })

    metric_names = (
        "rmse", "nmse", "mae", "psnr", "raw_rmse", "raw_nmse", "raw_mae", "raw_psnr",
        "mean_ensemble_std", "error_uncertainty_correlation", "coverage_90", "coverage_95",
        "mean_interval_width_90", "mean_interval_width_95", "gaussian_nll",
    )
    aggregate = {}
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in per_map], dtype=np.float64)
        aggregate[metric] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values, ddof=0))}

    output = args.output.expanduser().resolve()
    write_csv(output / "metrics_per_map.csv", per_map)
    write_csv(output / "risk_coverage.csv", risk_rows)
    write_json(output / "calibration.json", {
        "calibration_split": "scene-disjoint validation",
        "calibration_maps": len(calibration_dataset),
        "calibration_offset": args.calibration_offset,
        "checkpoint_selection_overlap": "exact validation map anchors excluded by deterministic offset",
        "calibration_pixels_used": len(conformity),
        "uncertainty_definition": "sample standard deviation across independent model seeds, normalized by 127.5",
        "uncertainty_floor": args.uncertainty_floor,
        "conformal_score": "abs(ensemble_mean-target)/(ensemble_std+floor)",
        "q90": q90,
        "q95": q95,
    })
    write_json(output / "summary.json", {
        "protocol": "three_seed_deep_ensemble_with_heldout_validation_map_conformal_style_calibration",
        "seeds": [int(loaded.config["project"]["seed"]) for loaded in loaded_models],
        "checkpoints": [str(loaded.checkpoint_path) for loaded in loaded_models],
        "test_maps": len(test_dataset),
        "support": "valid_rx_only",
        "primary_metric_domain": "normalized_minus1_to_1",
        "calibration": {"q90": q90, "q95": q95},
        "aggregate": aggregate,
        "global_error_uncertainty_correlation": uncertainty_correlation,
        "risk_coverage": risk_rows,
        "geometry_caches_built": built,
        "geometry_caches_reused": reused,
        "limitations": [
            "Three members estimate epistemic/model uncertainty only.",
            "Calibration uses additional maps from the validation scenes, so strict scene-level conformal exchangeability is not claimed.",
            "Pixelwise coverage does not imply independent spatial errors.",
            "Intervals are in the encoded normalized domain, not physical dB.",
        ],
    })
    print(json.dumps({"q90": q90, "q95": q95, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
