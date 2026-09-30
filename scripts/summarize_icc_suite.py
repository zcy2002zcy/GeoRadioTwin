"""Validate and aggregate the ICC sampling-free baseline/ablation suite."""

from __future__ import annotations

import csv
import argparse
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
METRICS = ("nmse", "rmse", "mae", "psnr", "raw_rmse")

RUNS = (
    ("HybridGeo", 42, "main", "hybrid_geo_valid_rx_v1"),
    ("HybridGeo", 43, "main", "hybrid_geo_valid_rx_seed43"),
    ("HybridGeo", 44, "main", "hybrid_geo_valid_rx_seed44"),
    ("UniRM adapted", 42, "main", "unirm_adapted_random0_seed42"),
    ("UniRM adapted", 43, "main", "unirm_adapted_random0_seed43"),
    ("UniRM adapted", 44, "main", "unirm_adapted_random0_seed44"),
    ("PEFNet-inspired VIE-regularized U-Net (adapted)", 42, "main", "pefnet_inspired_adapted_stable_v2_random0_seed42"),
    ("PEFNet-inspired VIE-regularized U-Net (adapted)", 43, "main", "pefnet_inspired_adapted_stable_v2_random0_seed43"),
    ("PEFNet-inspired VIE-regularized U-Net (adapted)", 44, "main", "pefnet_inspired_adapted_stable_v2_random0_seed44"),
    ("RMTransformer-style (height-adapted)", 42, "main", "rmtransformer_adapted_random0_seed42"),
    ("RMTransformer-style (height-adapted)", 43, "main", "rmtransformer_adapted_random0_seed43"),
    ("RMTransformer-style (height-adapted)", 44, "main", "rmtransformer_adapted_random0_seed44"),
    ("RadioDiff-3D-style conditional diffusion (paper-aligned adapted v2)", 42, "main", "radiodiff3d_paper_aligned_v2_random0_seed42"),
    ("RadioDiff-3D-style conditional diffusion (paper-aligned adapted v2)", 43, "main", "radiodiff3d_paper_aligned_v2_random0_seed43"),
    ("RadioDiff-3D-style conditional diffusion (paper-aligned adapted v2)", 44, "main", "radiodiff3d_paper_aligned_v2_random0_seed44"),
    ("RadioUNet adapted", 42, "baseline", "radiounet_adapted_random0_seed42"),
    ("RadioUNet adapted", 43, "baseline", "radiounet_adapted_random0_seed43"),
    ("RadioUNet adapted", 44, "baseline", "radiounet_adapted_random0_seed44"),
    ("Fitted log-distance", 42, "baseline", "log_distance_fitted_seed42"),
    ("Coordinate only", 42, "ablation", "coord_only_valid_rx_v1"),
    ("Coordinate only", 43, "ablation", "coord_only_valid_rx_seed43"),
    ("Coordinate only", 44, "ablation", "coord_only_valid_rx_seed44"),
    ("LOS only", 42, "ablation", "coord_los_valid_rx_v1"),
    ("Depth only", 42, "ablation", "coord_depth_valid_rx_v1"),
    ("LOS+Depth / no raster", 42, "ablation", "coord_los_depth_valid_rx_v1"),
    ("LOS+Depth / no raster", 43, "ablation", "coord_los_depth_valid_rx_seed43"),
    ("LOS+Depth / no raster", 44, "ablation", "coord_los_depth_valid_rx_seed44"),
    ("Hybrid no ray aggregation", 42, "ablation", "hybrid_no_ray_valid_rx_v1"),
    ("Hybrid no ray aggregation", 43, "ablation", "hybrid_no_ray_valid_rx_seed43"),
    ("Hybrid no ray aggregation", 44, "ablation", "hybrid_no_ray_valid_rx_seed44"),
    ("Hybrid no LOS", 42, "ablation", "hybrid_no_los_valid_rx_v1"),
    ("Hybrid no depth", 42, "ablation", "hybrid_no_depth_valid_rx_v1"),
    ("Hybrid no analytic geometry", 42, "ablation", "hybrid_no_geometry_valid_rx_v1"),
    ("Hybrid no analytic geometry", 43, "ablation", "hybrid_no_geometry_valid_rx_seed43"),
    ("Hybrid no analytic geometry", 44, "ablation", "hybrid_no_geometry_valid_rx_seed44"),
)

PEFNET_LABEL = "PEFNet-inspired VIE-regularized U-Net (adapted)"
PEFNET_DOI = "10.1109/TWC.2024.3429196"
PAPER_GUIDED_PROTOCOLS = {
    "RMTransformer-style (height-adapted)": {
        "doi": "10.1109/VTC2025-SPRING65109.2025.11174709",
        "implementation_status": "independent_paper_guided_adaptation",
    },
    "RadioDiff-3D-style conditional diffusion (paper-aligned adapted v2)": {
        "doi": "10.1109/TNSE.2025.3590545",
        "implementation_status": "independent_upstream_and_paper_informed_adaptation",
    },
}


def _finite_metric(summary: dict, metric: str, path: Path) -> float:
    value = summary.get(metric)
    if not isinstance(value, dict) or "mean" not in value:
        raise ValueError(f"Missing {metric}.mean in {path}")
    mean = float(value["mean"])
    if not math.isfinite(mean):
        raise ValueError(f"Non-finite {metric}.mean in {path}: {mean}")
    return mean


def _validate_optional_common_fields(summary: dict, *, seed: int, path: Path) -> None:
    if "seed" in summary and int(summary["seed"]) != seed:
        raise ValueError(f"Seed mismatch in {path}: expected {seed}, got {summary['seed']}")
    if "data_seed" in summary and int(summary["data_seed"]) != 42:
        raise ValueError(f"Data-seed mismatch in {path}: expected 42, got {summary['data_seed']}")
    if "sample_rate" in summary and float(summary["sample_rate"]) != 0.0:
        raise ValueError(f"Sampling-free suite received sample_rate={summary['sample_rate']} in {path}")


def _validate_pefnet_run(summary: dict, *, seed: int, directory: str, path: Path) -> None:
    required = {
        "seed": seed,
        "data_seed": 42,
        "sample_rate": 0.0,
        "method_label": PEFNET_LABEL,
        "primary_metric_domain": "normalized_minus1_to_1",
        "implementation_status": "independent_inspired_adaptation",
        "faithful_reproduction": False,
        "stability_revision": "groupnorm_bf16_refinement_fp32_warmup_v2",
        "normalization_layer": "group_norm",
        "force_refinement_fp32": True,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"PEFNet-inspired protocol mismatch in {path}: "
                f"{key}={summary.get(key)!r}, expected {expected!r}"
            )
    run_root = OUTPUTS / directory
    provenance_path = run_root / "provenance.json"
    completion_path = run_root / "run_complete.json"
    for required_path in (provenance_path, completion_path):
        if not required_path.is_file():
            raise ValueError(f"Required PEFNet-inspired audit artifact missing: {required_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("faithful_reproduction") is not False:
        raise ValueError(f"PEFNet-inspired provenance must declare faithful_reproduction=false: {provenance_path}")
    if provenance.get("implementation_status") != "independent_inspired_adaptation":
        raise ValueError(f"Unexpected implementation status in {provenance_path}")
    if provenance.get("stability_revision") != "groupnorm_bf16_refinement_fp32_warmup_v2":
        raise ValueError(f"Unexpected PEFNet stability revision in {provenance_path}")
    source = provenance.get("source", {})
    if not isinstance(source, dict) or source.get("doi") != PEFNET_DOI:
        raise ValueError(f"Missing or incorrect source DOI in {provenance_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if int(completion.get("seed", -1)) != seed or int(completion.get("data_seed", -1)) != 42:
        raise ValueError(f"Completion seed metadata mismatch in {completion_path}")
    if completion.get("stop_reason") not in {"early_stopping", "max_epochs"}:
        raise ValueError(f"Unknown stop_reason in {completion_path}: {completion.get('stop_reason')!r}")
    if int(completion.get("best_epoch", -1)) != int(summary.get("checkpoint_epoch", -2)):
        raise ValueError(
            f"PEFNet best/checkpoint epoch mismatch: {completion_path} vs {path}"
        )
    config_hashes = {
        summary.get("config_sha256"),
        provenance.get("config_sha256"),
        completion.get("config_sha256"),
    }
    if None in config_hashes or len(config_hashes) != 1:
        raise ValueError(f"PEFNet config hash mismatch for {run_root}")
    if completion.get("stability_revision") != summary.get("stability_revision"):
        raise ValueError(f"PEFNet stability metadata mismatch for {run_root}")


def _validate_paper_guided_run(
    summary: dict, *, method: str, seed: int, directory: str, path: Path
) -> None:
    protocol = PAPER_GUIDED_PROTOCOLS[method]
    required = {
        "seed": seed,
        "data_seed": 42,
        "sample_rate": 0.0,
        "method_label": method,
        "primary_metric_domain": "normalized_minus1_to_1",
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"Adapted-baseline protocol mismatch in {path}: "
                f"{key}={summary.get(key)!r}, expected {expected!r}"
            )
    run_root = OUTPUTS / directory
    provenance_path = run_root / "provenance.json"
    completion_path = run_root / "run_complete.json"
    for required_path in (provenance_path, completion_path):
        if not required_path.is_file():
            raise ValueError(f"Required adapted-baseline audit artifact missing: {required_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_required = {
        "implementation_status": protocol["implementation_status"],
        "faithful_reproduction": False,
        "external_pretraining": False,
    }
    for key, expected in provenance_required.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"Adapted-baseline provenance mismatch in {provenance_path}: "
                f"{key}={provenance.get(key)!r}, expected {expected!r}"
            )
    source = provenance.get("source", {})
    if not isinstance(source, dict) or source.get("doi") != protocol["doi"]:
        raise ValueError(f"Missing or incorrect source DOI in {provenance_path}")
    if method == "RadioDiff-3D-style conditional diffusion (paper-aligned adapted v2)":
        anchor_protocol = {
            "selection_protocol": "common_balanced_scene_anchor_maps",
            "conditioning_volumes": 128,
            "generated_slices_per_volume": 4,
            "scored_slices_per_volume": 1,
            "scored_maps": 128,
            "best_of_k": 1,
        }
        for key, expected in anchor_protocol.items():
            if summary.get(key) != expected:
                raise ValueError(
                    f"RadioDiff-3D anchor protocol mismatch in {path}: "
                    f"{key}={summary.get(key)!r}, expected {expected!r}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows = []
    missing = []
    for method, seed, category, directory in RUNS:
        path = OUTPUTS / directory / "evaluation" / "test" / "metrics_summary.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        support = data.get("support", data.get("evaluation_support"))
        if data.get("maps") != 128 or support != "valid_rx_only":
            raise ValueError(
                f"Protocol mismatch in {path}: maps={data.get('maps')} support={support}"
            )
        _validate_optional_common_fields(data, seed=seed, path=path)
        if method == PEFNET_LABEL:
            _validate_pefnet_run(data, seed=seed, directory=directory, path=path)
        elif method in PAPER_GUIDED_PROTOCOLS:
            _validate_paper_guided_run(
                data, method=method, seed=seed, directory=directory, path=path
            )
        row = {
            "method": method, "seed": seed, "category": category,
            "directory": directory, "maps": data["maps"], "support": support,
            "parameter_count": data.get("parameter_count", ""),
            "method_label": data.get("method_label", method),
            "primary_metric_domain": data.get(
                "primary_metric_domain", "normalized_minus1_to_1"
            ),
        }
        for metric in METRICS:
            row[metric] = _finite_metric(data, metric, path)
        rows.append(row)

    destination = OUTPUTS / "icc_suite_summary"
    destination.mkdir(parents=True, exist_ok=True)
    per_run_fields = (
        "method", "seed", "category", "directory", "maps", "support",
        "parameter_count", "method_label", "primary_metric_domain", *METRICS,
    )
    with (destination / "per_run.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_run_fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    methods = list(dict.fromkeys(row["method"] for row in rows))
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        parameter_counts = {
            str(row["parameter_count"]) for row in selected if row["parameter_count"] != ""
        }
        if len(parameter_counts) > 1:
            raise ValueError(f"Parameter count changes across seeds for {method}: {sorted(parameter_counts)}")
        record = {
            "method": method,
            "category": selected[0]["category"],
            "runs": len(selected),
            "seeds": ",".join(str(row["seed"]) for row in selected),
            "parameter_count": selected[0]["parameter_count"],
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            record[f"{metric}_mean"] = statistics.fmean(values)
            record[f"{metric}_std_across_seeds"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(record)
    aggregate_fields = (
        "method", "category", "runs", "seeds", "parameter_count",
        *(field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_std_across_seeds")),
    )
    with (destination / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate)
    report = {
        "protocol": {
            "split": "scene-disjoint test",
            "maps_per_run": 128,
            "support": "valid_rx_only",
            "primary_domain": "x / 127.5 - 1.0",
            "aggregation": "per-map metric, then macro mean; seed std is sample std",
            "data_seed": 42,
            "model_seeds": [42, 43, 44],
            "legacy_exclusions": ["PMNet-v1 (adapted)"],
            "radiodiff3d_test_selection": (
                "the same 128 balanced_scene anchor maps; one h1--h4 conditioning "
                "volume per anchor; only the anchor slice is scored"
            ),
        },
        "runs": rows,
        "aggregate": aggregate,
        "missing": missing,
    }
    (destination / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {destination / 'aggregate.csv'}")
    print(f"completed={len(rows)} missing={len(missing)}")
    for path in missing:
        print(f"missing: {path}")
    if args.require_complete and missing:
        raise SystemExit(f"ICC suite incomplete: {len(missing)} required results are missing")


if __name__ == "__main__":
    main()
