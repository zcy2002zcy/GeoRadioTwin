"""Evaluate paired building interventions with independently ray-traced labels.

This script deliberately refuses geometry-only edits without an edited PathLoss
label.  A model prediction is never used as counterfactual ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.coordinate_query_dataset import load_raw_pathloss
from geometry.coordinates import row_col_to_physical_xy
from geometry.ray_query import query_propagation_geometry
from scripts.twin_validation_common import delta_scores, load_hybrid, map_scores, predict_sample, write_csv, write_json


REQUIRED_FIELDS = (
    "pair_id", "scene_id", "intervention", "tx_x", "tx_y", "tx_z", "rx_height",
    "base_building_path", "base_pathloss_path", "edited_building_path", "edited_pathloss_path",
)
PROVENANCE_FIELDS = (
    "simulator_name", "simulator_version", "simulator_config_sha256",
    "intervention_metadata_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/counterfactual/manifest.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_geo.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/geometry_counterfactuals_seed42"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--affected-delta-threshold", type=float, default=1.0)
    parser.add_argument("--allow-missing-provenance", action="store_true")
    parser.add_argument("--allow-identical-labels", action="store_true")
    return parser.parse_args()


def _resolve(value: str, manifest: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def _load_rows(path: Path, *, require_provenance: bool) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Counterfactual manifest not found: {path}. Copy manifest_template.csv to manifest.csv "
            "after generating both base and edited labels with the same ray tracer."
        )
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or [])]
        if require_provenance:
            missing.extend(field for field in PROVENANCE_FIELDS if field not in (reader.fieldnames or []))
        if missing:
            raise ValueError(f"Counterfactual manifest missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("Counterfactual manifest contains no pairs")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_provenance(row: dict[str, str], manifest: Path) -> None:
    for field in ("simulator_name", "simulator_version"):
        value = row.get(field, "").strip()
        if not value or value == "FILL_REQUIRED":
            raise ValueError(f"Pair {row['pair_id']} has incomplete {field}")
    fingerprint = row.get("simulator_config_sha256", "").strip().lower()
    if len(fingerprint) != 64 or any(value not in "0123456789abcdef" for value in fingerprint):
        raise ValueError(
            f"Pair {row['pair_id']} must record the exact simulator configuration SHA-256"
        )
    metadata_path = _resolve(row.get("intervention_metadata_path", ""), manifest)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Intervention metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("scene_id") != int(row["scene_id"]):
        raise ValueError(f"Intervention metadata scene mismatch for pair {row['pair_id']}")


def _sample(loaded, building_path: Path, target_path: Path, row: dict[str, str]):
    if not building_path.is_file() or not target_path.is_file():
        raise FileNotFoundError(f"Missing paired input/label: {building_path} or {target_path}")
    with Image.open(building_path) as image:
        building = np.asarray(image, dtype=np.float32)
    target = load_raw_pathloss(target_path, int(loaded.config["data"]["pathloss_channel"]))
    if building.shape != target.shape:
        raise ValueError(f"Shape mismatch: {building_path} {building.shape}, {target_path} {target.shape}")
    tx = (float(row["tx_x"]), float(row["tx_y"]), float(row["tx_z"]))
    rx_height = float(row["rx_height"])
    height, width = target.shape
    rows, cols = np.indices((height, width))
    rx_x, rx_y = row_col_to_physical_xy(rows.ravel(), cols.ravel(), height)
    rx_x = np.asarray(rx_x, dtype=np.float32)
    rx_y = np.asarray(rx_y, dtype=np.float32)
    rx_z = np.full_like(rx_x, rx_height)
    tx_x = np.full_like(rx_x, tx[0])
    tx_y = np.full_like(rx_x, tx[1])
    tx_z = np.full_like(rx_x, tx[2])
    dx, dy, dz = rx_x - tx_x, rx_y - tx_y, rx_z - tx_z
    distance = np.sqrt(dx * dx + dy * dy + dz * dz)
    coordinates = np.stack((tx_x, tx_y, tx_z, rx_x, rx_y, rx_z, dx, dy, dz, distance), axis=-1)
    result = query_propagation_geometry(
        building, tx, np.stack((rx_x, rx_y, rx_z), axis=-1),
        samples_per_unit=float(loaded.config["geometry"]["samples_per_unit"]),
        chunk_size=int(loaded.config["geometry"]["ray_chunk_size"]),
    )
    geometry_parts = []
    if bool(loaded.config["experiment"]["use_los"]):
        geometry_parts.append(result.los.astype(np.float32))
    if bool(loaded.config["experiment"]["use_depth"]):
        geometry_parts.append(getattr(result, str(loaded.config["experiment"]["depth_feature"])))
    geometry = np.stack(geometry_parts, axis=-1).astype(np.float32)
    valid = building.ravel() < rx_height
    target_flat = target.ravel()
    for excluded in loaded.config["dataset"]["exclude_target_values"]:
        valid &= ~np.isclose(
            target_flat, float(excluded),
            atol=float(loaded.config["dataset"]["target_exclusion_tolerance"]), rtol=0.0,
        )
    source = np.zeros_like(target, dtype=np.float32)
    source[int(round((height - 1) - tx[0])), int(round(tx[1]))] = 1.0
    environment = np.stack((
        np.clip(building / 255.0, 0.0, 1.0), source,
        np.full_like(target, rx_height / 20.0, dtype=np.float32),
    ))
    return {
        "coordinate_features": torch.from_numpy(coordinates.astype(np.float32)),
        "geometry_features": torch.from_numpy(geometry),
        "environment_map": torch.from_numpy(environment.astype(np.float32)),
        "target": torch.from_numpy(target_flat.astype(np.float32)[:, None]),
        "valid_mask": torch.from_numpy(valid[:, None]),
    }


def main() -> None:
    args = parse_args()
    if args.affected_delta_threshold < 0:
        raise ValueError("--affected-delta-threshold must be non-negative")
    manifest = args.manifest.expanduser().resolve()
    rows = _load_rows(manifest, require_provenance=not args.allow_missing_provenance)
    loaded = load_hybrid(args.config, args.checkpoint, device_name=args.device)
    results = []
    for index, row in enumerate(rows):
        if not args.allow_missing_provenance:
            _validate_provenance(row, manifest)
        base_building_path = _resolve(row["base_building_path"], manifest)
        edited_building_path = _resolve(row["edited_building_path"], manifest)
        base_target_path = _resolve(row["base_pathloss_path"], manifest)
        edited_target_path = _resolve(row["edited_pathloss_path"], manifest)
        if base_building_path.resolve() == edited_building_path.resolve():
            raise ValueError(f"Base and edited building paths are identical for {row['pair_id']}")
        if base_target_path.resolve() == edited_target_path.resolve():
            raise ValueError(f"Base and edited label paths are identical for {row['pair_id']}")
        if not base_building_path.is_file() or not edited_building_path.is_file():
            raise FileNotFoundError(f"Building pair is incomplete for {row['pair_id']}")
        if _sha256(base_building_path) == _sha256(edited_building_path):
            raise ValueError(f"Building intervention has no raster change for {row['pair_id']}")
        if (
            not args.allow_identical_labels
            and base_target_path.is_file()
            and edited_target_path.is_file()
            and _sha256(base_target_path) == _sha256(edited_target_path)
        ):
            raise ValueError(
                f"Base and edited PathLoss labels are byte-identical for {row['pair_id']}; "
                "verify the external simulator job or pass --allow-identical-labels with justification"
            )
        base = _sample(
            loaded, base_building_path, base_target_path, row,
        )
        edited = _sample(
            loaded, edited_building_path, edited_target_path, row,
        )
        pred_base = predict_sample(loaded, base)
        pred_edited = predict_sample(loaded, edited)
        target_base = base["target"][:, 0].numpy()
        target_edited = edited["target"][:, 0].numpy()
        support = base["valid_mask"][:, 0].numpy().astype(bool) & edited["valid_mask"][:, 0].numpy().astype(bool)
        target_delta = target_edited - target_base
        affected = support & (np.abs(target_delta) >= args.affected_delta_threshold)
        if not affected.any():
            raise ValueError(
                f"Pair {row['pair_id']} has no affected valid pixels at raw threshold "
                f"{args.affected_delta_threshold:g}"
            )
        global_delta = delta_scores(
            pred_edited[support] - pred_base[support], target_delta[support]
        )
        affected_delta = delta_scores(
            pred_edited[affected] - pred_base[affected], target_delta[affected]
        )
        results.append({
            "pair_id": row["pair_id"],
            "scene_id": int(row["scene_id"]),
            "intervention": row["intervention"],
            "points": int(support.sum()),
            "affected_points": int(affected.sum()),
            "affected_fraction": float(affected.sum() / max(support.sum(), 1)),
            "mean_absolute_target_change_raw": float(np.mean(np.abs(target_delta[support]))),
            **{f"edited_{key}": value for key, value in map_scores(pred_edited, target_edited, support).items()},
            **global_delta,
            **{f"affected_{key}": value for key, value in affected_delta.items()},
        })
        print(f"counterfactual [{index + 1}/{len(rows)}] pair={row['pair_id']}")
    output = args.output.expanduser().resolve()
    write_csv(output / "counterfactual_pairs.csv", results)
    metrics = (
        "edited_rmse", "edited_nmse", "delta_rmse", "delta_nmse",
        "delta_correlation", "delta_sign_agreement", "affected_delta_rmse",
        "affected_delta_nmse", "affected_delta_correlation", "affected_delta_sign_agreement",
        "affected_fraction", "mean_absolute_target_change_raw",
    )
    by_intervention = []
    for intervention in sorted({str(row["intervention"]) for row in results}):
        group = [row for row in results if row["intervention"] == intervention]
        by_intervention.append({
            "intervention": intervention,
            "pairs": len(group),
            **{
                metric: float(np.nanmean([float(item[metric]) for item in group]))
                for metric in metrics
            },
        })
    write_csv(output / "summary_by_intervention.csv", by_intervention)
    summary = {
        "pairs": len(results),
        "ground_truth_policy": "both states independently labeled by an external simulator or measurement",
        "provenance_required": not args.allow_missing_provenance,
        "affected_delta_threshold_raw": args.affected_delta_threshold,
        **{metric: float(np.nanmean([float(row[metric]) for row in results])) for metric in metrics},
        "by_intervention": by_intervention,
    }
    write_json(output / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
