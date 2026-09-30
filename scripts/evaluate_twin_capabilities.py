"""Evaluate generalization, state updates, and planning utility of HybridGeo."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twin_validation_common import (
    build_missing_geometry,
    delta_scores,
    load_hybrid,
    make_dataset,
    map_scores,
    predict_sample,
    require_geometry,
    select_state_records,
    test_manifest,
    unique_records,
    write_csv,
    write_json,
)


DISTANCE_GROUPS = (
    ("0_64", 0.0, 64.0),
    ("64_128", 64.0, 128.0),
    ("128_192", 128.0, 192.0),
    ("192_plus", 192.0, math.inf),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_geo.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/twin_validation_seed42"))
    parser.add_argument("--max-maps", type=int, default=128)
    parser.add_argument("--transition-scenes", type=int, default=8)
    parser.add_argument("--planning-scenes", type=int, default=8)
    parser.add_argument("--planning-candidates", type=int, default=4)
    parser.add_argument("--build-missing-caches", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _density(record) -> float:
    assert record.building_height_path is not None
    with Image.open(record.building_height_path) as image:
        building = np.asarray(image, dtype=np.float32)
    return float(np.mean(building > 0.0))


def _density_labels(values: list[float]) -> tuple[np.ndarray, list[float]]:
    boundaries = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.5, 0.75])
    labels = np.searchsorted(boundaries, np.asarray(values), side="right") + 1
    return labels, [float(value) for value in boundaries]


def _append_group(
    rows: list[dict[str, object]],
    *,
    dimension: str,
    group: str,
    map_index: int,
    scene_id: int,
    prediction: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
) -> None:
    if int(support.sum()) == 0:
        return
    rows.append({
        "dimension": dimension,
        "group": group,
        "map_index": map_index,
        "scene_id": scene_id,
        "points": int(support.sum()),
        **map_scores(prediction, target, support),
    })


def _aggregate(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    metric_names = ("rmse", "nmse", "mae", "psnr", "raw_rmse", "raw_nmse", "raw_mae", "raw_psnr")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for key_values, group in sorted(groups.items(), key=lambda value: str(value[0])):
        summary: dict[str, object] = {key: value for key, value in zip(keys, key_values)}
        summary["maps_or_pairs"] = len(group)
        summary["points"] = int(sum(int(row.get("points", 0)) for row in group))
        for metric in metric_names:
            if metric not in group[0]:
                continue
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.nanmean(values))
            summary[f"{metric}_std"] = float(np.nanstd(values, ddof=0))
        result.append(summary)
    return result


def evaluate_generalization(loaded, max_maps: int) -> tuple[list[dict[str, object]], dict]:
    dataset = make_dataset(loaded, max_maps=max_maps, selection_strategy="balanced_scene")
    densities = [_density(record) for record in dataset.records]
    density_groups, boundaries = _density_labels(densities)
    rows: list[dict[str, object]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        record = dataset.records[index]
        prediction = predict_sample(loaded, sample)
        target = sample["target"][:, 0].numpy()
        valid = sample["valid_mask"][:, 0].numpy().astype(bool)
        distance = sample["coordinate_features"][:, 9].numpy()
        _append_group(
            rows, dimension="overall", group="all", map_index=index,
            scene_id=record.scene_id, prediction=prediction, target=target, support=valid,
        )
        _append_group(
            rows, dimension="rx_height", group=f"h{record.rx_height:g}", map_index=index,
            scene_id=record.scene_id, prediction=prediction, target=target, support=valid,
        )
        _append_group(
            rows, dimension="building_density_quartile", group=f"Q{density_groups[index]}",
            map_index=index, scene_id=record.scene_id, prediction=prediction,
            target=target, support=valid,
        )
        for label, lower, upper in DISTANCE_GROUPS:
            _append_group(
                rows, dimension="tx_rx_distance", group=label, map_index=index,
                scene_id=record.scene_id, prediction=prediction, target=target,
                support=valid & (distance >= lower) & (distance < upper),
            )
        geometry = sample.get("geometry_features")
        if geometry is not None and bool(loaded.config["experiment"]["use_los"]):
            los = geometry[:, 0].numpy() >= 0.5
            _append_group(
                rows, dimension="visibility", group="LOS", map_index=index,
                scene_id=record.scene_id, prediction=prediction, target=target,
                support=valid & los,
            )
            _append_group(
                rows, dimension="visibility", group="NLOS", map_index=index,
                scene_id=record.scene_id, prediction=prediction, target=target,
                support=valid & ~los,
            )
        print(f"generalization [{index + 1}/{len(dataset)}] scene={record.scene_id}")
    return rows, {"building_density_quartile_boundaries": boundaries, "maps": len(dataset)}


def _record_key(record) -> tuple[int, float, float, float]:
    return record.scene_id, record.tx_x, record.tx_y, record.rx_height


def evaluate_state_and_planning(loaded, transitions, planning):
    records = unique_records(transitions, planning)
    dataset = make_dataset(loaded, records=records)
    values: dict[tuple[int, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        record = dataset.records[index]
        values[_record_key(record)] = (
            predict_sample(loaded, sample),
            sample["target"][:, 0].numpy(),
            sample["valid_mask"][:, 0].numpy().astype(bool),
        )
        print(f"state map [{index + 1}/{len(dataset)}] scene={record.scene_id}")

    transition_rows = []
    for pair_index, (kind, before, after) in enumerate(transitions):
        pred_a, target_a, valid_a = values[_record_key(before)]
        pred_b, target_b, valid_b = values[_record_key(after)]
        support = valid_a & valid_b
        transition_rows.append({
            "pair_index": pair_index,
            "transition": kind,
            "scene_id": before.scene_id,
            "before_tx_x": before.tx_x,
            "before_tx_y": before.tx_y,
            "after_tx_x": after.tx_x,
            "after_tx_y": after.tx_y,
            "before_rx_height": before.rx_height,
            "after_rx_height": after.rx_height,
            "points": int(support.sum()),
            **delta_scores(pred_b[support] - pred_a[support], target_b[support] - target_a[support]),
        })

    planning_rows = []
    for (scene_id, rx_height), candidates in planning.items():
        common = np.logical_and.reduce([values[_record_key(record)][2] for record in candidates])
        predicted_utilities = []
        true_utilities = []
        for record in candidates:
            prediction, target, _ = values[_record_key(record)]
            predicted_utilities.append(float(np.mean(prediction[common])))
            true_utilities.append(float(np.mean(target[common])))
        predicted_utilities_np = np.asarray(predicted_utilities)
        true_utilities_np = np.asarray(true_utilities)
        selected = int(np.argmax(predicted_utilities_np))
        optimal = int(np.argmax(true_utilities_np))
        pred_rank = np.argsort(np.argsort(predicted_utilities_np))
        true_rank = np.argsort(np.argsort(true_utilities_np))
        rank_corr = (
            float(np.corrcoef(pred_rank, true_rank)[0, 1])
            if len(candidates) > 1 and np.std(pred_rank) > 0 and np.std(true_rank) > 0
            else math.nan
        )
        regret = float(true_utilities_np[optimal] - true_utilities_np[selected])
        planning_rows.append({
            "scene_id": scene_id,
            "rx_height": rx_height,
            "candidates": len(candidates),
            "points": int(common.sum()),
            "selected_index": selected,
            "optimal_index": optimal,
            "top1_correct": int(selected == optimal),
            "selected_tx_x": candidates[selected].tx_x,
            "selected_tx_y": candidates[selected].tx_y,
            "optimal_tx_x": candidates[optimal].tx_x,
            "optimal_tx_y": candidates[optimal].tx_y,
            "selected_true_mean_raw": float(true_utilities_np[selected]),
            "optimal_true_mean_raw": float(true_utilities_np[optimal]),
            "regret_raw": regret,
            "regret_normalized_255": regret / 255.0,
            "candidate_rank_correlation": rank_corr,
        })
    return transition_rows, planning_rows


def physical_scale_audit(loaded, limit: int = 16) -> dict:
    dataset = make_dataset(loaded, max_maps=limit, selection_strategy="balanced_scene")
    minimum = math.inf
    maximum = -math.inf
    channel_equal = True
    distance_correlations = []
    modes = set()
    dtypes = set()
    for record in dataset.records:
        with Image.open(record.pathloss_path) as image:
            modes.add(image.mode)
            array = np.asarray(image)
        dtypes.add(str(array.dtype))
        if array.ndim == 3:
            channel_equal &= bool(np.all(array[..., :3] == array[..., :1]))
            scalar = array[..., 0].astype(np.float64)
        else:
            scalar = array.astype(np.float64)
        minimum = min(minimum, float(scalar.min()))
        maximum = max(maximum, float(scalar.max()))
        rows, cols = np.indices(scalar.shape)
        rx_x = (scalar.shape[0] - 1) - rows
        distance = np.sqrt((rx_x - record.tx_x) ** 2 + (cols - record.tx_y) ** 2)
        distance_correlations.append(float(np.corrcoef(scalar.ravel(), distance.ravel())[0, 1]))
    mapping = loaded.config["data"].get("pathloss_db_mapping")
    return {
        "files_checked": len(dataset.records),
        "image_modes": sorted(modes),
        "dtypes": sorted(dtypes),
        "raw_min": minimum,
        "raw_max": maximum,
        "rgb_channels_identical": channel_equal,
        "mean_raw_distance_correlation": float(np.mean(distance_correlations)),
        "physical_unit_verified": mapping is not None,
        "configured_pathloss_db_mapping": mapping,
        "paper_rule": (
            "dB metrics may be reported only after an official, cited mapping is configured."
            if mapping is None
            else "Apply the configured, cited mapping before reporting dB metrics."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.max_maps = 2
        args.transition_scenes = 1
        args.planning_scenes = 1
        args.planning_candidates = 2
        args.output = args.output.with_name(args.output.name + "_smoke")
    loaded = load_hybrid(args.config, args.checkpoint, device_name=args.device)
    output = args.output.expanduser().resolve()
    generalization_rows, generalization_meta = evaluate_generalization(loaded, args.max_maps)

    transitions, planning = select_state_records(
        test_manifest(loaded),
        seed=int(loaded.config["training"].get("data_seed", loaded.config["project"]["seed"])),
        transition_scenes=args.transition_scenes,
        planning_scenes=args.planning_scenes,
        candidates=args.planning_candidates,
    )
    required = unique_records(transitions, planning)
    if args.build_missing_caches:
        built, skipped = build_missing_geometry(loaded, required)
    else:
        require_geometry(loaded, required)
        built, skipped = 0, len(required)
    transition_rows, planning_rows = evaluate_state_and_planning(loaded, transitions, planning)

    generalization_summary = _aggregate(generalization_rows, ("dimension", "group"))
    write_csv(output / "generalization_per_map.csv", generalization_rows)
    write_csv(output / "generalization_summary.csv", generalization_summary)
    write_csv(output / "state_transitions.csv", transition_rows)
    write_csv(output / "planning_utility.csv", planning_rows)
    audit = physical_scale_audit(loaded, min(args.max_maps, 16))
    write_json(output / "physical_scale_audit.json", audit)

    transition_summary = {}
    for kind in ("tx_position", "rx_height"):
        group = [row for row in transition_rows if row["transition"] == kind]
        if group:
            transition_summary[kind] = {
                "pairs": len(group),
                **{
                    metric: float(np.nanmean([float(row[metric]) for row in group]))
                    for metric in (
                        "delta_rmse", "delta_nmse", "delta_mae",
                        "delta_correlation", "delta_sign_agreement",
                    )
                },
            }
    planning_summary = {
        "scenes": len(planning_rows),
        "top1_accuracy": float(np.mean([row["top1_correct"] for row in planning_rows])) if planning_rows else math.nan,
        "mean_regret_normalized_255": float(np.mean([row["regret_normalized_255"] for row in planning_rows])) if planning_rows else math.nan,
        "mean_candidate_rank_correlation": float(np.nanmean([row["candidate_rank_correlation"] for row in planning_rows])) if planning_rows else math.nan,
        "utility_definition": "mean raw encoded propagation strength over common valid Rx support; larger is better",
    }
    summary = {
        "protocol": "scene-disjoint_test_only",
        "checkpoint": str(loaded.checkpoint_path),
        "checkpoint_epoch": loaded.checkpoint_epoch,
        "device": str(loaded.device),
        "generalization": generalization_meta,
        "state_transitions": transition_summary,
        "planning": planning_summary,
        "geometry_caches_built": built,
        "geometry_caches_reused": skipped,
        "physical_scale_verified": audit["physical_unit_verified"],
        "limitations": [
            "State transitions cover Tx position and Rx height, not temporal fading.",
            "Building interventions require paired ray-traced labels and are evaluated separately.",
            "Planning utility remains in the encoded raw domain until the official dB mapping is cited.",
        ],
    }
    write_json(output / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
