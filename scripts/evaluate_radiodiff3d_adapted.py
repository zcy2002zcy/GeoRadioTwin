"""Evaluate RadioDiff-3D-style adapted volumes on the common test protocol.

The common map-level selector first chooses the same 128 test maps used by all
other baselines.  Each selected map is completed to h1--h4 for one fixed-seed
DDIM volume generation, but only the selected anchor height is scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiodiff3d_volume_dataset import RadioDiff3DAnchorEvaluationDataset
from models.radiodiff3d_adapted import build_radiodiff3d_adapted
from scripts.evaluate import metrics, normalized_metrics
from utils.config import load_config
from utils.seed import seed_everything


METHOD_LABEL = "RadioDiff-3D-style conditional diffusion (adapted)"


def anchor_score_arrays(
    prediction_volume: np.ndarray,
    target_volume: np.ndarray,
    valid_volume: np.ndarray,
    score_depth_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid pixels from one anchor slice, never its companion heights."""

    if prediction_volume.shape != target_volume.shape or target_volume.shape != valid_volume.shape:
        raise ValueError("Prediction, target, and valid volumes must have identical [D,H,W] shapes")
    if not 0 <= score_depth_index < prediction_volume.shape[0]:
        raise IndexError(f"score_depth_index={score_depth_index} is outside the volume")
    support = valid_volume[score_depth_index].astype(bool)
    if not support.any():
        raise ValueError("Selected anchor height has no valid receiver support")
    return (
        prediction_volume[score_depth_index][support],
        target_volume[score_depth_index][support],
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/radiodiff3d_adapted.yaml")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument(
        "--sampling-algorithm",
        choices=("DDIM", "DDPM"),
        help="Override the configured sampler; DDPM always uses all diffusion steps.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{name}' must be a mapping")
    return value


def _experiment_name(config: dict[str, Any], seed: int, *, smoke_test: bool) -> str:
    base = str(
        _section(config, "experiment").get(
            "name", "radiodiff3d_adapted_random0"
        )
    ).strip()
    if not base:
        raise ValueError("experiment.name cannot be empty")
    name = f"{base}_seed{seed}"
    return f"{name}_smoke" if smoke_test else name


def main() -> None:
    args = arguments()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config, _ = load_config(config_path)
    data = _section(config, "data")
    diffusion = _section(config, "diffusion")
    evaluation = _section(config, "evaluation")
    experiment = _section(config, "experiment")
    if int(diffusion.get("best_of_k", 1)) != 1 or bool(evaluation.get("best_of_k_selection", False)):
        raise ValueError("Academic evaluation forbids best-of-K diffusion sample selection")
    sampling_seed = int(
        args.sampling_seed if args.sampling_seed is not None else diffusion.get("sampling_seed", 2027)
    )
    sampling_steps = int(
        args.sampling_steps if args.sampling_steps is not None else diffusion.get("sampling_steps", 50)
    )
    sampling_algorithm = str(
        args.sampling_algorithm
        if args.sampling_algorithm is not None
        else diffusion.get("sampling_algorithm", "DDIM")
    ).upper()
    if sampling_algorithm not in {"DDIM", "DDPM"}:
        raise ValueError("sampling_algorithm must be DDIM or DDPM")
    if args.smoke_test:
        sampling_steps = min(sampling_steps, 2)
    eta = float(diffusion.get("ddim_eta", 0.0))
    clip_denoised = bool(diffusion.get("clip_denoised", True))

    name = _experiment_name(config, args.seed, smoke_test=args.smoke_test)
    output = PROJECT_ROOT / "outputs" / name
    checkpoint_path = output / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_data_seed = int(checkpoint.get("metadata", {}).get("data_seed", args.data_seed))
    if checkpoint_data_seed != args.data_seed:
        raise ValueError(
            f"Evaluation data_seed={args.data_seed} does not match checkpoint data_seed={checkpoint_data_seed}"
        )
    checkpoint_condition_layout = str(
        checkpoint.get("metadata", {}).get(
            "condition_layout", data.get("condition_layout", "legacy_height")
        )
    )
    configured_condition_layout = str(
        data.get("condition_layout", "legacy_height")
    )
    if checkpoint_condition_layout != configured_condition_layout:
        raise ValueError(
            "Evaluation condition_layout does not match the checkpoint: "
            f"config={configured_condition_layout}, checkpoint={checkpoint_condition_layout}"
        )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint is missing the resolved model_config")
    model = build_radiodiff3d_adapted(**model_config)
    state = checkpoint.get("ema_model", checkpoint.get("model"))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint contains neither ema_model nor model weights")
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    seed_everything(args.seed, deterministic=True)

    test_anchor_maps = 1 if args.smoke_test else int(data.get("test_anchor_maps", 128))
    dataset = RadioDiff3DAnchorEvaluationDataset(
        PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv",
        max_anchor_maps=test_anchor_maps,
        selection_strategy="sequential" if args.smoke_test else str(data.get("selection_strategy", "balanced_scene")),
        seed=args.data_seed,
        expected_rx_heights=data.get("expected_rx_heights", (1.0, 2.0, 3.0, 4.0)),
        exclude_target_values=data.get("exclude_target_values", (0.0,)),
        building_scale=float(data.get("building_scale", 255.0)),
        rx_height_scale=float(data.get("rx_height_scale", 20.0)),
        condition_layout=configured_condition_layout,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0 if args.smoke_test else args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=(not args.smoke_test and args.num_workers > 0),
    )

    rows: list[dict[str, object]] = []
    used_seeds: list[int] = []
    with torch.inference_mode():
        for volume_index, batch in enumerate(loader):
            condition = batch["condition"].to(device, non_blocking=True)
            original_target = batch["target_raw"].numpy()[0, 0]
            valid = batch["valid_mask"].numpy()[0, 0].astype(bool)
            if args.smoke_test:
                depth = condition.shape[2]
                condition = F.interpolate(
                    condition, size=(depth, 64, 64), mode="trilinear", align_corners=False
                )
            volume_seed = sampling_seed + volume_index
            used_seeds.append(volume_seed)
            generator = torch.Generator(device=device.type)
            generator.manual_seed(volume_seed)
            if sampling_algorithm == "DDPM":
                prediction_normalized = model.ddpm_sample(
                    condition,
                    generator=generator,
                    clip_denoised=clip_denoised,
                )
            else:
                prediction_normalized = model.ddim_sample(
                    condition,
                    sampling_steps=sampling_steps,
                    generator=generator,
                    eta=eta,
                    clip_denoised=clip_denoised,
                )
            if args.smoke_test:
                prediction_normalized = F.interpolate(
                    prediction_normalized,
                    size=original_target.shape,
                    mode="trilinear",
                    align_corners=False,
                )
            prediction_raw = ((prediction_normalized[0, 0].float().cpu().numpy() + 1.0) * 127.5)
            depth_index = int(batch["score_depth_index"].item())
            rx_height = float(batch["score_rx_height"].item())
            support = valid[depth_index]
            scored_prediction, scored_target = anchor_score_arrays(
                prediction_raw, original_target, valid, depth_index
            )
            encoded = normalized_metrics(scored_prediction, scored_target)
            raw = metrics(scored_prediction, scored_target, 255.0)
            row: dict[str, object] = {
                "map_index": volume_index,
                "volume_index": volume_index,
                "scene_id": int(batch["scene_id"].item()),
                "tx_x": float(batch["tx_x"].item()),
                "tx_y": float(batch["tx_y"].item()),
                "tx_z": float(batch["tx_z"].item()),
                "rx_height": rx_height,
                "score_depth_index": depth_index,
                "valid_points": int(support.sum()),
                "sampling_seed": volume_seed,
                **encoded,
                **{f"raw_{key}": value for key, value in raw.items()},
            }
            rows.append(row)
            print(
                f"[{len(rows)}/{len(dataset)}] anchor_volume={volume_index} h={rx_height:g} "
                f"rmse={encoded['rmse']:.4f} nmse={encoded['nmse']:.6f}"
            )
    if not rows:
        raise ValueError("Test dataset is empty")

    result = output / "evaluation" / "test"
    result.mkdir(parents=True, exist_ok=True)
    with (result / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {}
    metric_names = (
        "rmse", "nmse", "mae", "psnr", "raw_rmse", "raw_nmse", "raw_mae", "raw_psnr"
    )
    for metric_name in metric_names:
        values = np.asarray([float(row[metric_name]) for row in rows], dtype=np.float64)
        summary[metric_name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    summary.update(
        {
            "maps": len(rows),
            "volumes": len(dataset),
            "conditioning_volumes": len(dataset),
            "generated_slices_per_volume": len(data.get("expected_rx_heights", (1, 2, 3, 4))),
            "scored_slices_per_volume": 1,
            "scored_maps": len(rows),
            "selection_protocol": "common_balanced_scene_anchor_maps",
            "unique_scenes": len({int(row["scene_id"]) for row in rows}),
            "seed": args.seed,
            "data_seed": args.data_seed,
            "sample_rate": 0.0,
            "support": "valid_rx_only",
            "primary_metric_domain": "normalized_minus1_to_1",
            "normalization": "prediction and target mapped from PNG code x to x / 127.5 - 1.0",
            "raw_domain": "UrbanRadio3D PNG code 0..255; not asserted to be dB",
            "per_map_macro_average": True,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "method_label": str(experiment.get("method_label", METHOD_LABEL)),
            "implementation_status": "independent_upstream_and_paper_informed_adaptation",
            "faithful_reproduction": False,
            "external_pretraining": False,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "sampling_algorithm": sampling_algorithm,
            "sampling_steps": model.timesteps if sampling_algorithm == "DDPM" else sampling_steps,
            "sampling_seed_base": sampling_seed,
            "volume_sampling_seeds": used_seeds,
            "ddim_eta": eta,
            "fixed_sampling_seed": True,
            "best_of_k": 1,
            "sample_selection": "none",
            "weights": "EMA" if "ema_model" in checkpoint else "raw",
            "condition_layout": configured_condition_layout,
            "loss_type": model.loss_type,
            "loss_support": model.loss_support,
            "parameterization": model.parameterization,
        }
    )
    (result / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (result / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "mean", "std"))
        writer.writeheader()
        for metric_name in metric_names:
            writer.writerow({"metric": metric_name, **summary[metric_name]})
    sampling_protocol = {
        "algorithm": sampling_algorithm,
        "steps": model.timesteps if sampling_algorithm == "DDPM" else sampling_steps,
        "eta": eta,
        "base_seed": sampling_seed,
        "per_volume_seed_rule": "base_seed + zero_based_volume_index",
        "generated_samples_per_condition": 1,
        "best_of_k": 1,
        "selection": "none",
        "test_selection_protocol": "common_balanced_scene_anchor_maps",
        "conditioning_volumes": len(dataset),
        "scored_maps": len(rows),
    }
    (result / "sampling_protocol.json").write_text(
        json.dumps(sampling_protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
