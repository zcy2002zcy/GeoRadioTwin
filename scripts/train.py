"""Train a continuous PathLoss field from map-level query sampling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.coordinate_query_dataset import CoordinateQueryDataset
from models.continuous_radio_field import ContinuousRadioField
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.seed import seed_everything


HISTORY_FIELDS = (
    "epoch", "train_loss", "train_rmse", "train_nmse", "train_mae",
    "val_loss", "val_rmse", "val_nmse", "val_mae", "learning_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve_path(value: object, config_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def make_dataset(
    config: dict[str, Any],
    config_path: Path,
    split: str,
    *,
    smoke: bool,
) -> CoordinateQueryDataset:
    data_root = resolve_path(config["data"]["output_root"], config_path)
    cache_root = resolve_path(config["geometry"]["cache_root"], config_path)
    experiment = config["experiment"]
    sparse = config.get("sparse_calibration", {})
    is_train = split == "train"
    max_records = (4 if is_train else 2) if smoke else int(
        config["training"]["train_max_maps" if is_train else "val_max_maps"]
    )
    points = 256 if smoke else int(config["dataset"]["points_per_map"])
    project_seed = int(config["project"]["seed"])
    data_seed = int(config["training"].get("data_seed", project_seed))
    return CoordinateQueryDataset(
        manifest_path=data_root / "splits" / f"{split}.csv",
        points_per_map=points,
        query_mode="fixed_random" if smoke or not is_train else config["dataset"]["train_query_mode"],
        seed=project_seed if is_train else data_seed,
        selection_seed=data_seed,
        pathloss_channel=int(config["data"]["pathloss_channel"]),
        max_records=max_records,
        geometry_cache_root=cache_root,
        use_los=bool(experiment["use_los"]),
        use_depth=bool(experiment["use_depth"]),
        depth_feature=str(experiment["depth_feature"]),
        exclude_building_occupied=bool(config["dataset"]["exclude_building_occupied"]),
        exclude_target_values=config["dataset"]["exclude_target_values"],
        target_exclusion_tolerance=float(config["dataset"]["target_exclusion_tolerance"]),
        selection_strategy=("sequential" if smoke else str(
            config["training"]["train_selection_strategy"]
            if is_train
            else config["dataset"]["val_selection_strategy"]
        )),
        sparse_sample_rate=float(sparse.get("sample_rate", 0.0)),
        sparse_context_radii=sparse.get("context_radii", (4, 16, 64)),
        include_environment_map=bool(experiment.get("use_environment_map", False)),
    )


def accumulate_metrics(
    totals: dict[str, float], prediction: Tensor, target: Tensor
) -> None:
    error = prediction.detach().float() - target.detach().float()
    totals["sse"] += error.square().sum().item()
    totals["sae"] += error.abs().sum().item()
    totals["target_sq"] += target.detach().float().square().sum().item()
    totals["count"] += target.numel()


def finish_metrics(totals: dict[str, float]) -> dict[str, float]:
    mse = totals["sse"] / max(totals["count"], 1.0)
    return {
        "rmse": math.sqrt(mse),
        "nmse": totals["sse"] / max(totals["target_sq"], 1e-12),
        "mae": totals["sae"] / max(totals["count"], 1.0),
    }


def run_epoch(
    model: ContinuousRadioField,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    target_scale: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    gradient_clip: float,
    max_steps: int | None,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    totals = {"sse": 0.0, "sae": 0.0, "target_sq": 0.0, "count": 0.0}
    steps = 0
    for batch in loader:
        coordinates = batch["coordinate_features"].to(device, non_blocking=True)
        target_raw = batch["target"].to(device, non_blocking=True)
        geometry = batch.get("geometry_features")
        if geometry is not None:
            geometry = geometry.to(device, non_blocking=True)
        sparse_context = batch.get("sparse_context_features")
        if sparse_context is not None:
            sparse_context = sparse_context.to(device, non_blocking=True)
        observation_mask = batch.get("observation_mask")
        if observation_mask is not None:
            observation_mask = observation_mask.to(device, non_blocking=True).bool()
        target = target_raw / target_scale
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(coordinates, geometry, sparse_context)
                loss_prediction = prediction if observation_mask is None else prediction[~observation_mask]
                loss_target = target if observation_mask is None else target[~observation_mask]
                loss = nn.functional.mse_loss(loss_prediction, loss_target)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
        loss_total += loss.detach().item()
        metric_prediction = prediction if observation_mask is None else prediction[~observation_mask]
        metric_target = target_raw if observation_mask is None else target_raw[~observation_mask]
        accumulate_metrics(totals, metric_prediction * target_scale, metric_target)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return loss_total / max(steps, 1), finish_metrics(totals)


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    experiment = config["experiment"]
    training = config["training"]
    model_config = config["model"]
    name = str(experiment["name"])
    run_tag = str(training.get("run_tag", "")).strip()
    if run_tag:
        name += f"_{run_tag}"
    if args.smoke_test:
        name += "_smoke"
    output_root = resolve_path(config["output_root"], config_path) / name
    checkpoint_root = output_root / "checkpoints"
    log_root = output_root / "logs"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.yaml").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    train_dataset = make_dataset(config, config_path, "train", smoke=args.smoke_test)
    val_dataset = make_dataset(config, config_path, "val", smoke=args.smoke_test)
    batch_size = 2 if args.smoke_test else int(training["batch_size"])
    workers = 0 if args.smoke_test else int(training["num_workers"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=torch.cuda.is_available(), generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    geometry_dim = int(bool(experiment["use_los"])) + int(bool(experiment["use_depth"]))
    sparse = config.get("sparse_calibration", {})
    calibration_dim = (
        2 * len(sparse.get("context_radii", (4, 16, 64))) + 1
        if float(sparse.get("sample_rate", 0.0)) > 0.0
        else 0
    )
    model = ContinuousRadioField(
        num_frequencies=int(model_config["num_frequencies"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        geometry_input_dim=geometry_dim,
        geometry_hidden_dim=int(model_config["geometry_hidden_dim"]),
        geometry_output_dim=int(model_config["geometry_output_dim"]),
        use_layer_norm=bool(model_config["use_layer_norm"]),
        xy_scale=float(model_config["xy_scale"]),
        z_scale=float(model_config["z_scale"]),
        distance_scale=float(model_config["distance_scale"]),
        calibration_input_dim=calibration_dim,
        calibration_hidden_dim=int(model_config.get("calibration_hidden_dim", 32)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pretrained = str(training.get("pretrained_checkpoint", "")).strip()
    if pretrained:
        pretrained_path = resolve_path(pretrained, config_path)
        state = torch.load(pretrained_path, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(state["model"], strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("calibration_")]
        if unexpected:
            raise RuntimeError(f"Unexpected pretrained keys: {unexpected}")
        print(f"initialized_from={pretrained_path} epoch={state.get('epoch', 'unknown')}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    epochs = 8 if args.smoke_test else int(training["epochs"])
    max_steps = 2 if args.smoke_test else int(training["steps_per_epoch"])
    target_scale = float(training["target_scale"])
    best_value = math.inf
    patience = 0
    history_path = log_root / "train_history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_dataset.set_epoch(epoch)
            train_loss, train_metrics = run_epoch(
                model, train_loader, device, target_scale, optimizer, scaler,
                amp_enabled, float(training["gradient_clip"]), max_steps,
            )
            with torch.no_grad():
                val_loss, val_metrics = run_epoch(
                    model, val_loader, device, target_scale, None, scaler,
                    amp_enabled, float(training["gradient_clip"]), None,
                )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                "val_loss": val_loss,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            history_file.flush()
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"train_rmse={train_metrics['rmse']:.4f} "
                f"val_rmse={val_metrics['rmse']:.4f} val_nmse={val_metrics['nmse']:.6f}"
            )
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "val_metrics": val_metrics,
            }
            save_checkpoint(checkpoint_root / "latest.pt", state)
            selection = val_metrics[str(training["selection_metric"])]
            if selection < best_value:
                best_value = selection
                patience = 0
                save_checkpoint(checkpoint_root / "best.pt", state)
            else:
                patience += 1
            if not args.smoke_test and patience >= int(training["early_stopping_patience"]):
                print(f"early_stopping epoch={epoch}")
                break
    (output_root / "run_complete.json").write_text(
        json.dumps({"epoch": epoch, "best": best_value, "device": str(device)}, indent=2),
        encoding="utf-8",
    )
    print(f"finished experiment={name} device={device} best={best_value:.6f} output={output_root}")


if __name__ == "__main__":
    main()
