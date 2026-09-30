"""Train the lightweight raster-context GeoRadioTwin residual model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.hybrid_radio_field import HybridGeoRadioField
from scripts.evaluate import build_model, resolve_path
from scripts.train import accumulate_metrics, finish_metrics, make_dataset
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.seed import seed_everything


def run_epoch(model, loader, device, optimizer, scaler, amp_enabled, target_scale, max_steps):
    training = optimizer is not None
    model.train(training)
    totals = {"sse": 0.0, "sae": 0.0, "target_sq": 0.0, "count": 0.0}
    loss_sum = 0.0
    steps = 0
    for batch in loader:
        coordinates = batch["coordinate_features"].to(device, non_blocking=True)
        geometry = batch.get("geometry_features")
        if geometry is not None:
            geometry = geometry.to(device, non_blocking=True)
        environment = batch["environment_map"].to(device, non_blocking=True)
        target_raw = batch["target"].to(device, non_blocking=True)
        target = target_raw / target_scale
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(coordinates, geometry, environment)
                loss = nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {steps + 1}")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        loss_sum += loss.item()
        accumulate_metrics(totals, prediction * target_scale, target_raw)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return loss_sum / max(steps, 1), finish_metrics(totals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_geo.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    if not torch.cuda.is_available() and not args.smoke_test:
        raise RuntimeError("Formal hybrid training requires CUDA")
    training = config["training"]
    name = config["experiment"]["name"] + "_" + training["run_tag"]
    if args.smoke_test:
        name += "_smoke"
    output = resolve_path(config["output_root"], config_path) / name
    checkpoint_dir = output / "checkpoints"
    log_dir = output / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train_set = make_dataset(config, config_path, "train", smoke=args.smoke_test)
    val_set = make_dataset(config, config_path, "val", smoke=args.smoke_test)
    batch_size = 2 if args.smoke_test else int(training["batch_size"])
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = build_model(config)
    pretrained_path = resolve_path(training["pretrained_checkpoint"], config_path)
    pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    base.load_state_dict(pretrained["model"], strict=True)
    model = HybridGeoRadioField(
        base,
        int(config["model"]["environment_dim"]),
        int(config["model"]["ray_samples"]),
        str(config["model"].get("context_mode", "ray")),
    ).to(device)
    print(
        f"initialized_from={pretrained_path} pretrained_epoch={pretrained.get('epoch', 'unknown')} "
        f"device={device} parameters={sum(parameter.numel() for parameter in model.parameters())}"
    )
    lr = float(training["learning_rate"])
    optimizer = torch.optim.AdamW([
        {"params": model.base.parameters(), "lr": lr * 0.1},
        {"params": list(model.environment_encoder.parameters()) + list(model.residual_head.parameters()), "lr": lr},
    ], weight_decay=float(training["weight_decay"]))
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    epochs = 2 if args.smoke_test else int(training["epochs"])
    max_steps = 2 if args.smoke_test else int(training["steps_per_epoch"])
    best = math.inf
    patience = 0
    fields = ("epoch", "train_loss", "train_rmse", "train_nmse", "val_loss", "val_rmse", "val_nmse")
    with (log_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_set.set_epoch(epoch)
            train_loss, train_metrics = run_epoch(model, train_loader, device, optimizer, scaler, amp_enabled, float(training["target_scale"]), max_steps)
            with torch.inference_mode():
                val_loss, val_metrics = run_epoch(model, val_loader, device, None, scaler, amp_enabled, float(training["target_scale"]), None)
            row = {"epoch": epoch, "train_loss": train_loss, "train_rmse": train_metrics["rmse"], "train_nmse": train_metrics["nmse"], "val_loss": val_loss, "val_rmse": val_metrics["rmse"], "val_nmse": val_metrics["nmse"]}
            writer.writerow(row); handle.flush()
            print(f"epoch={epoch:03d} train_rmse={train_metrics['rmse']:.4f} val_rmse={val_metrics['rmse']:.4f} val_nmse={val_metrics['nmse']:.6f}")
            state = {"epoch": epoch, "model": model.state_dict(), "config": config, "val_metrics": val_metrics}
            save_checkpoint(checkpoint_dir / "latest.pt", state)
            if val_metrics["nmse"] < best:
                best = val_metrics["nmse"]; patience = 0
                save_checkpoint(checkpoint_dir / "best.pt", state)
            else:
                patience += 1
            if not args.smoke_test and patience >= int(training["early_stopping_patience"]):
                print(f"early_stopping epoch={epoch}"); break
    (output / "run_complete.json").write_text(
        json.dumps({"epoch": epoch, "best": best, "device": str(device)}, indent=2),
        encoding="utf-8",
    )
    print(f"finished experiment={name} device={device} best={best:.6f} output={output}")


if __name__ == "__main__":
    main()
