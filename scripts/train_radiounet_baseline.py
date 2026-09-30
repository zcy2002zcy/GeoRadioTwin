"""Train a documented RadioUNet-style adapted baseline on the common split."""

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

from datasets.dense_baseline_dataset import DenseBaselineDataset
from models.radiounet_baseline import RadioUNetBaseline
from models.pmnet_baseline import build_pmnet_adapted
from utils.checkpoint import save_checkpoint
from utils.seed import seed_everything


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=float, choices=(0.0, 0.01, 0.05, 0.10), required=True)
    parser.add_argument("--architecture", choices=("radiounet", "pmnet"), default="radiounet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def masked_totals(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float, float, int]:
    error = prediction[mask] - target[mask]
    loss = error.square().mean()
    return loss, error.square().sum().item(), target[mask].square().sum().item(), error.numel()


def main() -> None:
    args = arguments()
    if args.architecture == "pmnet" and args.sample_rate != 0.0:
        raise ValueError("PMNet public baseline is sampling-free; use --sample-rate 0.0")
    if not torch.cuda.is_available() and not args.smoke_test:
        raise RuntimeError("Formal baseline training requires CUDA; activate rmdm5070")
    seed_everything(args.seed)
    rate_label = str(int(round(args.sample_rate * 100)))
    name = f"{args.architecture}_adapted_random{rate_label}_seed{args.seed}"
    if args.smoke_test:
        name += "_smoke"
    output = PROJECT_ROOT / "outputs" / name
    checkpoint_dir = output / "checkpoints"
    log_dir = output / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    is_pmnet = args.architecture == "pmnet"
    metadata = {
        "method_label": "PMNet-v1 (adapted)" if is_pmnet else "RadioUNet-style (adapted)",
        "upstream": "https://github.com/abman23/PMNet" if is_pmnet else "https://github.com/RonLevie/RadioUNet",
        "upstream_commit": "a0e0c5926de721074beeb23f630f2d313f6508dd" if is_pmnet else "36ab70663443af615c37051ce97da14d04925c7a",
        "license": "MIT",
        "adaptations": ["UrbanRadio3D", "Rx-height input channel", "scene-disjoint split", "valid-Rx loss"],
        "sample_rate": args.sample_rate,
        "seed": args.seed,
        "data_seed": args.data_seed,
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    split_root = PROJECT_ROOT / "data" / "processed" / "splits"
    train_set = DenseBaselineDataset(
        split_root / "train.csv", max_records=4 if args.smoke_test else 4096,
        selection_strategy="sequential" if args.smoke_test else "balanced_scene",
        seed=args.data_seed, sample_rate=args.sample_rate, training=True,
    )
    val_set = DenseBaselineDataset(
        split_root / "val.csv", max_records=2 if args.smoke_test else 128,
        selection_strategy="sequential" if args.smoke_test else "balanced_scene",
        seed=args.data_seed, sample_rate=args.sample_rate, training=False,
    )
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    model = (
        build_pmnet_adapted(3)
        if is_pmnet
        else RadioUNetBaseline(5 if args.sample_rate > 0 else 3)
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    # Official PMNet-v1 is numerically unstable with FP16 at 256x256 and a
    # small BatchNorm batch. Keep it in FP32; RadioUNet may still use AMP.
    amp_enabled = device.type == "cuda" and not is_pmnet
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    best = math.inf
    patience = 0
    epochs = 2 if args.smoke_test else args.epochs
    with (log_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "val_rmse", "val_nmse"))
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_set.set_epoch(epoch)
            model.train()
            loss_sum = 0.0
            steps = 0
            for batch in train_loader:
                inputs = batch["input"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                valid = batch["valid_mask"].to(device, non_blocking=True).bool()
                observed = batch["observation_mask"].to(device, non_blocking=True).bool()
                support = valid & ~observed
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    prediction = model(inputs)
                    loss, _, _, _ = masked_totals(prediction, target, support)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch={epoch}, step={steps + 1}; "
                        "the run was stopped before corrupt checkpoints were written"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item()
                steps += 1
                if args.smoke_test or steps >= 1000:
                    break
            model.eval()
            sse = target_sq = 0.0
            count = 0
            with torch.inference_mode():
                for batch in val_loader:
                    inputs = batch["input"].to(device, non_blocking=True)
                    target = batch["target"].to(device, non_blocking=True)
                    support = batch["valid_mask"].to(device).bool() & ~batch["observation_mask"].to(device).bool()
                    _, batch_sse, batch_target_sq, batch_count = masked_totals(model(inputs), target, support)
                    sse += batch_sse
                    target_sq += batch_target_sq
                    count += batch_count
            val_rmse = math.sqrt(sse / count)
            val_nmse = sse / max(target_sq, 1e-12)
            row = {"epoch": epoch, "train_loss": loss_sum / max(steps, 1), "val_rmse": val_rmse, "val_nmse": val_nmse}
            writer.writerow(row)
            handle.flush()
            print(f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} val_rmse={val_rmse:.6f} val_nmse={val_nmse:.6f}")
            state = {"epoch": epoch, "model": model.state_dict(), "metadata": metadata, "val_nmse": val_nmse}
            save_checkpoint(checkpoint_dir / "latest.pt", state)
            if val_nmse < best:
                best = val_nmse
                patience = 0
                save_checkpoint(checkpoint_dir / "best.pt", state)
            else:
                patience += 1
            if not args.smoke_test and patience >= 15:
                print(f"early_stopping epoch={epoch}")
                break
    (output / "run_complete.json").write_text(
        json.dumps({"epoch": epoch, "best": best, "device": str(device)}, indent=2),
        encoding="utf-8",
    )
    print(f"finished baseline={name} device={device} best={best:.6f} output={output}")


if __name__ == "__main__":
    main()
