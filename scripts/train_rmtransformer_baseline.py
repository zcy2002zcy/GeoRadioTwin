"""Train the independent height-adapted RMTransformer-style baseline.

The source paper specifies a four-stage MaxViT encoder and CNN decoder but
does not provide author code.  This script therefore records the run as a
paper-guided adaptation and trains it from scratch under the common
scene-disjoint, zero-measurement, valid-receiver protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dense_baseline_dataset import DenseBaselineDataset
from models.rmtransformer_baseline import build_rmtransformer_adapted
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.seed import seed_everything


METHOD_LABEL = "RMTransformer-style (height-adapted)"
SOURCE_DOI = "10.1109/VTC2025-SPRING65109.2025.11174709"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rmtransformer_adapted.yaml")
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{name}' must be a mapping")
    return value


def model_kwargs(config: dict[str, Any], *, smoke_test: bool) -> dict[str, Any]:
    model = _section(config, "model")
    kwargs: dict[str, Any] = {
        "in_channels": int(model.get("input_channels", 3)),
        "stem_channels": int(model.get("stem_channels", 128)),
        "encoder_channels": tuple(int(value) for value in model.get(
            "encoder_channels", (128, 256, 512, 1024)
        )),
        "num_heads": tuple(int(value) for value in model.get(
            "num_heads", (4, 8, 16, 32)
        )),
        "partition_size": int(model.get("partition_size", 8)),
        "mlp_ratio": float(model.get("mlp_ratio", 4.0)),
        "mbconv_expansion": float(model.get("mbconv_expansion", 4.0)),
        "attention_dropout": float(model.get("attention_dropout", 0.0)),
        "projection_dropout": float(model.get("projection_dropout", 0.0)),
        "use_gradient_checkpointing": bool(
            model.get("use_gradient_checkpointing", True)
        ),
    }
    if smoke_test:
        # Retain all architectural stages and both attention axes while making
        # CPU-only continuous-integration runs practical.
        kwargs.update(
            {
                "stem_channels": 16,
                "encoder_channels": (16, 32, 64, 128),
                "num_heads": (1, 2, 4, 8),
                "mlp_ratio": 2.0,
                "mbconv_expansion": 2.0,
                "use_gradient_checkpointing": False,
            }
        )
    return kwargs


def masked_statistics(
    prediction: torch.Tensor, target: torch.Tensor, support: torch.Tensor
) -> tuple[torch.Tensor, float, float, int]:
    prediction_valid = prediction[support].float()
    target_valid = target[support].float()
    if prediction_valid.numel() == 0:
        raise ValueError("A batch contains no valid receiver pixels")
    # Optimize the paper's [0, 1] image-regression MSE, but select checkpoints
    # in the project's declared primary domain x / 127.5 - 1 == 2*x - 1.
    unit_interval_error = prediction_valid - target_valid
    loss = unit_interval_error.square().mean()
    primary_error = (prediction_valid * 2.0 - 1.0) - (target_valid * 2.0 - 1.0)
    primary_target = target_valid * 2.0 - 1.0
    return (
        loss,
        primary_error.square().sum().item(),
        primary_target.square().sum().item(),
        primary_error.numel(),
    )


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    gradient_clip: float,
    max_steps: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = sse = target_sq = 0.0
    count = steps = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        support = (
            batch["valid_mask"].to(device, non_blocking=True).bool()
            & ~batch["observation_mask"].to(device, non_blocking=True).bool()
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(inputs)
            loss, batch_sse, batch_target_sq, batch_count = masked_statistics(
                prediction, target, support
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite RMTransformer loss at step {steps + 1}"
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
        loss_sum += loss.detach().item()
        sse += batch_sse
        target_sq += batch_target_sq
        count += batch_count
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    if steps == 0 or count == 0:
        raise ValueError("The selected split produced no evaluable batches")
    return {
        "loss": loss_sum / steps,
        "rmse": math.sqrt(sse / count),
        "nmse": sse / max(target_sq, 1e-12),
    }


def make_dataset(
    *,
    split: str,
    config: dict[str, Any],
    data_seed: int,
    smoke_test: bool,
    training: bool,
) -> DenseBaselineDataset:
    data = _section(config, "data")
    default_maximum = {"train": 4096, "val": 128}[split]
    maximum = int(data.get(f"{split}_maps", default_maximum))
    if smoke_test:
        maximum = 2 if split == "train" else 1
    return DenseBaselineDataset(
        PROJECT_ROOT / "data" / "processed" / "splits" / f"{split}.csv",
        max_records=maximum,
        selection_strategy=(
            "sequential" if smoke_test else str(data.get("selection_strategy", "balanced_scene"))
        ),
        seed=data_seed,
        sample_rate=0.0,
        training=training,
    )


def main() -> None:
    args = arguments()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config, resolved_config_path = load_config(config_path)
    experiment = _section(config, "experiment")
    training = _section(config, "training")
    data = _section(config, "data")
    if float(experiment.get("sample_rate", 0.0)) != 0.0:
        raise ValueError("RMTransformer baseline is registered only for zero sampling")
    seed = int(args.seed if args.seed is not None else experiment.get("seed", 42))
    data_seed = int(args.data_seed if args.data_seed is not None else data.get("data_seed", 42))
    epochs = int(args.epochs if args.epochs is not None else training.get("epochs", 100))
    batch_size = int(
        args.batch_size if args.batch_size is not None else training.get("batch_size", 1)
    )
    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else training.get("learning_rate", 1e-4)
    )
    num_workers = int(
        args.num_workers if args.num_workers is not None else training.get("num_workers", 0)
    )
    configured_steps = training.get("max_steps_per_epoch", 1000)
    max_steps = (
        args.max_steps_per_epoch
        if args.max_steps_per_epoch is not None
        else (None if configured_steps in (None, 0) else int(configured_steps))
    )
    selection_metric = str(training.get("selection_metric", "nmse"))
    if selection_metric not in {"rmse", "nmse"}:
        raise ValueError(
            "RMTransformer checkpoint selection_metric must be 'rmse' or 'nmse'"
        )
    if args.smoke_test:
        epochs, batch_size, num_workers, max_steps = 2, 1, 0, 1
    if not torch.cuda.is_available() and not args.smoke_test:
        raise RuntimeError("Formal RMTransformer training requires CUDA; activate rmdm5070")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    seed_everything(seed)

    name = f"rmtransformer_adapted_random0_seed{seed}"
    if args.smoke_test:
        name += "_smoke"
    output = PROJECT_ROOT / "outputs" / name
    checkpoint_dir = output / "checkpoints"
    log_dir = output / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    source_config_digest = hashlib.sha256(resolved_config_path.read_bytes()).hexdigest()
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    kwargs = model_kwargs(config, smoke_test=args.smoke_test)
    metadata = {
        "method_label": METHOD_LABEL,
        "implementation_status": "independent_paper_guided_adaptation",
        "faithful_reproduction": False,
        "official_code_available": False,
        "external_pretraining": False,
        "source": {
            "authors": "Y. Li, C. Zhang, W. Wang, and Y. Huang",
            "title": "RMTransformer: Accurate Radio Map Construction and Coverage Prediction",
            "venue": "IEEE VTC2025-Spring",
            "doi": SOURCE_DOI,
            "arxiv": "2501.05190",
        },
        "preserved_components": [
            "CNN stem at half input resolution",
            "four-stage multi-scale MaxViT-style encoder",
            "local block and global grid attention in every encoder stage",
            "reported encoder feature widths 128/256/512/1024",
            "CNN transposed-convolution decoder with multi-scale skip connections",
        ],
        "adaptations_and_assumptions": [
            "third Rx-height plane added to the paper's building/Tx input",
            "continuous building height replaces the paper's binary blockage map",
            "UrbanRadio3D scene-disjoint 4096/128/128 protocol",
            "valid-receiver-only masked loss and evaluation",
            "zero radio measurements and no external pretraining",
            "independent multi-axis attention implementation because no author code was identified",
            "common 100-epoch ceiling with early stopping instead of the paper's 50-epoch fixed run",
        ],
        "input_channels": [
            "building_height_normalized",
            "tx_source",
            "rx_height_plane",
        ],
        "seed": seed,
        "data_seed": data_seed,
        "sample_rate": 0.0,
        "support": "valid_rx_only",
        "training_loss_domain": "unit_interval_0_to_1_mse",
        "checkpoint_selection_metric": selection_metric,
        "checkpoint_selection_domain": "normalized_minus1_to_1",
        "config": str(resolved_config_path),
        "config_sha256": config_digest,
        "source_config_sha256": source_config_digest,
        "config_hash_scope": "canonical_resolved_config_including_base",
        "model_config": kwargs,
    }
    (output / "provenance.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    train_set = make_dataset(
        split="train", config=config, data_seed=data_seed,
        smoke_test=args.smoke_test, training=True,
    )
    val_set = make_dataset(
        split="val", config=config, data_seed=data_seed,
        smoke_test=args.smoke_test, training=False,
    )
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, shuffle=True, generator=generator, **loader_args
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_rmtransformer_adapted(**kwargs).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer_name = str(training.get("optimizer", "Adam")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported optimizer for RMTransformer: {optimizer_name}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(training.get("min_learning_rate", 1e-5)),
    )
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    gradient_clip = float(training.get("gradient_clip", 1.0))
    patience_limit = int(training.get("early_stopping_patience", 15))
    print(
        f"model={METHOD_LABEL} device={device} parameters={parameter_count} "
        f"amp={amp_enabled} batch_size={batch_size}"
    )

    best = math.inf
    best_epoch = 0
    best_val_nmse = math.inf
    patience = 0
    stop_reason = "max_epochs"
    history_fields = (
        "epoch", "train_loss", "train_rmse", "train_nmse",
        "val_loss", "val_rmse", "val_nmse", "learning_rate",
    )
    with (log_dir / "train_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_set.set_epoch(epoch)
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                gradient_clip=gradient_clip,
                max_steps=max_steps,
            )
            with torch.inference_mode():
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    gradient_clip=gradient_clip,
                    max_steps=None,
                )
            row = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} "
                f"val_nmse={val_metrics['nmse']:.6f}"
            )
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "model_config": kwargs,
                "metadata": metadata,
                "val_metrics": val_metrics,
            }
            save_checkpoint(checkpoint_dir / "latest.pt", state)
            best_val_nmse = min(best_val_nmse, val_metrics["nmse"])
            if val_metrics[selection_metric] < best:
                best = val_metrics[selection_metric]
                best_epoch = epoch
                patience = 0
                save_checkpoint(checkpoint_dir / "best.pt", state)
            else:
                patience += 1
            scheduler.step()
            if not args.smoke_test and patience >= patience_limit:
                stop_reason = "early_stopping"
                print(f"early_stopping epoch={epoch}")
                break
    (output / "run_complete.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "data_seed": data_seed,
                "best_epoch": best_epoch,
                "last_epoch": epoch,
                "stop_reason": stop_reason,
                "best": best,
                "best_val_nmse": best_val_nmse,
                "selection_metric": selection_metric,
                "selection_domain": "normalized_minus1_to_1",
                "device": str(device),
                "parameter_count": parameter_count,
                "config_sha256": config_digest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"finished baseline={name} device={device} best={best:.6f} output={output}"
    )


if __name__ == "__main__":
    main()
