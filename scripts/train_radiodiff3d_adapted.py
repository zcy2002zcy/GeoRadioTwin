"""Train the independent RadioDiff-3D-style conditional diffusion baseline.

Training uses random DDPM timesteps on strict h1--h4 volumes.  The configured
target is either epsilon or x0.  Validation uses one fixed-seed deterministic
DDIM sample per volume; there is no best-of-K selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiodiff3d_volume_dataset import RadioDiff3DVolumeDataset
from models.radiodiff3d_adapted import RadioDiff3DAdapted, build_radiodiff3d_adapted
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.seed import seed_everything


METHOD_LABEL = "RadioDiff-3D-style conditional diffusion (adapted)"
SOURCE_DOI = "10.1109/TNSE.2025.3590545"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/radiodiff3d_adapted.yaml")
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, dict):
        raise TypeError(f"Config section '{name}' must be a mapping")
    return section


def _build_model(config: dict[str, Any], *, smoke_test: bool) -> tuple[RadioDiff3DAdapted, dict[str, Any]]:
    model_config = _section(config, "model")
    diffusion = _section(config, "diffusion")
    base_channels = int(model_config.get("base_channels", 32))
    if smoke_test:
        base_channels = min(base_channels, 8)
    resolved = {
        "condition_channels": int(model_config.get("condition_channels", 3)),
        "target_channels": int(model_config.get("target_channels", 1)),
        "base_channels": base_channels,
        "channel_multipliers": tuple(int(v) for v in model_config.get("channel_multipliers", (1, 2, 4, 6))),
        "time_dimension": int(model_config.get("time_dimension", 128)),
        "dropout": float(model_config.get("dropout", 0.0)),
        "expected_depth": int(model_config.get("expected_depth", 4)),
        "multiscale_conditioning": bool(
            model_config.get("multiscale_conditioning", False)
        ),
        "timesteps": int(diffusion.get("timesteps", 1000)),
        "beta_schedule": str(diffusion.get("beta_schedule", "cosine")),
        "parameterization": str(diffusion.get("parameterization", "epsilon")),
        "loss_type": str(diffusion.get("loss_type", "l2")),
        "loss_support": str(diffusion.get("loss_support", "valid_rx")),
    }
    return build_radiodiff3d_adapted(**resolved), resolved


def _dataset(
    config: dict[str, Any], *, split: str, data_seed: int, smoke_test: bool
) -> RadioDiff3DVolumeDataset:
    data = _section(config, "data")
    maximum = int(data.get(f"{split}_volumes", {"train": 1024, "val": 32, "test": 32}[split]))
    if smoke_test:
        maximum = 1
    return RadioDiff3DVolumeDataset(
        PROJECT_ROOT / "data" / "processed" / "splits" / f"{split}.csv",
        max_volumes=maximum,
        selection_strategy="sequential" if smoke_test else str(data.get("selection_strategy", "balanced_scene")),
        seed=data_seed,
        expected_rx_heights=data.get("expected_rx_heights", (1.0, 2.0, 3.0, 4.0)),
        exclude_target_values=data.get("exclude_target_values", (0.0,)),
        building_scale=float(data.get("building_scale", 255.0)),
        rx_height_scale=float(data.get("rx_height_scale", 20.0)),
        condition_layout=str(data.get("condition_layout", "legacy_height")),
    )


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


def _resize_for_smoke(
    condition: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, size: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    depth = condition.shape[2]
    output_size = (depth, size, size)
    return (
        F.interpolate(condition, size=output_size, mode="trilinear", align_corners=False),
        F.interpolate(target, size=output_size, mode="trilinear", align_corners=False),
        F.interpolate(valid.float(), size=output_size, mode="nearest").bool(),
    )


@torch.no_grad()
def _update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


@torch.inference_mode()
def _validate(
    model: RadioDiff3DAdapted,
    loader: DataLoader,
    device: torch.device,
    *,
    sampling_steps: int,
    sampling_seed: int,
    eta: float,
    clip_denoised: bool,
    smoke_test: bool,
) -> dict[str, float]:
    model.eval()
    rmse_values: list[float] = []
    nmse_values: list[float] = []
    volume_index = 0
    for batch in loader:
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True).bool()
        if smoke_test:
            condition, target, valid = _resize_for_smoke(condition, target, valid)
        for local_index in range(condition.shape[0]):
            generator = torch.Generator(device=device.type)
            generator.manual_seed(sampling_seed + volume_index)
            prediction = model.ddim_sample(
                condition[local_index : local_index + 1],
                sampling_steps=sampling_steps,
                generator=generator,
                eta=eta,
                clip_denoised=clip_denoised,
            )
            target_normalized = target[local_index : local_index + 1] * 2.0 - 1.0
            for depth_index in range(prediction.shape[2]):
                support = valid[local_index, 0, depth_index]
                if not support.any():
                    raise ValueError("Validation map has no valid receiver pixels")
                error = prediction[0, 0, depth_index][support].float() - target_normalized[0, 0, depth_index][support].float()
                mse = error.square().mean()
                target_power = target_normalized[0, 0, depth_index][support].float().square().mean().clamp_min(1e-12)
                rmse_values.append(torch.sqrt(mse).item())
                nmse_values.append((mse / target_power).item())
            volume_index += 1
    if not nmse_values:
        raise ValueError("Validation dataset is empty")
    return {
        "rmse": sum(rmse_values) / len(rmse_values),
        "nmse": sum(nmse_values) / len(nmse_values),
        "maps": float(len(nmse_values)),
    }


def main() -> None:
    args = arguments()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config, config_path = load_config(config_path)
    experiment = _section(config, "experiment")
    data = _section(config, "data")
    training = _section(config, "training")
    diffusion = _section(config, "diffusion")
    evaluation = _section(config, "evaluation")
    if float(experiment.get("sample_rate", 0.0)) != 0.0:
        raise ValueError("This RadioDiff-3D baseline is defined for zero sparse measurements")
    if int(diffusion.get("best_of_k", 1)) != 1 or bool(evaluation.get("best_of_k_selection", False)):
        raise ValueError("Academic evaluation forbids best-of-K diffusion sample selection")

    seed = int(args.seed if args.seed is not None else experiment.get("seed", 42))
    data_seed = int(args.data_seed if args.data_seed is not None else data.get("data_seed", 42))
    epochs = int(args.epochs if args.epochs is not None else training.get("epochs", 100))
    batch_size = int(args.batch_size if args.batch_size is not None else training.get("batch_size", 1))
    learning_rate = float(args.learning_rate if args.learning_rate is not None else training.get("learning_rate", 5e-5))
    num_workers = int(args.num_workers if args.num_workers is not None else training.get("num_workers", 0))
    gradient_accumulation_steps = int(
        training.get("gradient_accumulation_steps", 1)
    )
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    max_steps = training.get("max_steps_per_epoch", 1000)
    max_steps = None if max_steps in (None, 0) else int(max_steps)
    if args.smoke_test:
        epochs, batch_size, num_workers, max_steps = 1, 1, 0, 1
        gradient_accumulation_steps = 1
    if not torch.cuda.is_available() and not args.smoke_test:
        raise RuntimeError("Formal RadioDiff-3D training requires CUDA; activate rmdm5070")
    seed_everything(seed)

    name = _experiment_name(config, seed, smoke_test=args.smoke_test)
    output = PROJECT_ROOT / "outputs" / name
    checkpoint_dir = output / "checkpoints"
    log_dir = output / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_set = _dataset(config, split="train", data_seed=data_seed, smoke_test=args.smoke_test)
    val_set = _dataset(config, split="val", data_seed=data_seed, smoke_test=args.smoke_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, resolved_model = _build_model(config, smoke_test=args.smoke_test)
    model.to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)

    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=float(training.get("weight_decay", 1e-6))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=float(training.get("minimum_learning_rate", 1e-6))
    )
    amp_enabled = device.type == "cuda" and bool(training.get("amp", True))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    ema_decay = float(training.get("ema_decay", 0.995))
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    gradient_clip = float(training.get("gradient_clip", 1.0))
    ema_update_every = int(training.get("ema_update_every", 1))
    if ema_update_every <= 0:
        raise ValueError("ema_update_every must be positive")
    validation_steps = 2 if args.smoke_test else int(diffusion.get("validation_sampling_steps", 20))
    sampling_seed = int(diffusion.get("sampling_seed", 2027))
    eta = float(diffusion.get("ddim_eta", 0.0))
    clip_denoised = bool(diffusion.get("clip_denoised", True))
    source_config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "method_label": str(experiment.get("method_label", METHOD_LABEL)),
        "implementation_status": "independent_upstream_and_paper_informed_adaptation",
        "faithful_reproduction": False,
        "official_3d_training_code_available": False,
        "external_pretraining": False,
        "source": {
            "title": "RadioDiff-3D / UrbanRadio3D",
            "doi": SOURCE_DOI,
            "radiodiff_repository": "https://github.com/UNIC-Lab/RadioDiff",
            "radiodiff_commit": "ee7c4c380037dd6aa30ab675d099b6f62199d3d5",
            "urbanradio3d_repository": "https://github.com/UNIC-Lab/UrbanRadio3D",
        },
        "adaptations_and_protocol": [
            "strictly paired h1-h4 target volumes",
            "3-D convolutional diffusion U-Net with spatial-only downsampling",
            f"condition layout: {data.get('condition_layout', 'legacy_height')}",
            f"diffusion parameterization: {resolved_model['parameterization']}",
            f"diffusion loss: {resolved_model['loss_type']} on {resolved_model['loss_support']} support",
            f"multiscale condition injection: {resolved_model['multiscale_conditioning']}",
            "direct voxel-space diffusion without external VAE or pretrained weights",
            "paper-reported learning rate 5e-5 and EMA decay 0.995 retained",
            "scene-disjoint common project split",
            "fixed-seed deterministic DDIM validation and test sampling",
            "exactly one generated volume per condition; no best-of-K selection",
            "UrbanRadio3D PNG code supervision; raw values are not asserted to be dB",
        ],
        "sample_rate": 0.0,
        "support": "valid_rx_only",
        "condition_layout": str(data.get("condition_layout", "legacy_height")),
        "seed": seed,
        "data_seed": data_seed,
        "train_volumes": len(train_set),
        "train_maps": len(train_set) * len(data.get("expected_rx_heights", (1, 2, 3, 4))),
        "training_slice_budget": len(train_set) * len(data.get("expected_rx_heights", (1, 2, 3, 4))),
        "val_volumes": len(val_set),
        "evaluation_selection_protocol": "common_balanced_scene_anchor_maps",
        "evaluation_conditioning_volumes": int(data.get("test_conditioning_volumes", 128)),
        "evaluation_scored_maps": int(data.get("test_scored_maps", 128)),
        "validation_sampling_seed": sampling_seed,
        "validation_ddim_steps": validation_steps,
        "ddim_eta": eta,
        "ema_update_every": ema_update_every,
        "micro_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "best_of_k": 1,
        "config": str(config_path),
        "config_sha256": config_digest,
        "source_config_sha256": source_config_digest,
        "config_hash_scope": "canonical_resolved_config_including_base",
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"baseline={name} device={device} parameters={parameter_count} "
        f"train_volumes={len(train_set)} val_volumes={len(val_set)}"
    )

    history_fields = (
        "epoch",
        "train_diffusion_loss",
        "loss_type",
        "loss_support",
        "val_rmse",
        "val_nmse",
        "learning_rate",
    )
    best = math.inf
    best_epoch = 0
    patience = 0
    configured_patience = training.get("early_stopping_patience", 15)
    patience_limit = (
        None if configured_patience in (None, 0) else int(configured_patience)
    )
    stop_reason = "max_epochs"
    epoch = 0
    global_step = 0
    with (log_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            steps = 0
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(train_loader):
                condition = batch["condition"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                valid = batch["valid_mask"].to(device, non_blocking=True).bool()
                if args.smoke_test:
                    condition, target, valid = _resize_for_smoke(condition, target, valid)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    loss, _ = model.training_loss(condition, target, valid)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite diffusion loss at epoch={epoch}, step={steps + 1}: {loss.item()}"
                    )
                scaler.scale(loss / gradient_accumulation_steps).backward()
                loss_sum += loss.item()
                steps += 1
                reached_loader_end = batch_index + 1 == len(train_loader)
                reached_max_steps = max_steps is not None and steps >= max_steps
                should_update = (
                    steps % gradient_accumulation_steps == 0
                    or reached_loader_end
                    or reached_max_steps
                )
                if should_update:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % ema_update_every == 0:
                        _update_ema(ema_model, model, ema_decay)
                if max_steps is not None and steps >= max_steps:
                    break
            validation = _validate(
                ema_model,
                val_loader,
                device,
                sampling_steps=validation_steps,
                sampling_seed=sampling_seed,
                eta=eta,
                clip_denoised=clip_denoised,
                smoke_test=args.smoke_test,
            )
            row = {
                "epoch": epoch,
                "train_diffusion_loss": loss_sum / max(steps, 1),
                "loss_type": resolved_model["loss_type"],
                "loss_support": resolved_model["loss_support"],
                "val_rmse": validation["rmse"],
                "val_nmse": validation["nmse"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch={epoch:03d} train_diffusion_loss={row['train_diffusion_loss']:.6f} "
                f"val_rmse={row['val_rmse']:.6f} val_nmse={row['val_nmse']:.6f}"
            )
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metadata": metadata,
                "model_config": resolved_model,
                "diffusion_sampling": {
                    "validation_steps": validation_steps,
                    "sampling_seed": sampling_seed,
                    "ddim_eta": eta,
                    "best_of_k": 1,
                },
                "val_metrics": validation,
            }
            save_checkpoint(checkpoint_dir / "latest.pt", state)
            if validation["nmse"] < best:
                best = validation["nmse"]
                best_epoch = epoch
                patience = 0
                save_checkpoint(checkpoint_dir / "best.pt", state)
            else:
                patience += 1
            scheduler.step()
            if (
                not args.smoke_test
                and patience_limit is not None
                and patience >= patience_limit
            ):
                stop_reason = "early_stopping"
                print(f"early_stopping epoch={epoch} best_epoch={best_epoch}")
                break

    completion = {
        "last_epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_nmse": best,
        "stop_reason": stop_reason,
        "device": str(device),
        "seed": seed,
        "data_seed": data_seed,
        "parameter_count": parameter_count,
        "global_steps": global_step,
        "micro_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "loss_type": resolved_model["loss_type"],
        "loss_support": resolved_model["loss_support"],
        "parameterization": resolved_model["parameterization"],
        "condition_layout": str(data.get("condition_layout", "legacy_height")),
        "config_sha256": config_digest,
        "source_config_sha256": source_config_digest,
        "config_hash_scope": "canonical_resolved_config_including_base",
    }
    (output / "run_complete.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(f"finished baseline={name} device={device} best={best:.6f} output={output}")


if __name__ == "__main__":
    main()
