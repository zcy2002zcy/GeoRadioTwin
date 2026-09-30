"""Fast overfit check for the RadioDiff-3D data/loss/sampling pipeline.

This is a diagnostic, not a paper result.  It trains a small denoiser on one
downsampled training volume and verifies that the reverse process can recover
that same volume.  Failure here indicates an implementation problem rather
than a generalization or dataset-size problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiodiff3d_volume_dataset import RadioDiff3DVolumeDataset
from models.radiodiff3d_adapted import build_radiodiff3d_adapted
from utils.seed import seed_everything


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--loss-type", choices=("l1", "l2"), default="l2")
    parser.add_argument(
        "--parameterization", choices=("epsilon", "x0"), default="epsilon"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/diagnostics/radiodiff3d_pipeline_overfit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.steps <= 0 or args.size <= 0 or args.diffusion_steps < 2:
        raise ValueError("steps/size must be positive and diffusion-steps >= 2")
    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RadioDiff3DVolumeDataset(
        PROJECT_ROOT / "data" / "processed" / "splits" / "train.csv",
        max_volumes=1,
        selection_strategy="sequential",
        seed=42,
        condition_layout="paper_v2",
    )
    sample = dataset[0]
    depth = sample["target"].shape[1]
    output_size = (depth, args.size, args.size)
    condition = F.interpolate(
        sample["condition"].unsqueeze(0),
        size=output_size,
        mode="trilinear",
        align_corners=False,
    ).to(device)
    target = F.interpolate(
        sample["target"].unsqueeze(0),
        size=output_size,
        mode="trilinear",
        align_corners=False,
    ).to(device)
    valid = F.interpolate(
        sample["valid_mask"].unsqueeze(0).float(),
        size=output_size,
        mode="nearest",
    ).bool().to(device)
    model = build_radiodiff3d_adapted(
        condition_channels=3,
        target_channels=1,
        base_channels=8,
        channel_multipliers=(1, 2, 4),
        time_dimension=32,
        expected_depth=4,
        multiscale_conditioning=True,
        timesteps=args.diffusion_steps,
        beta_schedule="cosine",
        parameterization=args.parameterization,
        loss_type=args.loss_type,
        loss_support="full",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    losses: list[float] = []
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.training_loss(condition, target, valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"step={step:04d} diffusion_loss={losses[-1]:.6f}")

    model.eval()
    generator = torch.Generator(device=device.type).manual_seed(2027)
    prediction = model.ddim_sample(
        condition,
        sampling_steps=args.diffusion_steps,
        generator=generator,
        eta=0.0,
        clip_denoised=True,
    )
    clean = target * 2.0 - 1.0
    support = valid.expand_as(prediction)
    valid_rmse = torch.sqrt(
        (prediction[support].float() - clean[support].float()).square().mean()
    ).item()
    full_rmse = torch.sqrt((prediction.float() - clean.float()).square().mean()).item()
    zero_condition_prediction = model.ddim_sample(
        torch.zeros_like(condition),
        sampling_steps=args.diffusion_steps,
        generator=torch.Generator(device=device.type).manual_seed(2027),
        eta=0.0,
        clip_denoised=True,
    )
    condition_effect_mae = (
        prediction.float() - zero_condition_prediction.float()
    ).abs().mean().item()
    single_step_denoising_rmse: dict[str, float] = {}
    fixed_noise = torch.randn(
        clean.shape,
        device=device,
        dtype=clean.dtype,
        generator=torch.Generator(device=device.type).manual_seed(3031),
    )
    for timestep in sorted({0, args.diffusion_steps // 2, args.diffusion_steps - 1}):
        timestep_tensor = torch.tensor([timestep], device=device, dtype=torch.long)
        noisy = model.q_sample(clean, timestep_tensor, fixed_noise)
        model_output = model.denoiser(noisy, condition, timestep_tensor)
        predicted_clean, _ = model.model_predictions(noisy, model_output, timestep)
        rmse = torch.sqrt(
            (
                predicted_clean[support].float()
                - clean[support].float()
            ).square().mean()
        ).item()
        single_step_denoising_rmse[str(timestep)] = rmse
    payload = {
        "diagnostic_only": True,
        "device": str(device),
        "seed": args.seed,
        "optimizer_steps": args.steps,
        "resolution": [args.size, args.size],
        "diffusion_steps": args.diffusion_steps,
        "condition_layout": "paper_v2",
        "loss_type": args.loss_type,
        "parameterization": args.parameterization,
        "loss_support": "full",
        "first_20_loss_mean": sum(losses[:20]) / min(20, len(losses)),
        "last_20_loss_mean": sum(losses[-20:]) / min(20, len(losses)),
        "overfit_valid_rmse": valid_rmse,
        "overfit_full_rmse": full_rmse,
        "condition_effect_mae": condition_effect_mae,
        "single_step_denoising_rmse": single_step_denoising_rmse,
        "interpretation": (
            "Pipeline can overfit one downsampled volume"
            if valid_rmse < 0.20
            else "Pipeline did not pass the strict one-volume overfit threshold"
        ),
    }
    output = args.output
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
