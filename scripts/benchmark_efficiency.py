"""Benchmark parameters, latency, VRAM, and scoped FLOPs at 256x256 resolution."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pefnet_inspired_dataset import PEFNetInspiredDataset
from datasets.radiodiff3d_volume_dataset import RadioDiff3DVolumeDataset
from models.pefnet_inspired_baseline import build_pefnet_inspired
from models.radiodiff3d_adapted import build_radiodiff3d_adapted
from models.radiounet_baseline import RadioUNetBaseline
from models.rmtransformer_baseline import build_rmtransformer_adapted
from models.unirm_baseline import build_unirm_adapted
from scripts.twin_validation_common import load_hybrid, make_dataset, write_csv, write_json
from utils.config import load_config


class HybridPointWrapper(nn.Module):
    def __init__(self, model: nn.Module, raster_shape: tuple[int, int]) -> None:
        super().__init__()
        self.model = model
        self.raster_shape = raster_shape

    def forward(self, coordinates: Tensor, geometry: Tensor, feature_map: Tensor) -> Tensor:
        return self.model.forward_with_encoded(
            coordinates, geometry, feature_map, self.raster_shape
        )


class PEFNetPredictionWrapper(nn.Module):
    """Expose only the refined map while preserving all benchmarked parameters."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)["prediction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_geo.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/twin_efficiency_seed42"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--diffusion-warmup", type=int, default=2)
    parser.add_argument("--diffusion-repeats", type=int, default=10)
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timings(function, device: torch.device, warmup: int, repeats: int) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        _sync(device)
        values = []
        for _ in range(repeats):
            start = time.perf_counter()
            function()
            _sync(device)
            values.append((time.perf_counter() - start) * 1000.0)
    return values


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _flops(module: nn.Module, inputs: tuple[Tensor, ...]) -> tuple[float | None, dict, str | None]:
    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(module, inputs)
        total = float(analysis.total())
        unsupported = {key: int(value) for key, value in analysis.unsupported_ops().items()}
        return total, unsupported, None
    except Exception as error:  # pragma: no cover - backend/version dependent
        return None, {}, f"{type(error).__name__}: {error}"


def _checkpoint_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _load_dense(method: str, device: torch.device):
    if method == "radiounet":
        model = RadioUNetBaseline(3)
        output = PROJECT_ROOT / "outputs" / "radiounet_adapted_random0_seed42"
    elif method == "unirm":
        model = build_unirm_adapted(enable_prompt=True)
        output = PROJECT_ROOT / "outputs" / "unirm_adapted_random0_seed42"
    elif method == "pefnet_inspired":
        model = None
        output = (
            PROJECT_ROOT
            / "outputs"
            / "pefnet_inspired_adapted_stable_v2_random0_seed42"
        )
    elif method == "rmtransformer":
        model = None
        output = PROJECT_ROOT / "outputs" / "rmtransformer_adapted_random0_seed42"
    else:
        raise ValueError(method)
    checkpoint_path = output / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        return None
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if method == "pefnet_inspired":
        model_config = state.get("model_config", {})
        model = build_pefnet_inspired(
            in_channels=int(model_config.get("in_channels", 5)),
            base_channels=int(model_config.get("base_channels", 128)),
            max_norm_groups=int(model_config.get("max_norm_groups", 32)),
            force_refinement_fp32=bool(
                model_config.get("force_refinement_fp32", True)
            ),
        )
    elif method == "rmtransformer":
        model_config = state.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError(f"RMTransformer checkpoint lacks model_config: {checkpoint_path}")
        model_config = dict(model_config)
        model_config["use_gradient_checkpointing"] = False
        model = build_rmtransformer_adapted(**model_config)
    assert model is not None
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint_path


def _load_radiodiff3d(device: torch.device):
    output = PROJECT_ROOT / "outputs" / "radiodiff3d_paper_aligned_v2_random0_seed42"
    checkpoint_path = output / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        return None
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = state.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(f"RadioDiff-3D checkpoint lacks model_config: {checkpoint_path}")
    weights = state.get("ema_model", state.get("model"))
    if not isinstance(weights, dict):
        raise ValueError(f"RadioDiff-3D checkpoint lacks model/EMA weights: {checkpoint_path}")
    model = build_radiodiff3d_adapted(**model_config)
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    return model, checkpoint_path, "EMA" if "ema_model" in state else "raw"


def _pefnet_input(device: torch.device) -> Tensor:
    config, _ = load_config(PROJECT_ROOT / "configs" / "pefnet_inspired_adapted.yaml")
    data = config.get("data", {})
    physics = config.get("physics", {})
    dataset = PEFNetInspiredDataset(
        PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv",
        max_records=1,
        selection_strategy=str(data.get("selection_strategy", "balanced_scene")),
        seed=int(data.get("data_seed", 42)),
        training=False,
        dimensionless_wavenumber=float(physics.get("dimensionless_wavenumber", 0.35)),
    )
    return dataset[0]["input"].unsqueeze(0).to(device)


def _radiodiff3d_input(device: torch.device) -> tuple[Tensor, dict[str, object]]:
    config, _ = load_config(
        PROJECT_ROOT / "configs" / "radiodiff3d_paper_aligned_v2.yaml"
    )
    data = config.get("data", {})
    diffusion = config.get("diffusion", {})
    dataset = RadioDiff3DVolumeDataset(
        PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv",
        max_volumes=1,
        selection_strategy=str(data.get("selection_strategy", "balanced_scene")),
        seed=int(data.get("data_seed", 42)),
        expected_rx_heights=data.get("expected_rx_heights", (1.0, 2.0, 3.0, 4.0)),
        exclude_target_values=data.get("exclude_target_values", (0.0,)),
        building_scale=float(data.get("building_scale", 255.0)),
        rx_height_scale=float(data.get("rx_height_scale", 20.0)),
        condition_layout=str(data.get("condition_layout", "paper_v2")),
    )
    protocol: dict[str, object] = {
        "sampling_steps": int(diffusion.get("sampling_steps", 50)),
        "sampling_seed": int(diffusion.get("sampling_seed", 2027)),
        "ddim_eta": float(diffusion.get("ddim_eta", 0.0)),
        "clip_denoised": bool(diffusion.get("clip_denoised", True)),
        "maps_per_volume": int(data.get("maps_per_volume", 4)),
    }
    return dataset[0]["condition"].unsqueeze(0).to(device), protocol


def _measure_row(
    name: str,
    model: nn.Module,
    checkpoint: Path,
    function,
    device: torch.device,
    warmup: int,
    repeats: int,
    flop_value: float | None,
    unsupported: dict,
    flop_error: str | None,
    *,
    benchmark_unit: str = "single_map",
    maps_per_inference: int = 1,
    scored_maps_per_inference: int | None = None,
    latency_map_basis: str = "output_map",
    sampling_steps: int | None = None,
    flop_scope: str = "full_model_forward",
    weights: str = "raw",
) -> dict[str, object]:
    if maps_per_inference < 1:
        raise ValueError("maps_per_inference must be positive")
    if scored_maps_per_inference is None:
        scored_maps_per_inference = maps_per_inference
    if scored_maps_per_inference < 1:
        raise ValueError("scored_maps_per_inference must be positive")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        baseline = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            function()
        _sync(device)
        peak = int(torch.cuda.max_memory_allocated(device))
        incremental_peak = max(peak - baseline, 0)
    else:
        peak = incremental_peak = 0
    values = _timings(function, device, warmup, repeats)
    mean_latency = float(statistics.mean(values))
    median_latency = float(statistics.median(values))
    p10_latency = _percentile(values, 10)
    p90_latency = _percentile(values, 90)
    latency_std = float(statistics.pstdev(values))
    is_diffusion_trace = flop_scope == "single_denoiser_forward"
    sampling_denoiser_flops = (
        flop_value * sampling_steps
        if is_diffusion_trace and flop_value is not None and sampling_steps is not None
        else None
    )
    return {
        "method": name,
        "device": str(device),
        "precision": "fp32",
        "output_grid": "256x256",
        "benchmark_unit": benchmark_unit,
        "maps_per_inference": maps_per_inference,
        "generated_maps_per_inference": maps_per_inference,
        "accuracy_scored_maps_per_inference": scored_maps_per_inference,
        "latency_map_basis": latency_map_basis,
        "weights": weights,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "checkpoint_bytes": _checkpoint_size(checkpoint),
        "latency_ms_mean": mean_latency,
        "latency_ms_median": median_latency,
        "latency_ms_p10": p10_latency,
        "latency_ms_p90": p90_latency,
        "latency_ms_std": latency_std,
        "latency_ms_per_map_mean": mean_latency / maps_per_inference,
        "latency_ms_per_map_median": median_latency / maps_per_inference,
        "latency_ms_per_map_p10": p10_latency / maps_per_inference,
        "latency_ms_per_map_p90": p90_latency / maps_per_inference,
        "maps_per_second_from_median": maps_per_inference * 1000.0 / max(median_latency, 1e-12),
        "generated_maps_per_second_from_median": (
            maps_per_inference * 1000.0 / max(median_latency, 1e-12)
        ),
        "peak_cuda_bytes": peak,
        "incremental_peak_cuda_bytes": incremental_peak,
        "fvcore_traced_flops": flop_value,
        "flop_scope": flop_scope,
        "flops_complete": (
            flop_value is not None and not unsupported and not is_diffusion_trace
        ),
        "sampling_steps": sampling_steps if sampling_steps is not None else "",
        "single_denoiser_forward_fvcore_flops": flop_value if is_diffusion_trace else "",
        "sampling_denoiser_flops_estimate": (
            sampling_denoiser_flops if sampling_denoiser_flops is not None else ""
        ),
        "sampling_denoiser_flops_estimate_complete": False if is_diffusion_trace else "",
        "unsupported_operators": json.dumps(unsupported, sort_keys=True),
        "flop_error": flop_error,
        "warmup": warmup,
        "repeats": repeats,
    }


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.warmup = 1
        args.repeats = 2
        args.diffusion_warmup = 1
        args.diffusion_repeats = 1
        args.output = args.output.with_name(args.output.name + "_smoke")
    loaded = load_hybrid(args.config, args.checkpoint, device_name=args.device)
    device = loaded.device
    dataset = make_dataset(loaded, max_maps=1, selection_strategy="balanced_scene")
    sample = dataset[0]
    coordinates = sample["coordinate_features"].unsqueeze(0).to(device)
    geometry = sample["geometry_features"].unsqueeze(0).to(device)
    environment = sample["environment_map"].unsqueeze(0).to(device)
    dense_input = environment
    height, width = environment.shape[-2:]

    def hybrid_forward():
        feature_map = loaded.model.encode_environment(environment)
        outputs = []
        for start in range(0, coordinates.shape[1], args.query_batch_size):
            end = min(start + args.query_batch_size, coordinates.shape[1])
            outputs.append(loaded.model.forward_with_encoded(
                coordinates[:, start:end], geometry[:, start:end], feature_map, (height, width)
            ))
        return torch.cat(outputs, dim=1)

    # Trace the encoder exactly and a fixed query probe, then scale only the pointwise/ray-query part.
    probe_count = min(1024, coordinates.shape[1])
    with torch.inference_mode():
        feature_probe = loaded.model.encode_environment(environment)
    encoder_flops, encoder_unsupported, encoder_error = _flops(
        loaded.model.environment_encoder, (environment,)
    )
    point_wrapper = HybridPointWrapper(loaded.model, (height, width)).to(device).eval()
    point_flops, point_unsupported, point_error = _flops(
        point_wrapper,
        (coordinates[:, :probe_count], geometry[:, :probe_count], feature_probe),
    )
    hybrid_flops = None
    if encoder_flops is not None and point_flops is not None:
        hybrid_flops = encoder_flops + point_flops * (coordinates.shape[1] / probe_count)
    hybrid_unsupported = dict(encoder_unsupported)
    for key, value in point_unsupported.items():
        hybrid_unsupported[key] = hybrid_unsupported.get(key, 0) + value
    hybrid_error = "; ".join(value for value in (encoder_error, point_error) if value) or None

    rows = [_measure_row(
        "HybridGeo", loaded.model, loaded.checkpoint_path, hybrid_forward, device,
        args.warmup, args.repeats, hybrid_flops, hybrid_unsupported, hybrid_error,
        flop_scope="encoder_plus_scaled_query_trace",
    )]
    del hybrid_forward, point_wrapper, feature_probe, coordinates, geometry, loaded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for method, label in (
        ("radiounet", "RadioUNet-style (adapted)"),
        ("unirm", "UniRM (adapted, no-measurement)"),
        ("pefnet_inspired", "PEFNet-inspired VIE-regularized U-Net (adapted)"),
        ("rmtransformer", "RMTransformer-style (height-adapted)"),
    ):
        loaded_dense = _load_dense(method, device)
        if loaded_dense is None:
            print(f"skip {method}: checkpoint missing")
            continue
        model, checkpoint = loaded_dense
        model_input = _pefnet_input(device) if method == "pefnet_inspired" else dense_input
        benchmark_model = (
            PEFNetPredictionWrapper(model).to(device).eval()
            if method == "pefnet_inspired"
            else model
        )

        def dense_forward(model=benchmark_model, inputs=model_input):
            return model(inputs)

        flop_value, unsupported, flop_error = _flops(benchmark_model, (model_input,))
        rows.append(_measure_row(
            label, benchmark_model, checkpoint, dense_forward, device,
            args.warmup, args.repeats, flop_value, unsupported, flop_error,
        ))
        del dense_forward, benchmark_model, model, model_input, loaded_dense
        if device.type == "cuda":
            torch.cuda.empty_cache()

    loaded_diffusion = _load_radiodiff3d(device)
    if loaded_diffusion is None:
        print("skip radiodiff3d: checkpoint missing")
    else:
        diffusion_model, diffusion_checkpoint, diffusion_weights = loaded_diffusion
        condition, diffusion_protocol = _radiodiff3d_input(device)
        sampling_steps = int(diffusion_protocol["sampling_steps"])
        sampling_seed = int(diffusion_protocol["sampling_seed"])
        eta = float(diffusion_protocol["ddim_eta"])
        clip_denoised = bool(diffusion_protocol["clip_denoised"])
        maps_per_volume = int(diffusion_protocol["maps_per_volume"])

        def diffusion_forward():
            generator = torch.Generator(device=device.type)
            generator.manual_seed(sampling_seed)
            return diffusion_model.ddim_sample(
                condition,
                sampling_steps=sampling_steps,
                generator=generator,
                eta=eta,
                clip_denoised=clip_denoised,
            )

        noisy_probe = torch.zeros(
            (condition.shape[0], diffusion_model.denoiser.target_channels, *condition.shape[2:]),
            device=device,
            dtype=condition.dtype,
        )
        timestep_probe = torch.full(
            (condition.shape[0],),
            diffusion_model.timesteps // 2,
            device=device,
            dtype=torch.long,
        )
        denoiser_flops, denoiser_unsupported, denoiser_error = _flops(
            diffusion_model.denoiser, (noisy_probe, condition, timestep_probe)
        )
        diffusion_row = _measure_row(
            "RadioDiff-3D-style conditional diffusion (paper-aligned adapted v2)",
            diffusion_model,
            diffusion_checkpoint,
            diffusion_forward,
            device,
            args.diffusion_warmup,
            args.diffusion_repeats,
            denoiser_flops,
            denoiser_unsupported,
            denoiser_error,
            benchmark_unit="four_height_volume",
            maps_per_inference=maps_per_volume,
            scored_maps_per_inference=1,
            latency_map_basis="generated_height_slice",
            sampling_steps=sampling_steps,
            flop_scope="single_denoiser_forward",
            weights=diffusion_weights,
        )
        diffusion_row.update({
            "sampling_seed": sampling_seed,
            "ddim_eta": eta,
            "generated_samples_per_condition": 1,
            "best_of_k": 1,
        })
        rows.append(diffusion_row)
        del diffusion_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = args.output.expanduser().resolve()
    write_csv(output / "efficiency.csv", rows)
    write_json(output / "efficiency.json", {
        "protocol": {
            "model_only_latency": True,
            "host_to_device_transfer_included": False,
            "precision": "fp32",
            "deterministic_batch_size_maps": 1,
            "diffusion_batch_size_volumes": 1,
            "diffusion_maps_per_volume": 4,
            "diffusion_accuracy_scored_slices_per_volume": 1,
            "output_grid": [height, width],
            "hybrid_query_batch_size": args.query_batch_size,
            "diffusion_unit": (
                "one h1--h4 volume (four generated 256x256 maps; the main accuracy "
                "protocol scores its canonical anchor slice only)"
            ),
            "diffusion_formal_sampling": "fixed-seed DDIM; one sample per condition; no best-of-K",
            "flop_note": (
                "FLOPs are fvcore traces; unsupported operators make a trace a lower bound. "
                "For RadioDiff-3D, fvcore_traced_flops is one denoiser call and "
                "sampling_denoiser_flops_estimate is that trace multiplied by the fixed "
                "DDIM step count; scheduler/random-number costs remain excluded, so "
                "flops_complete is always false for that row."
            ),
        },
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "results": rows,
    })
    for row in rows:
        print(
            f"{row['method']}: params={row['parameters']} "
            f"median_ms={row['latency_ms_median']:.3f} "
            f"per_map_ms={row['latency_ms_per_map_median']:.3f}"
        )


if __name__ == "__main__":
    main()
