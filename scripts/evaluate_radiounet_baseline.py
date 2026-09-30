"""Evaluate the adapted RadioUNet baseline on the common test protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dense_baseline_dataset import DenseBaselineDataset
from models.radiounet_baseline import RadioUNetBaseline
from models.pmnet_baseline import build_pmnet_adapted
from scripts.evaluate import metrics, normalized_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=float, choices=(0.0, 0.01, 0.05, 0.10), required=True)
    parser.add_argument("--architecture", choices=("radiounet", "pmnet"), default="radiounet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    args = parser.parse_args()
    if args.architecture == "pmnet" and args.sample_rate != 0.0:
        raise ValueError("PMNet public baseline is sampling-free")
    label = str(int(round(args.sample_rate * 100)))
    output = PROJECT_ROOT / "outputs" / f"{args.architecture}_adapted_random{label}_seed{args.seed}"
    checkpoint_path = output / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    dataset = DenseBaselineDataset(
        PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv",
        max_records=128, selection_strategy="balanced_scene", seed=args.data_seed,
        sample_rate=args.sample_rate, training=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        build_pmnet_adapted(3)
        if args.architecture == "pmnet"
        else RadioUNetBaseline(5 if args.sample_rate > 0 else 3)
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = []
    with torch.inference_mode():
        for index, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
            prediction = (model(batch["input"].to(device)).cpu().numpy()[0, 0] * 255.0)
            target = batch["target_raw"].numpy()[0, 0]
            valid = batch["valid_mask"].numpy()[0, 0].astype(bool)
            observed = batch["observation_mask"].numpy()[0, 0].astype(bool)
            support = valid & ~observed
            score = normalized_metrics(prediction[support], target[support])
            raw = metrics(prediction[support], target[support], 255.0)
            row = {"map_index": index, "scene_id": int(batch["scene_id"]), "valid_points": int(valid.sum()), "observed_points": int(observed.sum()), **score, **{f"raw_{k}": v for k, v in raw.items()}}
            rows.append(row)
            print(f"[{index + 1}/{len(dataset)}] rmse={score['rmse']:.4f} nmse={score['nmse']:.6f}")
    result = output / "evaluation" / "test"
    result.mkdir(parents=True, exist_ok=True)
    with (result / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for name in ("rmse", "nmse", "mae", "psnr", "raw_rmse", "raw_nmse", "raw_mae", "raw_psnr"):
        values = np.asarray([row[name] for row in rows])
        summary[name] = {"mean": float(values.mean()), "std": float(values.std())}
    summary.update({"maps": len(rows), "seed": args.seed, "data_seed": args.data_seed, "sample_rate": args.sample_rate, "support": "unobserved_valid_rx_only" if args.sample_rate else "valid_rx_only", "checkpoint_epoch": checkpoint["epoch"], "method_label": "PMNet-v1 (adapted)" if args.architecture == "pmnet" else "RadioUNet-style (adapted)", "parameter_count": sum(parameter.numel() for parameter in model.parameters())})
    (result / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
