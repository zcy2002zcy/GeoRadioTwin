"""Print a real CoordinateQueryDataset batch for STEP 4 validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.coordinate_query_dataset import CoordinateQueryDataset
from datasets.raw_urbanradio3d import ManifestBuilder, RawDataConfig
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--points-per-map", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    raw_config = RawDataConfig.from_mapping(config["data"], config_path.parent)
    records = ManifestBuilder(raw_config).inspect(args.batch_size)
    dataset = CoordinateQueryDataset(
        records=records,
        points_per_map=args.points_per_map,
        query_mode="fixed_random",
        seed=int(config["project"]["seed"]),
        pathloss_channel=int(config["data"]["pathloss_channel"]),
        exclude_building_occupied=bool(config["dataset"]["exclude_building_occupied"]),
        exclude_target_values=config["dataset"]["exclude_target_values"],
        target_exclusion_tolerance=float(config["dataset"]["target_exclusion_tolerance"]),
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)))
    for name, tensor in batch.items():
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    first = batch["coordinate_features"][0, 0]
    print(
        "first_query="
        f"tx={tuple(first[0:3].tolist())}, "
        f"rx={tuple(first[3:6].tolist())}, "
        f"delta={tuple(first[6:9].tolist())}, "
        f"distance={first[9].item():.6f}, "
        f"raw_target={batch['target'][0, 0, 0].item():.1f}"
    )


if __name__ == "__main__":
    main()
