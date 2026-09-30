"""Build a map-level manifest and scene-disjoint UrbanRadio3D splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.raw_urbanradio3d import ManifestBuilder, RawDataConfig
from utils.config import load_config
from utils.seed import seed_python


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print paired sample metadata without writing CSV files",
    )
    parser.add_argument(
        "--validate-index-only",
        action="store_true",
        help="Audit all names/building matches without writing CSV or reading all pixels",
    )
    parser.add_argument("--inspect-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    seed = int(config["project"]["seed"])
    seed_python(seed)
    raw_config = RawDataConfig.from_mapping(config["data"], config_path.parent)
    builder = ManifestBuilder(raw_config)

    if args.inspect_only and args.validate_index_only:
        raise ValueError("Choose only one of --inspect-only and --validate-index-only")

    if args.validate_index_only:
        summary = builder.validate_index()
        print("Index validation: " + ", ".join(f"{key}={value}" for key, value in summary.items()))
        return

    if args.inspect_only:
        for row in builder.inspect(args.inspect_count):
            print(
                f"scene_id={row['scene_id']} "
                f"Tx=({row['tx_x']}, {row['tx_y']}, {row['tx_z']}) "
                f"Tx_pixel=(row={row['tx_row']}, col={row['tx_col']}) "
                f"Rx_height={row['rx_height']} "
                f"PL_shape=({row['raster_height']}, {row['raster_width']}, "
                f"{row['pathloss_channels']}) "
                f"PL_mode={row['pathloss_mode']} "
                f"building_shape=({row['raster_height']}, {row['raster_width']}, "
                f"{row['building_channels']}) "
                f"building_mode={row['building_mode']}"
            )
        return

    split = config["split"]
    summary = builder.build(
        seed=seed,
        train_ratio=float(split["train_ratio"]),
        val_ratio=float(split["val_ratio"]),
        test_ratio=float(split["test_ratio"]),
    )
    print(f"Manifest root: {raw_config.output_root}")
    print("Summary: " + ", ".join(f"{key}={value}" for key, value in summary.items()))


if __name__ == "__main__":
    main()
