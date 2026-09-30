"""Raw UrbanRadio3D/RadioDiff-3D discovery and manifest generation.

This module indexes map-level metadata only. It never expands a raster into one
CSV row per pixel and deliberately does not interpret PathLoss pixel values.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from PIL import Image

from geometry.coordinates import physical_xy_to_row_col


MANIFEST_FIELDS = (
    "scene_id",
    "tx_x",
    "tx_y",
    "tx_row",
    "tx_col",
    "tx_z",
    "rx_height",
    "source_partition",
    "source_folder",
    "pathloss_path",
    "building_height_path",
    "raster_height",
    "raster_width",
    "pathloss_mode",
    "pathloss_channels",
    "building_mode",
    "building_channels",
)


@dataclass(frozen=True)
class RawDataConfig:
    """Validated configuration for raw map discovery."""

    pathloss_root: Path
    building_height_root: Path
    output_root: Path
    pathloss_folder_pattern: str
    pathloss_filename_pattern: str
    building_filename_pattern: str
    rx_heights: tuple[int, ...]
    tx_height: float = 1.5
    extensions: tuple[str, ...] = (".png",)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object], config_dir: Path
    ) -> "RawDataConfig":
        """Build the config, resolving relative paths against the YAML file."""

        def resolve(value: object) -> Path:
            path = Path(str(value)).expanduser()
            return path.resolve() if path.is_absolute() else (config_dir / path).resolve()

        return cls(
            pathloss_root=resolve(data["pathloss_root"]),
            building_height_root=resolve(data["building_height_root"]),
            output_root=resolve(data["output_root"]),
            pathloss_folder_pattern=str(data["pathloss_folder_pattern"]),
            pathloss_filename_pattern=str(data["pathloss_filename_pattern"]),
            building_filename_pattern=str(data["building_filename_pattern"]),
            rx_heights=tuple(int(value) for value in data["rx_heights"]),  # type: ignore[arg-type]
            tx_height=float(data.get("tx_height", 1.5)),
            extensions=tuple(str(value).lower() for value in data.get("extensions", [".png"])),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PathLossMap:
    """Parsed identity of one raw PathLoss map."""

    scene_id: int
    tx_x: int
    tx_y: int
    tx_z: float
    rx_height: int
    source_partition: str
    source_folder: str
    pathloss_path: Path


def _image_metadata(path: Path) -> tuple[int, int, str, int]:
    """Return (height, width, mode, channels) without transforming pixels."""

    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        channels = len(image.getbands())
    return height, width, mode, channels


class ManifestBuilder:
    """Scan raw files, match building rasters, and write scene-level splits."""

    def __init__(self, config: RawDataConfig) -> None:
        self.config = config
        self._folder_regex = re.compile(config.pathloss_folder_pattern)
        self._file_regex = re.compile(config.pathloss_filename_pattern, re.IGNORECASE)
        self._building_regex = re.compile(config.building_filename_pattern, re.IGNORECASE)
        # UrbanRadio3D uses a consistent image format within each source folder.
        # Cache representative metadata per folder and per building raster so a
        # 520k-row manifest does not reopen identical geometry 520k times.
        self._pathloss_metadata: dict[str, tuple[int, int, str, int]] = {}
        self._building_metadata: dict[Path, tuple[int, int, str, int]] = {}

    def validate_roots(self) -> None:
        """Fail early with explicit paths when a configured source is missing."""

        if not self.config.pathloss_root.is_dir():
            raise FileNotFoundError(
                f"PathLoss root does not exist: {self.config.pathloss_root}"
            )
        if not self.config.building_height_root.is_dir():
            raise FileNotFoundError(
                "Building-height root does not exist: "
                f"{self.config.building_height_root}"
            )

    def index_buildings(self) -> dict[int, Path]:
        """Index one building-height raster per numeric scene ID."""

        buildings: dict[int, Path] = {}
        for path in sorted(self.config.building_height_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.config.extensions:
                continue
            match = self._building_regex.fullmatch(path.name)
            if match is None:
                continue
            scene_id = int(match.group("scene_id"))
            if scene_id in buildings:
                raise ValueError(
                    f"Duplicate building rasters for scene {scene_id}: "
                    f"{buildings[scene_id]} and {path}"
                )
            buildings[scene_id] = path.resolve()
        if not buildings:
            raise ValueError(
                f"No building rasters matched in {self.config.building_height_root}"
            )
        return buildings

    def iter_maps(self) -> Iterator[PathLossMap]:
        """Yield maps lazily so the complete dataset is never expanded in RAM."""

        allowed_heights = set(self.config.rx_heights)
        for folder in sorted(self.config.pathloss_root.iterdir()):
            if not folder.is_dir():
                continue
            folder_match = self._folder_regex.fullmatch(folder.name)
            if folder_match is None:
                continue
            rx_height = int(folder_match.group("height"))
            if rx_height not in allowed_heights:
                continue
            partition = folder_match.group("partition")
            for path in sorted(folder.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in self.config.extensions:
                    continue
                file_match = self._file_regex.fullmatch(path.name)
                if file_match is None:
                    raise ValueError(f"Unexpected PathLoss filename: {path}")
                yield PathLossMap(
                    scene_id=int(file_match.group("scene_id")),
                    tx_x=int(file_match.group("tx_x")),
                    tx_y=int(file_match.group("tx_y")),
                    tx_z=self.config.tx_height,
                    rx_height=rx_height,
                    source_partition=partition,
                    source_folder=folder.name,
                    pathloss_path=path.resolve(),
                )

    def inspect(self, count: int = 5) -> list[dict[str, object]]:
        """Inspect real paired rasters and return printable map-level metadata."""

        self.validate_roots()
        buildings = self.index_buildings()
        rows: list[dict[str, object]] = []
        for item in self.iter_maps():
            building_path = buildings.get(item.scene_id)
            if building_path is None:
                raise ValueError(f"No building raster for scene {item.scene_id}")
            rows.append(self._make_row(item, building_path))
            if len(rows) >= count:
                break
        if not rows:
            raise ValueError("No PathLoss maps matched the configured heights/patterns")
        return rows

    def validate_index(self) -> dict[str, object]:
        """Scan every configured filename and verify scene-to-building coverage.

        This intentionally avoids opening every raster. Shape and image-mode
        checks belong to :meth:`inspect` and manifest writing, while this method
        provides a fast full-dataset naming and pairing audit.
        """

        self.validate_roots()
        buildings = self.index_buildings()
        scene_ids: set[int] = set()
        height_counts = {height: 0 for height in self.config.rx_heights}
        map_count = 0
        for item in self.iter_maps():
            if item.scene_id not in buildings:
                raise ValueError(f"No building raster for scene {item.scene_id}")
            scene_ids.add(item.scene_id)
            height_counts[item.rx_height] += 1
            map_count += 1
        if map_count == 0:
            raise ValueError("No PathLoss maps matched the configured heights/patterns")
        return {
            "maps": map_count,
            "scenes": len(scene_ids),
            "min_scene_id": min(scene_ids),
            "max_scene_id": max(scene_ids),
            "building_rasters": len(buildings),
            "height_counts": height_counts,
        }

    def build(
        self,
        seed: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> dict[str, int]:
        """Create the manifest and deterministic, scene-disjoint split files."""

        self.validate_roots()
        _validate_ratios(train_ratio, val_ratio, test_ratio)
        buildings = self.index_buildings()
        output_root = self.config.output_root
        split_root = output_root / "splits"
        output_root.mkdir(parents=True, exist_ok=True)
        split_root.mkdir(parents=True, exist_ok=True)

        manifest_path = output_root / "manifest.csv"
        manifest_temp_path = output_root / "manifest.csv.tmp"
        scene_ids: set[int] = set()
        row_count = 0
        with manifest_temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for item in self.iter_maps():
                building_path = buildings.get(item.scene_id)
                if building_path is None:
                    raise ValueError(f"No building raster for scene {item.scene_id}")
                writer.writerow(self._make_row(item, building_path))
                scene_ids.add(item.scene_id)
                row_count += 1

        if not scene_ids:
            raise ValueError("No PathLoss maps matched the configured heights/patterns")
        split_scenes = split_scene_ids(
            scene_ids, seed, train_ratio, val_ratio, test_ratio
        )
        scene_to_split = {
            scene_id: split_name
            for split_name, ids in split_scenes.items()
            for scene_id in ids
        }
        split_temp_paths = {
            name: split_root / f"{name}.csv.tmp"
            for name in ("train", "val", "test")
        }
        handles = {
            name: split_temp_paths[name].open(
                "w", newline="", encoding="utf-8"
            )
            for name in ("train", "val", "test")
        }
        try:
            writers = {
                name: csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
                for name, handle in handles.items()
            }
            for writer in writers.values():
                writer.writeheader()
            with manifest_temp_path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    writers[scene_to_split[int(row["scene_id"])]].writerow(row)
        finally:
            for handle in handles.values():
                handle.close()

        # Publish all large CSVs only after they have been written successfully.
        # An interrupted scan therefore cannot masquerade as a valid manifest.
        manifest_temp_path.replace(manifest_path)
        for name, temp_path in split_temp_paths.items():
            temp_path.replace(split_root / f"{name}.csv")

        for name, ids in split_scenes.items():
            text = "".join(f"{scene_id}\n" for scene_id in sorted(ids))
            (split_root / f"{name}_scenes.txt").write_text(text, encoding="utf-8")

        return {
            "maps": row_count,
            "scenes": len(scene_ids),
            **{f"{name}_scenes": len(ids) for name, ids in split_scenes.items()},
        }

    def _make_row(self, item: PathLossMap, building_path: Path) -> dict[str, object]:
        pl_metadata = self._pathloss_metadata.get(item.source_folder)
        if pl_metadata is None:
            pl_metadata = _image_metadata(item.pathloss_path)
            self._pathloss_metadata[item.source_folder] = pl_metadata
        building_metadata = self._building_metadata.get(building_path)
        if building_metadata is None:
            building_metadata = _image_metadata(building_path)
            self._building_metadata[building_path] = building_metadata
        pl_h, pl_w, pl_mode, pl_channels = pl_metadata
        b_h, b_w, b_mode, b_channels = building_metadata
        if (pl_h, pl_w) != (b_h, b_w):
            raise ValueError(
                f"Raster shape mismatch for scene {item.scene_id}: "
                f"PathLoss={(pl_h, pl_w)}, building={(b_h, b_w)}"
            )
        if not (0 <= item.tx_x < pl_h and 0 <= item.tx_y < pl_w):
            raise ValueError(
                f"Tx physical coordinate (X={item.tx_x}, Y={item.tx_y}) is outside "
                f"the {(pl_h, pl_w)} raster in {item.pathloss_path}"
            )
        tx_row, tx_col = physical_xy_to_row_col(item.tx_x, item.tx_y, pl_h)
        return {
            "scene_id": item.scene_id,
            "tx_x": item.tx_x,
            "tx_y": item.tx_y,
            "tx_row": int(tx_row),
            "tx_col": int(tx_col),
            "tx_z": item.tx_z,
            "rx_height": item.rx_height,
            "source_partition": item.source_partition,
            "source_folder": item.source_folder,
            "pathloss_path": str(item.pathloss_path),
            "building_height_path": str(building_path),
            "raster_height": pl_h,
            "raster_width": pl_w,
            "pathloss_mode": pl_mode,
            "pathloss_channels": pl_channels,
            "building_mode": b_mode,
            "building_channels": b_channels,
        }


def _validate_ratios(train: float, val: float, test: float) -> None:
    if min(train, val, test) < 0:
        raise ValueError("Split ratios must be non-negative")
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")


def split_scene_ids(
    scene_ids: Sequence[int] | set[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, set[int]]:
    """Shuffle unique scenes deterministically and return disjoint subsets."""

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    shuffled = sorted(set(scene_ids))
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_end = int(count * train_ratio)
    val_end = train_end + int(count * val_ratio)
    result = {
        "train": set(shuffled[:train_end]),
        "val": set(shuffled[train_end:val_end]),
        "test": set(shuffled[val_end:]),
    }
    if result["train"] & result["val"] or result["train"] & result["test"] or result["val"] & result["test"]:
        raise AssertionError("Scene split implementation produced overlap")
    return result
