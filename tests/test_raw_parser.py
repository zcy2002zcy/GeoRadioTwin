"""Tests for raw map parsing and scene-level splitting."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.raw_urbanradio3d import ManifestBuilder, RawDataConfig


class RawParserTest(unittest.TestCase):
    def test_manifest_and_scene_disjoint_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pathloss_root = root / "pl"
            building_root = root / "buildingsWHeight"
            output_root = root / "processed"
            building_root.mkdir(parents=True)

            for scene_id in range(10):
                Image.new("L", (8, 8), color=scene_id).save(
                    building_root / f"{scene_id}.png"
                )
                for height in (1, 2):
                    folder = pathloss_root / f"PathLoss_h{height}_train" / f"h{height}"
                    folder.mkdir(parents=True, exist_ok=True)
                    Image.new("RGBA", (8, 8), color=(1, 2, 3, 255)).save(
                        folder / f"{scene_id}_X2_Y3.png"
                    )

            config = RawDataConfig(
                pathloss_root=pathloss_root,
                building_height_root=building_root,
                output_root=output_root,
                pathloss_folder_pattern=r"^PathLoss_h(?P<height>[0-9]+)_(?P<partition>.+)$",
                pathloss_filename_pattern=r"^(?P<scene_id>[0-9]+)_X(?P<tx_x>[0-9]+)_Y(?P<tx_y>[0-9]+)\.png$",
                building_filename_pattern=r"^(?P<scene_id>[0-9]+)\.png$",
                rx_heights=(1, 2),
            )
            summary = ManifestBuilder(config).build(42, 0.8, 0.1, 0.1)
            self.assertEqual(summary["maps"], 20)
            self.assertEqual(summary["scenes"], 10)

            scene_sets = {}
            for name in ("train", "val", "test"):
                text = (output_root / "splits" / f"{name}_scenes.txt").read_text()
                scene_sets[name] = {int(value) for value in text.splitlines()}
            self.assertFalse(scene_sets["train"] & scene_sets["val"])
            self.assertFalse(scene_sets["train"] & scene_sets["test"])
            self.assertFalse(scene_sets["val"] & scene_sets["test"])

            with (output_root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)
            self.assertEqual(rows[0]["pathloss_mode"], "RGBA")
            self.assertEqual(rows[0]["building_mode"], "L")
            self.assertEqual(rows[0]["raster_height"], "8")
            self.assertEqual(rows[0]["tx_row"], "5")
            self.assertEqual(rows[0]["tx_col"], "3")


if __name__ == "__main__":
    unittest.main()
