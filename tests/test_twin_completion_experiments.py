from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.evaluate_ensemble_uncertainty import _conformal_quantile
from scripts.evaluate_unseen_heights import discover_unseen_records
from scripts.prepare_building_counterfactuals import _edit_building


class TwinCompletionExperimentTests(unittest.TestCase):
    def test_conformal_quantile_uses_finite_sample_correction(self) -> None:
        scores = np.arange(1, 101, dtype=np.float64)
        self.assertEqual(_conformal_quantile(scores, 0.90), 91.0)
        self.assertEqual(_conformal_quantile(scores, 0.95), 96.0)

    def test_building_interventions_change_only_real_geometry(self) -> None:
        building = np.zeros((32, 32), dtype=np.uint8)
        building[3:8, 4:9] = 6
        building[20:23, 20:23] = 4
        removed, removal_meta = _edit_building(
            building, "building_removal", height_delta=5,
            addition_size=4, addition_height=8, forbidden=set(),
        )
        self.assertEqual(int((removed != building).sum()), 25)
        self.assertEqual(removal_meta["component_pixels"], 25)
        raised, _ = _edit_building(
            building, "height_increase", height_delta=5,
            addition_size=4, addition_height=8, forbidden=set(),
        )
        self.assertTrue(np.all(raised[3:8, 4:9] == 11))
        added, addition_meta = _edit_building(
            building, "building_addition", height_delta=5,
            addition_size=4, addition_height=8, forbidden={(16, 16)},
        )
        self.assertEqual(addition_meta["component_pixels"], 16)
        self.assertEqual(int((added != building).sum()), 16)

    def test_unseen_height_discovery_keeps_common_test_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pathloss_root = root / "pl"
            building_root = root / "buildings"
            output_root = root / "processed"
            (output_root / "splits").mkdir(parents=True)
            building_root.mkdir()
            (output_root / "splits" / "test_scenes.txt").write_text("10\n11\n", encoding="utf-8")
            for scene in (10, 11):
                Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(building_root / f"{scene}.png")
            for height in (5, 8):
                folder = pathloss_root / f"PathLoss_h{height}_test" / f"h{height}"
                folder.mkdir(parents=True)
                for scene in (10, 11):
                    Image.fromarray(np.full((16, 16), scene + height, dtype=np.uint8)).save(
                        folder / f"{scene}_X5_Y6.png"
                    )
            config = {
                "data": {
                    "pathloss_root": str(pathloss_root),
                    "building_height_root": str(building_root),
                    "output_root": str(output_root),
                    "pathloss_folder_pattern": "^PathLoss_h(?P<height>[0-9]+)_(?P<partition>.+)$",
                    "pathloss_filename_pattern": "^(?P<scene_id>[0-9]+)_X(?P<tx_x>[0-9]+)_Y(?P<tx_y>[0-9]+)\\.png$",
                    "building_filename_pattern": "^(?P<scene_id>[0-9]+)\\.png$",
                    "rx_heights": [1, 2, 3, 4],
                    "tx_height": 1.5,
                    "extensions": [".png"],
                }
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            records = discover_unseen_records(config_path, [5, 8], anchors=2, seed=42)
            self.assertEqual(len(records), 4)
            keys = {(record.scene_id, record.tx_x, record.tx_y) for record in records}
            self.assertEqual(len(keys), 2)
            self.assertEqual({record.rx_height for record in records}, {5.0, 8.0})
            with self.assertRaises(ValueError):
                discover_unseen_records(config_path, [4, 5], anchors=1, seed=42)


if __name__ == "__main__":
    unittest.main()
