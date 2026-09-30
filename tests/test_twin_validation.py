from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.twin_validation_common import delta_scores, select_state_records, unique_records


class TwinValidationTests(unittest.TestCase):
    def test_delta_scores_are_zero_for_perfect_change(self) -> None:
        target = np.asarray([-20.0, 0.0, 15.0, 40.0], dtype=np.float32)
        score = delta_scores(target.copy(), target)
        self.assertAlmostEqual(score["delta_rmse"], 0.0)
        self.assertAlmostEqual(score["delta_nmse"], 0.0)
        self.assertAlmostEqual(score["delta_correlation"], 1.0)
        self.assertAlmostEqual(score["delta_sign_agreement"], 1.0)

    def test_state_selection_is_deterministic_and_paired(self) -> None:
        fields = (
            "scene_id", "tx_x", "tx_y", "tx_z", "rx_height",
            "pathloss_path", "building_height_path",
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "test.csv"
            rows = []
            for scene in (10, 20):
                for tx_index, (tx_x, tx_y) in enumerate(((10, 20), (30, 40), (50, 60))):
                    for height in (1, 4):
                        rows.append({
                            "scene_id": scene, "tx_x": tx_x, "tx_y": tx_y,
                            "tx_z": 1.5, "rx_height": height,
                            "pathloss_path": f"p_{scene}_{tx_index}_{height}.png",
                            "building_height_path": f"b_{scene}.png",
                        })
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            first = select_state_records(
                manifest, seed=42, transition_scenes=2, planning_scenes=2, candidates=3
            )
            second = select_state_records(
                manifest, seed=42, transition_scenes=2, planning_scenes=2, candidates=3
            )
            self.assertEqual(first, second)
            transitions, planning = first
            self.assertEqual(sum(kind == "rx_height" for kind, _, _ in transitions), 2)
            self.assertEqual(sum(kind == "tx_position" for kind, _, _ in transitions), 2)
            self.assertEqual(len(planning), 2)
            self.assertTrue(all(len(group) == 3 for group in planning.values()))
            keys = {
                (record.scene_id, record.tx_x, record.tx_y, record.rx_height)
                for record in unique_records(transitions, planning)
            }
            self.assertEqual(len(keys), len(unique_records(transitions, planning)))


if __name__ == "__main__":
    unittest.main()
