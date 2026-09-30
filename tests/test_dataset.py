"""Tests for lazy continuous coordinate queries."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.coordinate_query_dataset import (
    CoordinateQueryDataset,
    MapRecord,
    _balanced_scene_records,
)


class CoordinateQueryDatasetTest(unittest.TestCase):
    def test_shapes_values_and_epoch_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "map.png"
            raw = np.arange(64, dtype=np.uint8).reshape(8, 8)
            rgba = np.stack((raw, raw, raw, np.full_like(raw, 255)), axis=-1)
            Image.fromarray(rgba).save(path)
            record = {
                "scene_id": 7,
                "tx_x": 2,
                "tx_y": 3,
                "tx_z": 1.5,
                "rx_height": 4,
                "pathloss_path": str(path),
            }
            dataset = CoordinateQueryDataset(
                records=[record], points_per_map=16, query_mode="random", seed=42
            )
            first = dataset[0]
            repeated = dataset[0]
            self.assertEqual(first["coordinate_features"].shape, (16, 10))
            self.assertEqual(first["target"].shape, (16, 1))
            self.assertEqual(str(first["rx_position"].dtype), "torch.float32")
            self.assertTrue(np.array_equal(first["target"], repeated["target"]))

            row, col = first["pixel_row_col"][0].tolist()
            feature = first["coordinate_features"][0]
            self.assertEqual(feature[3].item(), 7 - row)
            self.assertEqual(feature[4].item(), col)
            self.assertEqual(first["target"][0, 0].item(), raw[row, col])
            expected_distance = math.sqrt(
                feature[6].item() ** 2
                + feature[7].item() ** 2
                + feature[8].item() ** 2
            )
            self.assertAlmostEqual(feature[9].item(), expected_distance, places=5)

            dataset.set_epoch(1)
            changed = dataset[0]
            self.assertFalse(np.array_equal(first["pixel_row_col"], changed["pixel_row_col"]))

    def test_full_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "map.png"
            Image.fromarray(np.arange(12, dtype=np.uint8).reshape(3, 4)).save(path)
            dataset = CoordinateQueryDataset(
                records=[{
                    "scene_id": 1,
                    "tx_x": 0,
                    "tx_y": 0,
                    "tx_z": 1.5,
                    "rx_height": 1,
                    "pathloss_path": str(path),
                }],
                points_per_map=1,
                query_mode="full_grid",
            )
            sample = dataset[0]
            self.assertEqual(sample["coordinate_features"].shape, (12, 10))
            self.assertEqual(sample["target"].flatten().tolist(), list(range(12)))

    def test_invalid_rx_pixels_are_not_sampled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pathloss_path = root / "pathloss.png"
            building_path = root / "building.png"
            target = np.arange(16, dtype=np.uint8).reshape(4, 4)
            building = np.zeros((4, 4), dtype=np.uint8)
            building[1, 1] = 10
            Image.fromarray(target).save(pathloss_path)
            Image.fromarray(building).save(building_path)
            dataset = CoordinateQueryDataset(
                records=[{
                    "scene_id": 1,
                    "tx_x": 0,
                    "tx_y": 0,
                    "tx_z": 1.5,
                    "rx_height": 1,
                    "pathloss_path": str(pathloss_path),
                    "building_height_path": str(building_path),
                }],
                points_per_map=64,
                query_mode="fixed_random",
                exclude_building_occupied=True,
                exclude_target_values=(0,),
            )
            sample = dataset[0]
            self.assertTrue(sample["valid_mask"].all())
            self.assertTrue((sample["target"] > 0).all())
            sampled = {tuple(value) for value in sample["pixel_row_col"].tolist()}
            self.assertNotIn((1, 1), sampled)

    def test_scene_balanced_record_selection(self) -> None:
        records = [
            MapRecord(scene, index, 0, 1.5, 1, Path(f"{scene}_{index}.png"))
            for scene in range(3)
            for index in range(10)
        ]
        selected = _balanced_scene_records(records, max_records=6, seed=42)
        counts = {scene: 0 for scene in range(3)}
        for record in selected:
            counts[record.scene_id] += 1
        self.assertEqual(counts, {0: 2, 1: 2, 2: 2})

    def test_random_sparse_context_is_deterministic_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pathloss_path = root / "pathloss.png"
            target = np.arange(1, 65, dtype=np.uint8).reshape(8, 8)
            Image.fromarray(target).save(pathloss_path)
            dataset = CoordinateQueryDataset(
                records=[{
                    "scene_id": 1,
                    "tx_x": 2,
                    "tx_y": 3,
                    "tx_z": 1.5,
                    "rx_height": 2,
                    "pathloss_path": str(pathloss_path),
                }],
                points_per_map=1,
                query_mode="full_grid",
                sparse_sample_rate=0.10,
                sparse_context_radii=(1, 2, 4),
                seed=42,
            )
            first = dataset[0]
            second = dataset[0]
            self.assertEqual(first["sparse_context_features"].shape, (64, 7))
            self.assertEqual(int(first["observation_mask"].sum()), 6)
            self.assertTrue(torch.equal(first["observation_mask"], second["observation_mask"]))
            self.assertTrue(torch.isfinite(first["sparse_context_features"]).all())


if __name__ == "__main__":
    unittest.main()
