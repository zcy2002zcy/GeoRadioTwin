"""Synthetic tests for LoS and first-obstruction geometry depth."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry.cache import geometry_cache_path, load_geometry_cache, save_geometry_cache
from geometry.ray_query import query_propagation_geometry


class GeometryQueryTest(unittest.TestCase):
    def test_los_and_first_obstruction(self) -> None:
        buildings = np.zeros((5, 5), dtype=np.float32)
        # Physical (X=2,Y=2) maps to raster (row=2,col=2).
        buildings[2, 2] = 10.0
        tx = (0.0, 2.0, 1.5)
        rx = np.asarray(((4.0, 2.0, 1.5), (4.0, 4.0, 1.5)), dtype=np.float32)
        result = query_propagation_geometry(
            buildings, tx, rx, samples_per_unit=4.0, chunk_size=2
        )
        self.assertFalse(result.los[0])
        self.assertTrue(result.los[1])
        self.assertGreater(result.num_obstruction_samples[0], 0)
        self.assertLess(result.first_obstruction_ratio[0], 1.0)
        self.assertAlmostEqual(result.depth[0], 4.0 * result.first_obstruction_ratio[0])
        self.assertAlmostEqual(result.depth[1], math.sqrt(20.0), places=5)
        self.assertEqual(result.normalized_depth[1], 1.0)

    def test_sloped_ray_clears_low_building(self) -> None:
        buildings = np.zeros((5, 5), dtype=np.float32)
        buildings[2, 2] = 2.0
        result = query_propagation_geometry(
            buildings,
            (0.0, 2.0, 1.5),
            np.asarray(((4.0, 2.0, 10.0),), dtype=np.float32),
            samples_per_unit=4.0,
        )
        self.assertTrue(result.los[0])

    def test_zero_horizontal_distance(self) -> None:
        result = query_propagation_geometry(
            np.zeros((3, 3), dtype=np.float32),
            (1.0, 1.0, 1.5),
            np.asarray(((1.0, 1.0, 1.5),), dtype=np.float32),
        )
        self.assertTrue(result.los[0])
        self.assertEqual(result.distance[0], 0.0)
        self.assertEqual(result.depth[0], 0.0)

    def test_atomic_cache_and_key(self) -> None:
        buildings = np.zeros((3, 3), dtype=np.float32)
        result = query_propagation_geometry(
            buildings,
            (0.0, 0.0, 1.5),
            np.asarray([(2.0, 2.0, 1.0)] * 9, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = geometry_cache_path(Path(temp_dir), 7, 10, 20, 3)
            save_geometry_cache(path, result, (3, 3), {"scene_id": 7})
            self.assertTrue(path.is_file())
            cache, metadata = load_geometry_cache(path)
            self.assertEqual(cache["los"].shape, (3, 3))
            self.assertEqual(cache["depth"].dtype, np.float32)
            self.assertEqual(metadata["scene_id"], 7)


if __name__ == "__main__":
    unittest.main()
