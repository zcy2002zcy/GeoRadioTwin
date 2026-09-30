"""Coordinate convention regression tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry.coordinates import physical_xy_to_row_col, row_col_to_physical_xy


class CoordinateConventionTest(unittest.TestCase):
    def test_known_256_grid_point(self) -> None:
        self.assertEqual(physical_xy_to_row_col(104, 153, 256), (151, 153))

    def test_round_trip(self) -> None:
        row_col = physical_xy_to_row_col(197, 50, 256)
        self.assertEqual(row_col_to_physical_xy(*row_col, height=256), (197, 50))


if __name__ == "__main__":
    unittest.main()
