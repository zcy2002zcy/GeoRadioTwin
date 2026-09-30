from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dense_baseline_dataset import random_observation_mask
from models.radiounet_baseline import RadioUNetBaseline
from models.pmnet_baseline import build_pmnet_adapted


class PublicBaselineTest(unittest.TestCase):
    def test_common_random_mask_is_valid_and_deterministic(self) -> None:
        valid = np.ones((20, 20), dtype=bool)
        valid[:2] = False
        first = random_observation_mask(valid, 0.10, seed=42, map_index=3)
        second = random_observation_mask(valid, 0.10, seed=42, map_index=3)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(first[~valid].any())
        self.assertEqual(int(first.sum()), round(valid.sum() * 0.10))

    def test_radiounet_output_shape(self) -> None:
        model = RadioUNetBaseline(in_channels=5, base_channels=8)
        output = model(torch.randn(2, 5, 64, 64))
        self.assertEqual(output.shape, (2, 1, 64, 64))

    def test_pmnet_adapted_output_shape(self) -> None:
        source = PROJECT_ROOT / "external" / "PMNet" / "models" / "pmnet_v1.py"
        if not source.exists():
            self.skipTest("optional pinned PMNet checkout is not present")
        model = build_pmnet_adapted(in_channels=3).eval()
        with torch.inference_mode():
            output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(output.shape, (1, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
