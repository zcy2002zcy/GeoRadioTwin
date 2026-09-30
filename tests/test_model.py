"""Forward-shape tests for all geometry ablations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.continuous_radio_field import ContinuousRadioField
from models.hybrid_radio_field import HybridGeoRadioField


class ContinuousRadioFieldTest(unittest.TestCase):
    def test_coordinate_only_shape_and_gradient(self) -> None:
        model = ContinuousRadioField(hidden_dim=32, num_layers=2)
        coordinates = torch.randn(2, 16, 10, requires_grad=True)
        output = model(coordinates)
        self.assertEqual(output.shape, (2, 16, 1))
        output.mean().backward()
        self.assertIsNotNone(coordinates.grad)

    def test_geometry_shape(self) -> None:
        model = ContinuousRadioField(
            hidden_dim=32, num_layers=2, geometry_input_dim=2
        )
        output = model(torch.randn(3, 8, 10), torch.randn(3, 8, 2))
        self.assertEqual(output.shape, (3, 8, 1))

    def test_sparse_calibration_shape_and_required_context(self) -> None:
        model = ContinuousRadioField(
            hidden_dim=32,
            num_layers=2,
            geometry_input_dim=2,
            calibration_input_dim=7,
            calibration_hidden_dim=8,
        )
        coordinates = torch.randn(2, 12, 10)
        geometry = torch.randn(2, 12, 2)
        context = torch.randn(2, 12, 7)
        self.assertEqual(model(coordinates, geometry, context).shape, (2, 12, 1))
        with self.assertRaises(ValueError):
            model(coordinates, geometry)

    def test_hybrid_field_shape_and_zero_residual_initialization(self) -> None:
        base = ContinuousRadioField(hidden_dim=32, num_layers=2, geometry_input_dim=2)
        model = HybridGeoRadioField(base, environment_dim=16, ray_samples=5)
        coordinates = torch.randn(2, 32, 10)
        geometry = torch.randn(2, 32, 2)
        environment = torch.randn(2, 3, 64, 64)
        with torch.inference_mode():
            expected = base(coordinates, geometry)
            actual = model(coordinates, geometry, environment)
        self.assertEqual(actual.shape, (2, 32, 1))
        self.assertTrue(torch.allclose(actual, expected))

    def test_hybrid_endpoints_without_analytic_geometry(self) -> None:
        base = ContinuousRadioField(hidden_dim=32, num_layers=2)
        model = HybridGeoRadioField(
            base, environment_dim=16, ray_samples=5, context_mode="endpoints"
        )
        coordinates = torch.randn(2, 16, 10)
        environment = torch.randn(2, 3, 64, 64)
        output = model(coordinates, None, environment)
        self.assertEqual(output.shape, (2, 16, 1))


if __name__ == "__main__":
    unittest.main()
