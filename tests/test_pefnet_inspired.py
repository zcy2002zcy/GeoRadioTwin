"""Tests for the independently adapted PEFNet-inspired baseline."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pefnet_inspired_dataset import PEFNetInspiredDataset
from models.pefnet_inspired_baseline import build_pefnet_inspired
from physics.vie_operator import VIEOperator, _vie_kernel, build_incident_field
from scripts.train_pefnet_inspired_baseline import _learning_rate_for_epoch


class PEFNetInspiredTest(unittest.TestCase):
    def test_stable_v2_uses_group_norm_without_batch_norm(self) -> None:
        model = build_pefnet_inspired(
            in_channels=5, base_channels=8, levels=3, max_norm_groups=32
        )
        group_norms = [module for module in model.modules() if isinstance(module, nn.GroupNorm)]
        batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        self.assertTrue(group_norms)
        self.assertFalse(batch_norms)
        for layer in group_norms:
            self.assertEqual(layer.num_channels % layer.num_groups, 0)
            self.assertGreaterEqual(layer.num_channels // layer.num_groups, 4)

    def test_zero_field_conversion_is_finite(self) -> None:
        model = build_pefnet_inspired(in_channels=5, base_channels=4, levels=2)
        rough = model.field_to_rough_pathloss(torch.zeros(1, 2, 8, 8))
        self.assertTrue(torch.isfinite(rough).all())
        rough.sum().backward()
        self.assertTrue(torch.isfinite(model.log_rough_scale.grad).all())

    def test_warmup_cosine_schedule_boundaries(self) -> None:
        kwargs = {
            "epochs": 100,
            "base_learning_rate": 1e-4,
            "minimum_learning_rate": 1e-6,
            "warmup_epochs": 5,
            "warmup_start_factor": 0.1,
        }
        self.assertAlmostEqual(_learning_rate_for_epoch(epoch=1, **kwargs), 1e-5)
        self.assertAlmostEqual(_learning_rate_for_epoch(epoch=5, **kwargs), 1e-4)
        self.assertAlmostEqual(_learning_rate_for_epoch(epoch=100, **kwargs), 1e-6)

    def test_batch_one_short_stress_remains_finite(self) -> None:
        torch.manual_seed(23)
        model = build_pefnet_inspired(
            in_channels=5, base_channels=4, levels=2
        ).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(torch.randn(1, 5, 16, 16))
            loss = outputs["prediction"].square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            self.assertTrue(torch.isfinite(norm))
            optimizer.step()
        self.assertTrue(
            all(
                torch.isfinite(value).all()
                for value in list(model.parameters()) + list(model.buffers())
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for BF16 autocast test")
    def test_refinement_stays_fp32_inside_bf16_autocast(self) -> None:
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA device does not support BF16")
        model = build_pefnet_inspired(
            in_channels=5, base_channels=4, levels=2,
            force_refinement_fp32=True,
        ).cuda().eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(torch.randn(1, 5, 16, 16, device="cuda"))
        self.assertEqual(outputs["total_field"].dtype, torch.bfloat16)
        for name in ("rough_pathloss", "residual", "prediction"):
            self.assertEqual(outputs[name].dtype, torch.float32, name)
            self.assertTrue(torch.isfinite(outputs[name]).all(), name)

    def test_model_outputs_have_expected_shapes_and_finite_values(self) -> None:
        model = build_pefnet_inspired(
            in_channels=5, base_channels=4, levels=3
        ).eval()
        inputs = torch.randn(2, 5, 32, 32)
        with torch.inference_mode():
            outputs = model(inputs)

        expected_shapes = {
            "prediction": (2, 1, 32, 32),
            "total_field": (2, 2, 32, 32),
            "rough_pathloss": (2, 1, 32, 32),
            "residual": (2, 1, 32, 32),
        }
        self.assertEqual(set(outputs), set(expected_shapes))
        for name, expected_shape in expected_shapes.items():
            self.assertEqual(tuple(outputs[name].shape), expected_shape, name)
            self.assertTrue(torch.isfinite(outputs[name]).all(), name)

    def test_joint_data_and_vie_loss_backpropagates(self) -> None:
        torch.manual_seed(7)
        model = build_pefnet_inspired(
            in_channels=5, base_channels=4, levels=2
        ).train()
        inputs = torch.randn(2, 5, 16, 16)
        outputs = model(inputs)
        incident = torch.randn(2, 2, 16, 16)
        contrast = torch.zeros(2, 2, 16, 16)
        contrast[:, 0, 4:12, 5:10] = 3.0
        vie = VIEOperator(grid_size=16, dimensionless_wavenumber=0.35)

        loss = outputs["prediction"].square().mean() + vie.loss(
            outputs["total_field"], incident, contrast
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_vie_zero_contrast_reduces_to_incident_field(self) -> None:
        incident_np = build_incident_field(
            (8, 8),
            (3, 5),
            dimensionless_wavenumber=0.35,
        )
        incident = torch.from_numpy(incident_np).unsqueeze(0)
        contrast = torch.zeros_like(incident)
        vie = VIEOperator(grid_size=(8, 8), dimensionless_wavenumber=0.35)

        residual = vie.residual(incident, incident, contrast)
        self.assertLess(float(residual.abs().max()), 1e-6)
        self.assertLess(
            float(vie.loss(incident, incident, contrast, epsilon=0.0)), 1e-7
        )

    def test_fft_vie_residual_matches_direct_small_grid_sum(self) -> None:
        torch.manual_seed(19)
        height, width = 4, 5
        wavenumber = 0.35
        total = torch.complex(
            torch.randn(1, height, width), torch.randn(1, height, width)
        )
        incident = torch.complex(
            torch.randn(1, height, width), torch.randn(1, height, width)
        )
        contrast = torch.complex(
            0.5 * torch.rand(1, height, width),
            -0.1 * torch.rand(1, height, width),
        )
        source = contrast * total
        kernel = torch.from_numpy(
            _vie_kernel(height, width, wavenumber, grid_spacing=1.0)
        )
        scattered = torch.zeros_like(total)
        for row in range(height):
            for col in range(width):
                value = torch.zeros((), dtype=total.dtype)
                for source_row in range(height):
                    for source_col in range(width):
                        value = value + source[0, source_row, source_col] * kernel[
                            height - 1 + row - source_row,
                            width - 1 + col - source_col,
                        ]
                scattered[0, row, col] = value
        expected_complex = total + scattered - incident
        expected = torch.stack(
            (expected_complex.real, expected_complex.imag), dim=1
        )

        def as_channels(value: torch.Tensor) -> torch.Tensor:
            return torch.stack((value.real, value.imag), dim=1)

        actual = VIEOperator(
            grid_size=(height, width), dimensionless_wavenumber=wavenumber
        ).residual(as_channels(total), as_channels(incident), as_channels(contrast))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_input_features_do_not_depend_on_target_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            building_path = root / "building.png"
            building = np.zeros((8, 8), dtype=np.uint8)
            building[1:3, 5:7] = 12
            Image.fromarray(building).save(building_path)

            target_a = root / "target_a.png"
            target_b = root / "target_b.png"
            Image.fromarray(np.full((8, 8), 25, dtype=np.uint8)).save(target_a)
            Image.fromarray(np.full((8, 8), 225, dtype=np.uint8)).save(target_b)

            manifests = []
            for suffix, target_path in (("a", target_a), ("b", target_b)):
                manifest = root / f"{suffix}.csv"
                with manifest.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=(
                            "scene_id",
                            "tx_x",
                            "tx_y",
                            "tx_z",
                            "rx_height",
                            "pathloss_path",
                            "building_height_path",
                        ),
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "scene_id": 11,
                            "tx_x": 3,
                            "tx_y": 4,
                            "tx_z": 1.5,
                            "rx_height": 4,
                            "pathloss_path": str(target_path),
                            "building_height_path": str(building_path),
                        }
                    )
                manifests.append(manifest)

            samples = []
            for manifest in manifests:
                dataset = PEFNetInspiredDataset(
                    manifest,
                    max_records=1,
                    selection_strategy="sequential",
                    seed=42,
                    exclude_building_occupied=False,
                )
                samples.append(dataset[0])

            self.assertFalse(torch.equal(samples[0]["target"], samples[1]["target"]))
            for name in ("input", "incident_field", "contrast", "building_mask"):
                self.assertTrue(torch.equal(samples[0][name], samples[1][name]), name)


if __name__ == "__main__":
    unittest.main()
