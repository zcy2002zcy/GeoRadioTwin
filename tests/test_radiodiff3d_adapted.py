from __future__ import annotations

import csv
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

from datasets.radiodiff3d_volume_dataset import (
    RadioDiff3DAnchorEvaluationDataset,
    RadioDiff3DVolumeDataset,
    read_complete_volume_records,
)
from models.radiodiff3d_adapted import (
    RadioDiff3DAdapted,
    build_radiodiff3d_adapted,
)
from scripts.evaluate_radiodiff3d_adapted import anchor_score_arrays


class RadioDiff3DAdaptedTest(unittest.TestCase):
    @staticmethod
    def _write_manifest(root: Path, heights: tuple[int, ...]) -> Path:
        building_path = root / "building.png"
        building = np.zeros((16, 16), dtype=np.uint8)
        building[0, 0] = 4
        Image.fromarray(building).save(building_path)
        rows = []
        for height in heights:
            target_path = root / f"pathloss_h{height}.png"
            Image.fromarray(
                np.full((16, 16), 40 + height, dtype=np.uint8)
            ).save(target_path)
            rows.append(
                {
                    "scene_id": 7,
                    "tx_x": 5,
                    "tx_y": 6,
                    "tx_z": 1.5,
                    "rx_height": height,
                    "pathloss_path": str(target_path),
                    "building_height_path": str(building_path),
                }
            )
        manifest = root / "split.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_volume_dataset_stacks_all_four_heights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_manifest(Path(directory), (1, 2, 3, 4))
            dataset = RadioDiff3DVolumeDataset(
                manifest,
                max_volumes=1,
                selection_strategy="sequential",
                seed=42,
            )
            sample = dataset[0]
            self.assertEqual(sample["condition"].shape, (3, 4, 16, 16))
            self.assertEqual(sample["target"].shape, (1, 4, 16, 16))
            self.assertEqual(sample["valid_mask"].shape, (1, 4, 16, 16))
            self.assertEqual(float(sample["condition"][0].sum()), 4.0)
            self.assertEqual(float(sample["condition"][2].sum()), 4.0)
            self.assertTrue(torch.equal(sample["rx_heights"], torch.tensor([1.0, 2.0, 3.0, 4.0])))

    def test_legacy_condition_layout_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_manifest(Path(directory), (1, 2, 3, 4))
            dataset = RadioDiff3DVolumeDataset(
                manifest,
                max_volumes=1,
                selection_strategy="sequential",
                seed=42,
                condition_layout="legacy_height",
            )
            sample = dataset[0]
            self.assertEqual(float(sample["condition"][1].sum()), 4.0)
            self.assertAlmostEqual(
                float(sample["condition"][2, :, 0, 0].sum()), 0.5, places=6
            )

    def test_incomplete_height_group_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_manifest(Path(directory), (1, 2, 4))
            with self.assertRaisesRegex(ValueError, "missing slices"):
                read_complete_volume_records(manifest)

    def test_anchor_evaluation_scores_only_selected_height(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            # Sequential canonical selection chooses h3 as the sole anchor;
            # h1, h2, and h4 remain conditioning companions only.
            manifest = self._write_manifest(Path(directory), (3, 1, 2, 4))
            dataset = RadioDiff3DAnchorEvaluationDataset(
                manifest,
                max_anchor_maps=1,
                selection_strategy="sequential",
                seed=42,
            )
            sample = dataset[0]
            self.assertEqual(sample["target"].shape[1], 4)
            self.assertEqual(int(sample["score_depth_index"]), 2)
            self.assertEqual(float(sample["score_rx_height"]), 3.0)

            prediction = np.stack(
                [np.full((2, 2), value, dtype=np.float32) for value in (10, 20, 30, 40)]
            )
            target = np.zeros_like(prediction)
            valid = np.ones_like(prediction, dtype=bool)
            scored_prediction, scored_target = anchor_score_arrays(
                prediction, target, valid, int(sample["score_depth_index"])
            )
            self.assertTrue(np.all(scored_prediction == 30))
            self.assertEqual(scored_prediction.size, 4)
            self.assertEqual(scored_target.size, 4)

    def test_noise_prediction_and_fixed_seed_ddim(self) -> None:
        model = build_radiodiff3d_adapted(
            base_channels=4,
            channel_multipliers=(1, 2),
            time_dimension=16,
            expected_depth=4,
            timesteps=8,
        ).eval()
        condition = torch.randn(1, 3, 4, 16, 16)
        target = torch.rand(1, 1, 4, 16, 16)
        valid = torch.ones_like(target, dtype=torch.bool)
        timesteps = torch.tensor([3], dtype=torch.long)
        noise = torch.randn_like(target)
        loss, returned_timesteps = model.training_loss(
            condition, target, valid, timesteps=timesteps, noise=noise
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(returned_timesteps, timesteps))

        first_generator = torch.Generator().manual_seed(2027)
        second_generator = torch.Generator().manual_seed(2027)
        different_generator = torch.Generator().manual_seed(2028)
        first = model.ddim_sample(condition, sampling_steps=4, generator=first_generator)
        second = model.ddim_sample(condition, sampling_steps=4, generator=second_generator)
        different = model.ddim_sample(condition, sampling_steps=4, generator=different_generator)
        self.assertEqual(first.shape, (1, 1, 4, 16, 16))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))

    def test_paper_v2_uses_full_volume_l1_x0_loss(self) -> None:
        class ZeroDenoiser(torch.nn.Module):
            target_channels = 1

            def forward(
                self,
                noisy_target: torch.Tensor,
                condition: torch.Tensor,
                timesteps: torch.Tensor,
            ) -> torch.Tensor:
                return torch.zeros_like(noisy_target)

        model = RadioDiff3DAdapted(
            ZeroDenoiser(),
            timesteps=8,
            parameterization="x0",
            loss_type="l1",
            loss_support="full",
        )
        target = torch.full((1, 1, 4, 2, 2), 0.75)
        condition = torch.zeros((1, 3, 4, 2, 2))
        valid = torch.zeros_like(target, dtype=torch.bool)
        valid[..., 0, 0] = True
        noise = torch.arange(target.numel(), dtype=torch.float32).reshape_as(target)
        loss, _ = model.training_loss(
            condition,
            target,
            valid,
            timesteps=torch.tensor([3]),
            noise=noise,
        )
        clean = target * 2.0 - 1.0
        self.assertAlmostEqual(float(loss), float(clean.abs().mean()), places=6)

    def test_multiscale_conditioning_and_ddpm_sampler(self) -> None:
        model = build_radiodiff3d_adapted(
            base_channels=4,
            channel_multipliers=(1, 2),
            time_dimension=16,
            expected_depth=4,
            timesteps=8,
            multiscale_conditioning=True,
            loss_type="l1",
            loss_support="full",
        ).eval()
        condition = torch.randn(1, 3, 4, 16, 16)
        first = model.ddpm_sample(
            condition,
            generator=torch.Generator().manual_seed(2027),
        )
        second = model.ddpm_sample(
            condition,
            generator=torch.Generator().manual_seed(2027),
        )
        self.assertEqual(first.shape, (1, 1, 4, 16, 16))
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
