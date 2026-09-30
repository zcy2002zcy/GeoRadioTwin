from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.rmtransformer_baseline import RMTransformerBaseline
from scripts.train_rmtransformer_baseline import masked_statistics


def tiny_model() -> RMTransformerBaseline:
    return RMTransformerBaseline(
        in_channels=3,
        stem_channels=8,
        encoder_channels=(8, 16, 32, 64),
        num_heads=(1, 2, 4, 8),
        partition_size=2,
        mlp_ratio=2.0,
        mbconv_expansion=2.0,
        use_gradient_checkpointing=False,
    )


class RMTransformerBaselineTest(unittest.TestCase):
    def test_dense_output_matches_input_resolution(self) -> None:
        model = tiny_model().eval()
        with torch.inference_mode():
            output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(output.shape, (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(output).all())

    def test_model_contains_both_attention_axes_and_cnn_decoder(self) -> None:
        model = tiny_model()
        names = set(dict(model.named_modules()))
        self.assertIn("encoder.0.block_attention", names)
        self.assertIn("encoder.0.grid_attention", names)
        self.assertIsInstance(model.decoder[0].upsample, torch.nn.ConvTranspose2d)

    def test_attention_checkpoint_path_backpropagates(self) -> None:
        model = tiny_model().train()
        model.use_gradient_checkpointing = True
        model(torch.randn(1, 3, 64, 64)).square().mean().backward()
        self.assertIsNotNone(model.stem[0].weight.grad)
        self.assertTrue(torch.isfinite(model.stem[0].weight.grad).all())

    def test_wrong_input_channel_count_is_rejected(self) -> None:
        model = tiny_model().eval()
        with self.assertRaisesRegex(ValueError, "Expected"):
            model(torch.randn(1, 2, 64, 64))

    def test_selection_statistics_use_primary_minus_one_to_one_domain(self) -> None:
        prediction = torch.tensor([[[[0.50]]]])
        target = torch.tensor([[[[0.75]]]])
        support = torch.ones_like(target, dtype=torch.bool)
        loss, sse, target_sq, count = masked_statistics(
            prediction, target, support
        )
        self.assertAlmostEqual(loss.item(), 0.25 ** 2)
        self.assertAlmostEqual(sse, 0.50 ** 2)
        self.assertAlmostEqual(target_sq, 0.50 ** 2)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
