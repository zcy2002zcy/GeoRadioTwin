"""Paper-guided RMTransformer-style baseline for dense radio-map prediction.

The implementation follows the architecture reported by Li et al.: a CNN
stem, four multi-scale MaxViT-style encoder stages, and a convolutional
decoder with skip connections.  No author implementation was available when
this adapter was written, so this module must not be presented as an official
or exact reproduction.  The extra receiver-height channel is the explicit
UrbanRadio3D adaptation used by this project.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ConvNormAct(nn.Sequential):
    """A conventional decoder convolution used after feature concatenation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.reduce(scale), inplace=True)
        return x * torch.sigmoid(self.expand(scale))


class MBConvDownsample(nn.Module):
    """Downsampling MBConv preceding each multi-axis attention stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expansion_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        expanded = max(int(round(in_channels * expansion_ratio)), out_channels)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, expanded, 1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                expanded,
                expanded,
                3,
                stride=2,
                padding=1,
                groups=expanded,
                bias=False,
            ),
            nn.BatchNorm2d(expanded),
            nn.SiLU(inplace=True),
            SqueezeExcitation(expanded),
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x) + self.shortcut(x)


def _block_partition(x: Tensor, size: int) -> tuple[Tensor, tuple[int, ...]]:
    batch, height, width, channels = x.shape
    if height % size or width % size:
        raise ValueError(
            f"Block attention requires H and W divisible by {size}, got {height}x{width}"
        )
    grid_h, grid_w = height // size, width // size
    tokens = (
        x.reshape(batch, grid_h, size, grid_w, size, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch * grid_h * grid_w, size * size, channels)
    )
    return tokens, (batch, height, width, channels, grid_h, grid_w)


def _block_reverse(tokens: Tensor, shape: tuple[int, ...], size: int) -> Tensor:
    batch, height, width, channels, grid_h, grid_w = shape
    return (
        tokens.reshape(batch, grid_h, grid_w, size, size, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height, width, channels)
    )


def _grid_partition(x: Tensor, size: int) -> tuple[Tensor, tuple[int, ...]]:
    """Group spatially separated cells so attention spans the whole map."""

    batch, height, width, channels = x.shape
    if height % size or width % size:
        raise ValueError(
            f"Grid attention requires H and W divisible by {size}, got {height}x{width}"
        )
    grid_h, grid_w = height // size, width // size
    tokens = (
        x.reshape(batch, grid_h, size, grid_w, size, channels)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch * size * size, grid_h * grid_w, channels)
    )
    return tokens, (batch, height, width, channels, grid_h, grid_w)


def _grid_reverse(tokens: Tensor, shape: tuple[int, ...], size: int) -> Tensor:
    batch, height, width, channels, grid_h, grid_w = shape
    return (
        tokens.reshape(batch, size, size, grid_h, grid_w, channels)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, height, width, channels)
    )


class PartitionAttention(nn.Module):
    """Pre-normalized multi-head attention over block or grid partitions."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        partition_size: int,
        partition_type: str,
        mlp_ratio: float,
        attention_dropout: float,
        projection_dropout: float,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"Embedding dimension {dim} must be divisible by {num_heads} heads")
        if partition_type not in {"block", "grid"}:
            raise ValueError(f"Unknown partition type: {partition_type}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.partition_size = int(partition_size)
        self.partition_type = partition_type
        self.attention_dropout = float(attention_dropout)
        self.norm_attention = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(projection_dropout),
        )
        hidden = max(int(round(dim * mlp_ratio)), dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(projection_dropout),
        )

    def _partition(self, x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        if self.partition_type == "block":
            return _block_partition(x, self.partition_size)
        return _grid_partition(x, self.partition_size)

    def _reverse(self, tokens: Tensor, shape: tuple[int, ...]) -> Tensor:
        if self.partition_type == "block":
            return _block_reverse(tokens, shape, self.partition_size)
        return _grid_reverse(tokens, shape, self.partition_size)

    def forward(self, x: Tensor) -> Tensor:
        tokens, shape = self._partition(x)
        residual = tokens
        normalized = self.norm_attention(tokens)
        groups, length, _ = normalized.shape
        qkv = self.qkv(normalized).reshape(
            groups, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(groups, length, self.dim)
        tokens = residual + self.projection(attended)
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        return self._reverse(tokens, shape)


class MultiAxisTransformerStage(nn.Module):
    """MaxViT-style downsampling followed by local and global attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_heads: int,
        partition_size: int,
        mlp_ratio: float,
        mbconv_expansion: float,
        attention_dropout: float,
        projection_dropout: float,
    ) -> None:
        super().__init__()
        self.downsample = MBConvDownsample(
            in_channels, out_channels, expansion_ratio=mbconv_expansion
        )
        # Depth-wise conditional positional encoding is resolution agnostic and
        # avoids claiming an unrecoverable positional-bias setup from the paper.
        self.position = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, groups=out_channels, bias=True
        )
        attention_args = dict(
            dim=out_channels,
            num_heads=num_heads,
            partition_size=partition_size,
            mlp_ratio=mlp_ratio,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
        )
        self.block_attention = PartitionAttention(
            partition_type="block", **attention_args
        )
        self.grid_attention = PartitionAttention(
            partition_type="grid", **attention_args
        )

    def prepare_attention(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        x = x + self.position(x)
        return x.permute(0, 2, 3, 1)

    @staticmethod
    def restore_channels_first(x: Tensor) -> Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        x = self.prepare_attention(x)
        x = self.block_attention(x)
        x = self.grid_attention(x)
        return self.restore_channels_first(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.refine = nn.Sequential(
            ConvNormAct(out_channels + skip_channels, out_channels),
            ConvNormAct(out_channels, out_channels),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                f"Decoder/skip spatial mismatch: {tuple(x.shape[-2:])} vs {tuple(skip.shape[-2:])}"
            )
        return self.refine(torch.cat((x, skip), dim=1))


class RMTransformerBaseline(nn.Module):
    """Height-adapted RMTransformer-style 256x256 image regressor."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        stem_channels: int = 128,
        encoder_channels: Sequence[int] = (128, 256, 512, 1024),
        num_heads: Sequence[int] = (4, 8, 16, 32),
        partition_size: int = 8,
        mlp_ratio: float = 4.0,
        mbconv_expansion: float = 4.0,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in encoder_channels)
        heads = tuple(int(value) for value in num_heads)
        if len(widths) != 4 or len(heads) != 4:
            raise ValueError("RMTransformer-style baseline requires exactly four encoder stages")
        if in_channels != 3:
            raise ValueError(
                "Height-adapted RMTransformer expects exactly three channels: "
                "building height, Tx source, and Rx-height plane"
            )
        self.in_channels = int(in_channels)
        self.partition_size = int(partition_size)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.stem = ConvNormAct(in_channels, stem_channels, stride=2)
        stage_inputs = (stem_channels, *widths[:-1])
        self.encoder = nn.ModuleList(
            MultiAxisTransformerStage(
                input_width,
                output_width,
                num_heads=head_count,
                partition_size=partition_size,
                mlp_ratio=mlp_ratio,
                mbconv_expansion=mbconv_expansion,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )
            for input_width, output_width, head_count in zip(
                stage_inputs, widths, heads
            )
        )
        self.decoder = nn.ModuleList(
            (
                DecoderBlock(widths[3], widths[2], widths[2]),
                DecoderBlock(widths[2], widths[1], widths[1]),
                DecoderBlock(widths[1], widths[0], widths[0]),
                DecoderBlock(widths[0], stem_channels, stem_channels),
            )
        )
        output_channels = max(stem_channels // 2, 16)
        self.output_upsample = nn.ConvTranspose2d(
            stem_channels, output_channels, 2, stride=2
        )
        self.output_head = nn.Sequential(
            ConvNormAct(output_channels + in_channels, output_channels),
            nn.Conv2d(output_channels, 1, 1),
        )

    def _run_stage(self, stage: MultiAxisTransformerStage, x: Tensor) -> Tensor:
        if self.use_gradient_checkpointing and self.training and x.requires_grad:
            # Keep the MBConv/BatchNorm path outside recomputation so running
            # statistics are updated exactly once.  Only the memory-intensive
            # attention/MLP paths are checkpointed.
            x = stage.prepare_attention(x)
            x = checkpoint(stage.block_attention, x, use_reentrant=False)
            x = checkpoint(stage.grid_attention, x, use_reentrant=False)
            return stage.restore_channels_first(x)
        return stage(x)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [B, {self.in_channels}, H, W], got {tuple(x.shape)}"
            )
        divisor = 2 ** (len(self.encoder) + 1)
        if x.shape[-2] % divisor or x.shape[-1] % divisor:
            raise ValueError(
                f"Input H and W must be divisible by {divisor}, got {tuple(x.shape[-2:])}"
            )
        original = x
        stem = self.stem(x)
        features: list[Tensor] = []
        encoded = stem
        for stage in self.encoder:
            encoded = self._run_stage(stage, encoded)
            features.append(encoded)
        decoded = features[-1]
        skips = (features[2], features[1], features[0], stem)
        for block, skip in zip(self.decoder, skips):
            decoded = block(decoded, skip)
        decoded = self.output_upsample(decoded)
        return self.output_head(torch.cat((decoded, original), dim=1))


def build_rmtransformer_adapted(
    *,
    in_channels: int = 3,
    stem_channels: int = 128,
    encoder_channels: Sequence[int] = (128, 256, 512, 1024),
    num_heads: Sequence[int] = (4, 8, 16, 32),
    partition_size: int = 8,
    mlp_ratio: float = 4.0,
    mbconv_expansion: float = 4.0,
    attention_dropout: float = 0.0,
    projection_dropout: float = 0.0,
    use_gradient_checkpointing: bool = True,
) -> RMTransformerBaseline:
    """Build the independent paper-guided, receiver-height adaptation."""

    return RMTransformerBaseline(
        in_channels=in_channels,
        stem_channels=stem_channels,
        encoder_channels=encoder_channels,
        num_heads=num_heads,
        partition_size=partition_size,
        mlp_ratio=mlp_ratio,
        mbconv_expansion=mbconv_expansion,
        attention_dropout=attention_dropout,
        projection_dropout=projection_dropout,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


__all__ = ["RMTransformerBaseline", "build_rmtransformer_adapted"]
