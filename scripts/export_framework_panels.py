#!/usr/bin/env python
"""Export the real-scene panels used by the GeoRadioTwin framework separately."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 9,
})

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_artifacts" / "source_data" / "framework_scene71.npz"
OUT = ROOT / "paper_artifacts" / "figures" / "framework_panels"

BUILDING_CMAP = LinearSegmentedColormap.from_list(
    "building", ["#0A1014", "#26343C", "#718995", "#D8E0E4"]
)
RADIO_CMAP = LinearSegmentedColormap.from_list(
    "radio", ["#111827", "#174D6A", "#247E88", "#65B9AD", "#E7D28A", "#D96B55"]
)
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error", ["#F7F7F7", "#F3C98B", "#D85C4A", "#7C2530"]
)
DEPTH_CMAP = plt.get_cmap("magma").copy()
LOS_CMAP = plt.get_cmap("gray").copy()
RADIO_CMAP.set_bad("#EEEEEE")
ERROR_CMAP.set_bad("#EEEEEE")
DEPTH_CMAP.set_bad("#EEEEEE")
LOS_CMAP.set_bad("#EEEEEE")


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / name
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def scalar_panel(data, *, title, cmap, vmin, vmax, cbar_label, mask=None, name):
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    shown = np.ma.array(data, mask=mask) if mask is not None else data
    im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontweight="bold", pad=7)
    ax.set_xlabel("raster column")
    ax.set_ylabel("raster row")
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.set_yticks([0, 64, 128, 192, 255])
    ax.tick_params(length=2.5, width=0.7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=7, length=2)
    fig.tight_layout(pad=0.8)
    save(fig, name)


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing {SOURCE}. Run: python scripts/export_framework_scene.py"
        )
    data = np.load(SOURCE)
    meta = json.loads(str(data["metadata"].item()))
    building = data["building"]
    target = data["target"]
    prediction = data["prediction"]
    valid = data["valid_mask"].astype(bool)
    los = data["los"]
    depth = data["depth"]
    error = np.abs(prediction - target)

    # Real building raster with the exact Tx and a deterministic illustrative query.
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(building, cmap=BUILDING_CMAP, vmin=0, vmax=255, interpolation="nearest")
    tx_row = 255.0 - float(meta["tx_x"])
    tx_col = float(meta["tx_y"])
    rx_row = float(meta["illustrative_rx_row"])
    rx_col = float(meta["illustrative_rx_col"])
    fractions = np.linspace(0, 1, 9)
    ray_rows = tx_row*(1-fractions) + rx_row*fractions
    ray_cols = tx_col*(1-fractions) + rx_col*fractions
    ax.plot(ray_cols, ray_rows, color="#FFD166", lw=1.5, zorder=5)
    ax.scatter(ray_cols, ray_rows, s=15, facecolor="#FFD166", edgecolor="#563A00", lw=0.35, zorder=6)
    ax.scatter([tx_col], [tx_row], marker="^", s=80, facecolor="#EF5A5A", edgecolor="white", lw=0.9, zorder=7)
    ax.scatter([rx_col], [rx_row], marker="o", s=60, facecolor="#2BD1C5", edgecolor="white", lw=0.9, zorder=7)
    ax.annotate("Tx", (tx_col, tx_row), xytext=(8, -8), textcoords="offset points",
                color="white", weight="bold", fontsize=8)
    ax.annotate("query q", (rx_col, rx_row), xytext=(-36, -8), textcoords="offset points",
                color="white", weight="bold", fontsize=8)
    ax.set_title(f"Scene {meta['scene_id']}: building state and ray query", fontweight="bold", pad=7)
    ax.set_xlabel("raster column")
    ax.set_ylabel("raster row")
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.set_yticks([0, 64, 128, 192, 255])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Building-height raster value", fontsize=7)
    cbar.ax.tick_params(labelsize=7, length=2)
    fig.tight_layout(pad=0.8)
    save(fig, "scene71_building_tx_ray")

    scalar_panel(
        los, title="Analytic line-of-sight map", cmap=LOS_CMAP,
        vmin=0, vmax=1, cbar_label="LoS indicator", mask=~valid,
        name="scene71_los",
    )
    scalar_panel(
        depth, title="Analytic obstruction-depth map", cmap=DEPTH_CMAP,
        vmin=0, vmax=1, cbar_label="Normalized obstruction depth", mask=~valid,
        name="scene71_depth",
    )
    scalar_panel(
        prediction, title="GeoRadioTwin prediction", cmap=RADIO_CMAP,
        vmin=0, vmax=255, cbar_label="Encoded radio value (not dB)", mask=~valid,
        name="scene71_prediction",
    )
    scalar_panel(
        target, title="Ground-truth radio field", cmap=RADIO_CMAP,
        vmin=0, vmax=255, cbar_label="Encoded radio value (not dB)", mask=~valid,
        name="scene71_ground_truth",
    )
    scalar_panel(
        error, title="Absolute prediction error", cmap=ERROR_CMAP,
        vmin=0, vmax=float(np.nanmax(error[valid])),
        cbar_label="Absolute encoded-value error", mask=~valid,
        name="scene71_absolute_error",
    )
    print(f"Exported six standalone panels to {OUT}")


if __name__ == "__main__":
    main()
