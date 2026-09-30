#!/usr/bin/env python
"""Generate a data-anchored GeoRadioTwin framework figure in Python."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
})

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_artifacts" / "source_data" / "framework_scene71.npz"
OUT = ROOT / "paper_artifacts" / "figures"

INK = "#243238"
MUTED = "#66757C"
BLUE = "#2166A5"
BLUE_SOFT = "#EAF2F8"
TEAL = "#238D8E"
TEAL_SOFT = "#E8F5F3"
AMBER = "#D8951E"
AMBER_SOFT = "#FFF5DD"
PURPLE = "#7565A8"
PURPLE_SOFT = "#F1EEF8"
LINE = "#C9D2D7"
WHITE = "#FFFFFF"

BUILDING_CMAP = LinearSegmentedColormap.from_list(
    "building", ["#0A1014", "#26343C", "#718995", "#D8E0E4"]
)
RADIO_CMAP = LinearSegmentedColormap.from_list(
    "radio", ["#111827", "#174D6A", "#247E88", "#65B9AD", "#E7D28A", "#D96B55"]
)
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error", ["#F7F7F7", "#F3C98B", "#D85C4A", "#7C2530"]
)


def rounded(ax, x, y, w, h, face=WHITE, edge=LINE, lw=0.8, radius=0.012, z=2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
    )
    ax.add_patch(box)
    return box


def text(ax, x, y, value, size=7, weight="normal", color=INK,
         ha="center", va="center", z=8, linespacing=1.12):
    ax.text(x, y, value, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, zorder=z, linespacing=linespacing)


def arrow(ax, start, end, color=INK, lw=1.0, rad=0.0, dashed=False, z=6):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8,
        color=color, linewidth=lw,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2, zorder=z,
    )
    ax.add_patch(patch)


def image_axes(fig, rect, data, cmap, vmin=None, vmax=None, mask=None):
    ax = fig.add_axes(rect, zorder=4)
    shown = np.ma.array(data, mask=mask) if mask is not None else data
    ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(WHITE)
        spine.set_linewidth(0.7)
    return ax


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing {SOURCE}. Run: python scripts/export_framework_scene.py"
        )
    source = np.load(SOURCE)
    metadata = json.loads(str(source["metadata"].item()))
    building = source["building"]
    target = source["target"]
    prediction = source["prediction"]
    valid = source["valid_mask"].astype(bool)
    los = source["los"]
    depth = source["depth"]
    error = np.abs(prediction - target)

    fig = plt.figure(figsize=(7.2, 4.55), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    text(ax, 0.035, 0.965, "GeoRadioTwin", size=11, weight="bold", color=BLUE, ha="left")
    text(ax, 0.215, 0.965, "Geometry-guided continuous radio-field digital twin",
         size=9.3, weight="bold", ha="left")
    text(ax, 0.965, 0.965, "measurement-free inference", size=6.2,
         weight="bold", color=TEAL, ha="right")

    # Section headers: small labels, not enclosing decorative panels.
    text(ax, 0.028, 0.908, "a", size=8, weight="bold", color=BLUE, ha="left")
    text(ax, 0.052, 0.908, "REAL ENVIRONMENT STATE", size=7.4, weight="bold", color=BLUE, ha="left")
    text(ax, 0.322, 0.908, "b", size=8, weight="bold", color=PURPLE, ha="left")
    text(ax, 0.346, 0.908, "GEOMETRY-GUIDED FIELD MODEL", size=7.4, weight="bold", color=PURPLE, ha="left")
    text(ax, 0.786, 0.908, "c", size=8, weight="bold", color=TEAL, ha="left")
    text(ax, 0.810, 0.908, "REAL FIELD QUERY", size=7.4, weight="bold", color=TEAL, ha="left")
    ax.plot([0.028, 0.292], [0.888, 0.888], color=BLUE, lw=1.0)
    ax.plot([0.322, 0.756], [0.888, 0.888], color=PURPLE, lw=1.0)
    ax.plot([0.786, 0.972], [0.888, 0.888], color=TEAL, lw=1.0)

    # Real building-height plate.
    rounded(ax, 0.025, 0.455, 0.270, 0.415, face="#0A1014", edge=BLUE, lw=1.0, radius=0.010)
    ax_build = image_axes(fig, [0.034, 0.475, 0.252, 0.365], building,
                          BUILDING_CMAP, vmin=0, vmax=255)
    tx_row = 255.0 - float(metadata["tx_x"])
    tx_col = float(metadata["tx_y"])
    rx_row = float(metadata["illustrative_rx_row"])
    rx_col = float(metadata["illustrative_rx_col"])
    fractions = np.linspace(0, 1, 9)
    ray_rows = tx_row * (1-fractions) + rx_row * fractions
    ray_cols = tx_col * (1-fractions) + rx_col * fractions
    ax_build.plot(ray_cols, ray_rows, color="#FFD166", lw=1.15, alpha=0.95, zorder=6)
    ax_build.scatter(ray_cols, ray_rows, s=7, facecolor="#FFD166", edgecolor="#563A00", lw=0.25, zorder=7)
    ax_build.scatter([tx_col], [tx_row], marker="^", s=48, facecolor="#EF5A5A", edgecolor=WHITE, lw=0.8, zorder=8)
    ax_build.scatter([rx_col], [rx_row], marker="o", s=36, facecolor="#2BD1C5", edgecolor=WHITE, lw=0.8, zorder=8)
    ax_build.text(tx_col+7, tx_row-8, "Tx", color=WHITE, fontsize=6.2, weight="bold", zorder=9)
    ax_build.text(rx_col-25, rx_row-8, "query q", color=WHITE, fontsize=6.0, weight="bold", zorder=9)
    ax_build.text(7, 244, "building height raster", color=WHITE, fontsize=5.4,
                  bbox=dict(facecolor="#0A1014", alpha=0.72, edgecolor="none", pad=1.6), zorder=9)

    text(ax, 0.035, 0.432, f"Scene {metadata['scene_id']}  |  Tx = ({metadata['tx_x']:.0f}, {metadata['tx_y']:.0f}, {metadata['tx_z']:.1f})",
         size=6.2, weight="bold", ha="left")
    text(ax, 0.035, 0.405, f"receiver height h{metadata['rx_height']:.0f}  |  nine deterministic ray samples",
         size=5.7, color=MUTED, ha="left")

    # Data-derived geometry thumbnails.
    rounded(ax, 0.025, 0.248, 0.129, 0.127, face=AMBER_SOFT, edge=AMBER, lw=0.7, radius=0.008)
    rounded(ax, 0.166, 0.248, 0.129, 0.127, face=AMBER_SOFT, edge=AMBER, lw=0.7, radius=0.008)
    ax_los = image_axes(fig, [0.050, 0.286, 0.078, 0.067], los, "gray", vmin=0, vmax=1, mask=~valid)
    ax_depth = image_axes(fig, [0.191, 0.286, 0.078, 0.067], depth, "magma", vmin=0, vmax=1, mask=~valid)
    text(ax, 0.089, 0.266, "LoS map", size=5.8, weight="bold", color="#91610D")
    text(ax, 0.230, 0.266, "depth map", size=5.8, weight="bold", color="#91610D")
    text(ax, 0.160, 0.225, "analytic propagation geometry", size=6.2, weight="bold", color=AMBER)

    # Central model, two information paths.
    rounded(ax, 0.330, 0.706, 0.124, 0.124, face=BLUE_SOFT, edge=BLUE, lw=0.8, radius=0.010)
    text(ax, 0.392, 0.794, "environment tensor M", size=6.2, weight="bold", color=BLUE)
    text(ax, 0.392, 0.758, "height · Tx impulse", size=5.6)
    text(ax, 0.392, 0.729, "Rx altitude", size=5.6)
    rounded(ax, 0.489, 0.706, 0.119, 0.124, face=BLUE_SOFT, edge=BLUE, lw=0.8, radius=0.010)
    text(ax, 0.5485, 0.790, "multiscale CNN", size=6.4, weight="bold", color=BLUE)
    text(ax, 0.5485, 0.754, "full-map context", size=5.5)
    text(ax, 0.5485, 0.726, "$F_\\theta(M)$", size=6.0, weight="bold")
    arrow(ax, (0.455, 0.768), (0.488, 0.768), color=BLUE)

    rounded(ax, 0.330, 0.505, 0.124, 0.140, face=PURPLE_SOFT, edge=PURPLE, lw=0.8, radius=0.010)
    text(ax, 0.392, 0.608, "continuous query", size=6.4, weight="bold", color=PURPLE)
    text(ax, 0.392, 0.573, "Tx, Rx, Δ, distance", size=5.5)
    text(ax, 0.392, 0.538, "+ LoS and depth", size=5.6, color=AMBER)
    rounded(ax, 0.489, 0.505, 0.119, 0.140, face=PURPLE_SOFT, edge=PURPLE, lw=0.8, radius=0.010)
    text(ax, 0.5485, 0.606, "Fourier + geometry", size=6.2, weight="bold", color=PURPLE)
    text(ax, 0.5485, 0.572, "continuous MLP", size=5.7)
    text(ax, 0.5485, 0.535, r"base field  $\hat{y}_0(q)$", size=5.8, weight="bold")
    arrow(ax, (0.455, 0.575), (0.488, 0.575), color=PURPLE)

    rounded(ax, 0.635, 0.664, 0.105, 0.166, face=AMBER_SOFT, edge=AMBER, lw=0.9, radius=0.010)
    text(ax, 0.6875, 0.794, "ray context", size=6.3, weight="bold", color=AMBER)
    ax.plot([0.650, 0.724], [0.757, 0.757], color=AMBER, lw=1.0, zorder=6)
    for xx in np.linspace(0.650, 0.724, 9):
        ax.scatter([xx], [0.757], s=8, color=AMBER, edgecolor=WHITE, lw=0.25, zorder=7)
    text(ax, 0.6875, 0.711, "Tx/Rx features", size=5.4)
    text(ax, 0.6875, 0.683, "+ ray mean/max", size=5.4)
    arrow(ax, (0.609, 0.768), (0.634, 0.752), color=BLUE)
    arrow(ax, (0.454, 0.550), (0.641, 0.675), color=AMBER, rad=-0.12)

    rounded(ax, 0.635, 0.455, 0.105, 0.154, face=WHITE, edge=PURPLE, lw=0.9, radius=0.010)
    text(ax, 0.6875, 0.574, "residual fusion", size=6.3, weight="bold", color=PURPLE)
    text(ax, 0.6875, 0.538, "context + state", size=5.4)
    text(ax, 0.6875, 0.510, r"+ geometry + $\hat{y}_0$", size=5.4)
    text(ax, 0.6875, 0.477, r"$\Delta\hat{y}(q)$", size=6.3, weight="bold", color=BLUE)
    arrow(ax, (0.6875, 0.664), (0.6875, 0.610), color=AMBER)
    arrow(ax, (0.609, 0.575), (0.634, 0.548), color=PURPLE)

    rounded(ax, 0.480, 0.319, 0.260, 0.080, face="#F7FAFC", edge=BLUE, lw=1.0, radius=0.010)
    text(ax, 0.610, 0.369, "continuous field query", size=5.5, color=MUTED)
    text(ax, 0.610, 0.341, r"$\hat{y}(q)=\hat{y}_0(q)+\Delta\hat{y}(q)$", size=7.2, weight="bold", color=BLUE)
    arrow(ax, (0.6875, 0.455), (0.666, 0.400), color=BLUE)

    # Cross-panel data flow.
    arrow(ax, (0.295, 0.700), (0.329, 0.768), color=BLUE)
    arrow(ax, (0.295, 0.315), (0.329, 0.558), color=AMBER, rad=-0.10)

    # Real model output and compact validation thumbnails.
    rounded(ax, 0.782, 0.455, 0.190, 0.415, face="#101820", edge=TEAL, lw=1.0, radius=0.010)
    ax_pred = image_axes(fig, [0.793, 0.573, 0.168, 0.267], prediction,
                         RADIO_CMAP, vmin=0, vmax=255, mask=~valid)
    ax_pred.scatter([metadata["tx_y"]], [255-metadata["tx_x"]], marker="^", s=34,
                    facecolor="#EF5A5A", edgecolor=WHITE, lw=0.7, zorder=8)
    ax_pred.text(7, 244, "GeoRadioTwin prediction", color=WHITE, fontsize=5.4,
                 bbox=dict(facecolor="#101820", alpha=0.72, edgecolor="none", pad=1.5), zorder=9)
    text(ax, 0.877, 0.551, "encoded radio value (not physical dB)", size=5.2, color=WHITE)

    ax_gt = image_axes(fig, [0.801, 0.478, 0.067, 0.067], target,
                       RADIO_CMAP, vmin=0, vmax=255, mask=~valid)
    err_vmax = float(np.nanpercentile(error[valid], 98))
    ax_err = image_axes(fig, [0.889, 0.478, 0.067, 0.067], error,
                        ERROR_CMAP, vmin=0, vmax=err_vmax, mask=~valid)
    text(ax, 0.8345, 0.464, "ground truth", size=5.1, color=WHITE)
    text(ax, 0.9225, 0.464, "absolute error", size=5.1, color=WHITE)
    arrow(ax, (0.741, 0.355), (0.782, 0.650), color=TEAL, rad=0.10)

    # Bottom: clear digital-twin loop and training boundary.
    rounded(ax, 0.025, 0.043, 0.947, 0.135, face="#F8FAFB", edge=LINE, lw=0.8, radius=0.012)
    text(ax, 0.043, 0.151, "DIGITAL-TWIN OPERATION", size=6.8, weight="bold", color=TEAL, ha="left")
    rounded(ax, 0.043, 0.071, 0.180, 0.056, face=TEAL_SOFT, edge=TEAL, lw=0.7, radius=0.007)
    text(ax, 0.133, 0.099, "update Tx position or Rx height", size=5.7, weight="bold", color=TEAL)
    arrow(ax, (0.224, 0.099), (0.257, 0.099), color=TEAL)
    rounded(ax, 0.259, 0.071, 0.170, 0.056, face=BLUE_SOFT, edge=BLUE, lw=0.7, radius=0.007)
    text(ax, 0.344, 0.099, "re-query the continuous field", size=5.7, weight="bold", color=BLUE)
    arrow(ax, (0.430, 0.099), (0.463, 0.099), color=TEAL)
    rounded(ax, 0.465, 0.071, 0.162, 0.056, face=WHITE, edge=LINE, lw=0.7, radius=0.007)
    text(ax, 0.546, 0.099, "dense map / site ranking", size=5.7, weight="bold")
    arrow(ax, (0.628, 0.099), (0.661, 0.099), color=TEAL)
    rounded(ax, 0.663, 0.071, 0.177, 0.056, face=PURPLE_SOFT, edge=PURPLE, lw=0.7, radius=0.007)
    text(ax, 0.7515, 0.099, "three-seed uncertainty", size=5.7, weight="bold", color=PURPLE)
    rounded(ax, 0.855, 0.071, 0.100, 0.056, face=AMBER_SOFT, edge=AMBER, lw=0.7, radius=0.007)
    text(ax, 0.905, 0.099, "no radio samples", size=5.5, weight="bold", color="#91610D")
    text(ax, 0.955, 0.151, "GT maps are used only for training", size=5.2, color=MUTED, ha="right")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig1_georadiotwin_framework"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Generated data-anchored framework figure: {stem}")


if __name__ == "__main__":
    main()
