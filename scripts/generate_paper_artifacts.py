#!/usr/bin/env python
"""Generate ICC-ready tables, figures, source data, and result prose.

All numerical content is read from protocol-checked experiment summaries. The
script never scans checkpoints or recomputes metrics, which keeps manuscript
artifacts traceable to the frozen evaluation outputs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_artifacts"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
DATA_DIR = OUT / "source_data"
MANUSCRIPT_DIR = OUT / "manuscript"

COLORS = {
    "ours": "#0F4D92",
    "adapted": "#7884B4",
    "adapted2": "#B4C0E4",
    "classical": "#B8B8B8",
    "teal": "#42949E",
    "gold": "#D39B2A",
    "oracle": "#767676",
    "black": "#272727",
}

DISPLAY = {
    "HybridGeo": "GeoRadioTwin",
    "UniRM adapted": "UniRM (adapted)",
    "PEFNet-inspired VIE-regularized U-Net (adapted)": "PEFNet-inspired\n(adapted)",
    "RMTransformer-style (height-adapted)": "RMTransformer-style\n(adapted)",
    "RadioDiff-3D-style conditional diffusion (paper-aligned adapted v2)": "RadioDiff-3D-style\n(adapted)",
    "RadioUNet adapted": "RadioUNet-style\n(adapted)",
    "Fitted log-distance": "Log-distance",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        FIG_DIR / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="bottom")


def generate_main_figure(aggregate: list[dict]) -> None:
    wanted = list(DISPLAY)
    rows = [next(row for row in aggregate if row["method"] == method) for method in wanted]
    source_rows = []
    for row in rows:
        source_rows.append(
            {
                "method": DISPLAY[row["method"]].replace("\n", " "),
                "seeds": row["seeds"],
                "runs": row["runs"],
                "normalized_rmse_mean": row["rmse_mean"],
                "normalized_rmse_seed_sample_std": row["rmse_std_across_seeds"],
                "normalized_nmse_mean": row["nmse_mean"],
                "normalized_nmse_seed_sample_std": row["nmse_std_across_seeds"],
                "parameter_count": row["parameter_count"],
            }
        )
    write_csv(DATA_DIR / "fig_main_comparison.csv", source_rows)

    labels = [DISPLAY[row["method"]] for row in rows]
    y = np.arange(len(rows))
    rmse = np.array([row["rmse_mean"] for row in rows])
    rmse_std = np.array([row["rmse_std_across_seeds"] for row in rows])
    params = np.array([float(row["parameter_count"]) for row in rows])
    colors = [COLORS["ours"]] + [COLORS["adapted"]] * 5 + [COLORS["classical"]]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.08, 2.75), gridspec_kw={"wspace": 0.58})
    ax_a.barh(y, rmse, xerr=rmse_std, color=colors, edgecolor="white", linewidth=0.5,
              error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8, "ecolor": COLORS["black"]})
    ax_a.set_yticks(y, labels)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Normalized RMSE (lower is better)")
    ax_a.set_xlim(0, max(rmse + rmse_std) * 1.08)
    ax_a.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax_a.set_axisbelow(True)
    panel_label(ax_a, "a")

    ax_b.barh(y, params, color=colors, edgecolor="white", linewidth=0.5)
    ax_b.set_yticks(y, labels)
    ax_b.invert_yaxis()
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Trainable parameters (log scale)")
    ax_b.grid(axis="x", which="both", color="#E6E6E6", linewidth=0.6)
    ax_b.set_axisbelow(True)
    panel_label(ax_b, "b")

    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.20, top=0.95)
    save_figure(fig, "fig2_main_comparison")


def generate_height_figure(height_summary: dict) -> None:
    rows = height_summary["metrics_by_height"]
    source_rows = [
        {
            "receiver_height": row["rx_height"],
            "seeds": row["seeds"],
            "maps_per_seed": row["maps_per_seed"],
            "normalized_rmse_mean": row["rmse_mean"],
            "normalized_rmse_seed_sample_std": row["rmse_seed_std"],
            "normalized_nmse_mean": row["nmse_mean"],
            "normalized_nmse_seed_sample_std": row["nmse_seed_std"],
            "psnr_db_mean": row["psnr_mean"],
            "psnr_db_seed_sample_std": row["psnr_seed_std"],
        }
        for row in rows
    ]
    write_csv(DATA_DIR / "fig_unseen_height.csv", source_rows)
    x = np.array([row["rx_height"] for row in rows])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.08, 2.55), gridspec_kw={"wspace": 0.32})
    for ax, key, std_key, ylabel in [
        (ax_a, "rmse_mean", "rmse_seed_std", "Normalized RMSE"),
        (ax_b, "nmse_mean", "nmse_seed_std", "Normalized NMSE"),
    ]:
        y = np.array([row[key] for row in rows])
        e = np.array([row[std_key] for row in rows])
        ax.errorbar(x, y, yerr=e, color=COLORS["ours"], marker="o", markersize=4,
                    capsize=3, markerfacecolor="white", markeredgewidth=1.2)
        ax.text(0.02, 0.96, "trained on h1--h4 only", transform=ax.transAxes,
                fontsize=6.2, color=COLORS["adapted"], va="top")
        ax.set_xlim(4.7, 8.3)
        ax.set_xticks(x, [f"h{int(value)}" for value in x])
        ax.set_xlabel("Unseen receiver height")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.94)
    save_figure(fig, "fig3_unseen_height")


def generate_uncertainty_figure(summary: dict) -> None:
    risk_rows = summary["risk_coverage"]
    write_csv(DATA_DIR / "fig_uncertainty_risk_coverage.csv", risk_rows)
    calibration_rows = [
        {
            "nominal_coverage": 0.90,
            "empirical_coverage_mean": summary["aggregate"]["coverage_90"]["mean"],
            "empirical_coverage_map_std": summary["aggregate"]["coverage_90"]["std"],
        },
        {
            "nominal_coverage": 0.95,
            "empirical_coverage_mean": summary["aggregate"]["coverage_95"]["mean"],
            "empirical_coverage_map_std": summary["aggregate"]["coverage_95"]["std"],
        },
    ]
    write_csv(DATA_DIR / "fig_uncertainty_calibration.csv", calibration_rows)

    retained = np.array([row["retained_fraction"] for row in risk_rows])
    uncertainty = np.array([row["uncertainty_ranked_rmse"] for row in risk_rows])
    oracle = np.array([row["oracle_ranked_rmse"] for row in risk_rows])
    random_ref = np.array([row["random_reference_rmse"] for row in risk_rows])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.08, 2.65), gridspec_kw={"wspace": 0.34})
    ax_a.plot(retained, uncertainty, "o-", color=COLORS["ours"], label="Ensemble uncertainty")
    ax_a.plot(retained, oracle, "s--", color=COLORS["oracle"], label="Oracle error ranking")
    ax_a.plot(retained, random_ref, ":", color=COLORS["black"], label="Random retention")
    ax_a.set_xlabel("Retained pixel fraction")
    ax_a.set_ylabel("Normalized RMSE")
    ax_a.set_xlim(1.02, 0.23)
    ax_a.grid(color="#E6E6E6", linewidth=0.6)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="lower left")
    panel_label(ax_a, "a")

    nominal = np.array([row["nominal_coverage"] for row in calibration_rows])
    empirical = np.array([row["empirical_coverage_mean"] for row in calibration_rows])
    width = 0.16
    x = np.arange(2)
    ax_b.bar(x - width / 2, nominal, width, color="#D8D8D8", label="Nominal")
    ax_b.bar(x + width / 2, empirical, width, color=COLORS["teal"], label="Empirical")
    ax_b.set_xticks(x, ["90% interval", "95% interval"])
    ax_b.set_ylabel("Pixelwise coverage")
    ax_b.set_ylim(0.84, 0.97)
    ax_b.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="upper left")
    ax_b.text(0.98, 0.04, f"global $r$ = {summary['global_error_uncertainty_correlation']:.3f}",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=6.5)
    panel_label(ax_b, "b")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.94)
    save_figure(fig, "fig4_uncertainty")


def aggregate_twin_validation() -> list[dict]:
    summaries = [
        load_json(ROOT / "outputs" / f"twin_validation_seed{seed}_extended" / "summary.json")
        for seed in (42, 43, 44)
    ]

    def values(path: tuple[str, ...]) -> list[float]:
        found = []
        for summary in summaries:
            item = summary
            for key in path:
                item = item[key]
            found.append(float(item))
        return found

    specs = [
        ("Tx-position update", "Delta correlation", ("state_transitions", "tx_position", "delta_correlation")),
        ("Tx-position update", "Sign agreement", ("state_transitions", "tx_position", "delta_sign_agreement")),
        ("Tx-position update", "Delta RMSE", ("state_transitions", "tx_position", "delta_rmse")),
        ("Receiver-height update", "Delta correlation", ("state_transitions", "rx_height", "delta_correlation")),
        ("Receiver-height update", "Sign agreement", ("state_transitions", "rx_height", "delta_sign_agreement")),
        ("Receiver-height update", "Delta RMSE", ("state_transitions", "rx_height", "delta_rmse")),
        ("Site planning", "Top-1 accuracy", ("planning", "top1_accuracy")),
        ("Site planning", "Rank correlation", ("planning", "mean_candidate_rank_correlation")),
        ("Site planning", "Normalized regret", ("planning", "mean_regret_normalized_255")),
    ]
    rows = []
    for capability, metric, path in specs:
        current = values(path)
        rows.append(
            {
                "capability": capability,
                "metric": metric,
                "mean": mean(current),
                "seed_sample_std": stdev(current),
                "seeds": "42,43,44",
            }
        )
    write_csv(DATA_DIR / "table_twin_validation.csv", rows)
    return rows


def main() -> None:
    for directory in (FIG_DIR, TABLE_DIR, DATA_DIR, MANUSCRIPT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    suite = load_json(ROOT / "outputs" / "icc_suite_summary" / "summary.json")
    height = load_json(ROOT / "outputs" / "unseen_height_generalization" / "summary.json")
    uncertainty = load_json(ROOT / "outputs" / "ensemble_uncertainty" / "summary.json")
    generate_main_figure(suite["aggregate"])
    generate_height_figure(height)
    generate_uncertainty_figure(uncertainty)
    aggregate_twin_validation()
    print(f"Generated paper artifacts under {OUT}")


if __name__ == "__main__":
    main()
