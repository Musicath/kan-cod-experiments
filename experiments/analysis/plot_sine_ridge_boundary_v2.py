#!/usr/bin/env python3
"""Build the revised sine-ridge figures reported in the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "experiments" / "data" / "raw" / "sine_ridge_v2"
PROCESSED = ROOT / "experiments" / "data" / "processed"
PREVIEW = ROOT / "experiments" / "figures"
PAPER = ROOT / "COD_of_KAN" / "figures"

DIMENSIONS = (4, 16)
RIDGE_COUNTS = (4, 16, 64)
BUDGETS = (250, 500, 1_000, 2_000, 5_000, 10_000, 30_000, 100_000)
CELL = ["dim", "fourier_rank", "interaction_support"]
COLORS = {
    1: "#0072B2",
    2: "#009E73",
    4: "#E69F00",
    8: "#CC79A7",
    16: "#D55E00",
}


def load_candidates(mlp_activations: tuple[str, ...] | None) -> pd.DataFrame:
    frames = [pd.read_csv(RAW / f"sine_seed{seed}_d4_d16.csv") for seed in (0, 1, 2)]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[
        frame.status.eq("ok")
        & frame.dim.isin(DIMENSIONS)
        & frame.fourier_rank.isin(RIDGE_COUNTS)
        & frame.params.le(max(BUDGETS))
    ].copy()
    if mlp_activations is None:
        return frame[frame.model.isin(("KAN", "MLP"))].copy()
    return frame[
        frame.model.eq("KAN")
        | (frame.model.eq("MLP") & frame.activation.isin(mlp_activations))
    ].copy()


def best_achieved_envelope(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for budget in BUDGETS:
        eligible = frame[frame.params.le(budget)]
        for model in ("KAN", "MLP"):
            family = eligible[eligible.model.eq(model)]
            if family.empty:
                continue
            selected = family.loc[family.groupby(CELL).train_probe_mse.idxmin()]
            for _, row in selected.iterrows():
                records.append(
                    {
                        "dim": int(row.dim),
                        "ridge_count": int(row.fourier_rank),
                        "support": int(row.interaction_support),
                        "budget": int(budget),
                        "model": model,
                        "selected_params": int(row.params),
                        "train_probe_mse": float(row.train_probe_mse),
                        "test_mse": float(row.test_mse),
                        "seed": int(row.seed),
                        "depth": int(row.hidden_depth),
                        "width": int(row.width),
                        "grid": int(row.grid) if model == "KAN" else -1,
                        "activation": "" if model == "KAN" else str(row.activation),
                        "lr": float(row.lr),
                    }
                )
    return pd.DataFrame(records)


def ratios(envelope: pd.DataFrame) -> pd.DataFrame:
    index = ["dim", "ridge_count", "support", "budget"]
    train = envelope.pivot(index=index, columns="model", values="train_probe_mse").dropna()
    test = envelope.pivot(index=index, columns="model", values="test_mse").dropna()
    result = pd.DataFrame(index=train.index)
    result["train_ratio"] = train.MLP / train.KAN
    result["test_ratio"] = test.MLP / test.KAN
    return result.reset_index()


def save(figure: plt.Figure, stem: str) -> None:
    for directory in (PREVIEW, PAPER):
        directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(PREVIEW / f"{stem}.png", dpi=300, bbox_inches="tight")


def draw_ratio_grid(
    data: pd.DataFrame,
    field: str,
    stem: str,
    comparison: str,
    subtitle: str,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 5.00), sharex=True, sharey=True)
    for row, dimension in enumerate(DIMENSIONS):
        for col, ridge_count in enumerate(RIDGE_COUNTS):
            axis = axes[row, col]
            panel = data[
                data.dim.eq(dimension) & data.ridge_count.eq(ridge_count)
            ]
            for support in sorted(panel.support.unique()):
                curve = panel[panel.support.eq(support)].sort_values("budget")
                axis.plot(
                    curve.budget,
                    curve[field],
                    color=COLORS[int(support)],
                    linewidth=1.65,
                    marker="o",
                    markersize=3.4,
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(220, 120_000)
            axis.set_ylim(1e-4, 1e6)
            axis.axhspan(1, 1e6, color="#CCFBF1", alpha=0.30, zorder=0)
            axis.axhspan(1e-4, 1, color="#FFEDD5", alpha=0.32, zorder=0)
            axis.axhline(1, color="#111827", linewidth=1.05, linestyle="--")
            axis.grid(True, which="major", color="#CBD5E1", linewidth=0.48, alpha=0.75)
            axis.grid(True, which="minor", color="#E2E8F0", linewidth=0.28, alpha=0.45)
            axis.set_title(rf"$d={dimension},\ R={ridge_count}$", fontsize=9.5)
            if row == len(DIMENSIONS) - 1:
                axis.set_xlabel(r"parameter budget $P$")
            if row == 0:
                axis.text(
                    0.97,
                    0.95,
                    "KAN lower error",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    color="#0F766E",
                    fontsize=7.4,
                    fontweight="bold",
                )
            if row == len(DIMENSIONS) - 1:
                axis.text(
                    0.97,
                    0.05,
                    "MLP lower error",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    color="#B45309",
                    fontsize=7.4,
                    fontweight="bold",
                )

    supports = sorted(data.support.unique())
    handles = [
        Line2D([0], [0], color=COLORS[int(s)], lw=2.1, marker="o", markersize=3.5,
               label=rf"$s={int(s)}$")
        for s in supports
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(handles),
        frameon=False,
    )
    figure.suptitle("Sine-ridge fitting boundary", fontsize=13.5, y=0.992)
    figure.text(0.5, 0.957, subtitle, ha="center", va="top", fontsize=8.2)
    figure.supylabel(rf"test-MSE ratio: {comparison} / KAN", x=0.012, fontsize=9.2)
    figure.text(
        0.5,
        0.012,
        "Models are selected by minimum train-probe MSE within each budget; "
        "the independent test MSE of the selected run is shown.",
        ha="center",
        fontsize=7.4,
    )
    figure.tight_layout(rect=(0.035, 0.055, 1, 0.925), h_pad=0.95, w_pad=0.55)
    save(figure, stem)
    plt.close(figure)


def summarize(all_activation_ratios: pd.DataFrame) -> None:
    endpoint_rows = []
    for (dimension, ridge_count, budget), group in all_activation_ratios.groupby(
        ["dim", "ridge_count", "budget"]
    ):
        sparse = group[group.support.eq(1)]
        dense = group[group.support.eq(dimension)]
        if sparse.empty or dense.empty:
            continue
        endpoint_rows.append(
            {
                "dim": dimension,
                "ridge_count": ridge_count,
                "budget": budget,
                "sparse_test_ratio": float(sparse.test_ratio.iloc[0]),
                "dense_test_ratio": float(dense.test_ratio.iloc[0]),
                "log10_density_shift": float(
                    np.log10(dense.test_ratio.iloc[0])
                    - np.log10(sparse.test_ratio.iloc[0])
                ),
            }
        )
    endpoints = pd.DataFrame(endpoint_rows)
    endpoints.to_csv(PROCESSED / "sine_ridge_v2_density_shift.csv", index=False)

    at_10k = all_activation_ratios[all_activation_ratios.budget.eq(10_000)].copy()
    at_10k.to_csv(PROCESSED / "sine_ridge_v2_ratios_at_10k.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "comparison_count": len(endpoints),
                "shifts_toward_mlp": int(endpoints.log10_density_shift.lt(0).sum()),
                "median_log10_density_shift": float(endpoints.log10_density_shift.median()),
                "kan_wins_at_10k": int(at_10k.test_ratio.gt(1).sum()),
                "mlp_wins_at_10k": int(at_10k.test_ratio.lt(1).sum()),
                "sparse_s1_kan_wins_at_10k": int(
                    at_10k[at_10k.support.eq(1)].test_ratio.gt(1).sum()
                ),
                "sparse_s1_cells_at_10k": int(at_10k.support.eq(1).sum()),
            }
        ]
    )
    summary.to_csv(PROCESSED / "sine_ridge_v2_summary.csv", index=False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "legend.fontsize": 8.0,
        }
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)

    comparisons = {
        "all": (None, "best MLP", "Best of ReLU, tanh, SiLU, and sine MLPs"),
        "non_sine": (("relu", "tanh", "silu"), "best non-sine MLP", "Best of ReLU, tanh, and SiLU MLPs"),
        "relu": (("relu",), "ReLU-MLP", "Fixed ReLU activation"),
    }
    ratio_tables: dict[str, pd.DataFrame] = {}
    for name, (activations, comparison, subtitle) in comparisons.items():
        envelope = best_achieved_envelope(load_candidates(activations))
        ratio = ratios(envelope)
        envelope.to_csv(PROCESSED / f"sine_ridge_v2_{name}_envelope.csv", index=False)
        ratio.to_csv(PROCESSED / f"sine_ridge_v2_{name}_ratios.csv", index=False)
        ratio_tables[name] = ratio
        draw_ratio_grid(
            ratio,
            "test_ratio",
            f"sine_ridge_v2_{name}_test_ratio",
            comparison,
            subtitle,
        )
        if name == "all":
            draw_ratio_grid(
                ratio,
                "train_ratio",
                "sine_ridge_v2_all_train_probe_ratio",
                comparison,
                subtitle + "; train-probe MSE",
            )
    summarize(ratio_tables["all"])


if __name__ == "__main__":
    main()
