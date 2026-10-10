#!/usr/bin/env python3
"""Plot absolute train-probe and test MSE for the reported sine-ridge envelope."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


COLORS = {1: "#0072B2", 2: "#009E73", 4: "#E69F00", 8: "#CC79A7", 16: "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.envelope)
    frame = frame[
        frame.dim.isin((4, 16))
        & frame.ridge_count.isin((4, 16, 64))
        & frame.budget.le(100_000)
    ].copy()

    figure, axes = plt.subplots(3, 4, figsize=(13.5, 8.2), sharex=True, sharey=True)
    columns = ((4, "KAN"), (4, "MLP"), (16, "KAN"), (16, "MLP"))
    for row, ridge_count in enumerate((4, 16, 64)):
        for col, (dimension, model) in enumerate(columns):
            axis = axes[row, col]
            panel = frame[
                frame.ridge_count.eq(ridge_count)
                & frame.dim.eq(dimension)
                & frame.model.eq(model)
            ]
            for support in sorted(panel.support.unique()):
                curve = panel[panel.support.eq(support)].sort_values("budget")
                color = COLORS[int(support)]
                axis.plot(curve.budget, curve.test_mse, color=color, marker="o",
                          markersize=2.8, linewidth=1.45)
                axis.plot(curve.budget, curve.train_probe_mse, color=color,
                          linewidth=1.05, linestyle="--", alpha=0.82)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(220, 120_000)
            axis.set_ylim(1e-13, 2)
            axis.axhline(1, color="#6B7280", linestyle=":", linewidth=1.0)
            axis.grid(True, which="major", color="#CBD5E1", linewidth=0.48, alpha=0.75)
            axis.grid(True, which="minor", color="#E2E8F0", linewidth=0.25, alpha=0.40)
            axis.set_title(rf"$d={dimension}$, {model}", fontsize=10)
            if col == 0:
                axis.set_ylabel(rf"$R={ridge_count}$" + "\nMSE")
            if row == 2:
                axis.set_xlabel(r"parameter budget $P$")

    support_handles = [
        Line2D([0], [0], color=COLORS[s], marker="o", linewidth=1.8,
               markersize=3.2, label=rf"$s={s}$")
        for s in (1, 2, 4, 8, 16)
    ]
    metric_handles = [
        Line2D([0], [0], color="#111827", linewidth=1.7, label="test MSE"),
        Line2D([0], [0], color="#111827", linewidth=1.2, linestyle="--",
               label="training-probe MSE"),
    ]
    figure.legend(handles=support_handles + metric_handles, loc="upper center",
                  bbox_to_anchor=(0.5, 0.945), ncol=7, frameon=False)
    figure.suptitle("Absolute errors of the selected sine-ridge models", fontsize=15, y=0.99)
    figure.text(
        0.5, 0.012,
        "Targets are standardized to unit variance; the dotted horizontal line at MSE = 1 "
        "is the zero-predictor baseline. Solid: independent test; dashed: training probe.",
        ha="center", fontsize=8.5,
    )
    figure.tight_layout(rect=(0.02, 0.045, 1, 0.91), h_pad=1.0, w_pad=0.65)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
