"""Draw a completed ridge-profile sweep in the main-text boundary style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


BUDGETS = (250, 500, 1000, 2000, 5000, 10000, 30000, 100000)
DIMENSIONS = (4, 16, 64)
RANKS = (4, 16, 64)
SUPPORT_COLORS = {
    1: "#0072B2", 2: "#009E73", 4: "#E69F00", 8: "#CC79A7",
    16: "#D55E00", 32: "#6A3D9A", 64: "#5B6573",
}


def build_envelope(frame: pd.DataFrame) -> pd.DataFrame:
    cells = ["dim", "fourier_rank", "interaction_support"]
    selected = []
    for budget in BUDGETS:
        eligible = frame[frame.params.le(budget)]
        for model in ("KAN", "MLP"):
            family = eligible[eligible.model.eq(model)]
            if family.empty:
                continue
            winners = family.loc[family.groupby(cells).train_probe_mse.idxmin()].copy()
            winners["budget"] = budget
            selected.append(winners)
    return pd.concat(selected, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("all_runs", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--profile", choices=("square", "sine", "silu", "tanh"), default="square")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--dimension", type=int, choices=DIMENSIONS,
                        help="Draw one three-panel dimension row instead of the full nine-panel grid")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.all_runs, low_memory=False)
    frame = frame[frame.profile.eq(args.profile) & frame.status.eq("ok")].copy()
    envelope = build_envelope(frame)
    envelope.to_csv(args.output_dir / f"{args.profile}_paper_style_envelope.csv", index=False)
    columns = ["dim", "fourier_rank", "interaction_support", "budget"]
    pivot = envelope.pivot(index=columns, columns="model", values="test_mse")
    ratios = (pivot.MLP / pivot.KAN).rename("test_mlp_over_kan").reset_index()
    ratios.to_csv(args.output_dir / f"{args.profile}_paper_style_ratios.csv", index=False)

    paper = args.paper
    dimensions = DIMENSIONS if args.dimension is None else (args.dimension,)
    display = ratios[ratios.dim.isin(dimensions)]
    row_only = args.dimension is not None
    if row_only:
        size = (5.5, 2.2) if paper else (11.2, 3.9)
    else:
        size = (7.15, 6.85) if paper else (11.2, 8.7)
    figure, axes = plt.subplots(len(dimensions), 3, figsize=size,
                               sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    positive = display.test_mlp_over_kan[display.test_mlp_over_kan.gt(0)]
    lower = (10 ** np.floor(np.log10(positive.min())) if row_only
             else min(1e-4, 10 ** np.floor(np.log10(positive.min()))))
    upper = max(1e2, 10 ** np.ceil(np.log10(positive.max())))
    for row, dimension in enumerate(dimensions):
        for col, rank in enumerate(RANKS):
            axis = axes[row, col]
            panel = ratios[ratios.dim.eq(dimension) & ratios.fourier_rank.eq(rank)]
            for support in sorted(panel.interaction_support.unique()):
                line = panel[panel.interaction_support.eq(support)].sort_values("budget")
                axis.plot(line.budget, line.test_mlp_over_kan,
                          color=SUPPORT_COLORS[int(support)], marker="o",
                          markersize=2.6 if paper else 3.4,
                          linewidth=1.15 if paper else 1.55)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(220, 120000)
            axis.set_ylim(lower, upper)
            axis.axhspan(1, upper, color="#CCFBF1", alpha=0.30, zorder=0)
            axis.axhspan(lower, 1, color="#FFEDD5", alpha=0.32, zorder=0)
            axis.axhline(1, color="#111827", linewidth=1.0, linestyle="--")
            axis.grid(True, which="major", color="#CBD5E1", linewidth=0.45, alpha=0.72)
            axis.grid(True, which="minor", color="#E2E8F0", linewidth=0.25, alpha=0.40)
            axis.set_title(rf"$d={dimension},\ R={rank}$", fontsize=8 if paper else 10)
            if row == len(dimensions) - 1:
                axis.set_xlabel(r"parameter budget $P$", fontsize=7.2 if paper else 10)
            if col == 0:
                axis.set_ylabel("test-MSE ratio: MLP / KAN", fontsize=7.2 if paper else 10)
            axis.tick_params(labelsize=6.3 if paper else 9)
    supports = sorted(display.interaction_support.unique())
    handles = [Line2D([0], [0], color=SUPPORT_COLORS[int(s)], marker="o", lw=2,
                      markersize=3.5, label=rf"$s={s}$") for s in supports]
    figure.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, 0.865 if row_only else 0.951),
                  ncol=len(supports), frameon=False, fontsize=6.7 if paper else 9)
    title = f"{args.profile.capitalize()}-ridge fitting boundary"
    if row_only:
        title += rf" ($d={args.dimension}$)"
    figure.suptitle(title,
                    fontsize=10 if paper else 15, y=0.985 if row_only else 0.995)
    figure.text(0.5, 0.915 if row_only else 0.965,
                "KAN lower test error above 1; best-of-four MLP lower below 1",
                ha="center", va="top", fontsize=6.8 if paper else 9)
    if not row_only:
        figure.text(0.5, 0.012,
                    "Best run across seeds 0, 1, 2 selected by training-probe MSE within each budget; independent test MSE shown.",
                    ha="center", fontsize=5.8 if paper else 8)
    if row_only:
        figure.subplots_adjust(left=0.10, right=0.985, bottom=0.205,
                               top=0.68, wspace=0.18)
    else:
        figure.tight_layout(rect=(0.035, 0.045, 1, 0.925), h_pad=1.0, w_pad=0.8)
    view = f"d{args.dimension}_row" if row_only else "nine_panel"
    stem = f"{args.profile}_ridge_24gpu_{view}" + ("_paper" if paper else "")
    figure.savefig(args.output_dir / f"{stem}.png", dpi=220)
    figure.savefig(args.output_dir / f"{stem}.pdf")
    plt.close(figure)
    print(f"{args.profile} rows={len(frame)}, plotted ratio cells={len(display)}")
    print(f"plotted ratio range={np.nanmin(display.test_mlp_over_kan):.3g} to {np.nanmax(display.test_mlp_over_kan):.3g}")


if __name__ == "__main__":
    main()
