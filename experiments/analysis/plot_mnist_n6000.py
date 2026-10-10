#!/usr/bin/env python3
"""Plot the complete N=6,000 MLP versus spline-only KAN MNIST scans."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / "experiments/data/raw/mnist_n6000"
OUT = ROOT / "experiments/figures"
PAPER_OUT = ROOT / "COD_of_KAN/figures"

NORMAL_MLP_MAIN = RAW / "normal_mlp.csv"
RANDOM_MLP_MAIN = RAW / "random_mlp.csv"
NORMAL_MLP_SMALL = RAW / "normal_mlp_low_budget.csv"
RANDOM_MLP_SMALL = RAW / "random_mlp_low_budget.csv"
NORMAL_SPLINE = RAW / "normal_kan.csv"
RANDOM_SPLINE = RAW / "random_kan.csv"

COLORS = {"ReLU-MLP": "#2459A6", "Spline-only KAN": "#15836F"}
MARKERS = {"ReLU-MLP": "o", "Spline-only KAN": "^"}


def frame(rows, model, metric):
    return pd.DataFrame(
        {
            "model": model,
            "depth": rows["depth"].astype(int),
            "width": rows["width"].astype(int),
            "params": rows["params"].astype(float),
            "accuracy": rows[metric].astype(float),
        }
    )


def aggregate_final_selected(path, metric, model=None):
    rows = pd.read_csv(path)
    if model is not None:
        rows = rows[rows.model.eq(model)].copy()
    selected = (
        rows.sort_values(
            ["final_train_acc", "final_train_loss"]
            if "final_train_loss" in rows.columns
            else ["final_train_acc"],
            ascending=[False, True]
            if "final_train_loss" in rows.columns
            else [False],
            kind="stable",
        )
        .groupby(["depth", "width", "seed"], as_index=False)
        .first()
    )
    grouped = []
    for (depth, width, params), group in selected.groupby(
        ["depth", "width", "params"], sort=True
    ):
        values = group[metric].to_numpy(float)
        grouped.append(
            {
                "depth": int(depth),
                "width": int(width),
                "params": int(params),
                "completed_seeds": len(values),
                "metric_median": float(np.median(values)),
            }
        )
    return pd.DataFrame(grouped)


def load_normal():
    main = aggregate_final_selected(NORMAL_MLP_MAIN, "final_test_acc", "MLP")
    small = aggregate_final_selected(NORMAL_MLP_SMALL, "final_test_acc", "MLP")
    spline = aggregate_final_selected(
        NORMAL_SPLINE, "final_test_acc"
    )
    return pd.concat(
        [
            frame(main[main.completed_seeds.eq(3)], "ReLU-MLP", "metric_median"),
            frame(small[small.completed_seeds.eq(3)], "ReLU-MLP", "metric_median"),
            frame(spline[spline.completed_seeds.eq(3)], "Spline-only KAN", "metric_median"),
        ],
        ignore_index=True,
    )


def load_random():
    main = aggregate_final_selected(RANDOM_MLP_MAIN, "final_train_acc", "MLP")
    small = aggregate_final_selected(RANDOM_MLP_SMALL, "final_train_acc", "MLP")
    spline = aggregate_final_selected(
        RANDOM_SPLINE, "final_train_acc"
    )
    return pd.concat(
        [
            frame(main[main.completed_seeds.eq(3)], "ReLU-MLP", "metric_median"),
            frame(small[small.completed_seeds.eq(3)], "ReLU-MLP", "metric_median"),
            frame(spline[spline.completed_seeds.eq(3)], "Spline-only KAN", "metric_median"),
        ],
        ignore_index=True,
    )


def draw(ax, data, title, ylabel, ylim):
    # Show the parameter range covered by both model families.
    xmax = 1.15 * data.loc[data.model.eq("Spline-only KAN"), "params"].max()
    xmin = 0.85 * data["params"].min()
    for model in ("ReLU-MLP", "Spline-only KAN"):
        points = data[data.model.eq(model) & data.params.le(xmax)].sort_values("params").copy()
        ax.scatter(
            points.params,
            points.accuracy,
            s=18,
            alpha=0.16,
            color=COLORS[model],
            marker=MARKERS[model],
            linewidths=0,
        )
        points["envelope"] = points.accuracy.cummax()
        ax.plot(
            points.params,
            points.envelope,
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=4,
            linewidth=1.65,
            label=model,
        )
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_xlabel("trainable parameter count")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.20, which="both")


def main():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    draw(axes[0], load_normal(), "Normal labels", "test accuracy", (0.35, 1.02))
    draw(axes[1], load_random(), "Random labels", "final training accuracy", (0.08, 1.02))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center")
    fig.suptitle("MNIST ($N=6{,}000$; spline-only KAN)", y=0.91, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.80), w_pad=1.15)
    for directory in (OUT, PAPER_OUT):
        directory.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "png"):
            fig.savefig(directory / f"mnist_normal_random_N6000_frontiers.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
