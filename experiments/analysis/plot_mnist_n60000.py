#!/usr/bin/env python3
"""Build the N=60,000 spline-only KAN versus MLP MNIST figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
PAPER_FIGURE_DIRS = (ROOT / "COD_of_KAN/figures",)
ANALYSIS_FIGURES = ROOT / "experiments/figures"
RAW = ROOT / "experiments/data/raw/mnist_n60000"
NORMAL_FULL = RAW / "normal_mlp.csv"
NORMAL_SPLINE = RAW / "normal_kan.csv"
RANDOM_FULL = RAW / "random_mlp.csv"
RANDOM_SPLINE = RAW / "random_kan.csv"
NORMAL_LOW_BUDGET = RAW / "normal_mlp_low_budget.csv"
RANDOM_LOW_BUDGET = RAW / "random_mlp_low_budget.csv"

MODEL_ORDER = ["ReLU-MLP", "Spline-only KAN"]
COLORS = {
    "ReLU-MLP": "#2459A6",
    "Spline-only KAN": "#15836F",
}
MARKERS = {"ReLU-MLP": "o", "Spline-only KAN": "^"}
BUDGETS = np.asarray([5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000])


def standardize(
    frame: pd.DataFrame,
    model: str,
    metric: str,
    q25: str,
    q75: str,
    params: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model,
            "depth": frame["depth"].astype(int),
            "width": frame["width"].astype(int),
            "params": frame[params].astype(float),
            "median": frame[metric].astype(float),
            "q25": frame[q25].astype(float),
            "q75": frame[q75].astype(float),
        }
    )


def aggregate_final_selected(path: Path, model: str | None, metric: str) -> pd.DataFrame:
    """Select optimizer/rate by fixed-model train accuracy, then aggregate seeds."""
    frame = pd.read_csv(path)
    if model is not None:
        frame = frame[frame.model.eq(model)].copy()
    selected = (
        frame.sort_values(
            ["final_train_acc", "final_train_loss"],
            ascending=[False, True],
            kind="stable",
        )
        .groupby(["depth", "width", "seed"], as_index=False)
        .first()
    )
    rows = []
    for (depth, width, params), group in selected.groupby(
        ["depth", "width", "params"], sort=True
    ):
        values = group[metric].to_numpy(float)
        rows.append(
            {
                "depth": int(depth),
                "width": int(width),
                "params": int(params),
                "completed_seeds": len(values),
                "metric_median": float(np.median(values)),
                "metric_q25": float(np.quantile(values, 0.25)),
                "metric_q75": float(np.quantile(values, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def load_normal(metric: str = "final_test_acc") -> pd.DataFrame:
    full = aggregate_final_selected(NORMAL_FULL, "MLP", metric)
    low_budget = aggregate_final_selected(
        NORMAL_LOW_BUDGET, "MLP", metric
    )
    full = pd.concat([full, low_budget], ignore_index=True)
    spline = aggregate_final_selected(NORMAL_SPLINE, None, metric)
    mlp = standardize(
        full[full.completed_seeds.eq(3)],
        "ReLU-MLP",
        "metric_median",
        "metric_q25",
        "metric_q75",
        "params",
    )
    spline_only = standardize(
        spline[spline.completed_seeds.eq(3)],
        "Spline-only KAN",
        "metric_median",
        "metric_q25",
        "metric_q75",
        "params",
    )
    return pd.concat([mlp, spline_only], ignore_index=True)


def load_random() -> pd.DataFrame:
    full = aggregate_final_selected(RANDOM_FULL, "MLP", "final_train_acc")
    low_budget = aggregate_final_selected(
        RANDOM_LOW_BUDGET, "MLP", "final_train_acc"
    )
    full = pd.concat([full, low_budget], ignore_index=True)
    spline = aggregate_final_selected(RANDOM_SPLINE, None, "final_train_acc")
    mlp = standardize(
        full[full.completed_seeds.eq(3)],
        "ReLU-MLP",
        "metric_median",
        "metric_q25",
        "metric_q75",
        "params",
    )
    spline_only = standardize(
        spline[spline.completed_seeds.eq(3)],
        "Spline-only KAN",
        "metric_median",
        "metric_q25",
        "metric_q75",
        "params",
    )
    return pd.concat([mlp, spline_only], ignore_index=True)


def budget_envelope(frame: pd.DataFrame) -> pd.DataFrame:
    points = frame.sort_values("params").copy()
    points["envelope"] = points["median"].cummax()
    return points


def budget_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = np.full((len(MODEL_ORDER), len(BUDGETS)), np.nan)
    for row, model in enumerate(MODEL_ORDER):
        subset = frame[frame.model.eq(model)]
        for col, budget in enumerate(BUDGETS):
            eligible = subset[subset.params.le(budget)]
            if not eligible.empty:
                values[row, col] = eligible["median"].max()
    return values


def draw_front(axis: plt.Axes, frame: pd.DataFrame, ylabel: str) -> None:
    for model in MODEL_ORDER:
        points = frame[frame.model.eq(model)].sort_values("params")
        axis.scatter(
            points["params"],
            points["median"],
            s=14,
            alpha=0.13,
            color=COLORS[model],
            marker=MARKERS[model],
            linewidths=0,
        )
        front = budget_envelope(points)
        axis.plot(
            front["params"],
            front["envelope"],
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=3.7,
            linewidth=1.55,
            label=model,
        )
    axis.set_xscale("log")
    axis.set_xlabel("trainable parameter budget")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.20, which="both")


def save(fig: plt.Figure, stem: str) -> None:
    for directory in PAPER_FIGURE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    ANALYSIS_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(ANALYSIS_FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(
        ANALYSIS_FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight"
    )


def draw_budget_heatmap(
    axis: plt.Axes, frame: pd.DataFrame, title: str
) -> plt.AxesImage:
    matrix = budget_matrix(frame)
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, cmap="YlGnBu", vmin=0.1, vmax=1.0, aspect="auto")
    axis.set_xticks(
        range(len(BUDGETS)),
        ["5k", "10k", "20k", "50k", "100k", "200k", "500k"],
    )
    axis.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    axis.set_xlabel("trainable parameter budget")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = "--" if np.isnan(value) else f"{100 * value:.1f}"
            color = "white" if not np.isnan(value) and value >= 0.62 else "#172033"
            axis.text(
                col, row, text, ha="center", va="center", fontsize=7.4, color=color
            )
    return image


def draw_budget_curves(
    axis: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    ylim: tuple[float, float],
) -> None:
    matrix = budget_matrix(frame)
    displayed_budgets = BUDGETS[[0, 2, 4, 6]]
    tick_labels = ["5k", "20k", "100k", "500k"]
    for row, model in enumerate(MODEL_ORDER):
        valid = np.isfinite(matrix[row])
        axis.plot(
            BUDGETS[valid],
            matrix[row, valid],
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=4.2,
            linewidth=1.8,
            label=model,
        )
    axis.set_xscale("log")
    axis.set_xlim(4_500, 550_000)
    axis.set_xticks(displayed_budgets, tick_labels)
    axis.set_ylim(*ylim)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("trainable parameter budget")
    axis.set_title(title)
    axis.grid(axis="both", which="major", alpha=0.22, linewidth=0.6)


def plot_main(
    normal_test: pd.DataFrame,
    random_train: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.35))
    draw_budget_curves(
        axes[0], normal_test, "Normal labels", (0.50, 1.01)
    )
    draw_budget_curves(
        axes[1], random_train, "Random labels", (0.08, 1.02)
    )
    axes[0].set_ylabel("best median test accuracy")
    axes[1].set_ylabel("best median training accuracy")
    for axis in axes:
        axis.title.set_fontsize(10.5)
        axis.xaxis.label.set_fontsize(9.5)
        axis.yaxis.label.set_fontsize(9.5)
        axis.tick_params(axis="both", labelsize=8.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        fontsize=8.7,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=1.1)
    save(fig, "mnist_normal_random_main")
    plt.close(fig)


def plot_random_main(random_train: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(4.25, 2.75))
    draw_budget_curves(
        axis,
        random_train,
        "MNIST with independent random labels",
        (0.08, 1.02),
    )
    axis.set_ylabel("best median training accuracy")
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, labels, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "mnist_random_labels_main")
    plt.close(fig)


def plot_appendix_frontiers(normal: pd.DataFrame, random: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    draw_front(axes[0], normal, "test accuracy")
    draw_front(axes[1], random, "final training accuracy")
    axes[0].set_title("Normal labels")
    axes[1].set_title("Random labels")
    axes[0].set_ylim(0.35, 1.02)
    axes[1].set_ylim(0.08, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.0)
    save(fig, "mnist_normal_random_frontiers_appendix")
    plt.close(fig)


def plot_architecture_heatmaps(normal: pd.DataFrame, random: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.7))
    image = None
    for row, (condition, frame) in enumerate((("Normal: test", normal), ("Random: final train", random))):
        for col, model in enumerate(MODEL_ORDER):
            axis = axes[row, col]
            subset = frame[frame.model.eq(model)]
            pivot = subset.pivot(index="depth", columns="width", values="median").sort_index()
            image = axis.imshow(pivot.to_numpy(), cmap="YlGnBu", vmin=0.1, vmax=1.0, aspect="auto")
            axis.set_xticks(range(len(pivot.columns)), [str(v) for v in pivot.columns])
            axis.set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
            axis.set_title(model if row == 0 else "", fontsize=9)
            axis.set_xlabel("width", fontsize=8)
            if col == 0:
                axis.set_ylabel(f"{condition}\ndepth", fontsize=8)
            for y in range(pivot.shape[0]):
                for x in range(pivot.shape[1]):
                    value = pivot.iloc[y, x]
                    color = "white" if value >= 0.62 else "#172033"
                    axis.text(x, y, f"{100 * value:.0f}", ha="center", va="center", fontsize=6.2, color=color)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.09, top=0.89, wspace=0.28, hspace=0.34)
    color_axis = fig.add_axes([0.91, 0.17, 0.018, 0.64])
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("median accuracy", fontsize=8)
    fig.suptitle("MNIST architecture scan ($N=60{,}000$)", fontsize=11)
    save(fig, "mnist_normal_random_architecture_heatmaps_appendix")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.5,
        }
    )
    normal = load_normal()
    random = load_random()
    plot_main(normal, random)
    plot_random_main(random)
    plot_architecture_heatmaps(normal, random)
    for condition, frame in (("normal", normal), ("random", random)):
        budget_table = pd.DataFrame(
            budget_matrix(frame), index=MODEL_ORDER, columns=BUDGETS
        )
        budget_table.to_csv(
            ANALYSIS_FIGURES / f"mnist_{condition}_budget_heatmap_values_N60000.csv"
        )


if __name__ == "__main__":
    main()
