#!/usr/bin/env python3
"""Compact main-text figure for four constructive Figure 3.1 targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd


TARGETS = (
    ("bessel", "(a) Bessel"),
    ("exp_sin_y2", "(b) Exp-sine"),
    ("exp100d", "(c) 100D composition"),
    ("nested4d", "(d) 4D nested"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 4, figsize=(5.65, 2.05))
    figure.subplots_adjust(left=0.105, right=0.995, bottom=0.27,
                           top=0.77, wspace=0.36)
    for ax, (name, title) in zip(axes, TARGETS):
        data = pd.read_csv(args.results_dir / f"{name}.csv").sort_values("grid")
        if list(data.grid) != [8, 16, 32, 64, 128]:
            raise ValueError(f"Incomplete grid sweep for {name}")
        ax.loglog(data.kan_coefficients, data.kan_test_rmse,
                  "o-", color="#0072B2", lw=1.7, ms=4,
                  label="Constructed spline-KAN")
        ax.loglog(data.mlp_full_dense_parameters,
                  data.mlp_selected_test_rmse,
                  "*-", color="#7B3294", lw=1.7, ms=6,
                  label=r"Transferred ReLU$^3$ MLP")
        ax.set_title(title, fontsize=8.2, pad=5)
        ax.grid(which="both", color="#BBBBBB", alpha=0.22, lw=0.4)
        ax.tick_params(labelsize=7.5, length=2, pad=1.5)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=3))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        x = np.r_[data.kan_coefficients, data.mlp_full_dense_parameters]
        y = np.r_[data.kan_test_rmse, data.mlp_selected_test_rmse]
        ax.set_xlim(10 ** (np.log10(x.min()) - 0.13),
                    10 ** (np.log10(x.max()) + 0.13))
        ax.set_ylim(10 ** (np.log10(y.min()) - 0.2),
                    10 ** (np.log10(y.max()) + 0.2))
    figure.supxlabel("Parameter count", fontsize=8.5, y=0.055)
    figure.supylabel("Test RMSE", fontsize=8.5, x=0.005)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2,
                  frameon=False, fontsize=8.5, bbox_to_anchor=(0.54, 0.982))

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".pdf"))
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=300)
    plt.close(figure)
    print(f"Wrote {args.output_stem}.pdf and .png")


if __name__ == "__main__":
    main()
