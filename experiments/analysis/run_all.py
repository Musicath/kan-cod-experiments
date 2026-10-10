#!/usr/bin/env python3
"""Regenerate the paper's original experiment figures from released tables."""

from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = (
    "plot_sine_ridge_boundary_v2.py",
    "plot_mnist_n60000.py",
    "plot_mnist_n6000.py",
)


def main() -> None:
    root = HERE.parents[1]
    data = root / "experiments/data/processed/ridge_profile_v2_24gpu_20260920"
    paper = root / "COD_of_KAN/figures"
    paper.mkdir(parents=True, exist_ok=True)
    for name in SCRIPTS:
        print(f"[analysis] {name}", flush=True)
        subprocess.run([sys.executable, str(HERE / name)], check=True)
    constructive = root / "experiments/diagnostics/constructive_example2"
    subprocess.run([
        sys.executable, str(constructive / "plot_four_function_main.py"),
        "--results-dir", str(constructive / "results"),
        "--output-stem", str(paper / "four_target_constructive_scaling"),
    ], check=True)
    subprocess.run([
        sys.executable, str(HERE / "plot_sine_absolute_mse.py"),
        "--envelope", str(root / "experiments/data/processed/sine_ridge_v2_all_envelope.csv"),
        "--output", str(paper / "sine_ridge_v2_absolute_mse_appendix.pdf"),
    ], check=True)
    for profile in ("square", "sine", "silu", "tanh"):
        subprocess.run([
            sys.executable, str(HERE / "plot_square_ridge_24gpu_like_paper.py"),
            str(data / "all_runs.csv.gz"), str(data), "--profile", profile, "--paper",
        ], check=True)
        shutil.copyfile(
            data / f"{profile}_ridge_24gpu_nine_panel_paper.pdf",
            paper / f"{profile}_ridge_24gpu_nine_panel.pdf",
        )
    subprocess.run([
        sys.executable, str(HERE / "plot_square_ridge_24gpu_like_paper.py"),
        str(data / "all_runs.csv.gz"), str(data), "--profile", "square", "--paper",
        "--dimension", "16",
    ], check=True)
    shutil.copyfile(
        data / "square_ridge_24gpu_d16_row_paper.pdf",
        paper / "square_ridge_24gpu_d16_row.pdf",
    )


if __name__ == "__main__":
    main()
