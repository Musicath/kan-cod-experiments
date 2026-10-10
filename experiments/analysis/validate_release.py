#!/usr/bin/env python3
"""Check that the current paper data and generated outputs are consistent."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "experiments/data/raw"
PROCESSED = ROOT / "experiments/data/processed"


def main() -> None:
    for seed in (0, 1, 2):
        path = RAW / f"sine_ridge_v2/sine_seed{seed}_d4_d16.csv"
        frame = pd.read_csv(path)
        assert len(frame) == 7_776, (path, len(frame))
        assert frame.status.eq("ok").all(), path
        assert set(frame.dim) == {4, 16}, path
        assert set(frame.fourier_rank) == {4, 16, 64}, path
        assert set(frame.seed) == {seed}, path

    for path in sorted(RAW.glob("mnist_*/*.csv")):
        frame = pd.read_csv(path)
        expected_model = "KAN" if path.stem.endswith("kan") else "MLP"
        assert set(frame.model) == {expected_model}, (path, set(frame.model))
        text = path.read_text(encoding="utf-8")
        assert not re.search(
            r"(?:[A-Za-z]:\\|/root/|/home/|seetacloud|autodl)", text
        ), path

    ratios = pd.read_csv(PROCESSED / "sine_ridge_v2_all_ratios.csv")
    summary = pd.read_csv(PROCESSED / "sine_ridge_v2_summary.csv").iloc[0]
    assert np.isfinite(ratios[["train_ratio", "test_ratio"]]).all().all()
    assert int(summary.comparison_count) == 48
    assert int(summary.shifts_toward_mlp) == 47
    assert int(summary.sparse_s1_kan_wins_at_10k) == 6
    assert int(summary.sparse_s1_cells_at_10k) == 6

    current = pd.read_csv(PROCESSED / "ridge_profile_v2_24gpu_20260920/all_runs.csv.gz")
    assert len(current) == 174_960
    assert current.status.eq("ok").all()
    assert set(current.profile) == {"square", "sine", "silu", "tanh"}
    assert set(current.dim) == {4, 16, 64}
    square = pd.read_csv(
        PROCESSED / "ridge_profile_v2_24gpu_20260920/square_paper_style_ratios.csv"
    )
    at_10k = square[square.budget.eq(10_000)]
    assert len(at_10k) == 45
    assert int(at_10k.test_mlp_over_kan.gt(1).sum()) == 38

    for path in (
        ROOT / "COD_of_KAN/figures/four_target_constructive_scaling.pdf",
        ROOT / "COD_of_KAN/figures/square_ridge_24gpu_d16_row.pdf",
        *(ROOT / f"COD_of_KAN/figures/{profile}_ridge_24gpu_nine_panel.pdf"
          for profile in ("square", "sine", "silu", "tanh")),
        ROOT / "COD_of_KAN/figures/sine_ridge_v2_all_test_ratio.pdf",
        ROOT / "COD_of_KAN/figures/sine_ridge_v2_non_sine_test_ratio.pdf",
        ROOT / "COD_of_KAN/figures/sine_ridge_v2_relu_test_ratio.pdf",
        ROOT / "COD_of_KAN/figures/sine_ridge_v2_all_train_probe_ratio.pdf",
        ROOT / "COD_of_KAN/figures/sine_ridge_v2_absolute_mse_appendix.pdf",
        ROOT / "COD_of_KAN/figures/mnist_normal_random_main.pdf",
    ):
        assert path.exists() and path.stat().st_size > 10_000, path

    print("release validation passed")


if __name__ == "__main__":
    main()
