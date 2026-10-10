#!/usr/bin/env python3
"""Construct Figure 3.1 target 2 as a spline-only KAN and transfer it to ReLU^3.

No optimization is performed.  The three KAN edges interpolate sin(pi*x),
y**2, and exp(t) with cubic splines.  The transferred MLP evaluates exactly
the same splines through truncated-power ReLU^3 units on the test domain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, CubicSpline
from scipy.stats import qmc


HERE = Path(__file__).resolve().parent


def cubic_spline_to_relu3(
    knots: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Represent a cubic interpolant by shifted ReLU-cubed units."""
    spline = CubicSpline(knots, values)
    left = float(knots[0])
    interval_length = float(knots[-1] - knots[0])
    shift_step = max(0.25, min(1.0, interval_length / 4.0))
    polynomial_shifts = left - shift_step * np.arange(1.0, 5.0)
    a, b, c, d = spline.c[:, 0]
    polynomial = np.array(
        [a, b - 3.0 * left * a,
         c - 2.0 * left * b + 3.0 * left**2 * a,
         d - left * c + left**2 * b - left**3 * a]
    )
    basis = np.stack(
        [np.ones(4), -3.0 * polynomial_shifts,
         3.0 * polynomial_shifts**2, -polynomial_shifts**3], axis=0
    )
    polynomial_weights = np.linalg.solve(basis, polynomial)
    interior_shifts = knots[1:-1]
    hinge_weights = spline.c[0, 1:] - spline.c[0, :-1]
    return (
        np.concatenate((polynomial_shifts, interior_shifts)),
        np.concatenate((polynomial_weights, hinge_weights)),
    )


def evaluate_relu3(
    points: np.ndarray, shifts: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    responses = np.maximum(points[:, None] - shifts[None, :], 0.0) ** 3
    return responses @ weights


def edge(function, left: float, right: float, grid: int):
    knots = np.linspace(left, right, grid + 1)
    spline = CubicSpline(knots, function(knots))
    # Re-express the interpolant in a clamped cubic B-spline basis with one
    # knot at each grid break.  This verifies the declared G+3 KAN slots.
    basis_knots = np.concatenate((np.repeat(left, 4), knots[1:-1], np.repeat(right, 4)))
    greville = np.array([np.mean(basis_knots[i + 1 : i + 4]) for i in range(grid + 3)])
    # Decimal endpoints (for example -0.1) can round one ulp outside the
    # B-spline support when averaged, which SciPy correctly rejects.
    greville = np.clip(greville, left, right)
    basis_matrix = BSpline.design_matrix(greville, basis_knots, 3).toarray()
    coefficients = np.linalg.solve(basis_matrix, spline(greville))
    bspline = BSpline(basis_knots, coefficients, 3)
    check = np.linspace(left, right, 257)
    if np.max(np.abs(bspline(check) - spline(check))) > 1e-10:
        raise AssertionError("Power-basis and B-spline representations disagree")
    shifts, weights = cubic_spline_to_relu3(knots, function(knots))
    assert len(shifts) == grid + 3
    return bspline, shifts, weights


def evaluate_models(points: np.ndarray, grid: int) -> dict[str, float | int]:
    sin_edge = edge(lambda x: np.sin(np.pi * x), -1.0, 1.0, grid)
    square_edge = edge(lambda y: y**2, -1.0, 1.0, grid)
    # The first two edges range within [-1, 2]; the margin also covers small
    # interpolation overshoots at coarse grids.
    exp_edge = edge(np.exp, -1.5, 2.5, grid)

    x, y = points[:, 0], points[:, 1]
    target = np.exp(np.sin(np.pi * x) + y**2)
    inner_kan = sin_edge[0](x) + square_edge[0](y)
    if np.min(inner_kan) < -1.5 or np.max(inner_kan) > 2.5:
        raise ValueError("Constructed inner KAN output left the outer spline domain")
    kan = exp_edge[0](inner_kan)

    # This is a two-hidden-layer ordinary ReLU^3 MLP.  Its second-layer
    # weight matrix has identical rows; factorizing that matrix during the
    # large evaluation avoids an unnecessary O(N G^2) dense multiplication.
    inner_mlp = evaluate_relu3(x, sin_edge[1], sin_edge[2])
    inner_mlp += evaluate_relu3(y, square_edge[1], square_edge[2])
    mlp = evaluate_relu3(inner_mlp, exp_edge[1], exp_edge[2])

    # Independently materialize the ordinary dense affine matrices on a small
    # subset, so the factorized evaluation is not the only transfer check.
    check_points = points[:16]
    m = grid + 3
    first_weights = np.zeros((2 * m, 2), dtype=np.float64)
    first_weights[:m, 0] = 1.0
    first_weights[m:, 1] = 1.0
    first_bias = -np.concatenate((sin_edge[1], square_edge[1]))
    first_hidden = np.maximum(check_points @ first_weights.T + first_bias, 0.0) ** 3
    shared_second_weights = np.concatenate((sin_edge[2], square_edge[2]))
    second_weights = np.tile(shared_second_weights, (m, 1))
    second_hidden = np.maximum(
        first_hidden @ second_weights.T - exp_edge[1], 0.0
    ) ** 3
    dense_mlp = second_hidden @ exp_edge[2]

    kan_rmse = float(np.sqrt(np.mean((kan - target) ** 2)))
    mlp_rmse = float(np.sqrt(np.mean((mlp - target) ** 2)))
    transfer_max = float(np.max(np.abs(mlp - kan)))
    dense_check_max = float(np.max(np.abs(dense_mlp - mlp[:16])))
    if transfer_max > 1e-8 or dense_check_max > 1e-8:
        raise AssertionError(
            f"G={grid}: transfer discrepancy {transfer_max:.3g}, "
            f"dense-matrix discrepancy {dense_check_max:.3g}"
        )

    # Each of the three spline edges has G+3 coefficient slots.  The MLP has
    # hidden widths (2m, m) and counts every ordinary dense affine entry,
    # including entries fixed to zero by this analytic construction.
    kan_coefficients = 3 * m
    mlp_full_affine = (2 + 1) * (2 * m) + (2 * m + 1) * m + (m + 1)
    return {
        "grid_intervals": grid,
        "kan_spline_coefficients": kan_coefficients,
        "mlp_full_affine_parameters": mlp_full_affine,
        "kan_test_rmse": kan_rmse,
        "mlp_test_rmse": mlp_rmse,
        "transfer_max_abs_difference": transfer_max,
        "dense_matrix_check_max_abs_difference": dense_check_max,
    }


def plot_results(results: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    grids = results["grid_intervals"].to_numpy()
    kan_error = results["kan_test_rmse"].to_numpy()
    mlp_error = results["mlp_test_rmse"].to_numpy()

    axes[0].loglog(grids, kan_error, "o-", color="#0072B2", label="Constructed spline-only KAN")
    axes[0].loglog(
        grids, mlp_error, "s", mfc="none", mec="#D55E00", ms=7,
        label=r"Exact transferred ReLU$^3$-MLP",
    )
    reference = kan_error[1] * (grids / grids[1]) ** -4
    axes[0].loglog(grids, reference, "--", color="#666666", label=r"Reference $G^{-4}$")
    axes[0].set(xlabel="Grid intervals per edge $G$", ylabel="Test RMSE", title="Error versus spline resolution")
    axes[0].legend(fontsize=8)

    axes[1].loglog(
        results["kan_spline_coefficients"], kan_error, "o-", color="#0072B2",
        label="KAN spline coefficients",
    )
    axes[1].loglog(
        results["mlp_full_affine_parameters"], mlp_error, "s-", color="#D55E00",
        label=r"Transferred ReLU$^3$-MLP dense parameters",
    )
    for _, row in results.iterrows():
        axes[1].plot(
            [row.kan_spline_coefficients, row.mlp_full_affine_parameters],
            [row.kan_test_rmse, row.mlp_test_rmse],
            color="#BBBBBB", linewidth=0.7, zorder=0,
        )
    axes[1].set(xlabel="Stored scalar parameters (declared count)", ylabel="Test RMSE", title="Same function, different parameter count")
    axes[1].legend(fontsize=8)
    figure.suptitle(r"Analytic construction for $f(x,y)=e^{\sin(\pi x)+y^2}$ on $[-1,1]^2$", fontsize=11)
    figure.savefig(output_dir / "constructive_kan_to_relu3.png", dpi=220)
    figure.savefig(output_dir / "constructive_kan_to_relu3.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sobol-power", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = 2.0 * qmc.Sobol(d=2, scramble=True, seed=20260924).random_base2(args.sobol_power) - 1.0
    results = pd.DataFrame(evaluate_models(points, grid) for grid in (4, 8, 16, 32, 64, 128, 256))
    results.to_csv(args.output_dir / "constructive_kan_to_relu3.csv", index=False)
    plot_results(results, args.output_dir)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
