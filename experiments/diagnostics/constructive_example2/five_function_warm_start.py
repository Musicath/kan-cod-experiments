#!/usr/bin/env python3
"""Target-aware spline-KAN constructions and exact ReLU^3 MLP warm starts.

These are new diagnostics for the five targets of Liu et al., Figure 3.1;
they are not reproductions of that paper's trained networks.  All evaluations
use float64, held-out Sobol points, and a full dense MLP parameter count.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import j0
from scipy.stats import qmc

from construct_and_plot import edge


FUNCTIONS = {
    "bessel": (1, -1.0, 1.0),
    "exp_sin_y2": (2, -1.0, 1.0),
    "product": (2, -1.0, 1.0),
    "exp100d": (100, 0.0, 1.0),
    "nested4d": (4, -1.0, 1.0),
}


@dataclass(frozen=True)
class EdgeSpec:
    source: int
    destination: int
    function: str
    left: float
    right: float


def function(name: str, x: np.ndarray) -> np.ndarray:
    if name == "bessel":
        return j0(20.0 * x)
    if name == "sine_pi":
        return np.sin(np.pi * x)
    if name == "square":
        return x * x
    if name == "exp":
        return np.exp(x)
    if name == "identity":
        return x
    if name == "negative_identity":
        return -x
    if name == "quarter_square":
        return 0.25 * x * x
    if name == "negative_quarter_square":
        return -0.25 * x * x
    if name == "scaled_sin_square":
        return np.sin(0.5 * np.pi * x) ** 2 / 100.0
    if name == "sine":
        return np.sin(x)
    raise ValueError(name)


def specification(name: str) -> list[tuple[int, list[EdgeSpec]]]:
    if name == "bessel":
        return [(1, [EdgeSpec(0, 0, "bessel", -1, 1)])]
    if name == "exp_sin_y2":
        return [
            (1, [EdgeSpec(0, 0, "sine_pi", -1, 1),
                 EdgeSpec(1, 0, "square", -1, 1)]),
            (1, [EdgeSpec(0, 0, "exp", -1.5, 2.5)]),
        ]
    if name == "product":
        # xy = ((x+y)^2-(x-y)^2)/4, an exact [2,2,1] KAN.
        return [
            (2, [EdgeSpec(0, 0, "identity", -1, 1),
                 EdgeSpec(1, 0, "identity", -1, 1),
                 EdgeSpec(0, 1, "identity", -1, 1),
                 EdgeSpec(1, 1, "negative_identity", -1, 1)]),
            (1, [EdgeSpec(0, 0, "quarter_square", -2.1, 2.1),
                 EdgeSpec(1, 0, "negative_quarter_square", -2.1, 2.1)]),
        ]
    if name == "exp100d":
        return [
            (1, [EdgeSpec(i, 0, "scaled_sin_square", 0, 1)
                 for i in range(100)]),
            (1, [EdgeSpec(0, 0, "exp", -0.1, 1.1)]),
        ]
    if name == "nested4d":
        # Match the *rendered figure* target, not the conflicting prose formula.
        return [
            (2, [EdgeSpec(i, i // 2, "square", -1, 1)
                 for i in range(4)]),
            (1, [EdgeSpec(i, 0, "sine", -0.1, 2.1)
                 for i in range(2)]),
            (1, [EdgeSpec(0, 0, "exp", -0.2, 2.2)]),
        ]
    raise ValueError(name)


def target(name: str, x: np.ndarray) -> np.ndarray:
    if name == "bessel":
        return j0(20 * x[:, 0])
    if name == "exp_sin_y2":
        return np.exp(np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2)
    if name == "product":
        return x[:, 0] * x[:, 1]
    if name == "exp100d":
        return np.exp(np.mean(np.sin(0.5 * np.pi * x) ** 2, axis=1))
    if name == "nested4d":
        return np.exp(np.sin(x[:, 0] ** 2 + x[:, 1] ** 2)
                      + np.sin(x[:, 2] ** 2 + x[:, 3] ** 2))
    raise ValueError(name)


def sample(name: str, power: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    dimension, left, right = FUNCTIONS[name]
    x = left + (right - left) * qmc.Sobol(
        dimension, scramble=True, seed=seed
    ).random_base2(power)
    return torch.from_numpy(x), torch.from_numpy(target(name, x))


class CubedMLP(torch.nn.Module):
    def __init__(self, widths: list[int]) -> None:
        super().__init__()
        self.hidden = torch.nn.ModuleList([
            torch.nn.Linear(a, b, dtype=torch.float64)
            for a, b in zip(widths[:-2], widths[1:-1])
        ])
        self.output = torch.nn.Linear(widths[-2], 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden:
            x = torch.relu(layer(x)) ** 3
        return self.output(x).squeeze(-1)


def construct(name: str, grid: int) -> tuple[CubedMLP, list, int]:
    spec = specification(name)
    cache: dict[tuple[str, float, float], tuple] = {}
    built = []
    for node_count, edge_specs in spec:
        this = []
        for item in edge_specs:
            key = (item.function, item.left, item.right)
            if key not in cache:
                cache[key] = edge(
                    lambda z, fn=item.function: function(fn, z),
                    item.left, item.right, grid,
                )
            this.append((item, cache[key]))
        built.append((node_count, this))

    dimension = FUNCTIONS[name][0]
    widths = [dimension] + [len(edges) * (grid + 3) for _, edges in built] + [1]
    model = CubedMLP(widths)
    m = grid + 3
    previous_coefficients = np.eye(dimension, dtype=np.float64)
    with torch.no_grad():
        for stage, (node_count, edges) in enumerate(built):
            matrix = np.zeros((len(edges) * m, previous_coefficients.shape[1]))
            bias = np.zeros(len(edges) * m)
            node_coefficients = np.zeros((node_count, len(edges) * m))
            for edge_index, (item, (_, shifts, coefficients)) in enumerate(edges):
                part = slice(edge_index * m, (edge_index + 1) * m)
                matrix[part] = previous_coefficients[item.source]
                bias[part] = -shifts
                node_coefficients[item.destination, part] += coefficients
            model.hidden[stage].weight.copy_(torch.from_numpy(matrix))
            model.hidden[stage].bias.copy_(torch.from_numpy(bias))
            previous_coefficients = node_coefficients
        model.output.weight.copy_(torch.from_numpy(previous_coefficients))
        model.output.bias.zero_()
    return model, built, sum(len(edges) * m for _, edges in built)


def kan_prediction(built: list, x: np.ndarray) -> np.ndarray:
    values = x
    for node_count, edges in built:
        next_values = np.zeros((len(x), node_count), dtype=np.float64)
        for item, (spline, _, _) in edges:
            coordinate = values[:, item.source]
            if coordinate.min() < item.left - 1e-8 or coordinate.max() > item.right + 1e-8:
                raise ValueError(f"Spline domain exceeded: {item}")
            next_values[:, item.destination] += spline(coordinate)
        values = next_values
    return values[:, 0]


@torch.no_grad()
def model_rmse(model: CubedMLP, data: tuple[torch.Tensor, torch.Tensor]) -> float:
    x, y = data
    square_sum = 0.0
    for begin in range(0, len(x), 512):
        difference = model(x[begin:begin + 512]) - y[begin:begin + 512]
        square_sum += float(torch.sum(difference * difference))
    return float(np.sqrt(square_sum / len(x)))


def train_one(
    warm: CubedMLP, train: tuple[torch.Tensor, torch.Tensor],
    validation: tuple[torch.Tensor, torch.Tensor], steps: int,
    learning_rate: float, seed: int,
) -> tuple[CubedMLP, int, float]:
    model = copy.deepcopy(warm)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_step = 0
    best_validation = model_rmse(model, validation)
    best_state = copy.deepcopy(model.state_dict())
    generator = torch.Generator().manual_seed(seed)
    train_x, train_y = train
    for step in range(1, steps + 1):
        indices = torch.randint(len(train_x), (256,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        error = model(train_x[indices]) - train_y[indices]
        loss = torch.mean(error * error)
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()
        if step in (1, 2, 5, 10, 20, 50, 100, 150, 200, steps):
            current = model_rmse(model, validation)
            if np.isfinite(current) and current < best_validation:
                best_step = step
                best_validation = current
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_step, best_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", choices=FUNCTIONS, required=True)
    parser.add_argument("--grids", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = sample(args.function, 10, 1001)
    validation = sample(args.function, 11, 2002)
    test = sample(args.function, 13, 3003)
    rows = []
    started = time.monotonic()
    for grid in args.grids:
        warm, built, coefficient_count = construct(args.function, grid)
        kan = kan_prediction(built, test[0].numpy())
        kan_rmse = float(np.sqrt(np.mean((kan - test[1].numpy()) ** 2)))
        initial_rmse = model_rmse(warm, test)
        with torch.no_grad():
            transfer_max = float(np.max(np.abs(
                warm(test[0][:64]).numpy() - kan[:64]
            )))
        if transfer_max > max(1e-7, 0.01 * kan_rmse):
            raise RuntimeError(
                f"{args.function} G={grid}: inaccurate float64 transfer {transfer_max:g}"
            )
        learning_rate = 1e-6 * (8 / grid) ** 4
        trained, best_step, best_validation = train_one(
            warm, train, validation, args.steps, learning_rate, 7409 + grid
        )
        row = {
            "function": args.function, "grid": grid,
            "kan_coefficients": coefficient_count,
            "mlp_full_dense_parameters": sum(p.numel() for p in warm.parameters()),
            "kan_test_rmse": kan_rmse,
            "mlp_initial_test_rmse": initial_rmse,
            "mlp_selected_test_rmse": model_rmse(trained, test),
            "best_validation_rmse": best_validation,
            "best_step": best_step,
            "learning_rate": learning_rate,
            "transfer_max_abs_64": transfer_max,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output_dir / f"{args.function}.csv", index=False)
        print(json.dumps(row), flush=True)
    metadata = {
        "function": args.function, "grids": args.grids, "steps": args.steps,
        "train_n": len(train[0]), "validation_n": len(validation[0]),
        "test_n": len(test[0]), "train_seed": 1001, "validation_seed": 2002,
        "test_seed": 3003, "dtype": "float64", "device": "server CPU",
        "activation": "ReLU cubed", "target_knowledge": True,
        "optimizer": "Adam", "mini_batch": 256,
        "selection": "lowest validation RMSE among monitored checkpoints; test once selected",
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / f"{args.function}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
