#!/usr/bin/env python3
"""Run one support-controlled dense-ridge profile experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F

import run_dense_fourier_v2 as common


PROFILES = ("square", "cubic", "sine", "tanh", "silu")
RESULT_FIELDS = (
    "status", "error", "model", "activation", "profile", "dim", "fourier_rank", "interaction_support",
    "hidden_depth", "width", "grid", "k", "lr", "lr_schedule", "min_lr_ratio", "steps", "warmup_steps",
    "batch_size", "seed", "teacher_seed", "data_seed", "params",
    "target_mean", "target_std", "best_val_mse", "best_val_step",
    "train_probe_mse", "val_mse", "test_mse", "final_online_mse",
    "train_seconds", "wall_seconds", "peak_cuda_mb",
)


class RidgeProfileTeacher:
    """Nested ridge teacher with controlled coordinate support."""

    def __init__(
        self,
        dim: int,
        rank: int,
        profile: str,
        teacher_seed: int,
        interaction_support: int,
        max_dim: int,
        device: torch.device,
    ) -> None:
        if profile not in PROFILES:
            raise ValueError(f"Unsupported profile: {profile}")
        if not 1 <= interaction_support <= dim <= max_dim:
            raise ValueError("Require 1 <= interaction_support <= dim <= max_dim")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(120_000 + teacher_seed)
        max_rank = 64
        directions = torch.randn(max_rank, max_dim, generator=generator)[:, :dim]
        directions /= directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        offsets = torch.empty(max_rank).uniform_(-0.5, 0.5, generator=generator)
        coefficients = torch.randn(max_rank, generator=generator)[:rank]
        coefficients /= coefficients.norm().clamp_min(1e-12)
        if interaction_support < dim:
            support_generator = torch.Generator(device="cpu")
            support_generator.manual_seed(80_000 + teacher_seed)
            sparse = torch.zeros_like(directions)
            for row in range(max_rank):
                selected = torch.randperm(dim, generator=support_generator)[:interaction_support]
                sparse[row, selected] = directions[row, selected]
            directions = sparse
            directions /= directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.directions = directions[:rank].to(device)
        self.offsets = offsets[:rank].to(device)
        self.coefficients = coefficients.to(device)
        self.profile = profile

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ridge = (2.0 * x - 1.0) @ self.directions.T + self.offsets
        if self.profile == "square":
            features = ridge.square()
        elif self.profile == "cubic":
            features = ridge.pow(3)
        elif self.profile == "sine":
            features = torch.sin(2.0 * torch.pi * ridge)
        elif self.profile == "tanh":
            features = torch.tanh(ridge)
        else:
            features = F.silu(ridge)
        return features @ self.coefficients[:, None]


def prepare_problem(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    teacher = RidgeProfileTeacher(
        args.dim,
        args.fourier_rank,
        args.profile,
        args.teacher_seed,
        args.interaction_support,
        args.max_dim,
        device,
    )
    with torch.no_grad():
        x_reference = common.sample_x(
            args.n_reference, args.dim, 50_000 + args.data_seed, device
        )
        y_reference = teacher(x_reference)
        target_mean = y_reference.mean()
        target_std = y_reference.std().clamp_min(1e-8)

        def fixed_split(count: int, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
            x = common.sample_x(
                count, args.dim, 50_000 + args.data_seed + offset, device
            )
            y = (teacher(x) - target_mean) / target_std
            return x, y

        x_probe, y_probe = fixed_split(args.n_train_probe, 1_000_000)
        x_val, y_val = fixed_split(args.n_val, 2_000_000)
        x_test, y_test = fixed_split(args.n_test, 3_000_000)
    return {
        "teacher": teacher,
        "mean": target_mean,
        "std": target_std,
        "x_probe": x_probe,
        "y_probe": y_probe,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def write_csv(path: Path, fields: tuple[str, ...], row: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("KAN", "MLP"), required=True)
    parser.add_argument("--activation", choices=("sine", "relu", "tanh", "silu"), default="silu")
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--fourier-rank", type=int, required=True)
    parser.add_argument("--interaction-support", type=int, required=True)
    parser.add_argument("--max-dim", type=int, default=32)
    parser.add_argument("--hidden-depth", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--n-reference", type=int, default=16384)
    parser.add_argument("--n-train-probe", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=16384)
    parser.add_argument("--n-test", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    # Accepted for compatibility with the shared manifest launcher; this teacher has no Fourier frequency scale.
    parser.add_argument("--freq-scale", type=float, default=0.35)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fourier_rank <= 0:
        raise ValueError("--fourier-rank must be a positive integer")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must lie in [0, 1]")
    wall_started = time.perf_counter()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "teacher_mode": "support_controlled_ridge_profile",
            "input_domain": "[0,1]^d; teacher uses z=2x-1",
            "online_sampling": True,
            "kan_base_fun": "zero",
            "kan_layer_residual": False,
        }
    )
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": args.device,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    row: dict[str, object] = {
        "status": "ok",
        "error": "",
        "model": args.model,
        "activation": args.activation if args.model == "MLP" else "",
        "profile": args.profile,
        "dim": args.dim,
        "fourier_rank": args.fourier_rank,
        "interaction_support": args.interaction_support,
        "hidden_depth": args.hidden_depth,
        "width": args.width,
        "grid": args.grid if args.model == "KAN" else "",
        "k": args.k if args.model == "KAN" else "",
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "min_lr_ratio": args.min_lr_ratio,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "teacher_seed": args.teacher_seed,
        "data_seed": args.data_seed,
    }
    try:
        common.set_seed(args.seed)
        device = torch.device(args.device)
        problem = prepare_problem(args, device)
        model = common.make_model(args, device)
        row["params"] = common.count_params(model)
        row["target_mean"] = float(problem["mean"].cpu())
        row["target_std"] = float(problem["std"].cpu())
        metrics, history = common.train(model, args, problem, device)
        row.update(metrics)
        history_path = output / "history.csv"
        temporary = history_path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("step", "online_mse", "val_mse", "lr"))
            writer.writeheader()
            writer.writerows(history)
        temporary.replace(history_path)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = repr(exc)
        for field in RESULT_FIELDS:
            row.setdefault(field, "")
    row["wall_seconds"] = time.perf_counter() - wall_started
    write_csv(output / "result.csv", RESULT_FIELDS, row)
    marker = "DONE" if row["status"] == "ok" else "FAILED"
    (output / marker).write_text(str(row["error"] or "ok") + "\n", encoding="utf-8")
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
