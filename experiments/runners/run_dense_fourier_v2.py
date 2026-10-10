#!/usr/bin/env python3
"""Run one reproducible online Dense Fourier approximation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
PYKAN_CANDIDATES = tuple(
    path
    for path in (
        Path(os.environ["KAN_PYKAN_PATH"]).expanduser()
        if os.environ.get("KAN_PYKAN_PATH")
        else None,
        SCRIPT_DIR / "vendor" / "pykan",
        SCRIPT_DIR.parent / "vendor" / "pykan",
    )
    if path is not None
)
for candidate in PYKAN_CANDIDATES:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from kan import KAN
except Exception as exc:  # pragma: no cover
    KAN = None
    KAN_IMPORT_ERROR = exc
else:
    KAN_IMPORT_ERROR = None


SINE_OMEGA_0 = 30.0
RESULT_FIELDS = (
    "status", "error", "model", "activation", "dim", "fourier_rank",
    "interaction_support", "hidden_depth", "width", "grid", "k", "lr",
    "lr_schedule", "min_lr_ratio", "steps", "warmup_steps", "batch_size", "seed", "teacher_seed",
    "data_seed", "params", "teacher_mode", "freq_scale", "target_mean",
    "target_std", "best_val_mse", "best_val_step", "train_probe_mse",
    "val_mse", "test_mse", "final_online_mse", "train_seconds",
    "wall_seconds", "peak_cuda_mb",
)


class Sine(nn.Module):
    def __init__(self, omega_0: float = SINE_OMEGA_0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


def activation_module(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sine":
        return Sine()
    raise ValueError(f"Unsupported activation: {name}")


class PlainMLP(nn.Module):
    def __init__(self, dim: int, width: int, hidden_depth: int, activation: str):
        super().__init__()
        layers: list[nn.Module] = []
        previous = dim
        for _ in range(hidden_depth):
            layers.extend((nn.Linear(previous, width), activation_module(activation)))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)
        self.activation_name = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear_layers = [module for module in self.modules() if isinstance(module, nn.Linear)]
        for index, module in enumerate(linear_layers):
            if self.activation_name == "relu":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif self.activation_name == "tanh":
                gain = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
            elif self.activation_name == "sine":
                if index == 0:
                    bound = 1.0 / module.in_features
                else:
                    bound = math.sqrt(6.0 / module.in_features) / SINE_OMEGA_0
                nn.init.uniform_(module.weight, -bound, bound)
                nn.init.uniform_(module.bias, -bound, bound)
            else:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierTeacher:
    """Rank-nested Fourier ridges with controlled interaction support."""

    def __init__(
        self,
        dim: int,
        rank: int,
        teacher_seed: int,
        freq_scale: float,
        interaction_support: int,
        max_dim: int,
        device: torch.device,
    ):
        max_rank = 64
        if rank > max_rank:
            raise ValueError(f"rank={rank} exceeds max_rank={max_rank}")
        if dim > max_dim:
            raise ValueError(f"dim={dim} exceeds max_dim={max_dim}")
        support = dim if interaction_support <= 0 else interaction_support
        if not 1 <= support <= dim:
            raise ValueError(f"interaction_support={support} is invalid for dim={dim}")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(40_000 + teacher_seed)
        full_omega = torch.randn(max_rank, max_dim, generator=generator)
        phase = 2.0 * math.pi * torch.rand(max_rank, generator=generator)
        raw_coef = torch.randn(max_rank, generator=generator)[:rank]
        omega = full_omega[:, :dim].clone()
        if support < dim:
            support_generator = torch.Generator(device="cpu")
            support_generator.manual_seed(80_000 + teacher_seed)
            sparse_omega = torch.zeros_like(omega)
            for row in range(max_rank):
                indices = torch.randperm(dim, generator=support_generator)[:support]
                sparse_omega[row, indices] = omega[row, indices]
            omega = sparse_omega
        omega /= omega.norm(dim=1, keepdim=True).clamp_min(1e-8)

        self.omega = (freq_scale * omega[:rank]).to(device)
        self.phase = phase[:rank].to(device)
        self.coef = (raw_coef / raw_coef.norm().clamp_min(1e-8)).to(device)
        self.support = support

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * (x @ self.omega.T) + self.phase
        return torch.cos(phase) @ self.coef[:, None]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    """Count trainable scalars used by the numerical forward branch."""
    return int(
        sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("symbolic_fun.")
        )
    )


def make_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.model == "MLP":
        return PlainMLP(args.dim, args.width, args.hidden_depth, args.activation).to(device)
    if KAN is None:
        raise RuntimeError(f"Could not import pykan: {KAN_IMPORT_ERROR}")
    shape = [args.dim] + [args.width] * args.hidden_depth + [1]
    model = KAN(
        width=shape,
        grid=args.grid,
        k=args.k,
        seed=args.seed,
        device=str(device),
        auto_save=False,
        symbolic_enabled=False,
        save_act=False,
        base_fun="zero",
        scale_base_sigma=0.0,
        sb_trainable=False,
    )
    if hasattr(model, "speed"):
        model.speed()
    return model


def sample_x(
    count: int,
    dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.rand(count, dim, generator=generator, device=device)


def prepare_problem(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    teacher = FourierTeacher(
        dim=args.dim,
        rank=args.fourier_rank,
        teacher_seed=args.teacher_seed,
        freq_scale=args.freq_scale,
        interaction_support=args.interaction_support,
        max_dim=args.max_dim,
        device=device,
    )
    with torch.no_grad():
        x_reference = sample_x(args.n_reference, args.dim, 50_000 + args.data_seed, device)
        y_reference = teacher(x_reference)
        target_mean = y_reference.mean()
        target_std = y_reference.std().clamp_min(1e-8)
        del x_reference, y_reference

        def fixed_split(count: int, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
            x = sample_x(count, args.dim, 50_000 + args.data_seed + offset, device)
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


@torch.no_grad()
def evaluate_mse(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    chunk_size: int,
) -> float:
    model.eval()
    squared_error = 0.0
    count = 0
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        prediction = model(x[start:stop])
        squared_error += float(torch.sum((prediction - y[start:stop]) ** 2).cpu())
        count += stop - start
    return squared_error / count


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train(
    model: nn.Module,
    args: argparse.Namespace,
    problem: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    loss_function = nn.MSELoss()
    online_generator = torch.Generator(device=device)
    online_generator.manual_seed(70_000 + args.data_seed)
    teacher = problem["teacher"]
    target_mean = problem["mean"]
    target_std = problem["std"]
    best_val = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    final_online = float("nan")
    history: list[dict[str, float | int]] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        if args.warmup_steps > 0 and step <= args.warmup_steps:
            current_lr = args.lr * step / args.warmup_steps
        elif args.lr_schedule == "cosine":
            decay_steps = max(args.steps - args.warmup_steps, 1)
            progress = min(max((step - args.warmup_steps) / decay_steps, 0.0), 1.0)
            min_lr = args.lr * args.min_lr_ratio
            current_lr = min_lr + 0.5 * (args.lr - min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        else:
            current_lr = args.lr
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        x = torch.rand(
            args.batch_size,
            args.dim,
            generator=online_generator,
            device=device,
        )
        with torch.no_grad():
            y = (teacher(x) - target_mean) / target_std
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = loss_function(prediction, y)
        loss.backward()
        optimizer.step()
        final_online = float(loss.detach().cpu())

        if step % args.eval_every == 0 or step == args.steps:
            val_mse = evaluate_mse(
                model,
                problem["x_val"],
                problem["y_val"],
                args.eval_batch_size,
            )
            history.append(
                {
                    "step": step,
                    "online_mse": final_online,
                    "val_mse": val_mse,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            if val_mse < best_val:
                best_val = val_mse
                best_step = step
                best_state = cpu_state_dict(model)
            model.train()

    train_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    train_probe_mse = evaluate_mse(
        model, problem["x_probe"], problem["y_probe"], args.eval_batch_size
    )
    val_mse = evaluate_mse(
        model, problem["x_val"], problem["y_val"], args.eval_batch_size
    )
    test_mse = evaluate_mse(
        model, problem["x_test"], problem["y_test"], args.eval_batch_size
    )
    peak_cuda_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda"
        else 0.0
    )
    metrics = {
        "best_val_mse": best_val,
        "best_val_step": best_step,
        "train_probe_mse": train_probe_mse,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "final_online_mse": final_online,
        "train_seconds": train_seconds,
        "peak_cuda_mb": peak_cuda_mb,
    }
    return metrics, history


def write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("KAN", "MLP"), required=True)
    parser.add_argument("--activation", choices=("sine", "relu", "tanh", "silu"), default="silu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--fourier-rank", type=int, choices=(4, 16, 64), required=True)
    parser.add_argument("--interaction-support", type=int, default=0)
    parser.add_argument("--hidden-depth", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--freq-scale", type=float, default=0.35)
    parser.add_argument("--max-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
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
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must lie in [0, 1]")
    wall_started = time.perf_counter()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["teacher_mode"] = "prefix_normalized"
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": args.device,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    base: dict[str, object] = {
        "status": "ok",
        "error": "",
        "model": args.model,
        "activation": args.activation if args.model == "MLP" else "",
        "dim": args.dim,
        "fourier_rank": args.fourier_rank,
        "interaction_support": args.dim if args.interaction_support <= 0 else args.interaction_support,
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
        "teacher_mode": "prefix_normalized",
        "freq_scale": args.freq_scale,
    }
    try:
        set_seed(args.seed)
        device = torch.device(args.device)
        problem = prepare_problem(args, device)
        model = make_model(args, device)
        base["params"] = count_params(model)
        base["target_mean"] = float(problem["mean"].detach().cpu())
        base["target_std"] = float(problem["std"].detach().cpu())
        metrics, history = train(model, args, problem, device)
        base.update(metrics)
        write_csv(
            output / "history.csv",
            ["step", "online_mse", "val_mse", "lr"],
            history,
        )
    except Exception as exc:
        base["status"] = "error"
        base["error"] = repr(exc)
        for field in RESULT_FIELDS:
            base.setdefault(field, "")
    base["wall_seconds"] = time.perf_counter() - wall_started
    write_csv(output / "result.csv", list(RESULT_FIELDS), [base])
    if base["status"] == "ok":
        (output / "DONE").write_text("ok\n", encoding="ascii")
        return 0
    (output / "FAILED").write_text(str(base["error"]) + "\n", encoding="utf-8")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
