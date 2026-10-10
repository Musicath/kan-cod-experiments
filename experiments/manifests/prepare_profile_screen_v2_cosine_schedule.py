#!/usr/bin/env python3
"""Generate the revised controlled-ridge scan with cosine LR decay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROFILES = ("cosine", "square", "sine", "tanh", "silu")
DEFAULT_PROFILES = ("square", "sine", "tanh", "silu")
DIMS = (4, 16, 64)
RIDGE_TERMS = (4, 16, 64)
DEPTHS = (2, 4, 6)
KAN_WIDTHS = (1, 4, 16, 64)
KAN_GRIDS = (3, 7, 15, 31)
MLP_WIDTHS = (1, 4, 16, 64, 256)
MLP_ACTIVATIONS = ("relu", "tanh", "silu", "sine")
LEARNING_RATES = (1e-3, 3e-3, 1e-2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profiles", nargs="+", choices=PROFILES, default=DEFAULT_PROFILES
    )
    return parser.parse_args()


def supports(dim: int) -> tuple[int, ...]:
    return tuple(value for value in (1, 2, 4, 8, 16, 32, 64) if value <= dim)


def teacher_seed(dim: int, terms: int, support: int) -> int:
    return 10_000 + 100 * dim + 10 * terms + support


def common(dim: int, terms: int, support: int, seed: int) -> dict[str, object]:
    return {
        "dim": dim,
        "fourier_rank": terms,
        "interaction_support": support,
        "steps": 5000,
        "warmup_steps": 500,
        "batch_size": 4096,
        "eval_every": 100,
        "eval_batch_size": 4096,
        "n_reference": 16384,
        "n_train_probe": 4096,
        "n_val": 16384,
        "n_test": 16384,
        "seed": seed,
        "teacher_seed": teacher_seed(dim, terms, support),
        "data_seed": seed,
        "freq_scale": 0.35,
        "max_dim": 64,
        "lr_schedule": "cosine",
        "min_lr_ratio": 0.01,
    }


def build(profile: str, seed: int = 0) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for dim in DIMS:
        for terms in RIDGE_TERMS:
            for support in supports(dim):
                base = common(dim, terms, support, seed)
                if profile != "cosine":
                    base["profile"] = profile
                for depth in DEPTHS:
                    for width in KAN_WIDTHS:
                        for grid in KAN_GRIDS:
                            for learning_rate in LEARNING_RATES:
                                configs.append(
                                    base
                                    | {
                                        "model": "KAN",
                                        "activation": "",
                                        "hidden_depth": depth,
                                        "width": width,
                                        "grid": grid,
                                        "k": 3,
                                        "lr": learning_rate,
                                    }
                                )
                    for width in MLP_WIDTHS:
                        for activation in MLP_ACTIVATIONS:
                            for learning_rate in LEARNING_RATES:
                                configs.append(
                                    base
                                    | {
                                        "model": "MLP",
                                        "activation": activation,
                                        "hidden_depth": depth,
                                        "width": width,
                                        "grid": 0,
                                        "k": 3,
                                        "lr": learning_rate,
                                    }
                                )

    for index, config in enumerate(configs):
        activation = str(config["activation"] or "none")
        config["job_id"] = (
            f"{index:05d}_{profile}_{config['model']}_d{config['dim']}_"
            f"R{config['fourier_rank']}_s{config['interaction_support']}_"
            f"L{config['hidden_depth']}_w{config['width']}_G{config['grid']}_"
            f"{activation}_lr{float(config['lr']):.0e}_cosine_seed{seed}"
        )
    return configs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for profile in args.profiles:
        configs = build(profile, args.seed)
        output = (
            args.output_dir
            / f"ridge_profile_v2_{profile}_cosine_seed{args.seed}.json"
        )
        output.write_text(
            json.dumps(
                {
                    "name": output.stem,
                    "description": (
                        "Revised controlled-ridge screen with 500-step warmup "
                        "and cosine decay to 1% of the base learning rate; "
                        f"experiment seed {args.seed}."
                    ),
                    "configs": configs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{profile}: {len(configs)} configurations -> {output}")


if __name__ == "__main__":
    main()
