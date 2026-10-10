#!/usr/bin/env python3
"""MNIST normal-label experiment aligned with the paper's random-label scan.

The full scan uses all 60,000 official training examples and the official
10,000-example test split. It is resumable at the individual-run level and can
be divided across multiple GPUs with --shard-index/--num-shards.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST


MLP_WIDTHS = [4, 8, 16, 32, 64, 128, 256, 512]
KAN_WIDTHS = [1, 2, 4, 8, 16, 32]
DEPTHS = [2, 4, 6, 8]
SEEDS = [0, 1, 2]
OPTIM_LRS = {
    "sgd": [0.1, 0.03, 0.01],
    "adam": [0.01, 0.003, 0.001],
}


def import_kan():
    configured = os.environ.get("KAN_PYKAN_PATH")
    candidates = [
        *([Path(configured).expanduser()] if configured else []),
        Path(__file__).resolve().parent / "vendor" / "pykan",
        Path(__file__).resolve().parent.parent / "vendor" / "pykan",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from kan import KAN
    except ImportError as exc:
        raise RuntimeError(
            "pykan is required. Install it, set KAN_PYKAN_PATH, or place it "
            "under experiments/vendor/pykan."
        ) from exc
    return KAN


@dataclass(frozen=True)
class RunConfig:
    n_train: int
    model: str
    width: int
    depth: int
    seed: int
    optimizer: str
    lr: float

    @property
    def key(self) -> str:
        lr_tag = f"{self.lr:g}".replace(".", "p").replace("-", "m")
        return (
            f"mnist_normal_N{self.n_train}_{self.model}_L{self.depth}_w{self.width}"
            f"_seed{self.seed}_{self.optimizer}_lr{lr_tag}"
        )


class PlainMLP(nn.Module):
    def __init__(self, width: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(784, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.ReLU()])
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(
    cfg: RunConfig,
    device: torch.device,
    kan_edge_base: str = "zero",
) -> nn.Module:
    set_seed(cfg.seed)
    if cfg.model == "MLP":
        return PlainMLP(cfg.width, cfg.depth).to(device)

    KAN = import_kan()
    shape = [784] + [cfg.width] * cfg.depth + [10]
    use_silu_base = kan_edge_base == "silu"
    model = KAN(
        width=shape,
        grid=3,
        k=3,
        seed=cfg.seed,
        device=str(device),
        auto_save=False,
        symbolic_enabled=False,
        save_act=False,
        base_fun="silu" if use_silu_base else "zero",
        scale_base_sigma=0.1 if use_silu_base else 0.0,
        sp_trainable=False,
        sb_trainable=use_silu_base,
    )
    if hasattr(model, "symbolic_fun"):
        for parameter in model.symbolic_fun.parameters():
            parameter.requires_grad_(False)
    if hasattr(model, "speed"):
        model.speed()
    return model


def count_params(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def make_loaders(
    data_root: Path,
    n_train: int,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    raw_train = MNIST(str(data_root), train=True, download=True)
    raw_test = MNIST(str(data_root), train=False, download=True)

    if not 1 <= n_train <= len(raw_train):
        raise ValueError(f"n_train must be in [1, {len(raw_train)}], got {n_train}.")

    # MNIST.data is uint8. Materialize the ToTensor-equivalent conversion once
    # instead of reconstructing a PIL image for every sample in every epoch.
    train_x = raw_train.data.unsqueeze(1).to(torch.float32).div_(255.0)
    train_y = raw_train.targets
    test_x = raw_test.data.unsqueeze(1).to(torch.float32).div_(255.0)
    test_y = raw_test.targets
    if n_train < len(raw_train):
        subset_generator = torch.Generator()
        subset_generator.manual_seed(20_000 + seed * 997 + n_train)
        indices = torch.randperm(len(raw_train), generator=subset_generator)[:n_train]
        train_x = train_x.index_select(0, indices)
        train_y = train_y.index_select(0, indices)

    train_set = TensorDataset(train_x, train_y)
    test_set = TensorDataset(test_x, test_y)

    generator = torch.Generator()
    generator.manual_seed(30_000 + seed * 997)

    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **common,
    )
    train_eval_loader = DataLoader(
        train_set, batch_size=eval_batch_size, shuffle=False, **common
    )
    test_loader = DataLoader(
        test_set, batch_size=eval_batch_size, shuffle=False, **common
    )
    return train_loader, train_eval_loader, test_loader


def flatten_to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.view(x.shape[0], -1).to(device, non_blocking=True)


def shutdown_loader(loader: DataLoader) -> None:
    """Release persistent workers before constructing the next configuration."""
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        loader._iterator = None


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for x, y in loader:
        x = flatten_to_device(x, device)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += loss_fn(logits, y).item() * y.numel()
        correct += (logits.argmax(dim=1) == y).sum().item()
        count += y.numel()
    return loss_sum / count, correct / count


def train_one(
    model: nn.Module,
    cfg: RunConfig,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    progress_every: int,
) -> dict[str, float | int]:
    if cfg.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=0.0
        )
    loss_fn = nn.CrossEntropyLoss()
    best_online_train_acc = 0.0
    best_epoch = 0

    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        count = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x = flatten_to_device(x, device)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += y.numel()

        online_acc = correct / count
        if online_acc > best_online_train_acc:
            best_online_train_acc = online_acc
            best_epoch = epoch
        if epoch == 1 or epoch % progress_every == 0 or epoch == epochs:
            print(
                f"  epoch={epoch:03d}/{epochs} online_train_loss={loss_sum / count:.6f} "
                f"online_train_acc={online_acc:.6f}",
                flush=True,
            )

    elapsed_sec = time.perf_counter() - start
    final_train_loss, final_train_acc = evaluate(
        model, train_eval_loader, device, loss_fn
    )
    final_test_loss, final_test_acc = evaluate(model, test_loader, device, loss_fn)
    return {
        "best_online_train_acc": best_online_train_acc,
        "best_online_train_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "elapsed_sec": elapsed_sec,
    }


def iter_configs(args: argparse.Namespace) -> Iterable[RunConfig]:
    models = ["MLP", "KAN"] if args.models is None else args.models
    for seed in args.seeds:
        for model in models:
            widths = args.mlp_widths if model == "MLP" else args.kan_widths
            for depth in args.depths:
                for width in widths:
                    for optimizer_name in args.optimizers:
                        for lr in OPTIM_LRS[optimizer_name]:
                            yield RunConfig(
                                args.n_train, model, width, depth, seed, optimizer_name, lr
                            )


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def aggregate(out_root: Path) -> None:
    rows: list[dict[str, object]] = []
    for result_path in sorted((out_root / "per_run").glob("*/result.json")):
        rows.append(json.loads(result_path.read_text(encoding="utf-8")))
    if not rows:
        print(f"No completed runs found under {out_root}", flush=True)
        return

    rows.sort(
        key=lambda row: (
            row["model"], row["depth"], row["width"], row["seed"],
            row["optimizer"], row["lr"],
        )
    )
    write_csv(out_root / "raw_results.csv", rows)

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["model"], row["depth"], row["width"], row["seed"])
        grouped.setdefault(key, []).append(row)
    best = []
    for candidates in grouped.values():
        best.append(
            max(
                candidates,
                key=lambda row: (
                    float(row["final_train_acc"]),
                    -float(row["final_train_loss"]),
                    float(row["best_online_train_acc"]),
                ),
            )
        )
    best.sort(key=lambda row: (row["model"], row["depth"], row["width"], row["seed"]))
    write_csv(out_root / "best_by_architecture_seed.csv", best)

    summaries: list[dict[str, object]] = []
    architecture_groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in best:
        key = (row["model"], row["depth"], row["width"], row["params"])
        architecture_groups.setdefault(key, []).append(row)
    metrics = [
        "best_online_train_acc",
        "final_train_acc",
        "final_train_loss",
        "final_test_acc",
        "final_test_loss",
    ]
    for key, candidates in architecture_groups.items():
        summary: dict[str, object] = {
            "model": key[0],
            "depth": key[1],
            "width": key[2],
            "params": key[3],
            "completed_seeds": len(candidates),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in candidates]
            summary[f"{metric}_median"] = percentile(values, 50)
            summary[f"{metric}_q25"] = percentile(values, 25)
            summary[f"{metric}_q75"] = percentile(values, 75)
        summaries.append(summary)
    summaries.sort(key=lambda row: (row["model"], row["params"], row["depth"], row["width"]))
    write_csv(out_root / "summary.csv", summaries)
    print(
        f"Aggregated {len(rows)} runs, {len(best)} selected architecture-seed rows, "
        f"and {len(summaries)} architecture summaries under {out_root}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="mnist_normal_labels_N60000")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-train", type=int, default=60000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--models", nargs="+", choices=["MLP", "KAN"])
    parser.add_argument("--depths", nargs="+", type=int, default=DEPTHS)
    parser.add_argument("--mlp-widths", nargs="+", type=int, default=MLP_WIDTHS)
    parser.add_argument("--kan-widths", nargs="+", type=int, default=KAN_WIDTHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--optimizers", nargs="+", choices=["sgd", "adam"], default=["sgd", "adam"])
    parser.add_argument(
        "--kan-edge-base",
        choices=["zero", "silu"],
        default="zero",
        help="Use a spline-only KAN (zero) or the practical SiLU edge-base branch.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Do not rewrite shared summary CSVs when several shards run concurrently.",
    )
    parser.add_argument("--max-runs", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards.")

    out_root = args.results_root / args.run_name
    if args.aggregate_only:
        aggregate(out_root)
        return 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_configs = list(iter_configs(args))
    shard_configs = [
        cfg for index, cfg in enumerate(all_configs)
        if index % args.num_shards == args.shard_index
    ]
    if args.max_runs > 0:
        shard_configs = shard_configs[: args.max_runs]

    out_root.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "created_at": datetime.now().isoformat(),
        "n_train": args.n_train,
        "n_test": 10000,
        "normal_labels": True,
        "grid": 3,
        "spline_degree": 3,
        "kan_base_fun": args.kan_edge_base,
        "kan_scale_base_sigma": 0.1 if args.kan_edge_base == "silu" else 0.0,
        "kan_sp_trainable": False,
        "kan_sb_trainable": args.kan_edge_base == "silu",
        "layerwise_residual": False,
        "weight_decay": 0.0,
        "optimizer_lrs": OPTIM_LRS,
        "total_cartesian_runs": len(all_configs),
        **vars(args),
    }
    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config_payload.items()
    }
    atomic_write_json(out_root / f"run_config_shard{args.shard_index}.json", config_payload)

    completed = 0
    for cfg in shard_configs:
        result_path = out_root / "per_run" / cfg.key / "result.json"
        if args.skip_existing and result_path.exists():
            print(f"Skipping existing {cfg.key}", flush=True)
            continue

        print(f"Starting {cfg.key} on {device}", flush=True)
        train_loader, train_eval_loader, test_loader = make_loaders(
            args.data_root,
            args.n_train,
            args.batch_size,
            args.eval_batch_size,
            cfg.seed,
            args.num_workers,
        )
        model = make_model(cfg, device, args.kan_edge_base)
        params = count_params(model)
        print(f"  params={params}", flush=True)
        metrics = train_one(
            model,
            cfg,
            train_loader,
            train_eval_loader,
            test_loader,
            device,
            args.epochs,
            args.progress_every,
        )
        row: dict[str, object] = {
            "run_key": cfg.key,
            **asdict(cfg),
            "params": params,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "n_train": args.n_train,
            "n_test": 10000,
            **metrics,
        }
        atomic_write_json(result_path, row)
        print(
            f"Completed {cfg.key}: train_acc={metrics['final_train_acc']:.6f} "
            f"test_acc={metrics['final_test_acc']:.6f} "
            f"elapsed={metrics['elapsed_sec']:.1f}s",
            flush=True,
        )
        completed += 1
        shutdown_loader(train_loader)
        shutdown_loader(train_eval_loader)
        shutdown_loader(test_loader)
        del model, train_loader, train_eval_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.no_aggregate:
        aggregate(out_root)
    print(f"Shard {args.shard_index} completed {completed} new runs.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
