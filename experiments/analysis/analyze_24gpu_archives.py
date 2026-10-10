"""Audit and summarize the three complete 2026-09-20 ridge-profile archives."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


PROFILES = ("sine", "silu", "tanh", "square")
BUDGETS = (250, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
FIELDS = (
    "status", "error", "model", "activation", "profile", "dim",
    "fourier_rank", "interaction_support", "hidden_depth", "width",
    "grid", "k", "lr", "lr_schedule", "min_lr_ratio", "steps",
    "warmup_steps", "batch_size", "seed", "teacher_seed", "data_seed",
    "params", "target_mean", "target_std", "best_val_mse",
    "best_val_step", "train_probe_mse", "val_mse", "test_mse",
    "final_online_mse", "train_seconds", "wall_seconds", "peak_cuda_mb",
)


def numeric(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return math.inf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[tuple[int, str, str]] = Counter()
    missing: dict[str, list[str]] = {}
    selected: dict[tuple[str, int, int, int, int, str], dict[str, str]] = {}
    statuses: dict[str, dict] = {}
    results_path = args.output_dir / "all_runs.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for seed in (0, 1, 2):
            archive = args.archive_dir / f"kan_dense_v2_seed{seed}_20260920.tar.gz"
            root = f"kan_dense_v2{f'_seed{seed}' if seed else ''}_20260920"
            seen: dict[str, set[str]] = defaultdict(set)
            with tarfile.open(archive, "r:gz") as tar:
                for item in tar:
                    if not item.isfile():
                        continue
                    parts = item.name.split("/")
                    if len(parts) == 4 and parts[2].endswith(f"_seed{seed}") and parts[3] == "scheduler_status.json":
                        file = tar.extractfile(item)
                        if file is not None:
                            jobs = json.load(file)
                            statuses[parts[2]] = {
                                "jobs": len(jobs),
                                "nonzero_returncodes": sum(job.get("returncode") != 0 for job in jobs),
                                "total_job_seconds": sum(job.get("seconds", 0) for job in jobs),
                            }
                        continue
                    if len(parts) != 6 or parts[0] != root or parts[1] != "results" or parts[3] != "jobs" or parts[5] != "result.csv":
                        continue
                    profile_seed, job_id = parts[2], parts[4]
                    if profile_seed not in {f"{p}_seed{seed}" for p in PROFILES}:
                        continue
                    file = tar.extractfile(item)
                    if file is None:
                        continue
                    row = next(csv.DictReader(io.TextIOWrapper(file, encoding="utf-8")))
                    writer.writerow({field: row.get(field, "") for field in FIELDS})
                    seen[profile_seed].add(job_id)
                    counts[(seed, row["profile"], row["status"])] += 1
                    if row["status"] != "ok":
                        continue
                    params = int(row["params"])
                    target = (
                        row["profile"], int(row["dim"]),
                        int(row["fourier_rank"]), int(row["interaction_support"]),
                        seed, row["model"],
                    )
                    train_mse = numeric(row, "train_probe_mse")
                    if not math.isfinite(train_mse) or not math.isfinite(numeric(row, "test_mse")):
                        continue
                    for budget in BUDGETS:
                        if params > budget:
                            continue
                        key = target[:4] + (budget,) + target[4:]
                        old = selected.get(key)
                        if old is None or (train_mse, numeric(row, "test_mse")) < (
                            numeric(old, "train_probe_mse"), numeric(old, "test_mse")
                        ):
                            selected[key] = row
                for profile in PROFILES:
                    profile_seed = f"{profile}_seed{seed}"
                    manifest = tar.getmember(f"{root}/manifests/ridge_profile_v2_{profile}_cosine_seed{seed}.json")
                    file = tar.extractfile(manifest)
                    assert file is not None
                    expected = {entry["job_id"] for entry in json.load(file)["configs"]}
                    absent = sorted(expected - seen[profile_seed])
                    if absent:
                        missing[profile_seed] = absent
            print(f"seed {seed}: " + ", ".join(
                f"{p}={counts[(seed, p, 'ok')]} ok/{counts[(seed, p, 'error')]} error"
                for p in PROFILES
            ), flush=True)

    rows = []
    for key in sorted({key[:5] for key in selected}):
        profile, dim, rank, support, budget = key
        for seed in (0, 1, 2):
            kan = selected.get(key + (seed, "KAN"))
            mlp = selected.get(key + (seed, "MLP"))
            if kan is None or mlp is None:
                continue
            rows.append({
                "profile": profile, "dim": dim, "rank": rank,
                "support": support, "budget": budget, "seed": seed,
                "kan_params": kan["params"], "kan_depth": kan["hidden_depth"],
                "kan_width": kan["width"], "kan_grid": kan["grid"],
                "kan_lr": kan["lr"], "kan_train_mse": kan["train_probe_mse"],
                "kan_test_mse": kan["test_mse"],
                "mlp_params": mlp["params"], "mlp_depth": mlp["hidden_depth"],
                "mlp_width": mlp["width"], "mlp_activation": mlp["activation"],
                "mlp_lr": mlp["lr"], "mlp_train_mse": mlp["train_probe_mse"],
                "mlp_test_mse": mlp["test_mse"],
                "test_log10_ratio": math.log10(numeric(mlp, "test_mse") / numeric(kan, "test_mse")),
                "train_log10_ratio": math.log10(numeric(mlp, "train_probe_mse") / numeric(kan, "train_probe_mse")),
            })
    comparison_path = args.output_dir / "budget_comparisons_by_seed.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "audit.json").write_text(json.dumps({
        "counts": {f"seed{seed}_{profile}_{status}": count
                   for (seed, profile, status), count in sorted(counts.items())},
        "missing_job_ids": missing,
        "scheduler_status": statuses,
        "total_result_rows": sum(counts.values()),
        "budget_comparison_rows": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"total rows={sum(counts.values())}, missing={sum(map(len, missing.values()))}, comparisons={len(rows)}")


if __name__ == "__main__":
    main()
