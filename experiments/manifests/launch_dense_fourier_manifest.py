#!/usr/bin/env python3
"""Run an explicit Dense Fourier experiment manifest across selected GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--runner", default="")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_capture(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def build_command(
    python: str,
    runner: Path,
    config: dict[str, Any],
    output: Path,
) -> list[str]:
    command = [
        python,
        str(runner),
        "--model", str(config["model"]),
        "--device", "cuda",
        "--dim", str(config["dim"]),
        "--fourier-rank", str(config["fourier_rank"]),
        "--interaction-support", str(config.get("interaction_support", 0)),
        "--hidden-depth", str(config["hidden_depth"]),
        "--width", str(config["width"]),
        "--grid", str(config.get("grid", 3)),
        "--k", str(config.get("k", 3)),
        "--lr", str(config["lr"]),
        "--lr-schedule", str(config.get("lr_schedule", "constant")),
        "--min-lr-ratio", str(config.get("min_lr_ratio", 0.01)),
        "--steps", str(config.get("steps", 5000)),
        "--warmup-steps", str(config.get("warmup_steps", 500)),
        "--batch-size", str(config.get("batch_size", 4096)),
        "--eval-every", str(config.get("eval_every", 100)),
        "--eval-batch-size", str(config.get("eval_batch_size", 4096)),
        "--n-reference", str(config.get("n_reference", 16384)),
        "--n-train-probe", str(config.get("n_train_probe", 4096)),
        "--n-val", str(config.get("n_val", 16384)),
        "--n-test", str(config.get("n_test", 16384)),
        "--seed", str(config["seed"]),
        "--teacher-seed", str(config.get("teacher_seed", config["seed"])),
        "--data-seed", str(config.get("data_seed", config["seed"])),
        "--freq-scale", str(config.get("freq_scale", 0.35)),
        "--max-dim", str(config.get("max_dim", 32)),

    ]
    if "profile" in config:
        command.extend(("--profile", str(config["profile"])))
    if config["model"] == "MLP":
        command.extend(("--activation", str(config["activation"])))
    command.extend(("--out-dir", str(output)))
    return command


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    runner = (
        Path(args.runner).resolve()
        if args.runner
        else script_dir.parent / "runners/run_dense_fourier_v2.py"
    )
    manifest_source = Path(args.manifest).resolve()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    jobs_dir = root / "jobs"
    snapshot = root / "code_snapshot"
    logs.mkdir(exist_ok=True)
    jobs_dir.mkdir(exist_ok=True)
    snapshot.mkdir(exist_ok=True)

    raw_manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    configs = raw_manifest.get("configs") if isinstance(raw_manifest, dict) else raw_manifest
    if not isinstance(configs, list):
        raise ValueError("Manifest must be a JSON list or an object with a configs list")
    for index, config in enumerate(configs):
        config.setdefault("job_id", f"{index:05d}")
    manifest_copy = root / "manifest.json"
    manifest_copy.write_text(
        json.dumps(configs, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for source in (Path(__file__), runner, manifest_source):
        shutil.copy2(source, snapshot / source.name)

    configured_pykan = os.environ.get("KAN_PYKAN_PATH")
    pykan_root = (
        Path(configured_pykan).expanduser()
        if configured_pykan
        else script_dir.parent / "vendor" / "pykan"
    )
    pykan_source = pykan_root / "kan" / "MultKAN.py"
    metadata = {
        "created_unix": time.time(),
        "hostname": socket.gethostname(),
        "launcher_python": platform.python_version(),
        "python_executable": args.python,
        "gpus": args.gpus,
        "manifest_source": str(manifest_source),
        "manifest_sha256": file_sha256(manifest_copy),
        "runner_sha256": file_sha256(runner),
        "launcher_sha256": file_sha256(Path(__file__)),
        "python_version": run_capture([args.python, "--version"]),
        "pip_freeze": run_capture([args.python, "-m", "pip", "freeze"]),
        "nvidia_smi": run_capture(["nvidia-smi"]),
        "pykan_source_path": str(pykan_source),
        "pykan_source_sha256": (
            file_sha256(pykan_source) if pykan_source.exists() else "missing"
        ),
    }
    (root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    pending: queue.Queue[dict[str, Any]] = queue.Queue()
    skipped = 0
    for config in configs:
        output = jobs_dir / config["job_id"]
        if (output / "DONE").exists():
            skipped += 1
        else:
            pending.put(config)

    records: list[dict[str, Any]] = []
    lock = threading.Lock()
    status_path = root / "scheduler_status.json"
    running_marker = root / "RUNNING"
    running_marker.write_text(f"{time.time()}\n", encoding="ascii")

    def save_status() -> None:
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(records, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(status_path)

    def worker(gpu: str) -> None:
        while True:
            try:
                config = pending.get_nowait()
            except queue.Empty:
                return
            output = jobs_dir / config["job_id"]
            output.mkdir(parents=True, exist_ok=True)
            command = build_command(args.python, runner, config, output)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            log_path = logs / f"{config['job_id']}.log"
            started = time.time()
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"\nSTART gpu={gpu} time={started} command={' '.join(command)}\n"
                )
                log.flush()
                result = subprocess.run(
                    command,
                    cwd=script_dir,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            record = {
                "job_id": config["job_id"],
                "gpu": gpu,
                "returncode": result.returncode,
                "seconds": time.time() - started,
                "out_dir": str(output),
            }
            with lock:
                records.append(record)
                save_status()
                print(
                    f"[{len(records) + skipped}/{len(configs)}] gpu={gpu} "
                    f"{config['job_id']} rc={result.returncode} "
                    f"sec={record['seconds']:.1f}",
                    flush=True,
                )
            pending.task_done()

    print(
        f"total={len(configs)} pending={pending.qsize()} skipped={skipped} "
        f"gpus={','.join(args.gpus)}",
        flush=True,
    )
    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=False)
        for gpu in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failures = [record for record in records if record["returncode"] != 0]
    if running_marker.exists():
        running_marker.unlink()
    summary = {
        "finished_unix": time.time(),
        "total": len(configs),
        "skipped": skipped,
        "launched": len(records),
        "failures": len(failures),
    }
    (root / "DONE").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"finished={len(records)} skipped={skipped} failures={len(failures)}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
