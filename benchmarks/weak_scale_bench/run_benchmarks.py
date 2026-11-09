#!/usr/bin/env python3
"""Compile and run independent weak-scaling trials, emitting raw reproducible CSV.

Default sweep starts at 1 thread and reaches the affinity-detected CPU count.
To reduce order bias, configurations are shuffled anew for each repetition.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIELDS = ["timestamp_utc", "run", "threads", "work_per_thread", "total_samples",
          "serial_seconds", "parallel_seconds", "scaled_speedup", "efficiency",
          "serial_checksum", "parallel_checksum", "relative_checksum_error", "cpu_model"]


def available_threads() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "not available"


def default_counts(n: int) -> list[int]:
    if n <= 16:
        return list(range(1, n + 1))
    seq = {1, n}
    x = 2
    while x < n:
        seq.add(x)
        x *= 2
    return sorted(seq)


def build(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["g++", "-O3", "-std=c++17", "-fopenmp", "-ffp-contract=fast",
           str(HERE / "bench.cpp"), "-o", str(output)]
    subprocess.run(cmd, check=True)


def run_one(binary: Path, p: int, k: int) -> dict:
    result = subprocess.run([str(binary), "--threads", str(p),
                             "--work-per-thread", str(k)], check=True,
                            capture_output=True, text=True)
    values = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            values[key] = val
    required = {"serial_seconds", "parallel_seconds", "serial_checksum",
                "parallel_checksum", "relative_checksum_error"}
    if not required <= values.keys():
        raise RuntimeError(f"Missing fields in benchmark output: {result.stdout}")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", default="auto", help="Comma-separated counts or auto (1 through max)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--work-per-thread", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "results.csv")
    ap.add_argument("--binary", type=Path, default=HERE / "bench")
    args = ap.parse_args()
    max_hw = available_threads()
    counts = default_counts(max_hw) if args.threads == "auto" else [int(v) for v in args.threads.split(",")]
    if not counts or len(set(counts)) != len(counts) or any(p < 1 or p > max_hw for p in counts):
        ap.error(f"Thread counts must be unique and between 1 and affinity count {max_hw}")
    if args.runs < 1 or args.work_per_thread < 1:
        ap.error("Runs and work-per-thread must be positive")
    if max_hw < 5:
        print("NOTE: fewer than five CPUs visible; cannot test five distinct counts on this host.", file=sys.stderr)
    build(args.binary)
    warmup = run_one(args.binary, max(counts), min(args.work_per_thread, 50_000))
    assert float(warmup["relative_checksum_error"]) < 1e-10
    rng = random.Random(args.seed)
    rows = []
    for run in range(1, args.runs + 1):
        order = list(counts)
        rng.shuffle(order)
        for p in order:
            result = run_one(args.binary, p, args.work_per_thread)
            seq = float(result["serial_seconds"])
            par = float(result["parallel_seconds"])
            row = {
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "run": run, "threads": p,
                "work_per_thread": args.work_per_thread,
                "total_samples": p * args.work_per_thread,
                "serial_seconds": format(seq, ".9f"),
                "parallel_seconds": format(par, ".9f"),
                "scaled_speedup": format(seq / par, ".9f"),
                "efficiency": format(seq / (p * par), ".9f"),
                "serial_checksum": result["serial_checksum"],
                "parallel_checksum": result["parallel_checksum"],
                "relative_checksum_error": result["relative_checksum_error"],
                "cpu_model": cpu_model(),
            }
            rows.append(row)
            print(f"run={run:02d} P={p:02d} T1(Pk)={seq:.5f}s TP(Pk)={par:.5f}s Sscaled={seq / par:.3f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["threads"], row["run"])))
    print(f"Saved {len(rows)} raw trials to {args.output}")


if __name__ == "__main__":
    main()
