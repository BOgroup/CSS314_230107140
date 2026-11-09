#!/usr/bin/env python3
"""Task 23: fit alpha to measured scaled speedup by constrained and unconstrained LS.

Gustafson S(P) = P - alpha*(P-1). For each P>1, regress (P-S(P))
onto (P-1) through the origin. A nonlinear overhead trend will violate
constant-alpha assumptions; residuals and raw fit are saved rather than hidden.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fit_alpha(points):
    x = np.asarray([p - 1 for p, s in points if p > 1], dtype=float)
    y = np.asarray([p - s for p, s in points if p > 1], dtype=float)
    if x.size < 1 or not np.any(x):
        raise ValueError("At least one observation for P > 1 is required")
    return float(np.dot(x, y) / np.dot(x, x))


def read_trials(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("Input CSV has no measurement rows")
    by_p = {}
    for row in rows:
        p = int(row["threads"])
        by_p.setdefault(p, []).append(row)
    return rows, by_p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "benchmarks" / "results.csv")
    parser.add_argument("--aggregate", type=Path, default=ROOT / "benchmarks" / "aggregate.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "benchmarks" / "fit_summary.json")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    rows, by_p = read_trials(args.input)
    aggregate = []
    for p, group in sorted(by_p.items()):
        speeds = np.array([float(r["scaled_speedup"]) for r in group])
        sequential = np.array([float(r["serial_seconds"]) for r in group])
        parallel = np.array([float(r["parallel_seconds"]) for r in group])
        n = len(speeds)
        std = float(speeds.std(ddof=1)) if n > 1 else 0.0
        aggregate.append({
            "threads": p, "runs": n,
            "mean_serial_seconds": float(sequential.mean()),
            "mean_parallel_seconds": float(parallel.mean()),
            "mean_scaled_speedup": float(speeds.mean()),
            "sample_std_scaled_speedup": std,
            "standard_error_scaled_speedup": std / np.sqrt(n),
            "mean_efficiency": float((speeds / p).mean()),
        })
    raw_alpha = fit_alpha([(int(r["threads"]), float(r["scaled_speedup"])) for r in rows])
    # The constrained estimate is the closest admissible alpha in [0, 1].
    bounded_alpha = max(0.0, min(1.0, raw_alpha))
    p_all = np.array([int(r["threads"]) for r in rows], dtype=float)
    s_all = np.array([float(r["scaled_speedup"]) for r in rows], dtype=float)
    pred = p_all - raw_alpha * (p_all - 1)
    pred_bounded = p_all - bounded_alpha * (p_all - 1)
    rmse = float(np.sqrt(np.mean((s_all - pred) ** 2)))
    r2 = float(1 - np.sum((s_all - pred) ** 2) / np.sum((s_all - s_all.mean()) ** 2))
    # Stratified bootstrap resamples runs WITHIN each thread count.
    rng = np.random.default_rng(20261109)
    bootstrap = []
    for _ in range(args.bootstrap):
        sampled = []
        for p, group in by_p.items():
            idx = rng.integers(0, len(group), len(group))
            sampled.extend((p, float(group[int(i)]["scaled_speedup"])) for i in idx)
        bootstrap.append(fit_alpha(sampled))
    ci = [float(v) for v in np.percentile(bootstrap, [2.5, 97.5])] if bootstrap else [None, None]
    args.aggregate.parent.mkdir(parents=True, exist_ok=True)
    with args.aggregate.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    payload = {
        "model": "S(P)=P-alpha*(P-1)",
        "regression": "unweighted OLS through origin of (P-S) on (P-1), on all individual trials P>1",
        "num_measurements": len(rows),
        "thread_counts": sorted(by_p),
        "runs_per_thread_count": {str(p): len(v) for p, v in sorted(by_p.items())},
        "alpha_raw_unconstrained": raw_alpha,
        "alpha_physical_bounded": bounded_alpha,
        "alpha_bootstrap_95pct_raw": ci,
        "rmse_speedup_unconstrained": rmse,
        "r_squared_unconstrained": r2,
        "rmse_speedup_constrained": float(np.sqrt(np.mean((s_all - pred_bounded) ** 2))),
        "warning": "Shared-fraction Amdahl comparison is illustrative: Amdahl s and Gustafson alpha have different reference frames. Fit estimates an effective phenomenological fraction, not an invariant application property.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
