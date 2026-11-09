#!/usr/bin/env python3
"""Task 24: empirical scaled speedup with standard-error bars and theoretical overlays."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregate", type=Path, default=ROOT / "benchmarks" / "aggregate.csv")
    p.add_argument("--summary", type=Path, default=ROOT / "benchmarks" / "fit_summary.json")
    p.add_argument("--out", type=Path, default=ROOT / "plots" / "scaling_comparison.png")
    args = p.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    with args.aggregate.open(newline="", encoding="utf-8") as fh:
        points = list(csv.DictReader(fh))
    ps = np.array([int(r["threads"]) for r in points], dtype=float)
    measured = np.array([float(r["mean_scaled_speedup"]) for r in points], dtype=float)
    se = np.array([float(r["standard_error_scaled_speedup"]) for r in points], dtype=float)
    a = float(summary["alpha_physical_bounded"])
    dense = np.linspace(1, max(ps), 300)
    ideal = dense
    gust = dense - a * (dense - 1)
    amdahl = 1.0 / (a + (1.0 - a) / dense) if a > 0 else dense

    fig, ax = plt.subplots(figsize=(9.6, 6.2), constrained_layout=True)
    ax.plot(dense, ideal, color="#4f5968", linestyle=":", linewidth=2.3, label="Ideal linear speedup")
    ax.plot(dense, gust, color="#155cb5", linewidth=2.4, label=f"Gustafson fitted (alpha={a:.4f})")
    ax.plot(dense, amdahl, color="#cf5a3b", linestyle="--", linewidth=2.3, label="Amdahl same numeric fraction")
    ax.errorbar(ps, measured, yerr=se, fmt="o", markersize=7, capsize=4,
                color="#111111", ecolor="#111111", label="Measured mean +/- SE (n=5)")
    ax.set_title("Empirical weak scaling vs ideal and theoretical curves", fontsize=14)
    ax.set_xlabel("OpenMP threads / available CPUs, P")
    ax.set_ylabel("Scaled speedup = T1(Pk) / TP(Pk)")
    ax.set_xlim(0.85, max(ps) + 0.25)
    ax.set_ylim(0.0, max(max(ps), measured.max()) * 1.12)
    ax.set_xticks(ps)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(alpha=0.24)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kw in [(".png", {"dpi": 260}), (".pdf", {}), (".svg", {})]:
        fig.savefig(args.out.with_suffix(suffix), bbox_inches="tight", **kw)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    ax.errorbar(ps, measured / ps, yerr=se / ps, fmt="o-", capsize=4, linewidth=2,
                color="#155cb5", label="Empirical scaled efficiency +/- SE")
    ax.plot(ps, (ps - a * (ps - 1)) / ps, linestyle="--", color="#c4543c", label="Fitted Gustafson efficiency")
    ax.axhline(1.0, color="#444444", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Threads P")
    ax.set_ylabel("E(P) = S(P)/P")
    ax.set_title("Weak-scaling efficiency")
    ax.set_xticks(ps)
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(args.out.with_name("efficiency_comparison.png"), dpi=260, bbox_inches="tight")
    fig.savefig(args.out.with_name("efficiency_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Plots saved to", args.out.parent)


if __name__ == "__main__":
    main()
