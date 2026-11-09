#!/usr/bin/env python3
"""Task 21: produce CSV and overlay plots for Amdahl and Gustafson models.

Run e.g.: python3 benchmarks/simulator.py --core-counts 1,2,4,8,16,128,1024,65536 --alphas .001,.04,.08,.2
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def amdahl(p: int, s: float) -> float:
    return 1.0 / (s + (1.0 - s) / p)


def gustafson(p: int, alpha: float) -> float:
    return p - alpha * (p - 1)


def read_values(raw: str, cast, label: str) -> list:
    try:
        result = [cast(item.strip()) for item in raw.split(",")]
        if not result:
            raise ValueError("empty list")
        return sorted(set(result))
    except ValueError as exc:
        raise ValueError(f"Cannot parse {label}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-counts", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536")
    parser.add_argument("--alphas", default="0.001,0.04,0.08,0.20")
    parser.add_argument("--csv", type=Path, default=ROOT / "benchmarks" / "simulated_scaling.csv")
    parser.add_argument("--plot", type=Path, default=ROOT / "plots" / "simulator_overlay.png")
    args = parser.parse_args()
    try:
        ps = read_values(args.core_counts, int, "core counts")
        fractions = read_values(args.alphas, float, "serial fractions")
        if any(p < 1 or p > 65536 for p in ps):
            raise ValueError("All core counts must satisfy 1 <= P <= 65,536")
        if any(a < 0.001 or a > 0.20 for a in fractions):
            raise ValueError("All serial fractions must satisfy 0.001 <= alpha <= 0.20")
    except ValueError as exc:
        parser.error(str(exc))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh)
        out.writerow(["cores", "serial_fraction", "amdahl_fixed_size_speedup",
                      "gustafson_scaled_speedup", "gustafson_parallel_efficiency"])
        for a in fractions:
            for p in ps:
                g = gustafson(p, a)
                out.writerow([p, f"{a:.6f}", f"{amdahl(p, a):.12f}", f"{g:.12f}", f"{g/p:.12f}"])

    fig, ax = plt.subplots(figsize=(10.7, 6.7), constrained_layout=True)
    ax.plot(ps, ps, linestyle=":", linewidth=2.5, color="#222222", label="Ideal linear")
    colors = ["#245ca6", "#d14d33", "#117766", "#9b5fad", "#dd9933"]
    for j, a in enumerate(fractions):
        color = colors[j % len(colors)]
        ax.plot(ps, [gustafson(p, a) for p in ps], label=f"Gustafson alpha={a:g}", color=color, linewidth=1.8)
        ax.plot(ps, [amdahl(p, a) for p in ps], label=f"Amdahl s={a:g}", color=color, linestyle="--", linewidth=1.4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Processors P (log2)")
    ax.set_ylabel("Predicted speedup (log10)")
    ax.set_title("Strong vs weak scaling under matched numeric fractions")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.savefig(args.plot, dpi=220, bbox_inches="tight")
    fig.savefig(args.plot.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"CSV: {args.csv}\nPlot: {args.plot}")


if __name__ == "__main__":
    main()
