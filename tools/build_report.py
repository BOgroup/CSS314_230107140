#!/usr/bin/env python3
"""Assemble all 25 markdown task artifacts into a paginated PDF via Pandoc/LaTeX."""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "theory/task01_scaling_models.md", "theory/task02_derivation.md",
    "theory/task03_serial_invariance.md", "theory/task04_asymptotic_limits.md",
    "theory/task05_historical_ncube.md", "calculations/task06_parametric_table.md",
    "calculations/task07_tolerable_serial.md", "calculations/task08_cross_divergence.md",
    "calculations/task09_multicomponent.md", "calculations/task10_memory_growth.md",
    "theory/task11_sun_ni_comparison.md", "theory/task12_communication_penalty.md",
    "theory/task13_failure_cases.md", "theory/task14_isoefficiency.md",
    "theory/task15_energy_scaling.md", "calculations/task16_gemm_scaling.md",
    "calculations/task17_stencil_scaling.md", "analysis/task18_nbody_scaling.md",
    "analysis/task19_monte_carlo.md", "analysis/task20_graph_processing.md",
    "analysis/task21_simulator_validation.md",
    "benchmarks/weak_scale_bench/README.md",
    "analysis/task23_empirical_regression.md", "analysis/task24_hardware_audit.md",
    "README.md",
]


def compose() -> str:
    front = """---
title: "Gustafson's Law & Weak Scaling"
subtitle: "Complete Solutions · Tasks 01–25 · Mathematical Derivations, Real Benchmarks, and Hardware Audit"
date: "9 November 2026"
abstract: |
  A complete practice portfolio for Parallel & Distributed Computing Architecture,
  based on the 25-task sheet by instructor Sufyan bin Uzayr. Contains structured
  solutions, verified numeric calculations, a running OpenMP benchmark and
  independent empirical measurements. Measurements are valid only for the
  explicitly documented container hardware; no personal Git remote was changed.
---

"""
    sections = [front]
    part_titles = {
        1: "Part I — Theoretical Foundations & Mathematical Derivations",
        6: "Part II — Mathematical Modeling & Scaled Calculations",
        11: "Part III — Comparative Scaling & Theoretical Limits",
        16: "Part IV — Domain-Specific Algorithmic Workloads",
        21: "Part V — Benchmarking, Empirical Research & Repository Artifacts",
    }
    for index, name in enumerate(FILES, start=1):
        material = (ROOT / name).read_text(encoding="utf-8").strip()
        # Task 22 is a benchmark README; the root README is Task 25.
        if index == 22:
            material = material.replace("# OpenMP weak-scaling benchmark (Task 22)", "# Task 22 — OpenMP Weak-Scaling Benchmark Suite", 1)
        if index == 25:
            material = material.replace("# Gustafson's Law & Weak Scaling — Complete Practice Repository", "# Task 25 — Master Repository and Executive Synthesis", 1)
        # Shift heading levels under each Part; avoid altering fenced source code.
        shifted = []
        fenced = False
        for line in material.splitlines():
            if line.lstrip().startswith("```"):
                fenced = not fenced
            if not fenced and re.match(r"^#{1,5} ", line):
                line = "#" + line
            shifted.append(line)
        if index in part_titles:
            sections.extend([r"\clearpage", "# " + part_titles[index]])
        sections.extend(["\n".join(shifted), "\n"])
    sections.extend([r"\clearpage", "# Appendix — Verified Benchmark Figures",
                    "![Observed mean scaled speedups, error bars, and model curves](plots/scaling_comparison.png){width=95%}",
                    "\n![Measured parallel efficiency and fitted efficiency](plots/efficiency_comparison.png){width=95%}",
                    "\n![Parametric simulator's fixed-size and fixed-time overlays](plots/simulator_overlay.png){width=95%}"])
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Complete_Solutions.pdf")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gustafson_report_") as temp:
        md = Path(temp) / "report.md"
        md.write_text(compose(), encoding="utf-8")
        cmd = ["pandoc", str(md), "-o", str(args.output), "--pdf-engine=xelatex",
               "--from=markdown+tex_math_dollars+pipe_tables", "--toc", "--toc-depth=2",
               "--resource-path=" + str(ROOT), "-V", "mainfont=Noto Serif",
               "-V", "sansfont=Inter", "-V", "monofont=DejaVu Sans Mono",
               "-V", "geometry:margin=20mm", "-V", "papersize=a4", "-V", "fontsize=10pt",
               "-V", "colorlinks=true", "-V", "linkcolor=MidnightBlue",
               "-V", "urlcolor=MidnightBlue", "--highlight-style=tango"]
        subprocess.run(cmd, cwd=ROOT, check=True)
    print(args.output)


if __name__ == "__main__":
    main()
