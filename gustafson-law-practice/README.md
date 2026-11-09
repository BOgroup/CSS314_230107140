# Gustafson's Law & Weak Scaling — Complete Practice Repository

**Course:** Parallel & Distributed Computing Architecture  
**Assignment:** 25-task comprehensive practice sheet (instructor: Sufyan bin Uzayr)  
**Submission target:** individual Git repository, branch or folder `gustafson-law-practice/`  
**Status:** all 25 tasks answered; executable simulation, OpenMP benchmarking, real measured data, regression analysis, validation tests, and publication-resolution plots included.

> **Attribution:** The assignment prompt is included in `references/Gustafson_Law_Practice_Tasks.pdf`. This directory contains an independent worked solution. The repository has a **local Git history** but has **not** been pushed to the student's personal Git hosting account. Review and personalize attribution before submitting.

## Table of contents

1. [Executive synthesis](#executive-synthesis)
2. [Essential mathematics](#essential-mathematics)
3. [Contents and task-by-task index](#contents-and-task-by-task-index)
4. [Reproduce the scripts and benchmarks](#reproduce-the-scripts-and-benchmarks)
5. [Actual experiment and its limits](#actual-experiment-and-its-limits)
6. [When to use each model](#when-to-use-each-model)
7. [Primary references](#primary-references)
8. [Submission checklist](#submission-checklist)

## Executive synthesis

**Amdahl's Law** models the speedup of the **same fixed problem** on more processors; when the original one-core serial work fraction is $s_A>0$, total speedup is bounded by $1/s_A$. **Gustafson's Law** models how much **more useful work** can be completed in nearly the **same wall-clock time**, with $\alpha$ measured on the *parallel scaled* job. In its ideal form, $S_G(P)=P-\alpha(P-1)$ and $E_G(P)=1-\alpha+\alpha/P$. Neither invalidates the other: they answer distinct questions.

The Sun–Ni model incorporates **aggregate memory capacity**, which may permit workload growth faster or slower than the fixed-time rule. Isoefficiency analysis makes the missing constraint explicit: constant local work preserves constant efficiency only when communication, imbalance, and coordination overhead do not outgrow useful work. Dense GEMM, regular stencils, N-body solvers, Monte Carlo, and graph workloads demonstrate the diverse behavior of realistic scaling strategies.

A complete engineering scaling report must always name **(i)** the problem-size schedule $W(P)$, **(ii)** the numerator of speedup (serial reference time for *the same job*), **(iii)** the measured serial fraction's frame, **(iv)** computation vs communication, and **(v)** memory and hardware constraints. A large theoretical scaled speedup is an upper-bound model, not a hardware guarantee.

## Essential mathematics

| Model | Formula | Experiment |
|---|---|---|
| Amdahl fixed size | $S_A(P)=1/[s_A+(1-s_A)/P]$ | $W(P)=W_0$, $s_A$ measured on one core |
| Gustafson scaled | $S_G(P)=P-\alpha(P-1)$ | $T_P(W(P))\approx T^*$, $\alpha$ measured on the scaled parallel run |
| Gustafson efficiency | $E_G(P)=1-\alpha+\alpha/P$ | Constant $\alpha$ idealization yields limiting efficiency $1-\alpha$ |
| Communication-aware | $S=(A+B)/(A+B/P+C)$ | Same scaled useful work $A+B$; $C$ is nonoverlapped parallel communication |
| Isoefficiency | $E=W/[W+T_o(P,W)]$ | Keep $W$ large enough relative to parallel overhead |
| Memory-bounded | $S_M=(A+h(P)B)/(A+h(P)B/P)$ | Scale useful work with memory via $h(P)=g(Pm)/g(m)$ |

## Contents and task-by-task index

All required files are included **at their specified paths**, with additional reproducibility artifacts where helpful.

| Task | Topic | Answer / submitted deliverable |
|---:|---|---|
| 01 | Strong vs. weak foundations | [`theory/task01_scaling_models.md`](theory/task01_scaling_models.md) |
| 02 | Scaled speedup derivation | [`theory/task02_derivation.md`](theory/task02_derivation.md) |
| 03 | Serial fraction reference frame | [`theory/task03_serial_invariance.md`](theory/task03_serial_invariance.md) |
| 04 | Asymptotic limits | [`theory/task04_asymptotic_limits.md`](theory/task04_asymptotic_limits.md) |
| 05 | Historical 1024-node nCUBE | [`theory/task05_historical_ncube.md`](theory/task05_historical_ncube.md) |
| 06 | Parametric table $\alpha=.04$ | [`calculations/task06_parametric_table.md`](calculations/task06_parametric_table.md) |
| 07 | Maximum $\alpha$ for $P=8192$ | [`calculations/task07_tolerable_serial.md`](calculations/task07_tolerable_serial.md) |
| 08 | Amdahl vs Gustafson comparison | [`calculations/task08_cross_divergence.md`](calculations/task08_cross_divergence.md) |
| 09 | Multi-phase workload | [`calculations/task09_multicomponent.md`](calculations/task09_multicomponent.md) |
| 10 | 3-D grid memory ceilings | [`calculations/task10_memory_growth.md`](calculations/task10_memory_growth.md) |
| 11 | Sun–Ni memory-bounded scaling | [`theory/task11_sun_ni_comparison.md`](theory/task11_sun_ni_comparison.md) |
| 12 | Communication-penalty model | [`theory/task12_communication_penalty.md`](theory/task12_communication_penalty.md) |
| 13 | Counterexamples | [`theory/task13_failure_cases.md`](theory/task13_failure_cases.md) |
| 14 | Isoefficiency proof | [`theory/task14_isoefficiency.md`](theory/task14_isoefficiency.md) |
| 15 | Energy & power scaling | [`theory/task15_energy_scaling.md`](theory/task15_energy_scaling.md) |
| 16 | Dense GEMM dimension scaling | [`calculations/task16_gemm_scaling.md`](calculations/task16_gemm_scaling.md) |
| 17 | 3-D stencil communication | [`calculations/task17_stencil_scaling.md`](calculations/task17_stencil_scaling.md) |
| 18 | Direct vs Barnes–Hut N-body | [`analysis/task18_nbody_scaling.md`](analysis/task18_nbody_scaling.md) |
| 19 | Monte Carlo reduction | [`analysis/task19_monte_carlo.md`](analysis/task19_monte_carlo.md) |
| 20 | BFS/PageRank graph scaling | [`analysis/task20_graph_processing.md`](analysis/task20_graph_processing.md) |
| 21 | Python parametric simulator | [`benchmarks/simulator.py`](benchmarks/simulator.py); [`simulated_scaling.csv`](benchmarks/simulated_scaling.csv); [`analysis/task21_simulator_validation.md`](analysis/task21_simulator_validation.md) |
| 22 | Real OpenMP benchmark harness | [`benchmarks/weak_scale_bench/bench.cpp`](benchmarks/weak_scale_bench/bench.cpp); [`run_benchmarks.py`](benchmarks/weak_scale_bench/run_benchmarks.py); [run instructions](benchmarks/weak_scale_bench/README.md) |
| 23 | 25 raw trials and regression | [`benchmarks/results.csv`](benchmarks/results.csv); [`fit_gustafson.py`](benchmarks/fit_gustafson.py); [`aggregate.csv`](benchmarks/aggregate.csv); [`fit_summary.json`](benchmarks/fit_summary.json); [analysis](analysis/task23_empirical_regression.md) |
| 24 | Publication-quality plots and audit | [`plots/scaling_comparison.pdf`](plots/scaling_comparison.pdf) / [PNG](plots/scaling_comparison.png) / [SVG](plots/scaling_comparison.svg); [`analysis/task24_hardware_audit.md`](analysis/task24_hardware_audit.md); [`hardware_audit.json`](analysis/hardware_audit.json); [`plot_comparison.py`](benchmarks/plot_comparison.py) |
| 25 | Executive master documentation | **This `README.md`** |

Additional diagrams: `plots/simulator_overlay.png/.pdf` and `plots/efficiency_comparison.png/.pdf`. Automated checks are in `tests/test_formulas.py`. Primary sources are cited inside relevant solutions; all reported experiment figures are locally produced, not fabricated.

## Reproduce the scripts and benchmarks

Tested with Python 3, `g++` 14 with OpenMP, NumPy and Matplotlib. From the **root directory**:

```bash
python3 -m pip install -r requirements.txt

# Task 21 — generate ideal model data and overlay figures
python3 benchmarks/simulator.py

# Task 22 — compile OpenMP and execute 5 independent runs at P=1..5
python3 benchmarks/weak_scale_bench/run_benchmarks.py \
  --threads 1,2,3,4,5 --runs 5 --work-per-thread 12000000

# Task 23 — aggregate, calculate standard errors, fit alpha by OLS
python3 benchmarks/fit_gustafson.py

# Task 24 — audit CPU/caches/quota and create raster/vector figures
python3 benchmarks/audit_hardware.py
python3 benchmarks/plot_comparison.py

# Verify numerical identities, dataset shape and exported figures
python3 -m unittest discover -s tests -v
```

**Hardware adaptation:** choose a subset of available thread counts if your environment has fewer than five affinity-visible processors. To reproduce the exact submitted 25-row format, use five counts and five runs; when fewer than five are visible on your device, execute in an appropriately provisioned environment. Without an explicit thread list, the script automatically samples from one CPU to its local affinity maximum. Do not add artificial trials to satisfy the count.

### Rebuild the consolidated PDF

The optional PDF builder requires [Pandoc](https://pandoc.org/) and a LaTeX installation with XeLaTeX and the **Noto Serif**, **Inter**, and **DejaVu Sans Mono** fonts; the executable calculations themselves do not require LaTeX. From the root directory:

```bash
python3 tools/build_report.py
```

This collects the full Task 01–25 Markdown solutions into `Complete_Solutions.pdf`, with a navigable table of contents, exact equations, readable tables, and measured plots. Any change to the Markdown answers should be followed by rebuilding this PDF.

The OpenMP benchmark measures scaled speedup correctly: the **single-core reference works on the same scaled $Pk$ samples** as the $P$-core run. It checks numerical equivalence in each execution and records raw timings rather than back-calculating them from ideal models. The compiled binary is intentionally untracked and rebuilt by the harness.

## Actual experiment and its limits

This project was measured in a Linux container with **five visible CPU IDs but a quota corresponding to four CPU cores**. There are five independent trials for each of five configurations; the empirical speedup data is noisy and includes an extreme $P=5$ outlier. The constant-$\alpha$ least-squares fit returns **$\hat\alpha\approx0.7304$** with a very poor model fit ($R^2\approx-0.0072$). That coefficient should **not** be interpreted as a causal property of the mathematical kernel. No reliable thermal-throttle telemetry, physical memory-channel count, or cross-NUMA experiment was available. Read [Task 23](analysis/task23_empirical_regression.md) and [Task 24](analysis/task24_hardware_audit.md) before comparing the figure with production cluster data.

## When to use each model

- **Choose Amdahl/strong scaling** for fixed-demand service latency, a fixed mesh, a fixed query, or a fixed input file when additional cores do *not* increase the scientific task.
- **Choose Gustafson/weak scaling** for simulations with meaningful higher resolution, independent Monte Carlo paths, or larger regular batch-processing jobs when constant local work and sufficient memory/network resources are plausible.
- **Choose Sun–Ni memory-bounded scaling** when a larger problem becomes possible specifically because more aggregate memory is available, even if it requires a longer runtime.
- **Use isoefficiency + profiling** when synchronization, communication, load imbalance, algorithmic work inflation, or memory capacity threaten the ideal scaling schedule.

Engineering rule: report both **scaled throughput** and **time-to-solution** where possible; include parallel efficiency, energy per result, and a clear description of scientific accuracy and dataset characteristics. Processor count alone never identifies quality of scaling.

## Primary references

1. Gene M. Amdahl, [“Validity of the single processor approach to achieving large scale computing capabilities,” AFIPS (1967), 483–485](https://doi.org/10.1145/1465482.1465560).
2. John L. Gustafson, [“Reevaluating Amdahl's Law,” *Communications of the ACM* 31(5), 532–533 (1988)](https://doi.org/10.1145/42411.42415).
3. John L. Gustafson, Gary R. Montry, Robert E. Benner, [“Development of Parallel Methods for a 1024-Processor Hypercube,” *SIAM J. Sci. Stat. Comput.* 9(4), 609–638 (1988)](https://doi.org/10.1137/0909041).
4. Xian-He Sun and Lionel M. Ni, [“Scalable Problems and Memory-Bounded Speedup,” *J. Parallel Distrib. Comput.* 19(1), 27–37 (1993)](https://doi.org/10.1006/jpdc.1993.1087).

## Submission checklist

- [x] All 25 tasks represented by solution files or executable code, with derivations and arithmetic shown.
- [x] Git-ready repository organized into `theory/`, `calculations/`, `benchmarks/`, `analysis/`, and `plots/`.
- [x] Parametric simulator with bounds, CSV, and overlay plot.
- [x] OpenMP benchmark, five thread counts, five independent real runs per count, and numerical correctness checks.
- [x] Least-squares fit, raw and aggregate CSV, standard errors, and fitted/ideal/Amdahl figures.
- [x] Hardware audit with truthful unknowns for unobservable telemetry.
- [x] Eight automated checks passing in this environment.
- [x] Local Git commits prepared for repository import.
- [ ] **Student action:** personalize repository ownership and `git remote`, review the analysis, and push to the designated personal Git repository.

**Academic integrity / reproducibility:** if rerunning on another machine, the measured values and hardware audit will differ. Keep an unmodified copy of each actual raw run and regenerate Task 23/24 reports to match your environment. Do not submit these container measurements as if you personally collected them on a different computer.
