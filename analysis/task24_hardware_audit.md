# Task 24 — Publication Plots and Hardware Audit

## Reproducible figures

The script [`benchmarks/plot_comparison.py`](../benchmarks/plot_comparison.py) reads [`benchmarks/aggregate.csv`](../benchmarks/aggregate.csv) and [`benchmarks/fit_summary.json`](../benchmarks/fit_summary.json) to produce:

- [`plots/scaling_comparison.png`](../plots/scaling_comparison.png) — 260-dpi raster, and matching **vector** [`plots/scaling_comparison.pdf`](../plots/scaling_comparison.pdf) and `.svg`.
- [`plots/efficiency_comparison.png`](../plots/efficiency_comparison.png) and vector PDF — empirical efficiency and fitted efficiency.

The main speedup plot contains all four requested traces: **(1)** ideal $S=P$, **(2)** Gustafson $S=P-\hat\alpha(P-1)$, **(3)** empirical measured mean $S$ with **standard-error bars**, **(4)** Amdahl $S=1/[\hat\alpha+(1-\hat\alpha)/P]$. Using the same numerical coefficient in both laws is illustrative; their serial shares have different definitions.

## Actual system audited

| Property | Observation | Confidence or limitation |
|---|---|---|
| CPU model | **Intel(R) Xeon(R) Platinum 8573C** | Visible vCPU model exposed by Linux `/proc/cpuinfo` |
| Visible logical processors | **5**, affinity CPUs `0–4` | Five runnable OS-visible CPUs, not proof of five dedicated physical cores |
| Cgroup quota | **400000 / 100000** microseconds = **4.0 CPU cores** | Important limit; five active threads can exceed the combined CPU allowance |
| OS / architecture | Linux 6.18.57, x86-64, virtualized environment | Host scheduling is not under experimental control |
| L1 data cache | **48 KiB** for the CPU0 cache entry, shared list `0–1` | Observed sysfs `cpu0/cache/index0`, not an assertion of physical device topology |
| L1 instruction cache | **32 KiB**, CPU0 entry | Observed sysfs cache entry |
| L2 unified cache | **2 MiB**, CPU0 entry | Observed sysfs cache entry |
| L3 unified cache | **260 MiB**, visible shared list `0–4` | Virtualized visible topology; actual host placement may differ |
| NUMA | **1 NUMA node visible (node 0)** | No NUMA crossing can be demonstrated on this exposure |
| Memory channels | **Not disclosed to unprivileged environment** | **Unknown**, no invented physical-channel count |
| Temperature and clock throttling | **No direct telemetry collected** | Cannot attribute a slowdown to throttling with confidence |
| Compiler | **g++ 14.2.0**, OpenMP with `-O3` | Recorded audit and reproducible build flags |

Detailed machine-readable evidence, including the full `lscpu` output, CPU affinity, cache index attributes, cgroup quota, and audit UTC timestamp, is in [`analysis/hardware_audit.json`](hardware_audit.json). Regenerate it with `python3 benchmarks/audit_hardware.py`.

## Explain the observed departure from ideal scaling

The empirical speedups in Task 23 fail to follow $P$ or a clean constant-$\alpha$ model. **Directly supported facts** are: the four-core CPU quota despite five visible CPU IDs; substantial between-trial variance; a slow $P=5$ run; and a poor $R^2$ for the constant-$\alpha$ regression. This is consistent with CPU-share contention, virtualization jitter, affinity/scheduling overhead, and OpenMP synchronization.

**Not established:** thermal throttling (no validated clock/temperature time series), memory-channel contention (channel count not available), and NUMA-boundary crossing (only one node exposed). These are plausible general causes of parallel-performance deviation, but they should **not** be asserted as measured causes for this dataset. The numerical kernel is compute-oriented with small memory demand, so its results do not generalize to cache- or bandwidth-intensive production codes.

## Verification and rerun

```bash
python3 benchmarks/audit_hardware.py
python3 benchmarks/fit_gustafson.py
python3 benchmarks/plot_comparison.py
```

To obtain publication-grade evidence rather than just publication-quality **figures**, repeat on a reserved machine, record per-core effective frequency and throttling counters, measure memory bandwidth with an appropriate tool, pin OpenMP threads, predeclare any exclusion rule, and extend the number of repetitions. The submitted figures correctly describe **this measured container snapshot**.
