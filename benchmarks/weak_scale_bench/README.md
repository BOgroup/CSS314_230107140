# OpenMP weak-scaling benchmark (Task 22)

## Design

The benchmark computes a deterministic numerical quadrature with **constant work per thread** `k`. At thread count `P`, the measured job contains `P*k` samples. The C++ executable measures both `T1(P*k)` (a single thread processes all P partitions) and `TP(P*k)` (an OpenMP team of P threads processes the same P partitions). These times share exactly the same scaled workload, so `S_scaled = T1(P*k) / TP(P*k)` is a valid scaled-speedup estimate.

The checksum verifies equivalence to relative error below `1e-10`. Threads perform independent math and synchronize only for the OpenMP reduction. The benchmark excludes process launch from timed sections but **includes** team creation, join, and reduction in the parallel timed section. Per-config timings cover compute only, not file I/O.

## Run

```bash
cd gustafson-law-practice
python3 benchmarks/weak_scale_bench/run_benchmarks.py --runs 5 --work-per-thread 2000000
python3 benchmarks/fit_gustafson.py
python3 benchmarks/plot_comparison.py
```

Requirements: Python 3, `g++` with OpenMP support, and the Python libraries in `requirements.txt`. `run_benchmarks.py` compiles `bench.cpp` into an untracked local executable. `--threads 1,2,3,4,5` specifies an explicit sweep, constrained by the host's available processor affinity; `auto` (default) includes 1 and the maximum CPU affinity count and uses all intermediate counts when at most 16 are available. The RNG seed controls **only execution order**, not the deterministic compute kernel. The script shuffles configuration order within each repetition to reduce sequential thermal bias.

The total dataset is `number_of_thread_counts × number_of_runs`; this submission includes five independent repetitions at each of five thread counts. Results are one experimental snapshot, not universal hardware performance claims. In a VM or container the CPU quota, competing tenants, and unseen actual memory channels can alter speedup.
