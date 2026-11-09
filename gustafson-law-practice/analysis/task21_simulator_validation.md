# Task 21 — Parametric Speedup Simulator Implementation

## Deliverables

The executable Python program [`benchmarks/simulator.py`](../benchmarks/simulator.py) generates two forms of output:

1. [`benchmarks/simulated_scaling.csv`](../benchmarks/simulated_scaling.csv), with processor count, serial fraction, Amdahl speedup, Gustafson scaled speedup, and Gustafson parallel efficiency.
2. [`plots/simulator_overlay.png`](../plots/simulator_overlay.png) and [`plots/simulator_overlay.pdf`](../plots/simulator_overlay.pdf), overlaying ideal, strong-scaling, and weak-scaling curves.

The command line accepts arbitrary comma-separated integer processor counts $1\le P\le65{,}536$ and floating serial fractions $0.001\le\alpha\le0.20$. It rejects out-of-range and malformed inputs instead of silently clipping them. This makes the simulator suitable for both a few hand-picked configurations and the maximum assigned range.

## Equations encoded

$$S_A(P;s)=\frac{1}{s+(1-s)/P},\qquad S_G(P;\alpha)=P-\alpha(P-1),\qquad E_G(P;\alpha)=S_G(P;\alpha)/P.$$

The same **numeric** fraction is used as $s$ and $\alpha$ in the comparison plot; as established in Task 08, these symbols belong to different workload frames. Plots therefore compare model curves rather than claiming they predict the same physical benchmark.

## Reproduce the default data and figure

```bash
python3 benchmarks/simulator.py
```

Try an arbitrary valid subset:

```bash
python3 benchmarks/simulator.py \
  --core-counts 1,3,17,256,65536 \
  --alphas 0.001,0.075,0.20 \
  --csv benchmarks/custom_simulated_scaling.csv \
  --plot plots/custom_overlay.png
```

The output CSV contains $17\times4=68$ rows for the shipped default sweep (17 different $P$ values and 4 fractions). The program writes both PNG and vector PDF plot for the same run. Plots use a log2 processor axis and log10 speedup axis to display the full range up to 65,536 without squeezing the small-$P$ behavior.

## Verification criteria

- At $P=1$, both models equal 1 for all valid fractions.
- For $\alpha=0.001$ and $P=65,536$, Gustafson predicts $65,470.465$; Amdahl predicts $1/(.001+.999/65536)\approx984.985$.
- All result rows satisfy $1\le S_G\le P$, $0<E_G\le1$, and $1\le S_A\le P$ in the parameter range.
- Automated formula boundary tests are included in `tests/test_formulas.py`.

**Quality note:** the simulator implements **ideal formulas**, not empirical capacity or network constraints. Benchmarks Tasks 22–24 analyze real deviations.
