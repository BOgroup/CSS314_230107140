# Task 23 — Experimental Measurements and Least-Squares Fit

## Experiment actually executed

The submitted [`benchmarks/results.csv`](../benchmarks/results.csv) contains **25 raw trials**: **five independent executions** for each of the **five distinct OpenMP thread counts** $P\in\{1,2,3,4,5\}$. Each worker performs $k=12{,}000{,}000$ deterministic quadrature samples; total samples for a run scale as $W(P)=kP$. Each trial independently measures

$$T_1(kP)\quad\text{and}\quad T_P(kP),$$

using the **same total workload**. Therefore the empirical scaled speedup is $S_{\mathrm{obs}}(P)=T_1(kP)/T_P(kP)$, not the strong-scaling speedup of one constant-sized job. Every trial includes two high-resolution timings, a numerical checksum comparison, and the run ID. The trial ordering was shuffled inside each repetition; all 25 raw observations, including slow outliers, are retained.

## Sample statistics (from the actual run)

| Threads | Trials | Mean serial ref. $T_1(kP)$, s | Mean parallel $T_P(kP)$, s | Mean observed $S$ | Sample std. of $S$ | Std. error of $S$ | Mean efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 0.06991 | 0.06699 | 1.04957 | 0.18391 | 0.08225 | 1.04957 |
| 2 | 5 | 0.14239 | 0.08054 | 1.77157 | 0.23324 | 0.10431 | 0.88579 |
| 3 | 5 | 0.24355 | 0.13430 | 1.97922 | 0.95621 | 0.42763 | 0.65974 |
| 4 | 5 | 0.26848 | 0.16034 | 1.68781 | 0.19240 | 0.08605 | 0.42195 |
| 5 | 5 | 0.38780 | 0.33772 | 1.82375 | 0.76730 | 0.34315 | 0.36475 |

For each processor count, sample standard deviation uses divisor $n-1$ and standard error uses $s/\sqrt n$ with $n=5$. The reported *mean of per-trial speedup ratios* is not generally equal to the ratio of aggregate mean times. Small deviations such as the $P=1$ mean speedup of 1.05 are measurement noise, not a claim that one worker intrinsically exceeds one worker.

## Explicit least-squares formulation

Gustafson's ideal model is $S(P)=P-\alpha(P-1)$. Rewrite as

$$y_i=P_i-S_{\mathrm{obs},i},\qquad x_i=P_i-1,\qquad y_i\approx\alpha x_i.$$

The **unweighted OLS** coefficient through the theoretical origin, using **all raw trial rows with $P>1$**, is

$$\boxed{\hat\alpha=\frac{\sum_i x_i y_i}{\sum_i x_i^2}=0.7303855508}.$$

Its physically constrained counterpart $\min(1,\max(0,\hat\alpha))$ is also **0.7303855508**. A stratified nonparametric bootstrap that resamples the five trials separately within each processor count (2,000 replicates, fixed seed) gives an indicative 95% interval approximately **[0.63898, 0.83589]** for the raw coefficient.

## Model-fit diagnostics and interpretation

The fitted unweighted speedup RMSE is **0.6050**, and the conventional speedup-space $R^2$ is **-0.0072**. A negative $R^2$ signals that a single constant-$\alpha$ line fits these noisy measurements poorly. The effective fitted $\alpha$ is **not** evidence that 73% of the numerical quadrature algorithm is inherently sequential. The kernel is largely data parallel; the fit absorbs runtime overhead, VM scheduling, core-quota limits, and benchmark-reference noise into one poor-fit phenomenological parameter.

The fifth trial at $P=5$ has a parallel time about **1.019 s**, versus roughly 0.150–0.183 s for the other four $P=5$ trials; it is kept because discarding it without a predefined exclusion rule would bias the study. Host affinity advertises five CPUs but the cgroup has a four-core CPU quota; these are not five unrestricted physical cores. Shared-system interference and unobserved power management are possible contributors, not confirmed causes.

**Limitations:** five trials per count are enough for the assignment but insufficient to characterize noisy virtual infrastructure robustly. The bootstrap interval conditions on this small trial collection; it does not capture hidden system-level biases or prove a constant causal serial fraction. Re-run the harness on a dedicated, frequency-controlled host for stronger inference.

**Reproduce:**

```bash
python3 benchmarks/weak_scale_bench/run_benchmarks.py --runs 5 --work-per-thread 12000000
python3 benchmarks/fit_gustafson.py
```
