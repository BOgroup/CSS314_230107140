# Task 19 — Monte Carlo: Near-Ideal Weak Scaling with Nonzero Overhead

## Ideal embarrassingly parallel case

Suppose $P$ workers each draw $k$ independent samples and evaluate a fixed-cost integrand. Useful serial-equivalent work scales as $W(P)\propto kP$, while per-worker computation takes approximately $\tau(k)$ seconds. If there is no cross-worker interaction and no fixed setup, then $T_P\approx\tau$, $T_1\approx P\tau$, $S(P)\approx P$, and $E(P)\approx1$.

Real Monte Carlo systems still contain overhead, even when sample evaluations are mutually independent:

- **PRNG seed/stream management:** central generation or validation, stream partitioning, or accidental serial initialization. Correct independent streams are important for statistical validity; assigning each worker an arbitrary unvalidated seed is not a guarantee of independence.
- **Reduction operations:** combining partial sums, variances, or histograms and synchronizing workers.
- **Output/checkpoint serialization:** logging, writing one file, serial aggregation, metadata updates, and reproducibility bookkeeping.
- **Load imbalance:** rejection sampling, adaptive paths, variable path lengths, and heterogeneous execution rates.

## Quantify tree-reduction latency and its effect on speedup

Let $a$ be constant serial setup/output seconds and $b$ be nonoverlapped seconds per reduction-tree level. A balanced binary reduction has $\lceil\log_2P\rceil$ stages, so a simplified runtime is

$$T_P(kP)=\tau(k)+a+b\lceil\log_2P\rceil.$$

If serial execution of all $kP$ samples takes $P\tau(k)$ and incurs no equivalent network reduction, the useful scaled speedup is

$$\boxed{S_{MC}(P)=\frac{P\tau(k)}{\tau(k)+a+b\lceil\log_2P\rceil}}.$$

The lost factor relative to ideal is $S_{MC}/P=\tau/[\tau+a+b\lceil\log_2P\rceil]$. Thus logarithmic overhead becomes significant when useful compute per worker is small. A naive sequential reduction that sends $P$ messages to one root could cost $O(Pb)$ stages instead of $O(b\log P)$, a potentially catastrophic difference.

## Worked numerical scenario

For illustration, let $P=1024$, $\tau=1.0$ s per worker, $a=.001$ s, and $b=.0001$ s per level. Since $\log_2(1024)=10$,

$$T_P=1.000+.001+.001=1.002\ \mathrm{s},$$
$$S_{MC}=1024/1.002\approx\boxed{1021.9561},\qquad E\approx0.99800399.$$

The observed setup-plus-reduction runtime fraction is $\gamma=.002/1.002\approx0.0019960$. The generalized overhead-aware formulation from Task 12 gives $S\approx P(1-\gamma)$ when all useful computation is parallel and the serial equivalent excludes the network overhead; this reproduces the example. If the same engine were run with $\tau=.01$ s, its overhead would be 20% of the useful compute and efficiency would fall to $0.01/(.012)\approx83.33\%$ despite an unchanged tree.

## Statistical correctness is not implied by speedup

Monte Carlo speedup measurements must preserve sampling distribution, counter-based streams or jump-ahead guarantees, deterministic reduction order if bitwise repeatability is desired, and uncertainty of the final estimator. Parallel reduction changes floating-point summation order, so exact bitwise equality can fail even when the statistical estimator is valid. Compute speedup without checking statistical correctness is incomplete.
