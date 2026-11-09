# Task 01 — Strong Scaling vs. Weak Scaling Foundations

## Definitions and experimental question

**Strong scaling (Amdahl, 1967)** asks: *How much faster can a **fixed** workload finish as we add processors?* Let a reference job require one-core time $T_1$, of which proportion $s_A\in[0,1]$ is inherently sequential. Assuming identical processor speeds, no scheduling overhead, and ideal distribution of parallel work,

$$T_P(W_0)=T_1(W_0)\left(s_A+\frac{1-s_A}{P}\right),\qquad S_A(P)=\frac{1}{s_A+(1-s_A)/P}.$$

Here $W(P)=W_0$ **for every $P$**. If $s_A>0$, then $T_P\to s_A T_1$ and $S_A(P)\to1/s_A$ as $P\to\infty$. In a geometric problem the physical or numerical dimension $N$ is unchanged; the subdomain assigned to each worker shrinks. Eventually, communication, minimum grain size, and launch overhead often invalidate the ideal formula even sooner.

**Weak/scaled scaling (Gustafson, 1988)** asks: *How much **more** work can finish in about the same time when we add processors?* Normalize the observed runtime of a scaled problem on $P$ processors to $T_P(W(P))=s_G+p_G=1$; $s_G$ and $p_G$ represent fractions of **parallel-machine elapsed time**, not the fixed-job one-core fractions. If $p_G$ can be multiplied by $P$ to reconstruct sequential work, then

$$S_G(P)=\frac{s_G+P p_G}{s_G+p_G}=s_G+Pp_G=P-s_G(P-1).$$

For an ideally divisible kernel, total operation count satisfies $W(P)=\Theta(P)$ and average work per core $W(P)/P=\Theta(1)$. The **dimension** scaling is determined by complexity: for a 3-D grid with $W=\Theta(N^3)$, $N(P)=\Theta(P^{1/3})$; for a dense matrix multiplication with $W=\Theta(N^3)$, the same cubic-root rule applies even though memory is $\Theta(N^2)$. It is generally **incorrect** to claim that each linear dimension must grow as $P$.

## Limiting behavior and why the laws are not contradictory

| Dimension | Amdahl/strong | Gustafson/weak |
|---|---|---|
| Controlled variable | Total work $W_0$ | Approximate wall-clock budget $T^*$ |
| $P\to\infty$ | Same global $N$, smaller local work | Increasing global $N(P)$, approximately constant local work |
| Serial fraction | $s_A$ on the original 1-core workload | $\alpha=s_G$ on the scaled $P$-core run |
| Question | Lower latency for one fixed result | More fidelity, samples, or throughput per time budget |
| Ideal speedup | Saturates at $1/s_A$ if $s_A>0$ | $P-\alpha(P-1)$ for constant $\alpha$ |

There is **no physical contradiction**: these are different experiments, with different workloads and different denominators for the serial fraction. If $W$ cannot expand meaningfully (e.g., a fixed-size interactive request), Amdahl's model is the appropriate planning tool. If a larger mesh, more simulations, or larger datasets produce extra scientific value and fit within capacity/time constraints, weak scaling is appropriate.

**Important boundary:** the word *fixed-time* describes the model's design objective, **not** a promise that production runtimes stay constant indefinitely. Communication, imbalance, memory bandwidth, and global reductions can make $T_P(W(P))$ grow.

**Sources:** [Amdahl (1967)](https://doi.org/10.1145/1465482.1465560); [Gustafson (1988), *Reevaluating Amdahl's Law*](https://doi.org/10.1145/42411.42415); [Gustafson, Montry & Benner (1988)](https://doi.org/10.1137/0909041).
