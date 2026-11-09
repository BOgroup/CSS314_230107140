# Task 11 — Gustafson vs. Sun–Ni Memory-Bounded Speedup

## Three different resource constraints

1. **Amdahl fixed-size:** $W(P)=W_0$; available processors increase but required result does not.
2. **Gustafson fixed-time:** choose $W(P)$ so $T_P(W(P))\approx T^*$; in the ideal two-phase model, $S_G(P)=P-\alpha(P-1)$.
3. **Sun–Ni memory-bounded:** choose the largest useful problem whose data fit into aggregated memory $M(P)$, even if runtime changes with $P$. With $m$ bytes accessible per node, $M(P)\le Pm$ and $W^*(P)=g(M(P))$, where $g$ maps memory footprint to work.

Sun and Ni's 1993 work explicitly treats memory capacity as another scalability constraint and includes the fixed-size and fixed-time models as special cases under simplified assumptions.

## Mathematical formulation

Let $A$ denote constant sequential work (in time units) and $B$ the parallelizable work at the baseline memory footprint $m$. Write the scaled parallel work as $B(P)=h(P)B$, where for a semihomogeneous memory–work map $g$ one may take

$$h(P)=\frac{g(Pm)}{g(m)}.$$

With perfect distribution and no communication, memory-bounded speedup on the **larger job** is

$$\boxed{S_M(P)=\frac{A+h(P)B}{A+h(P)B/P}}.$$

Let $s_0=A/(A+B)$, the baseline one-core fraction. Dividing by $A+B$ gives

$$S_M(P)=\frac{s_0+h(P)(1-s_0)}{s_0+h(P)(1-s_0)/P}.$$

- If $h(P)=1$, this reduces to **Amdahl** for the fixed workload.
- If $h(P)=P$, the denominator is $s_0+1-s_0=1$ and $S_M(P)=s_0+P(1-s_0)$: the **Gustafson** fixed-time form with $\alpha=s_0$ for this particular scaled schedule.
- If $g(M)\propto M^b$, then $h(P)=P^b$. For a dense $N\times N$ product, $M\propto N^2$ and work $\propto N^3$, so $b=3/2$; using all aggregate memory could make useful work grow as $P^{3/2}$ and compute time as $P^{1/2}$ rather than remain fixed.

A realistic generalization includes communication and imbalance:

$$S_M^{\mathrm{real}}(P)=\frac{A+h(P)B}{A+h(P)B/P+C(P)+I(P)}.$$

Here $C(P)$ and $I(P)$ are *time* costs, not improvements in useful work. Memory capacity is only an upper bound: inadequate network bandwidth or inaccessible remote memory can reduce usable $M(P)$.

## When Gustafson is too optimistic or too conservative

**Overestimates achievable throughput when** (i) all memory cannot be accessed efficiently, (ii) communication grows with domain size, (iii) aggregate memory capacity is exhausted before constant-runtime work grows to $P$ times baseline, (iv) dependencies or load imbalance increase, or (v) the best serial algorithm does fewer operations than the parallel implementation. Example: if $h(P)=P^{1/2}$ under a memory restriction and $A>0$, the serialized portion has a greater relative effect than an assumed $h(P)=P$ would suggest.

**Underestimates useful work that can be performed if the time budget may expand** and memory enables superlinear growth of operations relative to memory ($h(P)>P$): dense linear algebra can perform $\Theta(P^{3/2})$ arithmetic when matrix memory scales as $P$. This does **not** imply superlinear *speedup of the same algorithm in the same time*; the memory-bounded run may take longer. Another apparent underestimate arises if larger working sets improve throughput or the serial component shrinks, violating constant-$\alpha$ assumptions.

**Key distinction:** a *speedup ratio*, a *quantity of useful work*, and *throughput per fixed time* are different outcomes. Models are not directly rankable without declaring which constraint and numerator are held fixed.

**Primary reference:** X.-H. Sun and L. Ni, [“Scalable Problems and Memory-Bounded Speedup,” *Journal of Parallel and Distributed Computing* 19(1), 27–37 (1993)](https://doi.org/10.1006/jpdc.1993.1087). [NASA preprint record](https://ntrs.nasa.gov/citations/19930007365).
