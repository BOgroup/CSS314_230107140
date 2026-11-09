# Task 03 — The Serial Fraction Invariance Paradox

## Compare what each fraction actually divides by

Let the serial work take $A>0$ time on a single processor and let the ideally parallel work take $B(P)$ **one-core-equivalent** time. For the scaled job $W(P)$, ignoring communication,

$$T_1(P)=A+B(P),\qquad T_P(P)=A+\frac{B(P)}{P}.$$

The Amdahl-style fraction **for that job on one core** is

$$s_A(P)=\frac{A}{A+B(P)},$$

whereas the Gustafson fraction **measured during its $P$-core run** is

$$\alpha_G(P)=\frac{A}{A+B(P)/P}.$$

For the **same** workload, the two fractions satisfy

$$\boxed{\alpha_G(P)=\frac{P s_A(P)}{1+(P-1)s_A(P)}};\qquad
\boxed{s_A(P)=\frac{\alpha_G(P)}{P-(P-1)\alpha_G(P)}}.$$

Their numeric values differ, sometimes drastically, without changing the physics or violating either speedup law.

## Demonstrate the resolution

Assume weak scaling with $B(P)=P b$ and constant $A,b>0$. The parallel runtime is $A+b$, a constant, and hence

$$\alpha_G(P)=\frac{A}{A+b}\quad\text{(constant)},\qquad s_A(P)=\frac{A}{A+Pb}\xrightarrow[P\to\infty]{}0.$$

The fixed-work serial fraction is **not invariant when the workload itself grows**. In other words, the *same* $A$ is diluted by ever more useful total work in the one-core reconstruction. The Gustafson fraction on the parallel machine can remain unchanged because each node still receives $b$ units of parallel execution time.

For a concrete case, let $A=1$, $b=24$ and $P=1024$. Then $T_P=25$, $\alpha_G=.04$, $T_1=1+1024(24)=24577$, and

$$S=24577/25=983.08,\quad s_A(1024)=1/24577\approx0.00004069.$$

Plugging this **job-specific** $s_A(P)$ into Amdahl's formula gives exactly the same $983.08$, not a contradictory result. Using the fixed small-job value $s_A(1)=1/25=.04$ and treating it as unchanged for the expanded job would incorrectly impose an apparent ceiling of $25$.

## Critical note

It is therefore inaccurate to say that *measuring on the parallel machine alone removes all bottlenecks*. Constant $\alpha_G$ is a simplification; communication or additional coordination can make it rise. Hardware speed, cache effects, algorithmic work inflation, and one-core memory capacity also affect serial reconstruction. A 1024-node workload may be impossible to run on one physical node; its $T_1$ is then an **extrapolated serial equivalent** unless actually measured.

**Interpretation:** the paradox disappears when workload, time baseline, and fraction definition are written explicitly. Strong and weak scaling can both be valid simultaneously.

**References:** [Gustafson et al. (1988)](https://doi.org/10.1137/0909041), section on fixed-size and scaled-size evaluation.
