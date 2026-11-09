# Task 12 — Integrating Communication Overhead

## A time-domain extension

Gustafson's model assumes that all nonserial elapsed time corresponds to useful, perfectly distributable computation. That becomes misleading when workers also spend time communicating. For a scaled problem let

- $A(P)$ = useful sequential computation time,
- $B(P)$ = one-core-equivalent useful parallel computation time,
- $C(P)$ = elapsed, non-overlapped communication and synchronization time on $P$ processors.

Then an explicit extended runtime model is

$$T_P(P)=A(P)+\frac{B(P)}{P}+C(P).$$

Assume that the sequential reference performs the same useful computation but does **not** pay the multi-node network cost: $T_1(P)=A(P)+B(P)$. Measured scaled speedup becomes

$$\boxed{S_{G,C}(P)=\frac{A(P)+B(P)}{A(P)+B(P)/P+C(P)}}.$$

This form is often preferable because the time budget need not be fixed and the communication function need not be small.

## Recovery of Gustafson and a modified closed form

Normalize $T_P(P)=1$ and define $\alpha(P)=A(P)/T_P(P)$, $\gamma(P)=C(P)/T_P(P)$; therefore $B(P)/[P T_P(P)]=1-\alpha(P)-\gamma(P)$. Substituting yields

$$S_{G,C}(P)=\alpha(P)+P[1-\alpha(P)-\gamma(P)]$$
$$\boxed{S_{G,C}(P)=P-\alpha(P)(P-1)-P\gamma(P).}$$

If $C(P)=0$, the original expression is recovered. The penalty term $P\gamma(P)$ represents lost useful-work-equivalent speedup. **Do not** simply add $C(P)$ to a fraction already normalized as $s+p=1$ without renormalizing.

## Network cost examples

For a tree-based global collective with $k$ synchronization rounds,

$$C_{\mathrm{tree}}(P)=k\lambda\log_2 P+\frac{V(P)}{B_{\mathrm{net}}(P)},$$

where $\lambda$ is message-latency per tree stage and $V/B_{\mathrm{net}}$ a nonoverlapped effective bandwidth time. On a 2-D mesh, a diameter-sensitive exchange may have

$$C_{\mathrm{mesh}}(P)=c\sqrt P+\text{bandwidth term}.$$

The Big-O forms $O(\log P)$ and $O(P^{1/2})$ are examples of **network/algorithm combinations**, not universal laws for all communication on those topologies. Constant communication volume per worker may produce different costs, and overlapping data transfer with computation reduces the *nonoverlapped* $C(P)$.

## Fixed-time feasibility bound

With target $T^*=1$ and a fixed sequential component $A=a$, useful work that fits the budget is

$$B(P)=P[1-a-C(P)].$$

The implied useful workload is nonnegative only when $a+C(P)\le1$. If $C(P)=c\log_2P$ with $c>0$, this fails beyond $P>2^{(1-a)/c}$. Thus **constant wall-clock time cannot continue indefinitely** under an unbounded communication penalty.

Efficiency becomes $E(P)=S(P)/P=1-\alpha(P)+\alpha(P)/P-\gamma(P)$. This shows why it is possible to observe improving worker utilization but decreasing effective speedup when collective latency, contention, or redundant work increases.

**Use:** fit $A$, $B$, and $C$ separately from timed traces; avoid assigning all communication to inherently serial algorithms when parallel overlap is available.
