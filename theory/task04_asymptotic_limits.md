# Task 04 — Asymptotic Analysis of Parallel Efficiency

Let $P$ be a positive integer and hold the modeled Gustafson serial fraction $0\le\alpha\le1$ constant:

$$E_G(P)=\frac{P-\alpha(P-1)}{P}=1-\alpha+\frac{\alpha}{P}.$$

Because $\lim_{P\to\infty}\alpha/P=0$ for fixed finite $\alpha$,

$$\boxed{\lim_{P\to\infty}E_G(P)=1-\alpha}.$$

For $\alpha=.04$, the asymptotic parallel efficiency is $0.96$ (96%). The derivative of a real-valued relaxation, $E_G'(P)=-\alpha/P^2$, is nonpositive; efficiency decreases monotonically from $E_G(1)=1$ to its positive asymptote. The decrement from $P$ to $2P$ is $\alpha/(2P)$, hence falls rapidly at large $P$.

For a fixed **one-core** serial fraction $0<s_A\le1$, Amdahl gives

$$S_A(P)=\frac{1}{s_A+(1-s_A)/P}=\frac{P}{s_AP+1-s_A}.$$

Consequently,

$$E_A(P)=\frac{S_A(P)}{P}=\frac{1}{s_AP+1-s_A}\xrightarrow[P\to\infty]{}0.$$

The limit follows because the denominator diverges linearly when $s_A>0$. For the exceptional all-parallel case $s_A=0$, $E_A(P)=1$ at every $P$. Likewise, the Gustafson edge cases are $\alpha=0\implies E_G=1$ and $\alpha=1\implies E_G=1/P\to0$.

| Case | Scaled efficiency limit | Strong efficiency limit |
|---|---:|---:|
| Nonzero serial share, both symbols numerically .04 | 0.96 | 0 |
| Zero serial share | 1 | 1 |
| Fully serial | 0 | 0 |

**Why the limits differ:** Amdahl spreads unchanged work over more processors; Gustafson increases aggregate work with processor count while preserving a measured time composition. These do **not** concern the same sequence of jobs.

**Caveat on a changing fraction:** if $\alpha=\alpha(P)$ instead, then $E_G(P)=1-\alpha(P)+\alpha(P)/P$. A limit exists only with appropriate behavior of $\alpha(P)$ (for example, $\alpha(P)\to a$ implies $E_G(P)\to1-a$). No positive-efficiency limit is guaranteed in a real distributed system whose communication overhead grows with $P$.
