# Task 16 — Dense Matrix Multiplication (GEMM) Weak Scaling

## Work-derived matrix dimension

For conventional dense multiplication of square $N\times N$ matrices, useful arithmetic is $W(N)=\Theta(N^3)$ while input/output storage is $M(N)=\Theta(N^2)$. Write the leading work $W(N)=cN^3$. Under **ideal constant-time weak scaling** on $P$ identical processors, arithmetic handled by each core stays constant:

$$\frac{cN(P)^3}{P}=cN_0^3\quad\Longrightarrow\quad N(P)^3=PN_0^3.$$

Thus,

$$\boxed{N(P)=N_0P^{1/3}}.$$

The fixed-time computation assumption leads to a **cube-root** increase of matrix side length, *not* $N\propto P$. The result is approximate for integer matrix sizes and assumes that FLOP rate per processor, local arithmetic efficiency, and communication can remain stable.

## Tabular dimension growth targets

For an illustrative one-core baseline $N_0=256$:

| Processors | Multiplicative side factor $P^{1/3}$ | Target $N(P)$ | Aggregate FLOPs relative to baseline | Matrix storage relative to baseline |
|---:|---:|---:|---:|---:|
| 1 | 1 | 256 | 1× | 1× |
| 8 | 2 | 512 | 8× | 4× |
| 64 | 4 | 1,024 | 64× | 16× |
| 512 | 8 | 2,048 | 512× | 64× |
| 4,096 | 16 | 4,096 | 4,096× | 256× |
| 32,768 | 32 | 8,192 | 32,768× | 1,024× |

Since $M(N(P))=\Theta(P^{2/3})$, the ideal **storage per processor** is $M/P=\Theta(P^{-1/3})$. In this regime more processors are added for compute throughput, not to saturate their aggregate available memory. A memory-filling experiment would instead allow $M\propto P$, yielding $N\propto P^{1/2}$ and total GEMM work $\propto P^{3/2}$, so naive compute time $W/P\propto P^{1/2}$: that is Sun–Ni **memory-bounded** scaling, not fixed-time Gustafson scaling.

## Communication qualification

Distributed GEMM uses broadcasts/reductions of matrix blocks and may replicate panels for efficiency. An ideal 2-D process grid's bandwidth cost per processor has a nontrivial dependence on $N$ and $P$; a formula based only on $N^3/P$ omits it. For a real strong-scaling or weak-scaling claim, benchmark total wall time including block redistribution, initialization, and final reduction/checkpointing; report grid arrangement, BLAS library, and per-core memory.
