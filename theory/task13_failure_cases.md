# Task 13 — Concrete Cases Where Weak Scaling Breaks

## Case A — 3-D FFT with distributed transposes

A pencil-decomposed three-dimensional FFT can scale floating-point work by increasing the global grid. Suppose each worker owns a constant number $k$ of points, so total points $N_{\mathrm{tot}}=kP$. Local arithmetic is approximately $\Theta(k\log(kP))$ per worker, **not** a constant. Global FFT stages typically require all-to-all redistribution (or several transposes) across subsets of the process grid.

A conceptual per-iteration runtime is

$$T_P= a\,k\log(kP) + T_{\mathrm{alltoall}}(P,k) + T_{\mathrm{plan}}(P).$$

Even if a simplified all-to-all delivers a constant payload per worker, many small messages or increasing contention can raise latency, for example $T_{\mathrm{alltoall}}\gtrsim\lambda\sqrt P$ for some pencil communication patterns. More advanced all-to-all implementations may have better constants, but a **growing** $\log P$ compute term already violates a strict fixed-time claim at constant local grid size.

**Failure mechanism:** the algorithmic complexity per stored grid point depends on the global dimension, and global data redistribution requires nonlocal synchronization. Adding nodes does not preserve a constant amount of work *or* communication time per node.

**Mitigations:** larger batch sizes, overlapping transposes, topology-aware process layouts, fewer transform stages, mixed decomposition strategies. Mitigations can shift the break point but cannot make a universal constant-time proof.

## Case B — Breadth-first search on a scale-free graph

For BFS, expanding a graph with $V(P)=vP$ vertices and average degree near constant implies $E(P)=\Theta(P)$ total edges, seemingly an ideal constant local edge count. But the work per BFS level depends on the *frontier*, not merely total edge count. A few high-degree hubs in a power-law graph cause uneven partition loads; each frontier may require remote reads, atomics, and global decisions that cannot be kept fully local.

One coarse model is

$$T_P=\sum_{\ell=1}^{D(P)}\left(\frac{E_\ell(P)}{P r}+C_\ell(P)+I_\ell(P)\right),$$

where $D(P)$ is the number of BFS levels, $E_\ell$ edges visited at a level, $r$ the worker edge rate, $C_\ell$ communication/synchronization, and $I_\ell$ imbalance. Neither $D(P)$ nor the maximum frontier work per worker is guaranteed constant. For example, if one hub's degree grows as $P^\beta$ and its adjacency remains a single-owner partition, the maximum work on that owner grows as $\Omega(P^\beta)$ even though *mean* edges per worker remain constant.

**Failure mechanism:** skewed partitions and global frontier coordination dominate the nominal constant local volume. Average weak scaling hides tail latency from the most heavily loaded partition.

**Mitigations:** direction-optimizing BFS, 2-D partitioning, work stealing, hub replication, batching updates. Their own communication and replication costs must be measured.

## General lesson

The condition $W(P)=\Theta(P)$ is **necessary for ideal weak scaling of a constant-work-per-worker kernel, but not sufficient**. A compliant algorithm also needs bounded critical-path length, bounded imbalanced local work, bounded nonoverlapped communication, and bounded memory access costs. $\alpha(P)$ and $C(P)$ must be measured rather than inferred from the operation count alone.
