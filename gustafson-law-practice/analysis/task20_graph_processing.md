# Task 20 — Why BFS and PageRank Resist Regular Weak Scaling

## Weak-scaling promise vs. the graph execution reality

For an ideal dense regular kernel, work is distributed in equal blocks, arithmetic intensity is high, and communication patterns are predictable. Increasing global problem size with processor count can preserve **nearly identical local work and communication**. A power-law graph does not have these properties merely because the number of edges grows linearly with the number of processors.

**Breadth-First Search (BFS)** processes a sequence of frontiers. Work per level depends on visited vertices and frontier edges, which can vary by orders of magnitude. A few high-degree hubs account for disproportionate neighbor exploration. Fixed average edges per worker does not imply fixed **maximum** edges per worker, and the slowest partition determines completion of each barrier.

**PageRank** iteratively gathers and distributes contributions across irregular adjacency lists. Even if PageRank has a more regular iteration count than BFS, heavy-tailed in-degrees create hot destinations, skewed atomics, and memory bandwidth contention. Distributed graph partitions may need many remote messages and global convergence tests.

## Four scaling bottlenecks

| Source | BFS manifestation | PageRank manifestation | Effect on weak scaling |
|---|---|---|---|
| Irregular memory access | Frontier-dependent neighbor scans | Sparse gather/scatter | Cache/TLB misses, lower useful FLOP/s |
| Partition imbalance | Hubs and uneven frontiers | Unequal edge counts and high-in-degree vertices | Tail latency sets level/iteration time |
| Remote communication | Cross-partition frontier notifications | Contributions to remote vertex owners | Network startup, congestion, all-to-all-like traffic |
| Synchronization | Barrier at each BFS level | Residual/convergence reductions per iteration | Growing global collective costs |

## A simplified performance inequality

Let $E(P)$ be edges and assume $E(P)=eP$, so aggregate edge count grows linearly. Even with ideal average throughput $r$ edges per second per worker, the implementation may have

$$T_P\gtrsim\max_i\left(\frac{E_i(P)}r\right)+C(P)+I(P),$$

where $E_i(P)$ is the edge count actually visited by partition $i$ (not just average assigned edges), $C(P)$ communication and barriers, and $I(P)$ additional congestion/imbalance. Linear growth of $\sum_iE_i$ is insufficient to bound $\max_iE_i$ or $C(P)$. For BFS, varying graph diameter and frontier count add another serial critical-path factor. For PageRank, even a fixed iteration count can be dominated by random memory access and cache misses.

## Practical countermeasures and evaluations

Use edge-balanced/2-D partitioning, hotspot replication, push/pull BFS switching, dynamic frontier scheduling, compressed sparse layout, communication aggregation, and asynchronous or relaxed-accuracy convergence when scientifically acceptable. Each trade-off has its own cost: replication uses memory, asynchronous updates can change convergence, and repartitioning itself consumes time.

A robust study should compare at least (i) vertex count, edge count, graph degree distribution, and diameter; (ii) maximal and average edges handled by a worker; (iii) communication bytes and global synchronizations; (iv) total elapsed time and efficiency; and (v) output correctness. Merely plotting a weak-scaled vertex count against $P$ can conceal severe work skew.

**Contrast with dense GEMM (Task 16):** GEMM partitions uniform tiles and uses highly optimized contiguous BLAS kernels, while graph analytics operate on data-dependent sparse neighborhoods. Thus even when both have increasing workloads, the **critical path, data movement, and work variance** are fundamentally different.
