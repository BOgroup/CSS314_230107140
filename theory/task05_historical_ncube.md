# Task 05 — Reconstructing the 1988 nCUBE/ten Result

## Original experiment and source verification

John L. Gustafson, Gary R. Montry, and Robert E. Benner published **“Development of Parallel Methods for a 1024-Processor Hypercube”** in *SIAM Journal on Scientific and Statistical Computing* **9(4), 609–638 (July 1988)**. The work used a **1024-processor NCUBE/ten distributed-memory MIMD hypercube**, not a modern shared-memory multicore machine. The scientific applications were **wave mechanics** (explicit finite differences), **fluid dynamics** (flux-corrected transport), and **structural/beam strain analysis** (finite elements and conjugate gradients). An editorial note in the archived paper reports recognition with the Gordon Bell Award and the Karp Prize.

Crucially, the study reported **two different empirical speedup ranges**:

| Experiment | Measured speedup on 1,024 processors | Interpretation |
|---|---:|---|
| Problem size held fixed | 502×–637× | Strong scaling; shrinking work per node |
| Problem size **per processor** held fixed | 1009×–1020× | Scaled speedup; substantially larger global problems |

The original paper distinguishes fixed-job serial components of roughly **0.0006–0.001** from equivalent, workload-dependent serial components of roughly **3–10 parts per million** for scaled problems. It also discusses redundant operations, operation efficiency, load imbalance, communication, and hardware-dependent floating-point rates. Its performance range is **not** evidence that all algorithms attain a 1000× gain.

## Why the “0.4% serial fraction makes this impossible” statement is misleading

A **hypothetical** Amdahl calculation using a fixed-size, one-core serial share $s_A=.004$ gives a limit

$$S_A(P)\le1/s_A=250.$$

At $P=1024$, it predicts $1/[.004+.996/1024]\approx201.0$ for the fixed original job. This says nothing about a **different, scaled workload**, and the 0.4% example should not be falsely presented as the fixed-size fraction measured in the paper.

Under Gustafson's formulation, **if** the observed serial share of the **parallel scaled execution** were $\alpha_G=.004$, then

$$S_G(1024)=1024-.004(1023)=\boxed{1019.908}.$$

The actual reported range, $1009\text{–}1020$, corresponds (under the ideal two-component model) to effective $\alpha_G$ values

$$\frac{1024-1020}{1023}\approx0.00391\quad\text{through}\quad\frac{1024-1009}{1023}\approx0.01466.$$

These are *parallel-runtime* fractions; their values must not be swapped with the Amdahl fraction of a different job. The reported efficiencies for scaled runs were approximately 98.5%–99.6%; high efficiency was achieved by keeping much larger useful work per node and controlling communication.

## Architecture and practical mechanisms

The NCUBE/ten nodes communicated through explicit point-to-point links; each node had local memory. The scientists designed larger spatial domains, high-granularity updates, pipelined or localized messaging, and selective redundant computations to reduce communication stalls. The paper explicitly notes that for very large scaled domains a literal one-node serial run might be infeasible in memory; in such cases serial-equivalent speedups are **extrapolations** based on operation efficiency and timing, not stopwatch measurements of an impossible run.

## Conclusion

The breakthrough did **not** falsify Amdahl's mathematical bound. It exposed the error of silently extending **one small-job serial fraction** to a family of expanded scientific jobs. The general lesson is to report *both* strong and scaled speedups with workload size, core count, and serial-equivalence methodology.

**Primary references:** [Original SIAM paper](https://doi.org/10.1137/0909041); [archived full version](https://rsim.cs.illinois.edu/arch/qual_papers/systems/6.pdf); [Gustafson, “Reevaluating Amdahl's Law,” CACM 31(5), 1988](https://doi.org/10.1145/42411.42415).
