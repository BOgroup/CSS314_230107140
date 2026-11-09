# Task 02 — Formal Derivation of Scaled Speedup

## Starting assumptions

Consider a **particular scaled job** running on $P\ge1$ identical processors. Measure its elapsed time as a sequential component $t_s$ and an ideally parallel component $t_p$ in the parallel execution:

$$T_P(W(P))=t_s+t_p,\qquad s=\frac{t_s}{T_P},\qquad p=\frac{t_p}{T_P},\qquad s+p=1.$$

The parallel part contains $P$ equal-sized units of computation executed concurrently; one processor would need $P t_p$ time to process all units. The sequential part still takes $t_s$ on one processor. There is no extra serial-only cost for emulating interprocessor communication, and the one-core algorithm is assumed to perform comparable useful operations.

## Step-by-step reconstruction

1. **Parallel runtime:** $T_P=t_s+t_p$ by direct phase timing.
2. **Serial-equivalent time for the *same scaled problem*:** $T_1(W(P))=t_s+P t_p$. The factor $P$ applies to work that was concurrent, not to the unavoidable serial phase.
3. **Definition of speedup:**

   $$S(P)=\frac{T_1(W(P))}{T_P(W(P))}=\frac{t_s+P t_p}{t_s+t_p}.$$

4. **Normalize elapsed time:** divide numerator and denominator by $t_s+t_p$ and use $s+p=1$:

   $$S(P)=s+Pp.$$

5. **Eliminate $p$:** substitute $p=1-s$:

   $$S(P)=s+P(1-s)=P-s(P-1).$$

6. **Rename measured serial share:** with $\alpha\equiv s=t_s/T_P$,

   $$\boxed{S_G(P)=P-\alpha(P-1)}.$$

**Checkpoints:** $P=1\implies S=1$; $\alpha=0\implies S=P$; $\alpha=1\implies S=1$; for $0\le\alpha\le1$ the bound $1\le S\le P$ follows. Efficiency is

$$E_G(P)=\frac{S_G(P)}{P}=1-\alpha\left(1-\frac1P\right).$$

## Numerical consistency example

For $P=128$ and a measured $\alpha=.08$, a normalized parallel run spends $t_s=.08$ and $t_p=.92$. A single core would spend $.08+128(.92)=117.84$ units processing the **same larger job**. Therefore $S_G=117.84/1=117.84$, and $E_G=.920625$. This does **not** predict a $117.84\times$ strong-scaling speedup for the original $P=1$ problem: the job itself changed.

## Conditions for the derivation

The clean expression holds only if (a) $t_s$ and $t_p$ partition actual elapsed time without hidden communication, (b) the parallel phase performs useful work at a processor-independent rate, (c) the serial equivalent has the assumed cost, and (d) $\alpha$ is measured for the processor count/workload being described. $\alpha(P)$ may change with $P$; writing a constant $\alpha$ across a sweep is an additional modeling assumption, not part of the algebraic identity.

**Reference:** [Gustafson (1988)](https://doi.org/10.1145/42411.42415).
