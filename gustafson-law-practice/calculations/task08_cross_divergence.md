# Task 08 — Cross-Over Point and Divergence: Amdahl vs. Gustafson

**Given:** processor count $P=128$ and the **same numeric value** $0.08$ used as two conceptually different fractions: $s_A=.08$ of a one-core fixed workload and $\alpha_G=.08$ of a $128$-core scaled workload.

## Amdahl (fixed original job)

$$S_A(128)=\frac{1}{0.08+0.92/128}$$
$$=\frac{1}{0.08+0.0071875}=\frac{1}{0.0871875}$$
$$\boxed{S_A(128)=11.4695340502\times}.$$

The maximum possible fixed-job speedup for this $s_A$ is $1/.08=12.5$ as $P\to\infty$.

## Gustafson (larger job, constant time)

$$S_G(128)=128-0.08(128-1)=128-10.16$$
$$\boxed{S_G(128)=117.84\times},\qquad E_G=117.84/128=0.920625.$$

## Numerical discrepancy (explicit denominator)

Absolute difference:

$$\Delta S=S_G-S_A=117.84-11.4695340502=\boxed{106.3704659498}.$$

Percentage **relative to Amdahl** (chosen primary convention):

$$\frac{S_G-S_A}{S_A}\cdot100\%=\boxed{927.4175\%}.$$

Equivalently $S_G/S_A=10.274175$. If one instead uses Gustafson as the denominator, the discrepancy is about **90.2668584%**. Always state the denominator because “percentage difference” is otherwise ambiguous.

## Are these predictions contradictory?

No. The numbers use **equal labels but unequal measurement baselines**. $s_A$ refers to the one-core *fixed* job; $\alpha_G$ refers to runtime fractions during a *scaled* 128-core run. A valid same-workload comparison must transform the fractions. If $\alpha_G=.08$ on $P=128$, its equivalent Amdahl one-core fraction for the *scaled* job is

$$s_{A,\mathrm{scaled}}=\frac{\alpha_G}{P-(P-1)\alpha_G}=\frac{0.08}{117.84}\approx0.000678887.$$

Inserting this transformed share in Amdahl gives exactly $117.84$. Hence the task's “cross-over” comparison is a **cross-model numerical comparison**, not a physical intersection or contradiction. The apparent gap grows as the core count increases because one formula fixes useful work while the other expands it.
