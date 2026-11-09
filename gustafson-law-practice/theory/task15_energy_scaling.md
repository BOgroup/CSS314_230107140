# Task 15 — Energy and Power Proportionality under Scaled Speedup

## Distinguish power (watts) from processor count $P$

Let each **powered-on** node draw idle/static power $p_{\mathrm{idle}}$ watts, plus dynamic compute power $p_{\mathrm{active}}$ watts while active. Let a weak-scaled run finish in constant wall-clock time $T$ with serial elapsed fraction $\alpha$, and assume only one node performs useful compute during the serial phase while the other $P-1$ nodes remain powered on and idle. All $P$ compute during the parallel phase $1-\alpha$.

The idle/static draw is always $P p_{\mathrm{idle}}$. Dynamic power is $p_{\mathrm{active}}$ for $\alpha T$, and $P p_{\mathrm{active}}$ for $(1-\alpha)T$. Integrating power over elapsed time gives

$$\boxed{\mathcal E_P=T\left[P p_{\mathrm{idle}}+\left(\alpha+P(1-\alpha)\right)p_{\mathrm{active}}\right]}. $$

The energy for a **single core performing its own smaller one-core weak-scaling workload** in the same time $T$ is

$$\mathcal E_{1,\mathrm{small}}=T(p_{\mathrm{idle}}+p_{\mathrm{active}}).$$

Their ratio, which compares **different amounts of work**, is

$$\frac{\mathcal E_P}{\mathcal E_{1,\mathrm{small}}}=\frac{P p_{\mathrm{idle}}+[\alpha+P(1-\alpha)]p_{\mathrm{active}}}{p_{\mathrm{idle}}+p_{\mathrm{active}}}.$$

This ratio normally grows approximately linearly with $P$, because the larger system performs much more useful work. It is **not** proof of energy inefficiency.

## Apples-to-apples comparison: same scaled job

Under the ideal model, the same scaled job would take $T_1(W(P))=S_G(P)T$ on one core, with

$$S_G(P)=\alpha+P(1-\alpha).$$

Single-core energy for that large job is

$$\mathcal E_{1,\mathrm{same}}=S_G(P)T(p_{\mathrm{idle}}+p_{\mathrm{active}}).$$

Hence the energy ratio for equal work is

$$\boxed{\frac{\mathcal E_P}{\mathcal E_{1,\mathrm{same}}}=\frac{P p_{\mathrm{idle}}+S_G(P)p_{\mathrm{active}}}{S_G(P)(p_{\mathrm{idle}}+p_{\mathrm{active}})}.}$$

In this simple model dynamic energy for useful compute is **the same** on one core and $P$ cores ($S_GT p_{\mathrm{active}}$). The difference comes from powering idle baseline electronics on extra nodes. For $\alpha>0$, some static energy may be wasted while nodes wait at the serial phase.

## Numerical illustration

Take $P=128$, $\alpha=.04$, $T=100\,s$, $p_{\mathrm{idle}}=40\,W$, $p_{\mathrm{active}}=60\,W$. Then $S_G=122.92$, and

- Parallel run: $\mathcal E_P=100[128(40)+122.92(60)]=1{,}249{,}520\,J$.
- One core running its **small** job for 100 s: $\mathcal E_{1,\mathrm{small}}=10{,}000\,J$ (but processes much less work).
- One core running the **same large** job for $12{,}292\,s$: $\mathcal E_{1,\mathrm{same}}=1{,}229{,}200\,J$.

This simplified example gives about 1.0165× the energy for the parallel machine versus one core on identical work, while finishing 122.92× sooner. Real power curves depend on DVFS, leakage, DRAM, network, cooling, utilization, and node power-down policy. **Energy per solved timestep/sample and joules per useful result** should accompany scaled speedup.
