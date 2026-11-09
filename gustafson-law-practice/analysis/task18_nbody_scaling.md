# Task 18 — Direct $N$-Body vs. Barnes–Hut Tree Weak Scaling

## Two different work laws

**Direct all-pairs interactions:** With $N$ particles, the pairwise gravitational force evaluation requires $W_D(N)=\Theta(N^2)$ interactions. Ideal $P$-processor per-iteration compute time is proportional to $N^2/P$. To keep that compute time constant as processor count rises,

$$N_D(P)^2/P=N_0^2\quad\implies\quad\boxed{N_D(P)=N_0\sqrt P}.$$

**Barnes–Hut tree code:** Under conventional balanced-tree assumptions and a fixed acceptance criterion, expected computational work is approximately $W_{BH}(N)=\Theta(N\log N)$. To preserve constant per-worker compute time,

$$\frac{N_{BH}(P)\ln N_{BH}(P)}P=N_0\ln N_0.$$

Let $A=N_0\ln N_0>0$. Setting $N\ln N=AP$ and using the Lambert $W$ function ($W(z)e^{W(z)}=z$),

$$\boxed{N_{BH}(P)=\frac{AP}{W(AP)}}.$$

Asymptotically $N_{BH}(P)=\Theta(P/\ln P)$ up to lower-order log terms. The tree method permits faster particle-count growth with $P$ under the **same constant compute-time target** because the serial algorithm has a lower exponent.

## What happens to serial-to-parallel time ratios?

Neither $O(N^2)$ nor $O(N\log N)$ alone specifies the **serial overhead**. Let $A_{\mathrm{serial}}(P)$ be unoverlapped setup, tree build, communication, scheduling, or reduction time, and $B_{\mathrm{local}}(P)$ be parallel compute time per processor. The observed ratio is

$$r(P)=\frac{A_{\mathrm{serial}}(P)}{B_{\mathrm{local}}(P)}.$$

By construction, both methods target $B_{\mathrm{local}}(P)=\Theta(1)$. Therefore **any unbounded serial/control overhead makes $r(P)$ grow**, contradicting a constant-time premise. Specific examples illustrate the risk:

| Regime | Hypothesized non-overlapped overhead | Scaling at fixed compute per worker |
|---|---|---|
| Direct, only constant control setup | $\Theta(1)$ | $r_D=\Theta(1)$: plausible ideal weak scaling |
| Direct, centralized $\Theta(N)$ particle distribution | $\Theta(N)$ | $r_D=\Theta(\sqrt P)$: worsens |
| Barnes–Hut, centralized $\Theta(N)$ tree construction | $\Theta(N)$ | $r_{BH}=\Theta(P/\log P)$: severe growth |
| Either method, only logarithmic global reduction | $\Theta(\log P)$ | $r=\Theta(\log P)$: slow but unbounded growth |

These are **conditional illustrations**, not claims that a well-designed Barnes–Hut code necessarily builds its tree serially. Parallel tree construction, domain decomposition, hierarchical communication, and data redistribution can reduce these overheads. Direct methods may be communication-bound if particle positions must be replicated or exchanged globally each step. Barnes–Hut may incur remote multipole lookups, load imbalance in clustered domains, and tree-rebuilding costs.

## Algorithmic comparison and limitations

For equal *particle count* $N$, Barnes–Hut reduces arithmetic significantly but introduces approximate force error, tree storage, branching irregularity, and distributed data-dependence. For equal fixed-time target under increasing $P$, direct summation can only raise particle count like $\sqrt P$, while Barnes–Hut can raise it almost linearly (subject to its logarithmic factor). A fair benchmark must report accuracy criterion, timestep, particle distribution, force cutoff/opening criterion, and both serial and parallel reference execution methods.

**Conclusion:** the useful-compute scaling law determines $N(P)$; it does **not** uniquely determine $\alpha(P)$ without a model for setup, communication, and synchronization. Gustafson's law works best when these terms remain a small fraction of the measured runtime.
