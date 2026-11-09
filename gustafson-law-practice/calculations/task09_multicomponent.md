# Task 09 — Multi-Component Workload Partitioning

The parallel execution of a scientific simulation consists of:

| Phase | Activity | Share of observed parallel elapsed time | Model treatment |
|---|---|---:|---|
| A | Disk I/O + parameter broadcast | 2% | serial/nonoverlapped |
| B | Boundary exchange synchronization | 6% | serial/nonoverlapped (conservative assumption) |
| C | Core-domain flux computation | 92% | ideally parallelizable |
| **Total** | | **100%** | |

Under the exercise's simplified classification, the **aggregate serial fraction** on the parallel machine is

$$\boxed{\alpha=0.02+0.06=0.08},\qquad p=0.92.$$

The scaled-speedup prediction is

$$S_G(P)=P-0.08(P-1)=0.08+0.92P.$$

For **512 processors**,

$$S_G(512)=0.08+0.92(512)=\boxed{471.12},$$
$$E(512)=471.12/512=\boxed{0.92015625}\ (92.015625\%).$$

For **2,048 processors**,

$$S_G(2048)=0.08+0.92(2048)=\boxed{1884.24},$$
$$E(2048)=1884.24/2048=\boxed{0.9200390625}\ (92.00390625\%).$$

## Sensitivity and architectural interpretation

The 6% boundary exchange is **not inherently serial computation** in all implementations. If asynchronous halos hide a proportion $h$ of that 6%, then the non-overlapped overhead is $\alpha_{\mathrm{effective}}=0.02+0.06(1-h)$. For example, overlap of half that cost gives $\alpha=.05$ and $S_G(512)=512-.05(511)=486.45$. This is an alternative hypothetical, not the result dictated by the task.

Conversely, if the boundary cost rises with processor count, the constant 6% premise fails and the measured $\alpha(P)$ must be updated. A more accurate instrumentation plan separates disk time, serialized control work, wait time, network transit, and useful floating-point execution; communication may be tracked independently using Task 12's $\gamma(P)$ term.
