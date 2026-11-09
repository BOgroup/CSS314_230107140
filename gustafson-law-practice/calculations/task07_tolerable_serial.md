# Task 07 — Maximum Tolerable Parallel Serial Overhead

Given $P=8192$ processing cores and an operational minimum scaled speedup $S_G(P)\ge7500$, solve for the maximum fraction $\alpha$ of the **parallel machine's elapsed time** that can be serial.

$$S_G(8192)=8192-\alpha(8192-1)=8192-8191\alpha.$$

Apply the speedup requirement:

$$8192-8191\alpha\ge7500$$
$$-8191\alpha\ge-692$$
$$\boxed{\alpha\le\frac{692}{8191}\approx0.0844829691124}.$$

Therefore the maximum is **$\alpha_{\max}\approx8.44829691124\%$ of parallel runtime**.

## Verification

Substituting the *exact* rational limit gives $S_G=8192-8191(692/8191)=8192-692=7500$, which satisfies the boundary at equality. For any $\alpha>692/8191$, the predicted speedup is strictly less than 7500. For $\alpha<692/8191$, it exceeds the requirement.

The implied minimum scaled efficiency is

$$E_{\min}=\frac{7500}{8192}=\frac{1875}{2048}\approx0.91552734375=91.552734375\%.$$

## Practical reading

The allowed fraction includes **all nonparallel elapsed-time overhead classified as serial in the simplified Gustafson model**; it should not be confused with a fixed-job, single-core Amdahl fraction. If communication has a separate non-overlapped time fraction $\gamma$, the extended condition becomes

$$8192-8191\alpha-8192\gamma\ge7500\iff8191\alpha+8192\gamma\le692.$$

A real operational threshold should also account for trial-to-trial variance, outliers, thermal effects, and service-level safety margin.
