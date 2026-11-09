# Task 14 — Isoefficiency Relation with Gustafson Scaling

## Formal definitions

Let useful serial computation on a workload of size $W$ take $T_1(W)=W$ time units, and the parallel execution take $T_P(W)$. The total parallel overhead in **work units** is

$$T_o(P,W)=P T_P(W)-T_1(W).$$

Speedup and efficiency are

$$S(P,W)=\frac{W}{T_P(W)},\qquad E(P,W)=\frac{S}{P}=\frac{W}{W+T_o(P,W)}.$$

To maintain a prescribed efficiency $E_0\in(0,1)$,

$$E(P,W)\ge E_0 \iff W\ge\frac{E_0}{1-E_0}\,T_o(P,W).$$

The **isoefficiency function** is the growth rate of $W(P)$ necessary to keep this inequality satisfied as $P$ increases. Since $T_o$ can depend on $W$, the condition may have to be solved implicitly.

## Proof for a Gustafson-compliant constant-serial-time model

Take $A=a>0$ time of serial work and $B(P)=bP$ units of perfectly distributable work ($b>0$). Then

$$W(P)=a+bP,\qquad T_P(W(P))=a+b,\qquad T_o(P,W(P))=P(a+b)-(a+bP)=(P-1)a.$$

Hence $T_o=\Theta(P)$, and the equation for fixed positive efficiency requires $W(P)=\Omega(P)$; a linear workload $a+bP$ is sufficient. Indeed,

$$E(P)=\frac{a+bP}{P(a+b)}=\frac{b}{a+b}+\frac{a}{P(a+b)}\longrightarrow\frac{b}{a+b}=1-\alpha.$$

Thus, for this **particular** overhead model, the minimum isoefficiency is **$W(P)=\Theta(P)$**, which matches standard weak scaling. The claim is conditional, not a theorem that all programs with a `Gustafson` label have linear isoefficiency.

## What communication changes

Suppose nonoverlapped communication costs $C(P)=c\log P$ per fixed-size local partition. Then at $W=bP$,

$$T_o(P,W)=(P-1)a+P c\log P=\Theta(P\log P).$$

But $W(P)=\Theta(P)$ fails $W\ge K T_o$ as $P\to\infty$, so efficiency declines. Under a workload model where increasing local computation is permitted and communication remains $O(\log P)$ independent of local size, one would need $W(P)=\Omega(P\log P)$ to hide this cost at fixed efficiency, breaking *strict* constant local-work weak scaling. With additional per-message data movement the isoefficiency may be even worse.

**Takeaway:** Gustafson's simple expression implicitly assumes overhead grows no faster than total useful work, and that serial duration remains bounded when growing the job. Isoefficiency exposes the extra assumption: constant work **per processor** is viable only when the overhead-to-work ratio is controlled.

**Reference:** classic isoefficiency analysis as described by parallel-algorithm texts by Kumar, Grama, Gupta, and Karypis; derivation above follows directly from $E=W/(W+T_o)$.
