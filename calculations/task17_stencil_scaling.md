# Task 17 — 3-D Seven-Point Stencil and Surface-to-Volume Scaling

## Global and local decomposition

Take a global cubic grid of linear dimension $N$ (so $N^3$ update points) split over $P=q^3$ processors in a regular $q\times q\times q$ arrangement. Each owns a local cube of side

$$n=\frac{N}{q}=\frac{N}{P^{1/3}}.$$

Interior work per iteration per processor is approximately $c_w n^3$; for a 7-point finite-difference stencil, each interior update accesses six neighbors and the center. Exchanging one-cell-wide halos with six neighboring partitions transfers approximately $6n^2$ values per processor, excluding external domain boundaries. With $b$ bytes per value,

$$\text{compute/node}=\Theta(n^3),\quad\text{halo bytes/node}\approx6bn^2.$$

Hence the halo-volume-to-computation ratio for one iteration obeys

$$\boxed{R_{\mathrm{comm/comp}}\propto\frac{6bn^2}{c_w n^3}=\frac{6b}{c_w n}}.$$

## What happens under true weak scaling?

For fixed wall-time and constant local work, scale the **global number of cells** proportionally to processor count:

$$N(P)^3=P N_0^3\quad\Rightarrow\quad N(P)=N_0P^{1/3}.$$

Then $n=N(P)/P^{1/3}=N_0$ stays **constant**, giving

$$\boxed{R_{\mathrm{comm/comp}}=\Theta(1).}$$

Therefore the **per-processor** communication/computation ratio does **not** improve merely because the global domain expands. Each worker still computes on $N_0^3$ cells and exchanges $\sim6N_0^2$ halo elements. In an ideal nearest-neighbor network with constant latency and bandwidth, constant local runtime is plausible.

## Why a global surface-to-volume argument can mislead

The *outside surface* of the entire global cube is $\Theta(N^2)$ while its volume is $\Theta(N^3)$, so the global external ratio is $\Theta(1/N)=\Theta(P^{-1/3})$. But most interprocessor traffic comes from **internal partition interfaces**, not the outside domain boundary. Summing halo exchanges over $P$ local cubes yields $\Theta(Pn^2)$ words and useful updates $\Theta(Pn^3)$; their ratio remains $\Theta(1/n)$.

If instead local cubes become larger ($n\uparrow$) at a fixed processor count, the ratio **improves** as $1/n$. If strong scaling fixes global $N$ and increases $P$, then $n=N/P^{1/3}\downarrow$ and the ratio **degrades** as $P^{1/3}/N$.

## Latency and topology

A realistic step may cost

$$T_{\mathrm{step}}\approx c_w n^3+6\lambda+\frac{6bn^2}{B_{\mathrm{eff}}(P)}+T_{\mathrm{sync}}(P).$$

Even under weak scaling, a declining effective network bandwidth or growing global synchronization cost makes $T_{\mathrm{step}}$ increase. The result is therefore **conditional weak-scaling compliance**, not a guarantee of constant time on every cluster.
