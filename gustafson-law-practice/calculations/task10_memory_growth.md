# Task 10 — Memory Growth Rate vs. Scaled Speedup

## General capacity constraint

Under pure weak scaling, $W(P)=P W(1)$ and total memory $M(P)=cP$. Assume **$P$ nodes**, each with a hard cap of **32 GiB** of accessible memory ($1\,\mathrm{GiB}=2^{30}$ bytes). For a cubic 3-D grid with $N$ cells along each side and $b$ bytes of *total persistent memory per grid cell*, the leading storage requirement is

$$M_{\mathrm{grid}}(N)=bN^3\ \mathrm{bytes}.$$

The aggregate cap is $P\cdot32\cdot2^{30}$ bytes. Therefore,

$$bN^3\le32P\cdot2^{30},\quad \boxed{N_{\max}(P,b)=\left\lfloor\left(\frac{32P\cdot2^{30}}b\right)^{1/3}\right\rfloor}.$$

**Important:** the assignment does **not** specify $b$, nor halo buffers or runtime allocations. Consequently it does **not determine a unique numerical $N$**. To provide a concrete answer, the table explicitly assumes **one 64-bit (8-byte) scalar per cell**, zero halos, perfectly distributable storage, and that *all* 32 GiB on every node can be used. This is an optimistic **capacity ceiling**, not a production-safe grid size.

## Worked capacity ceilings, with stated assumptions

| Nodes $P$ | Aggregate memory | Assumed bytes/cell $b$ | Maximum integer global side $N$ | Total cells $N^3$ (approx.) |
|---:|---:|---:|---:|---:|
| 64 | 2 TiB | 8 | **6,501** | $2.747\times10^{11}$ |
| 512 | 16 TiB | 8 | **13,003** | $2.198\times10^{12}$ |
| 64 | 2 TiB | 16 | **5,160** | $1.374\times10^{11}$ |
| 512 | 16 TiB | 16 | **10,321** | $1.099\times10^{12}$ |

For $b=8$ and $P=64$:

$$N\le\left(\frac{64\cdot32\cdot2^{30}}8\right)^{1/3}\approx6501.995\implies N_{\max}=6501.$$

For $b=8$ and $P=512$:

$$N\le\left(\frac{512\cdot32\cdot2^{30}}8\right)^{1/3}\approx13003.989\implies N_{\max}=13003.$$

The eightfold increase in node count raises the ideal cubic grid **side** by a factor of $8^{1/3}=2$, while increasing total cell count by roughly $8$.

## Real grid allocation caveats

The formula assumes that the distributed grid maps perfectly onto nodes. A real solver needs multiple state arrays, halos, pressure/velocity fields, scratch storage, MPI buffers, checkpointing, and runtime reserves. If the usable fraction of node memory is $\eta_m<1$, replace 32 GiB with $32\eta_m$ GiB. Non-cubic decompositions may waste memory, and partitions should have convenient integer dimensions. If “32 GB” is defined in decimal rather than GiB, replace $2^{30}$ with $10^9$; the numerical ceilings decrease.

**Scaling conclusion:** total capacity $\Theta(P)$ is consistent with $N=\Theta(P^{1/3})$; however storage feasibility alone does not prove constant elapsed time or preserved Gustafson efficiency.
