# Task 06 — Parametric Speedup Across Processor Densities

**Given:** the observed serial fraction on the parallel machine is $\alpha=0.04$; $P\in\{16,64,256,1024,4096,16384\}$.

$$S_G(P)=P-0.04(P-1)=0.96P+0.04,$$
$$E_G(P)=\frac{S_G(P)}{P}=0.96+\frac{0.04}{P}.$$

## Complete computed table

| Processors $P$ | Serial penalty $0.04(P-1)$ | Scaled speedup $S_G$ | Efficiency $E_G$ | Efficiency (%) |
|---:|---:|---:|---:|---:|
| 16 | 0.60 | 15.40 | 0.9625000000 | 96.250000 |
| 64 | 2.52 | 61.48 | 0.9606250000 | 96.062500 |
| 256 | 10.20 | 245.80 | 0.9601562500 | 96.015625 |
| 1,024 | 40.92 | 983.08 | 0.9600390625 | 96.003906 |
| 4,096 | 163.80 | 3,932.20 | 0.9600097656 | 96.000977 |
| 16,384 | 655.32 | 15,728.68 | 0.9600024414 | 96.000244 |

## Calculation log

- $P=16:\ S=16-.04(15)=15.40;\ E=15.40/16=.9625.$
- $P=64:\ S=64-.04(63)=61.48;\ E=61.48/64=.960625.$
- $P=256:\ S=256-.04(255)=245.80;\ E=245.80/256=.96015625.$
- $P=1024:\ S=1024-.04(1023)=983.08;\ E=983.08/1024=.9600390625.$
- $P=4096:\ S=4096-.04(4095)=3932.20;\ E=3932.20/4096=.960009765625.$
- $P=16384:\ S=16384-.04(16383)=15728.68;\ E=15728.68/16384=.96000244140625.$

## Interpretation

Efficiency declines by **0.249756 percentage points** from 96.250000% at 16 processors to roughly 96.000244% at 16,384, approaching but never falling below its fixed-$\alpha$ asymptote of 96%. The *absolute* lost ideal speedup, $P-S_G(P)=.04(P-1)$, still grows linearly: missing 655.32 units of ideal speedup at 16,384 processors. This is a mathematical model, not a prediction that an actual 16k-core cluster maintains constant 4% nonparallel elapsed time.

**Independent verification:** `benchmarks/simulator.py --core-counts 16,64,256,1024,4096,16384 --alphas .04` generates the same figures in a CSV.
