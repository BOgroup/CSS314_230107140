# Cache Line & False Sharing Mechanics - Forensic Analysis

**Student ID:** 230107140  
**GitHub handle:** `BOgroup`  
**Instructor:** Sufyan bin Uzayr  
**Host CPU:** Apple M3 Pro  
**OS:** Darwin / arm64  
**Logical cores / available CPUs:** 12 / 12  
**Cache line:** 128 B; **affinity:** affinity_unavailable_on_this_OS  
**Pinned worker CPU IDs:** None  
**Iterations per worker:** 60,000,000; **threads:** 4; **Git/CPU SHA-256-derived 64-bit seed:** 11152264029803727661  
**Measured at UTC:** 2026-11-20T05:49:12.640873+00:00  
**Verification passed:** `false`

> These are observations from the runtime named above, not universal CPU performance
> guarantees. The four threads use native Numba kernels (`nogil=True`).

## Measured data (Task 3 and Task 4)

| Layout | Stride (elements) | Bytes between counters | Elapsed (ns) | ns / update | Voluntary switches | Involuntary switches |
|:--|--:|--:|--:|--:|--:|--:|
| contiguous | 1 | 8 | 96,996,667 | 0.404153 | 0 | 161 |
| half_line | 4 | 32 | 97,136,500 | 0.404735 | 0 | 174 |
| line_padded | 8 | 64 | 60,135,875 | 0.250566 | 0 | 146 |
| prefetch_safe | 16 | 128 | 56,069,167 | 0.233622 | 0 | 165 |
| register_private | 8 | 64 | 60,167 | 0.000251 | 0 | 29 |

Every row represents **240,000,000 logical counter updates**, across 4 threads, excluding JIT compilation time. All five layouts had **identical exact uint64 final counters**: `[60000000, 240000000, 360000000, 480000000]`. The array bases were 64-byte aligned.

## Q1 - MESI protocol state transitions

With `stride=1`, the four `uint64` counters occupy offsets 0, 8, 16, 24 B of the **same 64-byte cache line**. Actual zero-based relative line indices were `[0, 0, 0, 0]` for stride 1 and `[0, 0, 0, 0]` for stride 4, versus `[0, 0, 1, 1]` for stride 8 and `[0, 1, 2, 3]` for stride 16. MESI states are line-granular, not per variable. A representative sequence is: (1) after a cold read, Core 0 can obtain the line in **E**, while Core 1 is **I**; (2) Core 0 stores `buffer[0]`, changing **E to M** (or **S to M** via an upgrade/RFO if already shared); (3) Core 1 then needs ownership to write `buffer[1]`, sends a write-ownership/RFO request, and forces Core 0's **M to I** after the modified contents are forwarded or made visible; Core 1 gets **M**; (4) on Core 0's next update, ownership transfers back, flipping Core 1's **M to I**. If both cores initially read, both may be **S** before either one upgrades. States and transient details vary by CPU protocol. Core 1 cannot retire the dependent store into globally coherent memory until ownership is obtained; its pending store and subsequent dependent loads can stall. On modern coherent CPUs this typically uses a coherence interconnect, not necessarily a physically shared bus. The per-layout timings above quantify this test's impact.

## Q2 - Quantitative slowdown and coherency overhead

Measured `stride=1`: **96,996,667 ns**; `stride=4`: **97,136,500 ns**; `stride=8`: **60,135,875 ns**; `stride=16`: **56,069,167 ns**; register-private baseline: **60,167 ns**.

The observed `stride=1 / stride=8` slowdown factor is **1.612958x** (`96,996,667 / 60,135,875`); in this run, stride 1 was **slower** by a factor of **1.612958x**. The per-update numbers are elapsed wall nanoseconds divided by `4 * 60,000,000` total native increments, not a single thread's call duration. Shared-counter layouts expose read-for-ownership invalidations when actual cores execute simultaneously; the privatized control avoids the repeated memory access entirely. Hypervisor scheduling, power and frequency, and interference can change observed speed ratios.

## Q3 - L2 spatial prefetcher and 128-byte pairing

The measured `stride=8` time was **60,135,875 ns**, while the `stride=16` time was **56,069,167 ns**. Their ratio (`stride=8 / stride=16`) was **1.072530x** and the absolute difference was **+4,066,708 ns**. With 64-B cache lines and 8-byte counters, stride 8 puts each worker's counter in a distinct line. Stride 16 leaves a whole unused line between adjacent counters. Some processors implement adjacent-line spatial prefetchers that fetch nearby 64-byte lines as pairs (or operate on 128-byte aligned regions); 128-byte spacing can mitigate that secondary traffic. A speculative shared/read-only prefetch does **not itself** create a write invalidation; the impact depends on the actual CPU's prefetcher, cache hierarchy, ownership requests, and alignment. Thus 128-B padding is a conservative experimental control, not proof that an L2 prefetcher caused this specific timing difference. The actual CPU for this run was `Apple M3 Pro`.

## Q4 - Compiler register promotion versus real memory stores

A naive loop containing `buffer[slot] += delta` has a loop-carried data dependency, but that dependency **alone does not stop** an optimizing compiler from lifting the load out of the loop, doing all additions in registers, and sinking one store after the loop. Such a kernel would not measure false sharing. This implementation calls a zero-instruction **compiler-only memory clobber** in each iteration via Numba's LLVM intrinsic mechanism. It prevents the optimizer from proving that the buffer has unchanged memory across iterations, while adding no CPU `mfence` and no extra coherence traffic. The inspected *native* LLVM IR shows repeated 64-bit memory operations inside the loop and a side-effecting empty inline-assembly memory clobber. Exact static IR occurrences for this compilation are:

- `load i64`: **1**.
- `store i64`: **1**.
- `asm sideeffect "", "~{memory}"`: **1**.
- Native loop store/load audit: `true`; `numba_nogil_true=true`.
- Full IR SHA-256: `235624ab025b863ff55369052f3b60f3d6a02c86afd2abcb06b1258d5b883534`.

Representative IR instructions from the actual run:

```llvm
%1 = add i64 %arg.iters, 1
%.177 = load i64, ptr %sunkaddr1, align 8
%.179 = add i64 %.24, %.177
store i64 %.179, ptr %sunkaddr1, align 8
tail call void asm sideeffect "", "~{memory}"() #0
%lsr.iv.next = add i64 %lsr.iv, -1
```

The register-private control instead computes an intermediate value locally and writes its result once. Counts above are *static* LLVM IR occurrences, not dynamically executed instruction counts.

## Q5 - CPython GIL versus `nogil=True`

In a standard GIL-enabled CPython build, bytecode loops in separate `threading.Thread`s serialize their Python execution because only one thread can hold the Global Interpreter Lock at a time. This prevents sustained concurrent writes on multiple cores and thus masks the four-core coherence ping-pong that the test is intended to isolate. Adding Python threads often increases elapsed time through GIL handoffs, interpreter overhead, and operating-system scheduling instead. Here `@njit(nogil=True)` compiles the worker to native machine code and releases the GIL for its execution, and each worker is mapped to an available CPU core where OS support permits. For the current run, `numba_nogil_true=true` and affinity mode was `affinity_unavailable_on_this_OS`. This is a contrast in execution mechanisms, **not** a separate measured Python-bytecode benchmark (the specification did not require one).

## Verification and provenance

- Full assigned sweep (60,000,000 increments per thread): `true`.
- Bit-exact final counter parity: `true`.
- LLVM audit: `true`.
- Verification passed: `false`.

This analysis was generated directly from `cache_lab_manifest.json`. Re-run on the student's actual machine and GitHub identity before submitting.
