#!/usr/bin/env python3
"""Cache Line & False Sharing Mechanics in Multi-Core Python.

Student ID: 230107140
Instructor: Sufyan bin Uzayr

Reproducible 4-thread benchmark. Real, per-run CPU measurements are used to
create cache_lab_manifest.json and ANALYSIS.md. Neither artifact is fabricated.

Run on a machine with >= 4 AVAILABLE logical cores:
    python3 -m pip install numpy numba
    python3 false_sharing_lab.py --git-username YOUR_REAL_GITHUB_USERNAME

The Git username is deliberately required at execution time and is hashed
with the detected CPU model to create INSTANCE_SEED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numba import njit, types
from numba.extending import intrinsic
from llvmlite import ir

try:
    import resource
except ImportError:
    resource = None

STUDENT_ID = "230107140"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)
REPO_PREFIX = "cache-line-false-sharing-"


def detect_cache_line_size() -> int:
    """Prefer Linux L1 coherency line size, then macOS sysctl, then 64 B."""
    linux_path = Path("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size")
    if linux_path.exists():
        try:
            value = int(linux_path.read_text(encoding="ascii").strip())
            if value > 0:
                return value
        except (OSError, ValueError):
            pass
    if sys.platform == "darwin":
        try:
            value = int(subprocess.check_output(
                ["sysctl", "-n", "hw.cachelinesize"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip())
            if value > 0:
                return value
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    return 64


def _cpu_model_string() -> str:
    """Collect an actual host CPU model, including in Linux containers."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name") and ":" in line:
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                value = subprocess.check_output(
                    ["sysctl", "-n", key], text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if value:
                    return value
            except (OSError, subprocess.CalledProcessError):
                pass
    return platform.processor() or platform.uname().processor or platform.machine()


def _core_key(logical_cpu: int) -> tuple:
    """Select separate physical cores when topology metadata is accessible."""
    topo = Path(f"/sys/devices/system/cpu/cpu{logical_cpu}/topology")
    try:
        package = (topo / "physical_package_id").read_text().strip()
        core = (topo / "core_id").read_text().strip()
        return (package, core)
    except OSError:
        return ("logical", str(logical_cpu))


def _choose_cores() -> tuple[list[int] | None, str, list[int]]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None, "affinity_unavailable_on_this_OS", list(range(os.cpu_count() or 1))
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < NUM_THREADS:
        raise RuntimeError(
            f"Need {NUM_THREADS} CPU cores in the process affinity mask; "
            f"only {len(allowed)} available: {allowed}"
        )
    physical = []
    seen = set()
    for logical_cpu in allowed:
        key = _core_key(logical_cpu)
        if key not in seen:
            physical.append(logical_cpu)
            seen.add(key)
        if len(physical) == NUM_THREADS:
            break
    if len(physical) == NUM_THREADS:
        return physical, "linux_pinned_distinct_physical_cores", allowed
    return allowed[:NUM_THREADS], "linux_pinned_distinct_logical_cpus", allowed


def init_system_context(git_user: str) -> dict:
    """Task 1: Detect core topology and hash Git identity + CPU model."""
    git_user = git_user.strip()
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", git_user):
        raise ValueError("Provide your real GitHub handle (letters, digits, hyphens).")
    if git_user.upper() in {"REPLACE_WITH_YOUR_GIT_USERNAME", "YOUR_REAL_GITHUB_USERNAME"}:
        raise ValueError("A placeholder is not an acceptable GitHub username.")
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Need >= {NUM_THREADS} logical CPU cores; host reports {logical_cores}.")
    line_size = detect_cache_line_size()
    model = _cpu_model_string()
    identity = f"{git_user}::{model}"
    instance_seed = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16)
    pin_cores, affinity_mode, allowed_cores = _choose_cores()
    if len(allowed_cores) < NUM_THREADS:
        raise RuntimeError("Fewer than four logical cores are available to this process.")
    return {
        "student_id": STUDENT_ID,
        "git_username": git_user,
        "cpu_model": model,
        "cpu_signature": f"{platform.machine()}::{model}::{logical_cores}",
        "logical_cores": logical_cores,
        "available_logical_cpus": allowed_cores,
        "chosen_worker_cpus": pin_cores,
        "affinity_mode": affinity_mode,
        "cache_line_size_bytes": line_size,
        "instance_seed": instance_seed,
        "machine_arch": platform.machine(),
        "os": platform.system(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


@intrinsic
def compiler_memory_barrier(typingctx):
    """A compiler-only clobber, not a hardware memory fence.

    This is essential: a loop-carried data dependency by itself DOES NOT force
    LLVM to issue a load/store in each iteration. Empty, side-effecting inline
    assembly with a memory clobber prevents register promotion and exposes
    actual native i64 loads/stores in inspect_llvm(). It emits NO CPU fence.
    """
    signature = types.void()

    def codegen(context, builder, signature, args):
        asm = ir.InlineAsm(
            ir.FunctionType(ir.VoidType(), []),
            "",
            "~{memory}",
            side_effect=True,
        )
        return builder.call(asm, [])

    return signature, codegen


@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, thread_seed):
    """Task 2: Perform an actual native read-modify-write on EVERY iteration."""
    delta = np.uint64((thread_seed & np.uint64(7)) + np.uint64(1))
    for _ in range(iters):
        buffer[slot] = buffer[slot] + delta
        compiler_memory_barrier()


@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, thread_seed):
    """Task 4: Accumulate in a register; commit to shared memory just once."""
    delta = np.uint64((thread_seed & np.uint64(7)) + np.uint64(1))
    local_value = buffer[slot]
    for _ in range(iters):
        local_value = local_value + delta
    buffer[slot] = local_value


def _aligned_zeros(elements: int) -> np.ndarray:
    """A 64-byte-aligned, 1D, writable, contiguous uint64 NumPy view."""
    raw = np.zeros(elements + 8, dtype=np.uint64)
    aligned_elem_offset = ((-raw.ctypes.data) % 64) // raw.itemsize
    out = raw[aligned_elem_offset:aligned_elem_offset + elements]
    assert out.ctypes.data % 64 == 0
    assert out.flags.c_contiguous and out.dtype == np.uint64
    return out


def _thread_seeds(instance_seed: int) -> list[np.uint64]:
    seeds = []
    for worker in range(NUM_THREADS):
        digest = hashlib.sha256(f"{instance_seed}::{worker}".encode("utf-8")).digest()
        seeds.append(np.uint64(int.from_bytes(digest[:8], "little")))
    return seeds


def _rusage_snapshot() -> dict:
    if resource is None:
        return {"voluntary": None, "involuntary": None}
    data = resource.getrusage(resource.RUSAGE_SELF)
    return {"voluntary": int(data.ru_nvcsw), "involuntary": int(data.ru_nivcsw)}


def _rusage_delta(a: dict, b: dict) -> dict:
    return {
        "voluntary": b["voluntary"] - a["voluntary"] if a["voluntary"] is not None else None,
        "involuntary": b["involuntary"] - a["involuntary"] if a["involuntary"] is not None else None,
    }


def _run_workers(context: dict, *, stride: int, iters: int, is_private: bool) -> dict:
    """Tasks 3 & 4: Pin, synchronize four threads, measure true wall time."""
    slots = [thread * stride for thread in range(NUM_THREADS)]
    buffer = _aligned_zeros(slots[-1] + 1)
    seeds = _thread_seeds(context["instance_seed"])
    expected = [int((int(seed) & 7) + 1) * iters for seed in seeds]
    barrier = threading.Barrier(NUM_THREADS + 1, timeout=60)
    errors = [None] * NUM_THREADS
    worker_cpus = context["chosen_worker_cpus"]
    pin_success = [False] * NUM_THREADS
    used_kernel = worker_private_accumulate if is_private else worker_strided_accumulate
    thread_rusage = [None] * NUM_THREADS

    def runner(worker_id: int) -> None:
        try:
            if worker_cpus is not None:
                # Linux affinity is per-kernel-thread. pid=0 targets THIS thread.
                os.sched_setaffinity(0, {worker_cpus[worker_id]})
                pin_success[worker_id] = True
            barrier.wait()
            before_t = None
            if resource is not None and hasattr(resource, "RUSAGE_THREAD"):
                before_t = resource.getrusage(resource.RUSAGE_THREAD)
            used_kernel(buffer, slots[worker_id], iters, seeds[worker_id])
            if before_t is not None:
                after_t = resource.getrusage(resource.RUSAGE_THREAD)
                thread_rusage[worker_id] = {
                    "voluntary": int(after_t.ru_nvcsw - before_t.ru_nvcsw),
                    "involuntary": int(after_t.ru_nivcsw - before_t.ru_nivcsw),
                }
        except BaseException as exc:
            errors[worker_id] = f"{type(exc).__name__}: {exc}"
            try:
                barrier.abort()
            except BaseException:
                pass

    threads = [
        threading.Thread(target=runner, name=f"numba-worker-{i}", args=(i,))
        for i in range(NUM_THREADS)
    ]
    before = _rusage_snapshot()
    for thread in threads:
        thread.start()
    start = time.perf_counter_ns()
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter_ns() - start
    after = _rusage_snapshot()
    if any(errors):
        raise RuntimeError(f"Worker failures: {errors}")
    actual = [int(buffer[slot]) for slot in slots]
    parity = actual == expected
    return {
        "layout": "register_private" if is_private else {
            1: "contiguous", 4: "half_line", 8: "line_padded", 16: "prefetch_safe",
        }[stride],
        "stride_elements": stride,
        "spacing_bytes": stride * 8,
        "iterations_per_thread": iters,
        "total_updates": NUM_THREADS * iters,
        "elapsed_ns": int(elapsed),
        "elapsed_ms": elapsed / 1_000_000.0,
        "ns_per_update": elapsed / (NUM_THREADS * iters),
        "context_switches": _rusage_delta(before, after),
        "per_thread_context_switches": thread_rusage,
        "all_threads_affinity_pinned": bool(worker_cpus is not None and all(pin_success)),
        "buffer_base_address_mod_64": buffer.ctypes.data % 64,
        "buffer_indices": slots,
        "relative_cache_line_ids": [(slot * 8) // context["cache_line_size_bytes"] for slot in slots],
        "distinct_cache_lines": len({(slot * 8) // context["cache_line_size_bytes"] for slot in slots}),
        "final_counter_values": actual,
        "expected_counter_values": expected,
        "parity_passed": parity,
    }


def run_stride_benchmark(context: dict, *, iters: int) -> dict[str, dict]:
    """Task 3: Four topologies, four real concurrent OS threads per sweep."""
    benchmark = {}
    for stride in STRIDES:
        result = _run_workers(context, stride=stride, iters=iters, is_private=False)
        benchmark[str(stride)] = result
        print(f"stride={stride:2} {result['elapsed_ns']:>15,d} ns "
              f"({result['ns_per_update']:.5f} ns/update) parity={result['parity_passed']}",
              flush=True)
    return benchmark


def _native_ir_function(full_ir: str) -> str:
    # Capture the actual nopython native function, excluding its CPython wrapper.
    blocks = re.findall(r"(?ms)^define\b[^\n]*\bworker_strided_accumulate\b[^\n]*\{.*?^\}", full_ir)
    if not blocks:
        blocks = re.findall(r"(?ms)^define\b[^\n]*worker_strided_accumulate[^\n]*\{.*?^\}", full_ir)
    native = [s for s in blocks if "@_ZN7cpython" not in s and "@cfunc." not in s]
    if not native:
        raise RuntimeError("Could not isolate nopython native LLVM function.")
    return native[0]


def _llvm_audit() -> dict:
    signatures = worker_strided_accumulate.signatures
    if not signatures:
        raise RuntimeError("The strided worker was not JIT compiled.")
    llvm = worker_strided_accumulate.inspect_llvm(signatures[0])
    native = _native_ir_function(llvm)
    # A compiler-only memory clobber must remain INSIDE the native worker loop.
    loads = len(re.findall(r"\bload i64\b", native))
    stores = len(re.findall(r"\bstore i64\b", native))
    barriers = len(re.findall(r'asm sideeffect "", "~\{memory\}"', native))
    memory_loop_verified = any(
        re.search(r"\bload i64\b", section)
        and re.search(r"\bstore i64\b", section)
        and 'asm sideeffect "", "~{memory}"' in section
        and re.search(r"\bbr i1\b[^\n]*\blabel\b", section)
        for section in re.split(r"(?m)^\w[^\n]*:\s*(?:;[^\n]*)?$", native)
    )
    active_nogil = bool(worker_strided_accumulate.targetoptions.get("nogil", False))
    return {
        "llvm_sha256": hashlib.sha256(llvm.encode("utf-8")).hexdigest(),
        "llvm_native_function_sha256": hashlib.sha256(native.encode("utf-8")).hexdigest(),
        "llvm_native_i64_load_count": loads,
        "llvm_native_i64_store_count": stores,
        "llvm_native_memory_barrier_count": barriers,
        "llvm_loop_contains_load_store_and_clobber": bool(memory_loop_verified),
        "numba_nogil_true": active_nogil,
        "nopython_native_function": "PyObject" not in native,
        "audit_passed": bool(loads >= 1 and stores >= 1 and barriers >= 1
                             and memory_loop_verified and active_nogil
                             and "PyObject" not in native),
        "llvm_excerpt": "\n".join(
            line.strip() for line in native.splitlines()
            if "load i64" in line or "store i64" in line
            or "asm sideeffect" in line or "add i64" in line
        )[:2400],
    }


def audit_llvm_and_parity(context: dict, strides: dict, private: dict, *, iters: int) -> dict:
    """Task 5: bit-for-bit counter checks and real LLVM native store audit."""
    llvm = _llvm_audit()
    layouts = list(strides.values()) + [private]
    reference = layouts[0]["final_counter_values"]
    all_parity = all(result["parity_passed"] and result["final_counter_values"] == reference
                     for result in layouts)
    all_aligned = all(result["buffer_base_address_mod_64"] == 0 for result in layouts)
    full_run = iters == ITERATIONS_PER_THREAD
    mode = "complete_assigned_run" if full_run else "DEVELOPMENT_ONLY_SHORT_RUN"
    cpu_line_assumption = context["cache_line_size_bytes"] == 64
    verification = bool(all_parity and all_aligned and llvm["audit_passed"]
                        and full_run and cpu_line_assumption)
    return {
        "llvm": llvm,
        "exact_state_parity_passed": all_parity,
        "all_buffers_64_byte_aligned": all_aligned,
        "complete_60m_iterations_per_thread": full_run,
        "hardware_64_byte_cache_lines": cpu_line_assumption,
        "execution_mode": mode,
        "verification_passed": verification,
    }


def _fnum(value: float, digits: int = 4) -> str:
    return f"{value:,.{digits}f}"


def _analysis_markdown(manifest: dict) -> str:
    """Generate all five answers from this run's exact measured numbers."""
    ctx = manifest["context"]
    results = manifest["benchmarks"]
    baseline = manifest["private_baseline"]
    audit = manifest["verification"]["llvm"]
    n = manifest["iterations_per_thread"]
    d1, d4, d8, d16 = (results[str(s)]["elapsed_ns"] for s in STRIDES)
    bp = baseline["elapsed_ns"]
    ratio_18 = d1 / d8
    ratio_816 = d8 / d16
    header = (
        "# Cache Line & False Sharing Mechanics - Forensic Analysis\n\n"
        f"**Student ID:** {STUDENT_ID}  \n"
        f"**GitHub handle:** `{ctx['git_username']}`  \n"
        "**Instructor:** Sufyan bin Uzayr  \n"
        f"**Host CPU:** {ctx['cpu_model']}  \n"
        f"**OS:** {ctx['os']} / {ctx['machine_arch']}  \n"
        f"**Logical cores / available CPUs:** {ctx['logical_cores']} / "
        f"{len(ctx['available_logical_cpus'])}  \n"
        f"**Cache line:** {ctx['cache_line_size_bytes']} B; "
        f"**affinity:** {ctx['affinity_mode']}  \n"
        f"**Pinned worker CPU IDs:** {ctx['chosen_worker_cpus']}  \n"
        f"**Iterations per worker:** {n:,}; **threads:** {NUM_THREADS}; "
        f"**Git/CPU SHA-256-derived 64-bit seed:** {ctx['instance_seed']}  \n"
        f"**Measured at UTC:** {manifest['measured_at_utc']}  \n"
        f"**Verification passed:** `{str(manifest['verification']['verification_passed']).lower()}`\n\n"
        "> These are observations from the runtime named above, not universal CPU performance\n"
        "> guarantees. The four threads use native Numba kernels (`nogil=True`).\n\n"
    )
    table = (
        "## Measured data (Task 3 and Task 4)\n\n"
        "| Layout | Stride (elements) | Bytes between counters | Elapsed (ns) | "
        "ns / update | Voluntary switches | Involuntary switches |\n"
        "|:--|--:|--:|--:|--:|--:|--:|\n"
    )
    for s in STRIDES:
        r = results[str(s)]
        sw = r["context_switches"]
        table += (f"| {r['layout']} | {s} | {r['spacing_bytes']} | "
                  f"{r['elapsed_ns']:,} | {_fnum(r['ns_per_update'], 6)} | "
                  f"{sw['voluntary'] if sw['voluntary'] is not None else 'N/A'} | "
                  f"{sw['involuntary'] if sw['involuntary'] is not None else 'N/A'} |\n")
    sw = baseline["context_switches"]
    table += (f"| register_private | {baseline['stride_elements']} | "
              f"{baseline['spacing_bytes']} | {bp:,} | "
              f"{_fnum(baseline['ns_per_update'], 6)} | "
              f"{sw['voluntary'] if sw['voluntary'] is not None else 'N/A'} | "
              f"{sw['involuntary'] if sw['involuntary'] is not None else 'N/A'} |\n\n")
    table += (
        f"Every row represents **{NUM_THREADS*n:,} logical counter updates**, across "
        f"{NUM_THREADS} threads, excluding JIT compilation time. All five layouts had "
        f"**identical exact uint64 final counters**: "
        f"`{results['1']['final_counter_values']}`. The array bases were 64-byte aligned.\n\n"
    )
    q1 = (
        "## Q1 - MESI protocol state transitions\n\n"
        "With `stride=1`, the four `uint64` counters occupy offsets 0, 8, 16, 24 B "
        "of the **same 64-byte cache line**. Actual zero-based relative line indices "
        f"were `{results['1']['relative_cache_line_ids']}` for stride 1 and "
        f"`{results['4']['relative_cache_line_ids']}` for stride 4, versus "
        f"`{results['8']['relative_cache_line_ids']}` for stride 8 and "
        f"`{results['16']['relative_cache_line_ids']}` for stride 16. "
        "MESI states are line-granular, not per variable. "
        "A representative sequence is: (1) after a cold read, Core 0 can obtain the line "
        "in **E**, while Core 1 is **I**; (2) Core 0 stores `buffer[0]`, changing **E to M** "
        "(or **S to M** via an upgrade/RFO if already shared); (3) Core 1 then needs "
        "ownership to write `buffer[1]`, sends a write-ownership/RFO request, and forces "
        "Core 0's **M to I** after the modified contents are forwarded or made visible; "
        "Core 1 gets **M**; (4) on Core 0's next update, ownership transfers back, "
        "flipping Core 1's **M to I**. If both cores initially read, both may be **S** "
        "before either one upgrades. States and transient details vary by CPU protocol. "
        "Core 1 cannot retire the dependent store into globally coherent memory until "
        "ownership is obtained; its pending store and subsequent dependent loads can stall. "
        "On modern coherent CPUs this typically uses a coherence interconnect, not "
        "necessarily a physically shared bus. The per-layout timings above quantify "
        "this test's impact.\n\n"
    )
    direction = "slower" if ratio_18 > 1 else "faster"
    q2 = (
        "## Q2 - Quantitative slowdown and coherency overhead\n\n"
        f"Measured `stride=1`: **{d1:,} ns**; `stride=4`: **{d4:,} ns**; "
        f"`stride=8`: **{d8:,} ns**; `stride=16`: **{d16:,} ns**; "
        f"register-private baseline: **{bp:,} ns**.\n\n"
        f"The observed `stride=1 / stride=8` slowdown factor is "
        f"**{ratio_18:.6f}x** (`{d1:,} / {d8:,}`); in this run, stride 1 was "
        f"**{direction}** by a factor of **{max(ratio_18, 1/ratio_18):.6f}x**. "
        f"The per-update numbers are elapsed wall nanoseconds divided by "
        f"`4 * {n:,}` total native increments, not a single thread's call duration. "
        "Shared-counter layouts expose read-for-ownership invalidations when actual "
        "cores execute simultaneously; the privatized control avoids the repeated "
        "memory access entirely. Hypervisor scheduling, power and frequency, and "
        "interference can change observed speed ratios.\n\n"
    )
    q3 = (
        "## Q3 - L2 spatial prefetcher and 128-byte pairing\n\n"
        f"The measured `stride=8` time was **{d8:,} ns**, while the `stride=16` "
        f"time was **{d16:,} ns**. Their ratio (`stride=8 / stride=16`) was "
        f"**{ratio_816:.6f}x** and the absolute difference was "
        f"**{d8 - d16:+,} ns**. With 64-B cache lines and 8-byte counters, "
        "stride 8 puts each worker's counter in a distinct line. Stride 16 "
        "leaves a whole unused line between adjacent counters. Some processors "
        "implement adjacent-line spatial prefetchers that fetch nearby 64-byte "
        "lines as pairs (or operate on 128-byte aligned regions); 128-byte spacing "
        "can mitigate that secondary traffic. A speculative shared/read-only "
        "prefetch does **not itself** create a write invalidation; the impact depends "
        "on the actual CPU's prefetcher, cache hierarchy, ownership requests, and "
        "alignment. Thus 128-B padding is a conservative experimental control, "
        "not proof that an L2 prefetcher caused this specific timing difference. "
        f"The actual CPU for this run was `{ctx['cpu_model']}`.\n\n"
    )
    q4 = (
        "## Q4 - Compiler register promotion versus real memory stores\n\n"
        "A naive loop containing `buffer[slot] += delta` has a loop-carried data "
        "dependency, but that dependency **alone does not stop** an optimizing "
        "compiler from lifting the load out of the loop, doing all additions in "
        "registers, and sinking one store after the loop. Such a kernel would "
        "not measure false sharing. This implementation calls a zero-instruction "
        "**compiler-only memory clobber** in each iteration via Numba's LLVM "
        "intrinsic mechanism. It prevents the optimizer from proving that the "
        "buffer has unchanged memory across iterations, while adding no CPU "
        "`mfence` and no extra coherence traffic. The inspected *native* LLVM "
        "IR shows repeated 64-bit memory operations inside the loop and a "
        "side-effecting empty inline-assembly memory clobber. Exact static IR "
        "occurrences for this compilation are:\n\n"
        f"- `load i64`: **{audit['llvm_native_i64_load_count']}**.\n"
        f"- `store i64`: **{audit['llvm_native_i64_store_count']}**.\n"
        f"- `asm sideeffect \"\", \"~{{memory}}\"`: "
        f"**{audit['llvm_native_memory_barrier_count']}**.\n"
        f"- Native loop store/load audit: `{str(audit['llvm_loop_contains_load_store_and_clobber']).lower()}`; "
        f"`numba_nogil_true={str(audit['numba_nogil_true']).lower()}`.\n"
        f"- Full IR SHA-256: `{audit['llvm_sha256']}`.\n\n"
        "Representative IR instructions from the actual run:\n\n"
        "```llvm\n" + audit['llvm_excerpt'] + "\n```\n\n"
        "The register-private control instead computes an intermediate value "
        "locally and writes its result once. Counts above are *static* LLVM "
        "IR occurrences, not dynamically executed instruction counts.\n\n"
    )
    q5 = (
        "## Q5 - CPython GIL versus `nogil=True`\n\n"
        "In a standard GIL-enabled CPython build, bytecode loops in separate "
        "`threading.Thread`s serialize their Python execution because only one "
        "thread can hold the Global Interpreter Lock at a time. This prevents "
        "sustained concurrent writes on multiple cores and thus masks the "
        "four-core coherence ping-pong that the test is intended to isolate. "
        "Adding Python threads often increases elapsed time through GIL "
        "handoffs, interpreter overhead, and operating-system scheduling "
        "instead. Here `@njit(nogil=True)` compiles the worker to native machine "
        "code and releases the GIL for its execution, and each worker is mapped "
        "to an available CPU core where OS support permits. For the current run, "
        f"`numba_nogil_true={str(audit['numba_nogil_true']).lower()}` and "
        f"affinity mode was `{ctx['affinity_mode']}`. This is a contrast in "
        "execution mechanisms, **not** a separate measured Python-bytecode "
        "benchmark (the specification did not require one).\n\n"
    )
    verification = (
        "## Verification and provenance\n\n"
        f"- Full assigned sweep (60,000,000 increments per thread): "
        f"`{str(manifest['verification']['complete_60m_iterations_per_thread']).lower()}`.\n"
        f"- Bit-exact final counter parity: "
        f"`{str(manifest['verification']['exact_state_parity_passed']).lower()}`.\n"
        f"- LLVM audit: `{str(audit['audit_passed']).lower()}`.\n"
        f"- Verification passed: `{str(manifest['verification']['verification_passed']).lower()}`.\n\n"
        "This analysis was generated directly from `cache_lab_manifest.json`. "
        "Re-run on the student's actual machine and GitHub identity before submitting.\n"
    )
    return header + table + q1 + q2 + q3 + q4 + q5 + verification


def _atomic_json(path: Path, manifest: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_submission_zip(folder: Path) -> Path:
    name = f"CACHE_LAB_{STUDENT_ID}_GITHUB_FILES.zip"
    output = folder / name
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename in ("false_sharing_lab.py", "cache_lab_manifest.json", "ANALYSIS.md"):
            zf.write(folder / filename, filename)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache Line & False Sharing benchmark (Student 230107140)")
    parser.add_argument(
        "--git-username", default=os.environ.get("GIT_USERNAME"),
        help="Exact real GitHub username; required unless GIT_USERNAME is exported",
    )
    parser.add_argument(
        "--iterations", type=int, default=ITERATIONS_PER_THREAD,
        help="Iterations per worker (default 60000000; smaller is DEV ONLY)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not args.git_username and sys.stdin.isatty():
        args.git_username = input("Enter your exact GitHub username (NOT Student ID unless identical): ").strip()
    if not args.git_username:
        parser.error("Pass --git-username YOUR_ACTUAL_GITHUB_USERNAME (not Student ID unless it is your handle).")
    if args.iterations <= 0:
        parser.error("--iterations must be > 0")
    context = init_system_context(args.git_username)
    print(f"Student: {STUDENT_ID} | GitHub: {context['git_username']}", flush=True)
    print(f"CPU: {context['cpu_model']} | logical cores: {context['logical_cores']}", flush=True)
    print(f"L1 cache line: {context['cache_line_size_bytes']} B | "
          f"worker CPUs: {context['chosen_worker_cpus']}", flush=True)
    print(f"Derived INSTANCE_SEED: {context['instance_seed']}", flush=True)

    # JIT compile BEFORE timing. Also validate a short, untimed counter update.
    warm = _aligned_zeros(48)
    worker_strided_accumulate(warm, 0, 7, np.uint64(3))
    worker_private_accumulate(warm, 8, 7, np.uint64(3))
    if int(warm[0]) != 28 or int(warm[8]) != 28:
        raise RuntimeError("JIT smoke test failed.")
    print(f"Running four threads x {args.iterations:,} updates per topology...", flush=True)
    stride_results = run_stride_benchmark(context, iters=args.iterations)
    private_result = _run_workers(context, stride=8, iters=args.iterations, is_private=True)
    print(f"private   {private_result['elapsed_ns']:>15,d} ns "
          f"({private_result['ns_per_update']:.5f} ns/update) "
          f"parity={private_result['parity_passed']}", flush=True)
    verification = audit_llvm_and_parity(context, stride_results, private_result, iters=args.iterations)
    manifest = {
        "assignment": "Cache Line & False Sharing Mechanics in Multi-Core Python",
        "student_id": STUDENT_ID,
        "instructor": "Sufyan bin Uzayr",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_username": context["git_username"],
        "instance_seed": context["instance_seed"],
        "context": context,
        "num_threads": NUM_THREADS,
        "iterations_per_thread": args.iterations,
        "benchmark_strides": list(STRIDES),
        "benchmarks": stride_results,
        "private_baseline": private_result,
        "ratios": {
            "stride1_over_stride8": stride_results["1"]["elapsed_ns"] / stride_results["8"]["elapsed_ns"],
            "stride8_over_stride16": stride_results["8"]["elapsed_ns"] / stride_results["16"]["elapsed_ns"],
            "stride1_over_private": stride_results["1"]["elapsed_ns"] / private_result["elapsed_ns"],
        },
        "verification": verification,
        "verification_passed": verification["verification_passed"],
    }
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    # Copy source into selected output folder if the user chooses a different directory.
    script_path = Path(__file__).resolve()
    output_script = folder / "false_sharing_lab.py"
    if script_path != output_script:
        output_script.write_bytes(script_path.read_bytes())
    _atomic_json(folder / "cache_lab_manifest.json", manifest)
    (folder / "ANALYSIS.md").write_text(_analysis_markdown(manifest), encoding="utf-8")
    print(f"LLVM: {verification['llvm']['llvm_native_i64_load_count']} native load i64; "
          f"{verification['llvm']['llvm_native_i64_store_count']} native store i64; "
          f"{verification['llvm']['llvm_native_memory_barrier_count']} barriers", flush=True)
    print(f"EXACT STATE PARITY: {verification['exact_state_parity_passed']}", flush=True)
    print(f"VERIFICATION PASSED: {verification['verification_passed']}", flush=True)
    if verification["verification_passed"]:
        archive = _write_submission_zip(folder)
        print(f"GitHub files ready in: {folder}", flush=True)
        print(f"Archive (UNZIP before GitHub upload): {archive}", flush=True)
        print(f"Repository name: {REPO_PREFIX}{context['git_username']}", flush=True)
        return 0
    print("WARNING: This run does not satisfy all assigned verification requirements.", flush=True)
    if not verification["complete_60m_iterations_per_thread"]:
        print("Use the default 60,000,000 iterations for the actual submission.", flush=True)
    if not verification["llvm"]["audit_passed"]:
        print("LLVM verification failed; inspect the generated manifest.", flush=True)
    if not verification["hardware_64_byte_cache_lines"]:
        print("Host cache-line size is not 64 bytes; assumptions differ.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
