"""OpenMP Multithreading & Performance Engineering in Python.
Student: Bazarbek Omar, 230107140. Instructor: Sufyan bin Uzayr.
Run on a local workstation; the JSON is generated from the actual run.
"""
# SECTION 0: LOCAL OPENMP ENVIRONMENT & DETERMINISTIC SEED HARNESS
import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"
os.environ["OMP_DYNAMIC"] = "FALSE"

import time
import hashlib
import json
import math
import platform
from pathlib import Path
import numpy as np
import numba as nb
from numba import njit, prange, set_num_threads, get_num_threads, get_thread_id
from numba import types
from numba.extending import intrinsic
from llvmlite import ir

STUDENT_NAME = "Bazarbek Omar"
STUDENT_ID = 230107140
THREAD_COUNTS = (1, 2, 4, 8)
SWEEPS_T4 = 600
N_T1 = 4_000_000 + (STUDENT_ID % 991) * 1024
BINS_T1 = 48 + (STUDENT_ID % 17)
ITERS_T2 = 35_000_000 + (STUDENT_ID % 503) * 10_000
PAD_STRIDE = 8 + (STUDENT_ID % 5) * 8
ROWS_T3 = 12_000 + (STUDENT_ID % 257) * 16
SKEW_MOD = 37 + (STUDENT_ID % 19)
GRID_H = 384 + (STUDENT_ID % 4) * 32
GRID_W = 512 + (STUDENT_ID % 5) * 32
OMEGA_SOR = np.float64(1.72 + (STUDENT_ID % 11) * 0.015)

@njit(parallel=True)
def _omp_probe():
    acc = 0
    for i in prange(64):
        acc += get_thread_id() + 1
    return acc

# TASK 1: RACE-CONDITION FORENSICS AND DETERMINISTIC HISTOGRAM
@njit(parallel=True)
def buggy_hist_omp(indices, weights, num_bins):
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    hits = np.zeros((get_num_threads(), PAD_STRIDE), dtype=np.int64)
    for i in prange(indices.shape[0]):
        b = indices[i]
        hist[b] += 1  # Deliberately unsafe read-modify-write.
        w_sums[b] += weights[i]
        hits[get_thread_id(), 0] += 1
    return hist, w_sums, hits[:, 0].copy()

@njit(parallel=True)
def safe_hist_omp(indices, weights, num_bins):
    n = get_num_threads()
    h = np.zeros((n, num_bins), dtype=np.int64)
    hi = np.zeros((n, num_bins), dtype=np.int64)
    lo = np.zeros((n, num_bins), dtype=np.int64)
    worker_hits = np.zeros(n, dtype=np.int64)
    # All supplied float64 weights in [0.25, 2.50] are multiples of 2^-54.
    # Split exact integer units into two limbs, preventing int64 overflow.
    for i in prange(indices.shape[0]):
        tid = get_thread_id()
        b = indices[i]
        units = np.int64(weights[i] * 18014398509481984.0)  # 2**54
        h[tid, b] += 1
        hi[tid, b] += units >> 32
        lo[tid, b] += units & np.int64(0xFFFFFFFF)
        worker_hits[tid] += 1
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for b in range(num_bins):
        high = np.int64(0)
        low = np.int64(0)
        for tid in range(n):  # Deterministic row-wise reduction.
            hist[b] += h[tid, b]
            high += hi[tid, b]
            low += lo[tid, b]
        high += low >> 32
        low &= np.int64(0xFFFFFFFF)
        # Both integer limbs are exactly representable as float64 for N_T1.
        # Their sum is rounded once, independently of thread count.
        w_sums[b] = math.ldexp(float(high), -22) + math.ldexp(float(low), -54)
    return hist, w_sums, worker_hits

DIAGNOSIS_T1 = (
    "The shared histogram increments are read-modify-write operations, so two "
    "threads can read the same old value and overwrite each other's updates. "
    "An atomic integer increment would protect the count but would not make "
    "floating-point addition associative. Different thread schedules and "
    "partition sizes change the order of floating-point additions and their rounding. "
    "Private rows remove shared writes, and the exact two-limb integer accumulator "
    "preserves all binary64 weight units without overflow for this input. "
    "Merging the rows in thread-ID order and rounding once gives bit-identical "
    "weighted sums at every required thread count."
)

# TASK 2: CACHE-LINE FALSE SHARING
# A volatile load and store keep actual memory traffic inside the loop.
# LLVM is otherwise allowed to keep a private worker's state in a register.
class _VolatileLoad(ir.instructions.LoadInstr):
    def descr(self, buf):
        lines = []
        super().descr(lines)
        buf.extend(line.replace("load ", "load volatile ", 1) for line in lines)

class _VolatileStore(ir.instructions.StoreInstr):
    def descr(self, buf):
        lines = []
        super().descr(lines)
        buf.extend(line.replace("store ", "store volatile ", 1) for line in lines)

@intrinsic
def _volatile_xor(typingctx, array, index, value):
    if not (isinstance(array, types.Array) and array.dtype == types.int64
            and array.layout == 'C' and isinstance(index, types.Integer)
            and value == types.int64):
        return None
    sig = types.void(array, index, value)
    def codegen(context, builder, signature, args):
        a = context.make_array(signature.args[0])(context, builder, args[0])
        ptr = builder.gep(a.data, [args[1]])
        load = _VolatileLoad(builder.block, ptr, typ=ir.IntType(64))
        load.align = 8
        builder._insert(load)
        result = builder.xor(load, args[2])
        store = _VolatileStore(builder.block, result, ptr)
        store.align = 8
        builder._insert(store)
        return context.get_dummy_value()
    return sig, codegen

@njit
def _aligned_zeros(n):
    # Make the eight contiguous slots share one aligned 64-byte line.
    raw = np.zeros(n + 7, dtype=np.int64)
    offset = ((64 - raw.ctypes.data % 64) % 64) // 8
    return raw[offset:offset+n]

@njit(inline='always')
def _next_state(x):
    return (x * np.int64(6364136223846793005)
            + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)

@njit(parallel=True)
def unpadded_state_omp(iters, seed_val):
    n_threads = get_num_threads()
    state = _aligned_zeros(n_threads)
    hits = np.zeros(n_threads, dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = _next_state(x)
            _volatile_xor(state, tid, x >> 17)
        hits[tid] += iters
    return state, hits

@njit(parallel=True)
def padded_state_omp(iters, seed_val):
    n_threads = get_num_threads()
    padded_state = _aligned_zeros(n_threads * PAD_STRIDE).reshape((np.int64(n_threads), np.int64(PAD_STRIDE)))
    hits = np.zeros(n_threads, dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = _next_state(x)
            # Exactly the same volatile read-modify-write, different stride.
            _volatile_xor(padded_state, tid * PAD_STRIDE, x >> 17)
        hits[tid] += iters
    return padded_state, hits

DIAGNOSIS_T2 = (
    "Under MESI-style coherence, writing a cache line requires exclusive ownership. "
    "Adjacent int64 counters can occupy the same 64-byte line, so otherwise "
    "independent writers repeatedly invalidate each other's copies. This is false "
    "sharing, not an update race on the same counter. Padding separates active "
    "slots by at least 64 bytes, reducing ownership transfers on 64-byte-line CPUs. "
    "Both kernels perform a volatile memory update on every iteration to prevent "
    "compiler register promotion from hiding the effect. The benchmark keeps work "
    "per thread constant, so its times are weak-scaling times; the eight-thread "
    "padded/unpadded comparison measures the padding benefit. Hardware with larger "
    "coherence lines may exhibit a different benefit from the assigned stride."
)

# TASK 3: IRREGULAR WORKLOAD AND DYNAMIC CHUNK CLAIMING
@njit(inline='always')
def _row_score(r, count):
    total = 0.0
    for step in range(count):
        # The handout specifies the expression, not x; use this fixed formula.
        x = (r + 1) * 0.001 + (step + 1) * 0.0001
        total += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
    return total

@njit(parallel=True)
def static_skew_omp(work_units):
    out = np.empty(work_units.size, dtype=np.float64)
    ops = np.zeros((get_num_threads(), PAD_STRIDE), dtype=np.int64)
    for r in prange(work_units.size):
        tid = get_thread_id()
        count = work_units[r]
        out[r] = _row_score(r, count)
        ops[tid, 0] += count
    return out, ops[:, 0].copy()

@intrinsic
def _claim_chunk(typingctx, cursor):
    if not (isinstance(cursor, types.Array) and cursor.ndim == 1
            and cursor.dtype == types.int64 and cursor.layout == 'C'):
        return None
    sig = types.int64(cursor)
    def codegen(context, builder, signature, args):
        a = context.make_array(signature.args[0])(context, builder, args[0])
        return builder.atomic_rmw('add', a.data, ir.Constant(ir.IntType(64), 1), 'monotonic')
    return sig, codegen

def build_chunks(work_units, threads, chunk_size=16):
    # Snake-interleave rows sorted by cost into small, near-equal chunks.
    # Number of chunks is divisible by threads, including a partial last layer.
    n_chunks = threads * ((work_units.size + chunk_size*threads - 1) // (chunk_size*threads))
    order = np.argsort(-work_units, kind='stable')
    chunks = np.full((n_chunks, chunk_size), -1, dtype=np.int64)
    for i, r in enumerate(order):
        layer, col = divmod(i, n_chunks)
        chunk = col if layer % 2 == 0 else n_chunks - 1 - col
        chunks[chunk, layer] = r
    valid = chunks[chunks >= 0]
    assert np.array_equal(np.sort(valid), np.arange(work_units.size))
    return chunks

@njit(parallel=True)
def balanced_skew_omp(work_units, chunks):
    n = get_num_threads()
    out = np.empty(work_units.size, dtype=np.float64)
    ops = np.zeros((n, PAD_STRIDE), dtype=np.int64)
    claimed = np.zeros((n, PAD_STRIDE), dtype=np.int64)
    cursor = np.zeros(1, dtype=np.int64)
    quota = chunks.shape[0] // n
    for worker in prange(n):
        tid = get_thread_id()
        # Equal chunk quotas prevent one fast worker from monopolizing work.
        # The atomic cursor assigns chunks dynamically; no fixed chunk ownership.
        for j in range(quota):
            k = _claim_chunk(cursor)
            for col in range(chunks.shape[1]):
                r = chunks[k, col]
                if r >= 0:
                    count = work_units[r]
                    out[r] = _row_score(r, count)
                    ops[tid, 0] += count
            claimed[tid, 0] += 1
    return out, ops[:, 0].copy(), claimed[:, 0].copy(), cursor[0]

DIAGNOSIS_T3 = (
    "Static prange gives contiguous ranges to workers and can leave a barrier "
    "waiting for the most expensive range. The specified modulo workload is "
    "periodic, so static scheduling can already be close to balanced for this ID. "
    "The balanced scheduler sorts rows by cost and snake-interleaves them into "
    "small batches of at most 16 rows with similar operation counts. Each worker "
    "claims its next batch through an atomic fetch-add on a shared cursor; no "
    "Python lock or static chunk ownership is used. Equal batch quotas limit "
    "monopolization, and each row is evaluated exactly once using the same score "
    "formula as the static kernel. Load balance is validated by actual per-worker "
    "operation counts, separately from elapsed time and scheduling overhead."
)

# TASK 4: RED-BLACK SOR, WITH A BARRIER BETWEEN COLORS

def initial_grid(h=GRID_H, w=GRID_W):
    u = np.zeros((h, w), dtype=np.float64)
    u[0, :] = 100.0
    u[-1, :] = -50.0
    u[:, 0] = 25.0 * np.sin(np.linspace(0, np.pi, h))
    u[:, -1] = 75.0 * np.cos(np.linspace(0, np.pi / 2, h))
    # Assignment order gives the side conditions precedence at corners.
    return u

@njit(parallel=True)
def sor_red_black_step(u, omega):
    n = get_num_threads()
    maxima = np.zeros((n, PAD_STRIDE), dtype=np.float64)
    hits = np.zeros((n, PAD_STRIDE), dtype=np.int64)
    h, w = u.shape
    for color in range(2):
        # Each prange completion is a barrier before the next color.
        for r in prange(1, h - 1):
            tid = get_thread_id()
            start = 1 + ((r + 1 + color) % 2)
            local_max = 0.0
            count = 0
            for c in range(start, w - 1, 2):
                old = u[r, c]
                new = ((1.0 - omega) * old + 0.25 * omega *
                       (u[r-1, c] + u[r+1, c] + u[r, c-1] + u[r, c+1]))
                u[r, c] = new
                delta = abs(new - old)
                if delta > local_max:
                    local_max = delta
                count += 1
            if local_max > maxima[tid, 0]:
                maxima[tid, 0] = local_max
            hits[tid, 0] += count
    max_delta = 0.0
    for tid in range(n):
        max_delta = max(max_delta, maxima[tid, 0])
    return max_delta, hits[:, 0].copy()

@njit
def solve_pde_omp(grid, omega, sweeps):
    u = grid.copy()
    history = np.empty(sweeps, dtype=np.float64)
    worker_hits = np.zeros(get_num_threads(), dtype=np.int64)
    for k in range(sweeps):
        delta, hits = sor_red_black_step(u, omega)
        history[k] = delta
        worker_hits += hits
    return u, history, worker_hits

# BENCHMARK AND VERIFICATION HELPERS

def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def check_hits(hits, n, total):
    assert hits.shape == (n,), (hits.shape, n)
    assert np.all(hits > 0), f"Inactive workers: {hits}"
    assert int(np.sum(hits)) == int(total), (hits, total)
    return [int(x) for x in hits]


def imbalance(ops):
    assert np.all(ops > 0)
    return float(np.max(ops) / np.min(ops))


def log(label, value):
    print(label, json.dumps(value, allow_nan=False), flush=True)


def run_task1(data_t1, weights_t1):
    print("\nTASK 1 - RACE CONDITIONS", flush=True)
    set_num_threads(1)
    buggy_hist_omp(data_t1[:128], weights_t1[:128], BINS_T1)
    safe_hist_omp(data_t1[:128], weights_t1[:128], BINS_T1)
    (ref_hist, ref_w, hits), ref_time = timed(buggy_hist_omp, data_t1, weights_t1, BINS_T1)
    assert np.array_equal(ref_hist, np.bincount(data_t1, minlength=BINS_T1))
    check_hits(hits, 1, N_T1)
    races = []
    buggy_scaling = [{"threads": 1, "seconds": ref_time, "worker_hits": hits.tolist(),
                      "lost_counts": 0, "max_bin_count_discrepancy": 0,
                      "max_bin_weight_discrepancy": 0.0}]
    for n in (2, 4, 8):
        set_num_threads(n)
        for run in range(1, (10 if n == 8 else 1) + 1):
            (h, w, hits), sec = timed(buggy_hist_omp, data_t1, weights_t1, BINS_T1)
            row = {"threads": n, "run": run, "seconds": sec,
                   "lost_counts": N_T1 - int(np.sum(h)),
                   "max_bin_count_discrepancy": int(np.max(np.abs(h-ref_hist))),
                   "max_bin_weight_discrepancy": float(np.max(np.abs(w-ref_w))),
                   "worker_hits": check_hits(hits, n, N_T1)}
            if run == 1:
                buggy_scaling.append(row)
            if n == 8:
                races.append(row)
            log("Unsafe histogram:", row)
    assert any(row["lost_counts"] > 0 for row in races), "No lost updates observed in 10 runs."
    safe_rows = []
    for n in THREAD_COUNTS:
        set_num_threads(n)
        (hist, wsums, hits), sec = timed(safe_hist_omp, data_t1, weights_t1, BINS_T1)
        assert np.array_equal(hist, ref_hist)
        assert np.allclose(wsums, ref_w, rtol=1e-12, atol=1e-9)
        if n == 1:
            base_h, base_w, base_sec = hist.copy(), wsums.copy(), sec
            # Independent accurately rounded scalar summation of each bin.
            exact_ref = np.array([math.fsum(weights_t1[data_t1 == b]) for b in range(BINS_T1)])
            assert np.array_equal(base_w.view(np.uint64), exact_ref.view(np.uint64))
        assert np.array_equal(hist, base_h)
        assert np.array_equal(wsums.view(np.uint64), base_w.view(np.uint64))
        row = {"threads": n, "seconds": sec, "speedup_vs_1": base_sec / sec,
               "worker_hits": check_hits(hits, n, N_T1),
               "total_count": int(np.sum(hist)), "total_weight": float(np.sum(wsums)),
               "max_weight_difference_vs_naive_1_thread": float(np.max(np.abs(wsums-ref_w))),
               "bit_identical": True}
        safe_rows.append(row)
        log("Safe histogram:", row)
    return hist, wsums, {"unsafe_scaling": buggy_scaling, "unsafe_8_thread_runs": races,
                         "safe_scaling": safe_rows, "diagnosis": DIAGNOSIS_T1}


def run_task2():
    print("\nTASK 2 - FALSE SHARING", flush=True)
    set_num_threads(1)
    unpadded_state_omp(1000, STUDENT_ID)
    padded_state_omp(1000, STUDENT_ID)
    # Fail if optimizer removed the intended volatile stores.
    for fn in (unpadded_state_omp, padded_state_omp):
        llvm = fn.inspect_llvm(fn.signatures[0])
        assert "store volatile i64" in llvm and "load volatile i64" in llvm
    rows = []
    for n in THREAD_COUNTS:
        set_num_threads(n)
        (plain, p_hits), plain_sec = timed(unpadded_state_omp, ITERS_T2, STUDENT_ID)
        (padded, q_hits), pad_sec = timed(padded_state_omp, ITERS_T2, STUDENT_ID)
        assert plain.ctypes.data % 64 == 0 and padded.ctypes.data % 64 == 0
        assert np.array_equal(plain, padded[:, 0])
        if n == 1:
            base_p, base_q = plain_sec, pad_sec
        row = {"threads": n, "unpadded_seconds": plain_sec, "padded_seconds": pad_sec,
               "padding_speedup": plain_sec / pad_sec,
               "unpadded_throughput_speedup": n * base_p / plain_sec,
               "padded_throughput_speedup": n * base_q / pad_sec,
               "unpadded_worker_hits": check_hits(p_hits, n, n*ITERS_T2),
               "padded_worker_hits": check_hits(q_hits, n, n*ITERS_T2),
               "exact_equality": True, "state": plain.tolist()}
        rows.append(row)
        log("False sharing:", row)
    return padded[:, 0].copy(), {"scaling": rows, "speedup_at_8_threads": rows[-1]["padding_speedup"],
                                 "stride_bytes": PAD_STRIDE * 8, "diagnosis": DIAGNOSIS_T2}


def run_task3():
    print("\nTASK 3 - LOAD BALANCING", flush=True)
    work_units = 40 + ((np.arange(ROWS_T3, dtype=np.int64) * 7919 + STUDENT_ID) % SKEW_MOD)**2
    total = int(np.sum(work_units))
    set_num_threads(1)
    static_skew_omp(work_units[:32])
    balanced_skew_omp(work_units[:32], build_chunks(work_units[:32], 1))
    rows = []
    for n in THREAD_COUNTS:
        set_num_threads(n)
        chunks, prep_sec = timed(build_chunks, work_units, n)
        (out_s, ops_s), sec_s = timed(static_skew_omp, work_units)
        (out_b, ops_b, claimed, cursor), sec_b = timed(balanced_skew_omp, work_units, chunks)
        assert np.all(claimed == chunks.shape[0] // n)
        assert int(cursor) == chunks.shape[0]
        diff = float(np.max(np.abs(out_s-out_b)))
        assert diff < 1e-11, diff
        if n == 1:
            ref = out_s.copy()
            base_s, base_b = sec_s, sec_b
        assert float(np.max(np.abs(ref-out_b))) < 1e-11
        ratio = imbalance(ops_b)
        row = {"threads": n, "static_seconds": sec_s, "balanced_seconds": sec_b,
               "schedule_preparation_seconds": prep_sec,
               "balanced_end_to_end_seconds": prep_sec+sec_b,
               "static_speedup_vs_1": base_s/sec_s, "balanced_speedup_vs_1": base_b/sec_b,
               "static_worker_ops": check_hits(ops_s, n, total),
               "balanced_worker_ops": check_hits(ops_b, n, total),
               "static_imbalance_ratio": imbalance(ops_s), "balanced_imbalance_ratio": ratio,
               "chunks_per_worker": claimed.tolist(), "total_chunks": int(cursor),
               "max_abs_diff": diff}
        log("Workload:", row)
        if n == 8:
            assert ratio < 1.15, f"Load imbalance {ratio} must be < 1.15"
        rows.append(row)
    return out_b, {"scaling": rows, "total_work_units": total, "diagnosis": DIAGNOSIS_T3}


def run_task4():
    print("\nTASK 4 - RED-BLACK SOR", flush=True)
    original = initial_grid()
    set_num_threads(1)
    solve_pde_omp(original[:12, :12].copy(), OMEGA_SOR, 1)
    rows = []
    for n in THREAD_COUNTS:
        set_num_threads(n)
        (grid, residual, hits), sec = timed(solve_pde_omp, original, OMEGA_SOR, SWEEPS_T4)
        assert np.all(np.isfinite(grid)) and np.all(np.isfinite(residual))
        assert np.array_equal(grid[0, :], original[0, :])
        assert np.array_equal(grid[-1, :], original[-1, :])
        assert np.array_equal(grid[:, 0], original[:, 0])
        assert np.array_equal(grid[:, -1], original[:, -1])
        assert residual[-1] < residual[0]
        if n == 1:
            ref, ref_res, base_sec = grid.copy(), residual.copy(), sec
        diff = float(np.max(np.abs(grid-ref)))
        residual_diff = float(np.max(np.abs(residual-ref_res)))
        assert diff < 1e-9 and residual_diff < 1e-9, (diff, residual_diff)
        # Additional discrete-equation defect; not the required update residual.
        defect = float(np.max(np.abs(grid[1:-1, 1:-1] - 0.25*(
            grid[:-2, 1:-1]+grid[2:, 1:-1]+grid[1:-1, :-2]+grid[1:-1, 2:]))))
        row = {"threads": n, "seconds": sec, "speedup_vs_1": base_sec/sec,
               "worker_hits": check_hits(hits, n, SWEEPS_T4*(GRID_H-2)*(GRID_W-2)),
               "sweeps": SWEEPS_T4, "max_abs_diff": diff,
               "residual_history_max_abs_diff": residual_diff,
               "first_max_delta": float(residual[0]), "final_max_delta": float(residual[-1]),
               "discrete_laplace_defect": defect,
               "max_delta_history": residual.tolist()}
        rows.append(row)
        log("SOR:", {k: v for k, v in row.items() if k != "max_delta_history"})
    return grid, {"scaling": rows, "omega": float(OMEGA_SOR), "grid_shape": [GRID_H, GRID_W]}


def main():
    assert isinstance(STUDENT_ID, int) and STUDENT_ID > 0
    assert (os.cpu_count() or 1) >= 4, "Local multi-core CPU (>=4 logical cores) required; no Colab!"
    assert nb.config.NUMBA_NUM_THREADS >= 8, (
        "The required eight-thread gate needs a Numba pool of at least 8 threads. "
        "Use a local workstation with at least 8 logical CPUs and no restrictive NUMBA_NUM_THREADS.")
    set_num_threads(8)
    _omp_probe()
    assert nb.threading_layer() == "omp", f"Expected 'omp', got {nb.threading_layer()}"
    rng = np.random.default_rng(STUDENT_ID)
    data_t1 = rng.integers(0, BINS_T1, size=N_T1, dtype=np.int32)
    weights_t1 = rng.uniform(0.25, 2.50, size=N_T1).astype(np.float64)
    print(f"Backend: {nb.threading_layer()} | Logical Cores: {os.cpu_count()} | Student: {STUDENT_ID}", flush=True)
    params = {"N_T1": N_T1, "BINS_T1": BINS_T1, "ITERS_T2": ITERS_T2,
              "PAD_STRIDE": PAD_STRIDE, "ROWS_T3": ROWS_T3, "SKEW_MOD": SKEW_MOD,
              "GRID_H": GRID_H, "GRID_W": GRID_W, "OMEGA_SOR": float(OMEGA_SOR)}
    log("Parameters:", params)
    hist_t1, wsums_t1, t1 = run_task1(data_t1, weights_t1)
    state_t2, t2 = run_task2()
    out_t3, t3 = run_task3()
    grid_t4, t4 = run_task4()

    # FINAL SECTION: DETERMINISTIC VERIFICATION & TELEMETRY BUNDLE
    set_num_threads(8 if (os.cpu_count() or 1) >= 8 else os.cpu_count())
    for idx, diag in enumerate([DIAGNOSIS_T1, DIAGNOSIS_T2, DIAGNOSIS_T3], start=1):
        assert isinstance(diag, str) and len(diag.strip()) >= 80, f"Complete DIAGNOSIS_T{idx}!"
    checksum_blob = (
        f"{STUDENT_ID}|"
        f"{int(np.sum(hist_t1))}|{float(np.sum(wsums_t1)):.4f}|"
        f"{int(np.bitwise_xor.reduce(state_t2))}|"
        f"{float(np.sum(out_t3)):.6f}|"
        f"{float(np.sum(grid_t4)):.6f}"
    )
    lab_digest = hashlib.sha256(checksum_blob.encode()).hexdigest()[:20].upper()
    bundle = {
        "instructor": "Instructor - Sufyan bin Uzayr",
        "student_name": STUDENT_NAME,
        "student_id": STUDENT_ID,
        "omp_layer": nb.threading_layer(),
        "cpu_cores": os.cpu_count(),
        "verification_digest": lab_digest,
        "checksum_blob": checksum_blob,
        "all_assertions_passed": True,
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "python": platform.python_version(), "numpy": np.__version__,
                        "numba": nb.__version__},
        "parameters": params,
        "task_1": t1, "task_2": t2, "task_3": t3, "task_4": t4
    }
    text = json.dumps(bundle, indent=2, allow_nan=False)
    path = Path(__file__).resolve().with_name(f"openmp_lab_{STUDENT_ID}_verification.json")
    path.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved verification JSON: {path}", flush=True)
    print("=== OPENMP PYTHON LAB VERIFICATION BUNDLE ===")
    print(text)

if __name__ == "__main__":
    main()
