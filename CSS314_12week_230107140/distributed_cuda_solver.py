"""4-rank MPI + CUDA reaction-diffusion stencil laboratory.

Student ID: 230107140.  Execute on a CUDA-enabled Google Colab runtime:
    mpirun --allow-run-as-root --oversubscribe -np 4 \
        python3 distributed_cuda_solver.py YOUR_ACTUAL_GITHUB_USERNAME

The per-run GitHub/GPU seed, timings, PTX counts, and energy values are
measured at runtime.  No hardware-dependent results are prepopulated.
"""

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from numba import cuda, float64

STUDENT_ID = "230107140"
GLOBAL_NX = 512
GLOBAL_NY = 512
TILE_DIM = 16
SHARED_DIM = 18
STEPS = 400
LAMBDA_DAMP = 0.15

UP, DOWN, LEFT, RIGHT = range(4)
TAGS = (101, 102, 103, 104)
OPPOSITE = (DOWN, UP, RIGHT, LEFT)
DIRECTION_NAMES = ("up", "down", "left", "right")


def get_gpu_uuid() -> str:
    """Identify the *active* GPU; do not substitute a placeholder UUID."""
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    )
    uuid = raw.strip().splitlines()[0].strip()
    if not uuid or not uuid.startswith("GPU-"):
        raise RuntimeError(f"Invalid active GPU UUID: {uuid!r}")
    return uuid


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Starter specification: first 64 bits of SHA256(username::GPU-UUID)."""
    message = f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")
    return int(hashlib.sha256(message).hexdigest()[:16], 16)


def init_distributed_context(github_user: str) -> dict:
    world = MPI.COMM_WORLD
    size = world.Get_size()
    rank = world.Get_rank()
    if size != 4:
        raise RuntimeError(f"Exactly 4 MPI processes are required, received {size}")
    if not github_user.strip() or github_user == "REPLACE_WITH_YOUR_GITHUB_USERNAME":
        raise ValueError("Use the actual GitHub username on the command line")

    cuda.select_device(0)
    cart = world.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    coords = tuple(int(v) for v in cart.Get_coords(cart.Get_rank()))
    nbr_up, nbr_down = cart.Shift(0, 1)
    nbr_left, nbr_right = cart.Shift(1, 1)
    uuid = get_gpu_uuid() if rank == 0 else None
    uuid = world.bcast(uuid, root=0)
    instance_seed = derive_student_seed(github_user, uuid)
    ny, nx = GLOBAL_NY // 2, GLOBAL_NX // 2
    dx = dy = 1.0 / (GLOBAL_NX + 1)
    device = cuda.get_current_device()

    return {
        "world": world,
        "cart": cart,
        "rank": rank,
        "size": size,
        "coords": coords,
        "neighbors": (nbr_up, nbr_down, nbr_left, nbr_right),
        "gpu_uuid": uuid,
        "gpu_name": device.name.decode() if isinstance(device.name, bytes) else str(device.name),
        "compute_capability": list(device.compute_capability),
        "instance_seed": instance_seed,
        "github_username": github_user.strip(),
        "ny": ny,
        "nx": nx,
        "dx": dx,
        "dy": dy,
    }


def initialize_subdomains(ctx: dict) -> tuple:
    """Construct only local (258,258) host fields, including analytic ghost data."""
    ny, nx = ctx["ny"], ctx["nx"]
    row_rank, col_rank = ctx["coords"]
    gi = row_rank * ny + np.arange(ny + 2, dtype=np.float64)
    gj = col_rank * nx + np.arange(nx + 2, dtype=np.float64)
    y = gi[:, None] * ctx["dy"]
    x = gj[None, :] * ctx["dx"]

    rng = np.random.default_rng(np.uint64(ctx["instance_seed"]))
    phi_x, phi_y = rng.uniform(0.0, 2.0 * np.pi, size=2)
    kappa = 0.16 + 0.08 * np.sin(2.0 * np.pi * x + phi_x) * np.cos(
        2.0 * np.pi * y + phi_y
    )

    p1x, p1y = rng.uniform(0.30, 0.47, size=2)
    p2x, p2y = rng.uniform(0.53, 0.70, size=2)
    a1, a2 = rng.uniform(0.8, 1.2, size=2)
    g1 = np.exp(-((x - p1x) ** 2 + (y - p1y) ** 2) / (2.0 * 0.085**2))
    g2 = np.exp(-((x - p2x) ** 2 + (y - p2y) ** 2) / (2.0 * 0.095**2))
    u0 = a1 * g1 + a2 * g2

    # Localized source/sink pair mirrored across the domain center.  Its
    # spatial integral is (up to roundoff) zero; damping dissipates mass.
    q1 = np.exp(-((x - 0.32) ** 2 + (y - 0.48) ** 2) / (2.0 * 0.035**2))
    q2 = np.exp(-((x - 0.68) ** 2 + (y - 0.52) ** 2) / (2.0 * 0.035**2))
    source = 0.02 * (q1 - q2)

    for field in (u0, kappa, source):
        assert field.shape == (ny + 2, nx + 2)
    assert np.all(kappa >= 0.08 - 1e-14) and np.all(kappa <= 0.24 + 1e-14)

    up, down, left, right = ctx["neighbors"]
    if up == MPI.PROC_NULL:
        u0[0, :] = 0.0
    if down == MPI.PROC_NULL:
        u0[-1, :] = 0.0
    if left == MPI.PROC_NULL:
        u0[:, 0] = 0.0
    if right == MPI.PROC_NULL:
        u0[:, -1] = 0.0

    return (
        np.ascontiguousarray(u0, dtype=np.float64),
        np.ascontiguousarray(kappa, dtype=np.float64),
        np.ascontiguousarray(source, dtype=np.float64),
    )


@cuda.jit
def pack_halos_kernel(u, packed, ny, nx):
    """One GPU thread packs one element in each of four strided borders."""
    k = cuda.grid(1)
    if k < nx:
        packed[UP, k] = u[1, k + 1]
        packed[DOWN, k] = u[ny, k + 1]
    if k < ny:
        packed[LEFT, k] = u[k + 1, 1]
        packed[RIGHT, k] = u[k + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, packed, ny, nx):
    """One GPU thread injects 4 received values into ghost rows/columns."""
    k = cuda.grid(1)
    if k < nx:
        u[0, k + 1] = packed[UP, k]
        u[ny + 1, k + 1] = packed[DOWN, k]
    if k < ny:
        u[k + 1, 0] = packed[LEFT, k]
        u[k + 1, nx + 1] = packed[RIGHT, k]


@cuda.jit
def tiled_stencil_kernel(
    u, u_next, kappa, source, dt, inv_dx2, inv_dy2,
    ny, nx, r_start, r_end, c_start, c_end,
):
    """16x16 owned cells staged in a synchronized 18x18 float64 tile.

    All threads, including those outside a short boundary bounding box,
    participate in shared-memory loading and the barrier.  Coordinates
    past an array edge are clamped to the valid padded-array indices;
    only threads within [r_start:r_end, c_start:c_end] write output.
    """
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    r = r_start + cuda.blockIdx.y * TILE_DIM + ty
    c = c_start + cuda.blockIdx.x * TILE_DIM + tx

    rr = max(0, min(r, ny + 1))
    cc = max(0, min(c, nx + 1))
    smem = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)
    sy = ty + 1
    sx = tx + 1
    smem[sy, sx] = u[rr, cc]
    if tx == 0:
        smem[sy, 0] = u[rr, max(0, cc - 1)]
    if tx == TILE_DIM - 1:
        smem[sy, SHARED_DIM - 1] = u[rr, min(nx + 1, cc + 1)]
    if ty == 0:
        smem[0, sx] = u[max(0, rr - 1), cc]
    if ty == TILE_DIM - 1:
        smem[SHARED_DIM - 1, sx] = u[min(ny + 1, rr + 1), cc]
    # Complete all four corners of the 18x18 shared tile.  The five-point
    # stencil does not read diagonal neighbors, but a fully populated apron
    # makes the staging invariant explicit and easy to inspect.
    if ty == 0 and tx == 0:
        smem[0, 0] = u[max(0, rr - 1), max(0, cc - 1)]
    if ty == 0 and tx == TILE_DIM - 1:
        smem[0, SHARED_DIM - 1] = u[max(0, rr - 1), min(nx + 1, cc + 1)]
    if ty == TILE_DIM - 1 and tx == 0:
        smem[SHARED_DIM - 1, 0] = u[min(ny + 1, rr + 1), max(0, cc - 1)]
    if ty == TILE_DIM - 1 and tx == TILE_DIM - 1:
        smem[SHARED_DIM - 1, SHARED_DIM - 1] = u[
            min(ny + 1, rr + 1), min(nx + 1, cc + 1)
        ]
    cuda.syncthreads()

    if r <= r_end and c <= c_end and r >= r_start and c >= c_start:
        center = smem[sy, sx]
        up = smem[sy - 1, sx]
        down = smem[sy + 1, sx]
        left = smem[sy, sx - 1]
        right = smem[sy, sx + 1]
        kc = kappa[r, c]
        k_up = 0.5 * (kc + kappa[r - 1, c])
        k_down = 0.5 * (kc + kappa[r + 1, c])
        k_left = 0.5 * (kc + kappa[r, c - 1])
        k_right = 0.5 * (kc + kappa[r, c + 1])
        diffusion_y = (k_down * (down - center) - k_up * (center - up)) * inv_dy2
        diffusion_x = (k_right * (right - center) - k_left * (center - left)) * inv_dx2
        u_next[r, c] = center + dt * (
            diffusion_x + diffusion_y - LAMBDA_DAMP * center**3 + source[r, c]
        )


class SolverBuffers:
    """Persistent device buffers + genuinely page-locked MPI staging buffers."""

    def __init__(self, ctx: dict, u0: np.ndarray, kappa: np.ndarray, source: np.ndarray):
        self.ctx = ctx
        self.u0 = u0
        ny, nx = ctx["ny"], ctx["nx"]
        self.d_u = cuda.to_device(u0)
        self.d_v = cuda.device_array(u0.shape, dtype=np.float64)
        self.d_kappa = cuda.to_device(kappa)
        self.d_source = cuda.to_device(source)
        self.d_send = cuda.device_array((4, nx), dtype=np.float64)
        self.d_recv = cuda.device_array((4, nx), dtype=np.float64)
        self.h_send = cuda.pinned_array((4, nx), dtype=np.float64)
        self.h_recv = cuda.pinned_array((4, nx), dtype=np.float64)
        self.h_recv.fill(0.0)
        self.stream_boundary = cuda.stream()
        self.stream_interior = cuda.stream()
        self.dt = ctx["dt"]
        self.inv_dx2 = 1.0 / (ctx["dx"] ** 2)
        self.inv_dy2 = 1.0 / (ctx["dy"] ** 2)

    def reset(self):
        self.d_u.copy_to_device(self.u0)
        self.d_v.copy_to_device(np.zeros_like(self.u0))
        self.h_recv.fill(0.0)
        cuda.synchronize()

    def _launch_region(self, r0, r1, c0, c1, stream):
        if r1 < r0 or c1 < c0:
            return
        blocks = (
            math.ceil((c1 - c0 + 1) / TILE_DIM),
            math.ceil((r1 - r0 + 1) / TILE_DIM),
        )
        tiled_stencil_kernel[blocks, (TILE_DIM, TILE_DIM), stream](
            self.d_u, self.d_v, self.d_kappa, self.d_source,
            self.dt, self.inv_dx2, self.inv_dy2,
            self.ctx["ny"], self.ctx["nx"], r0, r1, c0, c1,
        )

    def _launch_deep_interior(self, stream):
        self._launch_region(2, self.ctx["ny"] - 1, 2, self.ctx["nx"] - 1, stream)

    def _launch_boundary_rings(self, stream):
        ny, nx = self.ctx["ny"], self.ctx["nx"]
        self._launch_region(1, 1, 1, nx, stream)
        self._launch_region(ny, ny, 1, nx, stream)
        self._launch_region(2, ny - 1, 1, 1, stream)
        self._launch_region(2, ny - 1, nx, nx, stream)

    def _post_halo_messages(self):
        cart = self.ctx["cart"]
        requests = []
        for direction, neighbor in enumerate(self.ctx["neighbors"]):
            if neighbor != MPI.PROC_NULL:
                requests.append(cart.Irecv(
                    self.h_recv[direction],
                    source=neighbor, tag=TAGS[OPPOSITE[direction]],
                ))
        for direction, neighbor in enumerate(self.ctx["neighbors"]):
            if neighbor != MPI.PROC_NULL:
                requests.append(cart.Isend(
                    self.h_send[direction],
                    dest=neighbor, tag=TAGS[direction],
                ))
        return requests

    def exchange_halos_async(self, mode: str, sample_events: bool = False):
        """Pack -> pinned D2H -> nonblocking MPI -> pinned H2D -> unpack.

        In overlapped mode, interior computation is launched immediately
        after posting non-blocking MPI and progresses while MPI Waitall runs.
        """
        ny, nx = self.ctx["ny"], self.ctx["nx"]
        bstream = self.stream_boundary
        istream = self.stream_interior
        sample = {}
        if sample_events:
            for name in ("pack_start", "pack_end", "inner_start", "inner_end", "ring_start", "ring_end"):
                sample[name] = cuda.event(timing=True)
            sample["pack_start"].record(bstream)

        pack_halos_kernel[math.ceil(max(ny, nx) / 256), 256, bstream](
            self.d_u, self.d_send, ny, nx
        )
        # Asynchronous DMA is valid because h_send is allocated via pinned_array.
        self.d_send.copy_to_host(self.h_send, stream=bstream)
        if sample_events:
            sample["pack_end"].record(bstream)
        bstream.synchronize()  # Host must not send until the D2H DMA completes.

        self.h_recv.fill(0.0)  # MPI.PROC_NULL: homogeneous Dirichlet ghosts.
        requests = self._post_halo_messages()
        if mode == "overlapped":
            if sample_events:
                sample["inner_start"].record(istream)
            self._launch_deep_interior(istream)
            if sample_events:
                sample["inner_end"].record(istream)

        MPI.Request.Waitall(requests)
        if sample_events:
            sample["ring_start"].record(bstream)
        self.d_recv.copy_to_device(self.h_recv, stream=bstream)
        unpack_halos_kernel[math.ceil(max(ny, nx) / 256), 256, bstream](
            self.d_u, self.d_recv, ny, nx
        )
        # The serialized control path waits for all halo updates before any
        # stencil work; the overlapped path only needs the boundary stream.
        if mode == "serialized":
            bstream.synchronize()
            if sample_events:
                sample["inner_start"].record(bstream)
            self._launch_deep_interior(bstream)
            if sample_events:
                sample["inner_end"].record(bstream)
        self._launch_boundary_rings(bstream)
        if sample_events:
            sample["ring_end"].record(bstream)
        bstream.synchronize()
        istream.synchronize()
        self.d_u, self.d_v = self.d_v, self.d_u

        if sample_events:
            return {
                "pack_d2h_ms": float(cuda.event_elapsed_time(sample["pack_start"], sample["pack_end"])),
                "deep_interior_ms": float(cuda.event_elapsed_time(sample["inner_start"], sample["inner_end"])),
                "h2d_unpack_and_rings_ms": float(cuda.event_elapsed_time(sample["ring_start"], sample["ring_end"])),
            }
        return None

    def warm_up(self):
        # Compile both GPU kernels and populate Numba's inspect_asm cache.
        stream = self.stream_boundary
        pack_halos_kernel[1, 256, stream](self.d_u, self.d_send, self.ctx["ny"], self.ctx["nx"])
        self.d_send.copy_to_host(self.h_send, stream=stream)
        self.d_recv.copy_to_device(self.h_recv, stream=stream)
        unpack_halos_kernel[1, 256, stream](self.d_u, self.d_recv, self.ctx["ny"], self.ctx["nx"])
        self._launch_deep_interior(stream)
        self._launch_boundary_rings(stream)
        stream.synchronize()
        self.reset()


def exchange_halos_async(buffers: SolverBuffers, mode="overlapped", sample_events=False):
    """Expose the assignment's required named exchange operation."""
    return buffers.exchange_halos_async(mode=mode, sample_events=sample_events)


def step_overlapped(buffers: SolverBuffers, sample_events=False):
    return exchange_halos_async(buffers, mode="overlapped", sample_events=sample_events)


def step_serialized(buffers: SolverBuffers, sample_events=False):
    return exchange_halos_async(buffers, mode="serialized", sample_events=sample_events)


def distributed_metrics(ctx: dict, current, previous, step: int) -> dict:
    """MPI Allreduce on two scalars; NEVER gather any global field."""
    u = current.copy_to_host()
    owned = u[1:-1, 1:-1]
    local_mass = float(np.sum(owned, dtype=np.float64)) * ctx["dx"] * ctx["dy"]
    if previous is None:
        local_rsq = 0.0
    else:
        prior = previous.copy_to_host()[1:-1, 1:-1]
        local_rsq = float(np.sum((owned - prior) ** 2, dtype=np.float64)) * ctx["dx"] * ctx["dy"]
    send = np.asarray((local_mass, local_rsq), dtype=np.float64)
    recv = np.zeros(2, dtype=np.float64)
    ctx["cart"].Allreduce(send, recv, op=MPI.SUM)
    result = {"step": int(step), "energy": float(recv[0]), "l2_update": float(math.sqrt(recv[1]))}
    if ctx["rank"] == 0:
        print(
            f"  step={step:3d}  energy={result['energy']:.15e}"
            f"  global_L2_update={result['l2_update']:.15e}", flush=True
        )
    return result


def ptx_audit() -> dict:
    # inspect_asm returns a dict when the signature is unspecified in some
    # Numba versions and returns a PTX string in others.
    assembly = tiled_stencil_kernel.inspect_asm()
    if isinstance(assembly, dict):
        assembly = "\n".join(str(value) for value in assembly.values())
    if not assembly:
        raise RuntimeError("No PTX generated for tiled_stencil_kernel")

    def count(pattern):
        return len(re.findall(pattern, assembly, flags=re.MULTILINE))

    result = {
        "st_shared_f64": count(r"\bst\.shared\.f64\b"),
        "ld_shared_f64": count(r"\bld\.shared\.f64\b"),
        "bar_sync": count(r"\bbar\.sync\b"),
        "st_shared_any": count(r"\bst\.shared(?:\.[A-Za-z0-9_]+)+\b"),
        "ld_shared_any": count(r"\bld\.shared(?:\.[A-Za-z0-9_]+)+\b"),
        "ptx_sha256": hashlib.sha256(assembly.encode("utf-8")).hexdigest(),
        "ptx_bytes": len(assembly.encode("utf-8")),
    }
    result["ptx_audit_passed"] = bool(
        result["st_shared_any"] and result["ld_shared_any"] and result["bar_sync"]
    )
    return result


def benchmark_mode(ctx: dict, buffers: SolverBuffers, mode: str) -> dict:
    buffers.reset()
    cart = ctx["cart"]
    cart.Barrier()
    if ctx["rank"] == 0:
        print(f"\n=== {mode.upper()} (400 steps, 4 ranks) ===", flush=True)
    trace = [distributed_metrics(ctx, buffers.d_u, None, step=0)]
    cart.Barrier()
    started = MPI.Wtime()
    events_last_step = {}
    for step in range(1, STEPS + 1):
        events = exchange_halos_async(buffers, mode=mode, sample_events=(step == STEPS))
        if events is not None:
            events_last_step = events
        if step % 100 == 0:
            trace.append(distributed_metrics(ctx, buffers.d_u, buffers.d_v, step=step))
    cuda.synchronize()
    cart.Barrier()
    local_wall_ms = 1000.0 * (MPI.Wtime() - started)
    wall_ms = float(cart.allreduce(local_wall_ms, op=MPI.MAX))
    max_event = {
        name: float(cart.allreduce(val, op=MPI.MAX))
        for name, val in events_last_step.items()
    }
    if ctx["rank"] == 0:
        print(f"{mode}_ms={wall_ms:.5f}", flush=True)
        print("CUDA event ms (last step, max across ranks):", max_event, flush=True)

    return {
        "mode": mode,
        "wall_ms": wall_ms,
        "energy_trace": trace,
        "cuda_events_ms": max_event,
        "final_host": buffers.d_u.copy_to_host(),
    }


def run_solver(github_username: str):
    ctx = init_distributed_context(github_username)
    cart, rank = ctx["cart"], ctx["rank"]
    u0, kappa, source = initialize_subdomains(ctx)
    local_max = np.array(np.max(kappa[1:-1, 1:-1]), dtype=np.float64)
    global_max = np.array(0.0, dtype=np.float64)
    cart.Allreduce(local_max, global_max, op=MPI.MAX)
    ctx["kappa_max"] = float(global_max)
    ctx["dt_max"] = ctx["dx"] ** 2 / (4.0 * ctx["kappa_max"])
    ctx["dt"] = 0.80 * ctx["dt_max"]

    local_source_integral = float(np.sum(source[1:-1, 1:-1])) * ctx["dx"] * ctx["dy"]
    net_source = float(cart.allreduce(local_source_integral, op=MPI.SUM))

    if rank == 0:
        print("== STUDENT / HARDWARE IDENTITY ==", flush=True)
        print(f"student_id={STUDENT_ID}", flush=True)
        print(f"github_username={github_username}", flush=True)
        print(f"gpu_name={ctx['gpu_name']}", flush=True)
        print(f"gpu_uuid={ctx['gpu_uuid']}", flush=True)
        print(f"INSTANCE_SEED={ctx['instance_seed']}", flush=True)
        print("mpi_ranks=4; process_grid=2x2; non_periodic=True", flush=True)
        print("global_shape=512x512; local_padded_shape=258x258", flush=True)
        print(f"dx={ctx['dx']:.15e}, dy={ctx['dy']:.15e}", flush=True)
        print(f"kappa_max={ctx['kappa_max']:.15e}", flush=True)
        print(f"dt_max={ctx['dt_max']:.15e}; dt={ctx['dt']:.15e}", flush=True)
        print(f"global_integral_source={net_source:.15e}", flush=True)
    for p in range(4):
        cart.Barrier()
        if p == rank:
            print(
                f"rank={rank}, coords={ctx['coords']}, neighbors="
                f"{dict(zip(DIRECTION_NAMES, ctx['neighbors']))}", flush=True
            )
    cart.Barrier()

    buffers = SolverBuffers(ctx, u0, kappa, source)
    buffers.warm_up()
    audit = ptx_audit()
    all_audit_passed = bool(cart.allreduce(int(audit["ptx_audit_passed"]), op=MPI.MIN))
    audit["ptx_audit_passed"] = all_audit_passed
    if rank == 0:
        print("\n== PTX AUDIT (actual emitted PTX) ==", flush=True)
        print(json.dumps(audit, indent=2), flush=True)
    if not all_audit_passed:
        raise RuntimeError("PTX audit failed: required shared-memory or barrier instruction missing")

    serialized = benchmark_mode(ctx, buffers, "serialized")
    overlapped = benchmark_mode(ctx, buffers, "overlapped")
    ny, nx = ctx["ny"], ctx["nx"]
    local_parity = float(np.max(np.abs(
        serialized["final_host"][1:ny + 1, 1:nx + 1]
        - overlapped["final_host"][1:ny + 1, 1:nx + 1]
    )))
    parity = float(cart.allreduce(local_parity, op=MPI.MAX))
    energy_monotone = all(
        overlapped["energy_trace"][i + 1]["energy"]
        <= overlapped["energy_trace"][i]["energy"] + 1e-13
        for i in range(len(overlapped["energy_trace"]) - 1)
    )
    speedup = serialized["wall_ms"] / overlapped["wall_ms"]
    if rank == 0:
        print("\n== FINAL VERIFICATION ==", flush=True)
        print(f"serialized_ms={serialized['wall_ms']:.6f}", flush=True)
        print(f"overlapped_ms={overlapped['wall_ms']:.6f}", flush=True)
        print(f"speedup_serialized_over_overlapped={speedup:.6f}", flush=True)
        print(f"parity_max_diff={parity:.15e}", flush=True)
        print(f"parity_passed={parity < 1e-12}", flush=True)
        print(f"energy_monotone={energy_monotone}", flush=True)

    manifest = {
        "student_id": STUDENT_ID,
        "github_username": ctx["github_username"],
        "gpu_uuid": ctx["gpu_uuid"],
        "gpu_name": ctx["gpu_name"],
        "compute_capability": ctx["compute_capability"],
        "INSTANCE_SEED": ctx["instance_seed"],
        "mpi_ranks": ctx["size"],
        "cartesian_dims": [2, 2],
        "cartesian_periods": [False, False],
        "global_shape": [GLOBAL_NY, GLOBAL_NX],
        "local_padded_shape": [ny + 2, nx + 2],
        "steps": STEPS,
        "dx": ctx["dx"],
        "dy": ctx["dy"],
        "lambda_damp": LAMBDA_DAMP,
        "kappa_max": ctx["kappa_max"],
        "dt": ctx["dt"],
        "dt_max": ctx["dt_max"],
        "source_integral_initial": net_source,
        "ptx_audit": audit,
        "ptx_audit_passed": audit["ptx_audit_passed"],
        "serialized_ms": serialized["wall_ms"],
        "overlapped_ms": overlapped["wall_ms"],
        "speedup": speedup,
        "parity_max_diff": parity,
        "parity_passed": bool(parity < 1e-12),
        "energy_monotone": bool(energy_monotone),
        "energy_trace_serialized": serialized["energy_trace"],
        "energy_trace_overlapped": overlapped["energy_trace"],
        "cuda_events_ms_serialized": serialized["cuda_events_ms"],
        "cuda_events_ms_overlapped": overlapped["cuda_events_ms"],
        "completed_on_real_gpu": True,
    }
    if rank == 0:
        Path("submission_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print("Wrote submission_manifest.json", flush=True)
    if parity >= 1e-12:
        raise AssertionError(f"Serialized vs overlapped results differ: {parity}")
    if not energy_monotone:
        raise AssertionError("Recorded energy did not decrease at each 100-step checkpoint")
    cart.Barrier()
    cart.Free()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python distributed_cuda_solver.py ACTUAL_GITHUB_USERNAME")
    run_solver(sys.argv[1])
