#!/usr/bin/env python3
"""Record reproducible, nonprivileged hardware/OS environment metadata (Task 24)."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def command(argv):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else f"unavailable ({r.stderr.strip()})"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable ({e})"


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return "unavailable"


def main():
    cpu_aff = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    caches = []
    for d in sorted(Path("/sys/devices/system/cpu/cpu0/cache").glob("index*")):
        caches.append({k: read(d / k) for k in ["level", "type", "size", "shared_cpu_list", "coherency_line_size"]})
    quota = read("/sys/fs/cgroup/cpu.max")
    parts = quota.split()
    effective_cores = None
    if len(parts) >= 2 and parts[0] != "max":
        try:
            effective_cores = int(parts[0]) / int(parts[1])
        except (ValueError, ZeroDivisionError):
            pass
    data = {
        "audit_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "cpu_model": next((line.partition(":")[2].strip() for line in read("/proc/cpuinfo").splitlines() if line.startswith("model name")), "unavailable"),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "logical_cpu_count_visible": os.cpu_count(),
        "process_cpu_affinity": cpu_aff,
        "cpu_affinity_count": len(cpu_aff),
        "cgroup_cpu_max": quota,
        "effective_quota_cores": effective_cores,
        "numa_online_nodes": read("/sys/devices/system/node/online"),
        "numa_possible_nodes": read("/sys/devices/system/node/possible"),
        "cpu0_cache_index_entries": caches,
        "lscpu_summary": command(["lscpu"]),
        "compiler": command(["g++", "--version"]).splitlines()[0],
        "python": platform.python_version(),
        "memory_channels": "Not exposed to unprivileged OS tools in this environment; no reliable channel count.",
        "thermal_throttling": "No temperature or throttling telemetry captured; cannot establish causation.",
    }
    out = ROOT / "analysis" / "hardware_audit.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
