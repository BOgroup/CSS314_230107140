"""Numerical and cross-artifact checks for the complete practice repository."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("simulator", ROOT / "benchmarks" / "simulator.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def integer_cuberoot(n: int) -> int:
    lo, hi = 0, 1
    while hi**3 <= n:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid**3 <= n:
            lo = mid
        else:
            hi = mid
    return lo


class FormulaTests(unittest.TestCase):
    def test_endpoints(self):
        for a in [0, 0.001, 0.04, 0.08, 0.20, 1.0]:
            self.assertAlmostEqual(MODULE.amdahl(1, a), 1.0)
            self.assertAlmostEqual(MODULE.gustafson(1, a), 1.0)
        self.assertAlmostEqual(MODULE.gustafson(128, .08), 117.84)
        self.assertAlmostEqual(MODULE.amdahl(128, .08), 11.469534050179211)

    def test_task06_and_task07(self):
        self.assertAlmostEqual(MODULE.gustafson(1024, .04), 983.08)
        self.assertAlmostEqual(MODULE.gustafson(16384, .04), 15728.68)
        self.assertAlmostEqual(MODULE.gustafson(8192, 692/8191), 7500.0)

    def test_task09(self):
        self.assertAlmostEqual(MODULE.gustafson(512, .08), 471.12)
        self.assertAlmostEqual(MODULE.gustafson(2048, .08), 1884.24)

    def test_task10_integer_memory(self):
        self.assertEqual(integer_cuberoot(64 * 32 * 2**30 // 8), 6501)
        self.assertEqual(integer_cuberoot(512 * 32 * 2**30 // 8), 13003)
        self.assertEqual(integer_cuberoot(64 * 32 * 2**30 // 16), 5160)
        self.assertEqual(integer_cuberoot(512 * 32 * 2**30 // 16), 10321)

    def test_model_speedup_bounds(self):
        for p in [1, 2, 16, 1024, 65536]:
            for a in [.001, .04, .08, .2]:
                self.assertLessEqual(MODULE.gustafson(p, a), p + 1e-12)
                self.assertGreaterEqual(MODULE.gustafson(p, a), 1.0 - 1e-12)
                self.assertLessEqual(MODULE.amdahl(p, a), p + 1e-12)
                self.assertGreaterEqual(MODULE.amdahl(p, a), 1.0 - 1e-12)


class ArtifactTests(unittest.TestCase):
    def test_raw_counts_and_checksum(self):
        with (ROOT / "benchmarks" / "results.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 25)
        self.assertEqual(sorted(set(int(r["threads"]) for r in rows)), [1,2,3,4,5])
        for p in range(1, 6):
            self.assertEqual(sum(int(r["threads"]) == p for r in rows), 5)
        for r in rows:
            self.assertEqual(int(r["work_per_thread"]), 12000000)
            self.assertEqual(int(r["total_samples"]), 12000000 * int(r["threads"]))
            self.assertLess(float(r["relative_checksum_error"]), 1e-10)

    def test_alpha_and_exported_models(self):
        with (ROOT / "benchmarks" / "results.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        points = [(int(r['threads']), float(r['scaled_speedup'])) for r in rows if int(r['threads']) > 1]
        num = sum((p - 1)*(p - s) for p, s in points)
        den = sum((p-1)**2 for p,s in points)
        fit = json.loads((ROOT / "benchmarks" / "fit_summary.json").read_text())
        self.assertAlmostEqual(num / den, fit['alpha_raw_unconstrained'])
        with (ROOT / "benchmarks" / "simulated_scaling.csv").open() as fh:
            rows_sim = list(csv.DictReader(fh))
        self.assertEqual(len(rows_sim), 68)
        for r in rows_sim:
            p, a = int(r['cores']), float(r['serial_fraction'])
            self.assertAlmostEqual(float(r['gustafson_scaled_speedup']), MODULE.gustafson(p,a))

    def test_plots_are_actual_pdf_and_png(self):
        for name in ['scaling_comparison', 'efficiency_comparison', 'simulator_overlay']:
            self.assertTrue((ROOT / 'plots' / (name + '.png')).read_bytes().startswith(b'\x89PNG'))
            self.assertTrue((ROOT / 'plots' / (name + '.pdf')).read_bytes().startswith(b'%PDF'))


if __name__ == '__main__':
    unittest.main()
