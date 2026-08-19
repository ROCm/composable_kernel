#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU correctness test for the MX (microscaling) GEMM dispatcher bridge (PR #9329).

MX GEMM computes a block-scaled low-precision GEMM: A/B are fp8 (e4m3) or fp4
(e2m1) with a per-32-K e8m0 block scale on each of A and B, accumulated in fp32
to fp16. This test builds a real mx_gemm dispatcher .so, runs it on-device via
``GpuMxGemmRunner``, and compares C to the block-scaled fp32 numpy reference
``mx_gemm_reference`` within a low-precision (5e-2) tolerance.

MX GEMM is gfx950-ONLY (CDNA4 / MI350): the C++ bridge static_asserts
GFX_ARCH == "gfx950" and uses the gfx950 pre-shuffle scale helper, and the fp8
codec is OCP e4m3 (gfx950 default). This test therefore ARCH-GATES to gfx950 --
on gfx942 / any other arch it SKIPs cleanly so it never reds CI on those nodes.
Verification of the assertions themselves is pending a gfx950 node.

Run (on a gfx950 node):
  python3 -m pytest tests/test_mx_gemm_gpu_correctness.py -v
  python3 tests/test_mx_gemm_gpu_correctness.py
"""

import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from mx_gemm_utils import (  # noqa: E402
    GpuMxGemmRunner,
    MxGemmProblem,
    default_fp4_config,
    default_fp8_config,
    setup_multiple_mx_gemm_dispatchers,
)

_TOL = 5e-2  # fp8/fp4 block-scaled precision floor


def _detect_arch():
    # subprocess.run with a timeout (rocminfo can hang on misconfigured ROCm
    # installs and would otherwise stall test discovery). Validate the parsed
    # token is a real gfx string before returning it. Mirrors
    # dispatcher/python/ctypes_utils.py:detect_gpu_arch.
    import subprocess
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        )
    except Exception:
        return None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Name:") and "gfx" in stripped:
            name = stripped.split(":", 1)[1].strip()
            if name.startswith("gfx") and name[3:].isdigit():
                return name
    return None


def _max_rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    g = got.astype(np.float32)
    r = ref.astype(np.float32)
    ref_max = float(np.abs(r).max())
    den = np.abs(r) + max(ref_max * 1e-2, 1e-6)
    return float(np.max(np.abs(g - r) / den))


class TestMxGemmGpu(unittest.TestCase):
    ARCH = _detect_arch()

    def setUp(self):
        # Hard arch-gate: mx_gemm is gfx950-only. Skip (not fail) elsewhere so
        # gfx942 / CPU-only CI stays green.
        if self.ARCH is None:
            self.skipTest("no GPU / rocminfo not available")
        if self.ARCH != "gfx950":
            self.skipTest(f"mx_gemm is gfx950-only; detected {self.ARCH}")
        if shutil.which("hipcc") is None and not Path("/opt/rocm/bin/hipcc").exists():
            self.skipTest("hipcc not found")

    def _run_dtype(self, dtype: str, cfg):
        so_paths = setup_multiple_mx_gemm_dispatchers(
            [cfg], gfx_arch="gfx950", parallel=False,
        )
        so = so_paths[0]
        if so is None:
            self.fail(f"mx_gemm {dtype} kernel failed to build")

        M, N, K = 512, 512, 512  # K % 32 == 0 required
        problem = MxGemmProblem(M=M, N=N, K=K, k_batch=1)

        runner = GpuMxGemmRunner(so, dtype=dtype, arch="gfx950")
        A_deq, B_deq, A_bytes, B_bytes, sa, sb = runner.make_inputs(
            problem, scale=1.0, seed=5,
        )
        result = runner.run(problem, A_bytes, B_bytes, sa, sb)
        C_got = np.asarray(result.C).astype(np.float32)
        self.assertGreater(result.time_ms, 0.0,
                           f"mx_gemm {dtype} time_ms not positive")
        self.assertFalse(np.all(C_got == 0.0),
                         f"mx_gemm {dtype} output all-zero")
        self.assertTrue(np.all(np.isfinite(C_got)),
                        f"mx_gemm {dtype} output NaN/Inf")

        C_ref = runner.reference(A_deq, B_deq, sa, sb, problem).astype(np.float32)
        mre = _max_rel_err(C_got, C_ref)
        self.assertLessEqual(
            mre, _TOL,
            f"mx_gemm {dtype} max_rel={mre:.4f} > tol={_TOL} "
            f"(M={M} N={N} K={K})",
        )
        print(f"[mx_gemm/{dtype}] max_rel={mre:.4e}, time_ms={result.time_ms:.3f}")

    def test_fp8(self):
        self._run_dtype("fp8", default_fp8_config(gfx_arch="gfx950"))

    def test_fp4(self):
        self._run_dtype("fp4", default_fp4_config(gfx_arch="gfx950"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
