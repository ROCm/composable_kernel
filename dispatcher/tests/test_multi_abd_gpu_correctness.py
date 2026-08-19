#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU correctness test for the MULTI-ABD GEMM dispatcher bridge (PR #9305).

Multi-ABD combines groups of A and B tensors element-wise into a single A / B,
does one GEMM, then folds D tensors in the epilogue:
``E = CDE( AB(A0,A1,..) @ BB(B0,B1,..), D0, D1, .. )``. This test builds a real
multi_abd dispatcher .so, runs it on-device via ``GpuMultiABDRunner``, and uses
the runner's built-in numeric verification (``verify=True``) -- which mirrors
ck_tile::reference_gemm_multiple_abd in fp32 -- to assert ``max_rel`` within the
fp16 tolerance. A separate non-zero / finite guard on the device output catches
an all-zero (mis-launched) kernel.

multi_abd is registry-BYPASS (its .so links only the force-included kernel), so
it needs NO dispatcher static lib -- only a GPU + hipcc. The TE multi_abd op is
fp16-only, so only fp16 is exercised.

Runs green on gfx942 (MI300X). SKIPs cleanly with no GPU / hipcc.

Run:
  python3 -m pytest tests/test_multi_abd_gpu_correctness.py -v
  python3 tests/test_multi_abd_gpu_correctness.py
"""

import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    GemmProblem,
    GpuMultiABDRunner,
    setup_multiple_gemm_dispatchers,
)

_TOL = 2e-2  # fp16 multi-tensor combine + GEMM + D-fuse precision band


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


class TestMultiAbdGemmGpu(unittest.TestCase):
    ARCH = _detect_arch()

    def setUp(self):
        if self.ARCH is None:
            self.skipTest("no GPU / rocminfo not available")
        if shutil.which("hipcc") is None and not Path("/opt/rocm/bin/hipcc").exists():
            self.skipTest("hipcc not found")

    def test_fp16_multiabd_add(self):
        # A/B groups summed (MultiDAdd), D tensors summed into the epilogue
        # (MultiDAdd) -- the fully-fused path, not a PassThrough no-op.
        na, nb, nd = 2, 2, 2
        cfg = GemmKernelConfig(
            dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32",
            layout_a="row", layout_b="col", layout_c="row", layout_d="row",
            tile_m=128, tile_n=128, tile_k=32,
            wave_m=2, wave_n=2, wave_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
            pad_m=True, pad_n=True, pad_k=True, persistent=False,
            variant="multi_abd",
            num_a_tensors=na, num_b_tensors=nb, num_d_tensors=nd,
            a_elementwise_op="MultiDAdd",
            b_elementwise_op="MultiDAdd",
            cde_elementwise_op="MultiDAdd",
            gfx_arch=self.ARCH,
        )
        so_paths = setup_multiple_gemm_dispatchers([cfg], verbose=False)
        so = so_paths[0]
        if so is None:
            self.fail("multi_abd fp16 kernel failed to build")

        M, N, K = 512, 512, 512
        problem = GemmProblem(M=M, N=N, K=K)

        runner = GpuMultiABDRunner(
            so,
            layout4=cfg.layout4,
            a_elementwise_op="MultiDAdd",
            b_elementwise_op="MultiDAdd",
            cde_elementwise_op="MultiDAdd",
        )
        # The runner generates A/B/D itself and owns the fp32 reference, so a
        # verify=True run returns max_rel = max|E_gpu - E_ref| / max|E_ref|.
        result = runner.run(problem, seed=3, verify=True, verify_tol=_TOL)
        self.assertEqual(result.status, 0,
                         f"multi_abd run status={result.status}")
        self.assertGreater(result.time_ms, 0.0,
                           "multi_abd time_ms not positive")

        E_got = np.asarray(result.output).astype(np.float32)
        self.assertFalse(np.all(E_got == 0.0), "multi_abd output all-zero")
        self.assertTrue(np.all(np.isfinite(E_got)), "multi_abd output NaN/Inf")

        self.assertIsNotNone(result.max_rel,
                             "runner did not compute max_rel (verify failed)")
        self.assertLessEqual(
            result.max_rel, _TOL,
            f"multi_abd fp16 max_rel={result.max_rel:.4f} > tol={_TOL} "
            f"(M={M} N={N} K={K})",
        )
        print(f"[multi_abd/fp16] max_rel={result.max_rel:.4e}, "
              f"time_ms={result.time_ms:.3f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
