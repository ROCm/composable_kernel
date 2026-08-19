#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU correctness test for the GROUPED GEMM dispatcher bridge (PR #9000).

The grouped bridge (Tile Engine -> Dispatcher) launches one persistent kernel
over a batch of independent (M, N, K) sub-problems. This test builds a real
grouped dispatcher .so, runs it on-device via ``GpuGroupedGemmRunner``, and
compares every group's output to an independent fp32 numpy reference
(``C_g = A_g @ B_g``) within a dtype-appropriate max-relative-error tolerance.

It is a REAL numeric check -- no trivial/always-pass asserts:
  * random A/B per group, distinct shapes,
  * device output validated element-wise vs. a fp32 matmul reference,
  * a non-zero / finite guard so an all-zero (mis-launched) kernel FAILS.

Runs green on gfx942 (MI300X). SKIPs cleanly when no GPU / hipcc / static lib
is present, so it never reds CPU-only CI.

Run:
  python3 -m pytest tests/test_grouped_gemm_gpu_correctness.py -v
  python3 tests/test_grouped_gemm_gpu_correctness.py            # standalone
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
    GpuGroupedGemmRunner,
    GroupedGemmProblem,
    numpy_dtype_for,
    setup_multiple_gemm_dispatchers,
)

# fp16/bf16 GEMM at K~512 accumulates enough rounding that a per-element floor
# is needed; 1e-2 max-rel (with a max-magnitude denominator floor) is the same
# precision band the bquant GPU test uses for its 2-byte outputs.
_TOL = {"fp16": 1e-2, "bf16": 1e-2}


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


def _static_lib_present():
    try:
        import ctypes_utils as _cu
        return (_cu.get_build_dir() / "libck_tile_dispatcher.a").exists()
    except Exception:
        return False


def _max_rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    g = got.astype(np.float32)
    r = ref.astype(np.float32)
    ref_max = float(np.abs(r).max())
    den = np.abs(r) + max(ref_max * 1e-2, 1e-6)
    return float(np.max(np.abs(g - r) / den))


class TestGroupedGemmGpu(unittest.TestCase):
    ARCH = _detect_arch()

    def setUp(self):
        if self.ARCH is None:
            self.skipTest("no GPU / rocminfo not available")
        if not _static_lib_present():
            self.skipTest(
                "dispatcher static lib (libck_tile_dispatcher.a) not built; "
                "grouped is registry-routed and needs it"
            )
        if shutil.which("hipcc") is None and not Path("/opt/rocm/bin/hipcc").exists():
            self.skipTest("hipcc not found")

    def _run_dtype(self, dtype: str):
        layout = "rcr"  # grouped C is always row-major; rcr = A row, B col.
        cfg = GemmKernelConfig(
            dtype_a=dtype, dtype_b=dtype, dtype_c=dtype, dtype_acc="fp32",
            layout_a="row", layout_b="col", layout_c="row",
            tile_m=128, tile_n=128, tile_k=32,
            wave_m=2, wave_n=2, wave_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
            pad_m=True, pad_n=True, pad_k=True, persistent=False,
            variant="grouped", gfx_arch=self.ARCH,
        )
        so_paths = setup_multiple_gemm_dispatchers([cfg], verbose=False)
        so = so_paths[0]
        if so is None:
            self.fail(f"grouped {dtype} kernel failed to build")

        # Three distinct sub-problems -> exercises per-group offset math.
        groups = [(512, 512, 512), (256, 384, 512), (128, 512, 256)]
        problem = GroupedGemmProblem(groups=groups)

        rng = np.random.default_rng(7)
        A_list, B_list = [], []
        for (M, N, K) in groups:
            A_list.append(rng.uniform(-1.0, 1.0, (M, K)).astype(np.float32))
            B_list.append(rng.uniform(-1.0, 1.0, (K, N)).astype(np.float32))

        runner = GpuGroupedGemmRunner(so, dtype=dtype, layout=layout)
        result = runner.run(A_list, B_list, problem)
        self.assertEqual(result.status, 0,
                         f"grouped {dtype} run status={result.status}")
        self.assertGreater(result.time_ms, 0.0,
                           f"grouped {dtype} time_ms not positive")

        # The runner casts A/B to the kernel's host codec (fp16 -> np.float16,
        # bf16 -> ml_dtypes.bfloat16) and returns C in that same codec WITHOUT
        # decoding, so build the reference by round-tripping through the identical
        # numpy dtype and reading the device output back as fp32 from it.
        in_nd = numpy_dtype_for(dtype)

        worst = 0.0
        for gi, ((M, N, K), C_gpu) in enumerate(zip(groups, result.outputs)):
            A_q = A_list[gi].astype(in_nd).astype(np.float32)
            B_q = B_list[gi].astype(in_nd).astype(np.float32)
            C_ref = (A_q @ B_q).astype(np.float32)
            C_got = np.asarray(C_gpu).astype(np.float32)

            self.assertFalse(np.all(C_got == 0.0),
                             f"grouped {dtype} group {gi} output all-zero")
            self.assertTrue(np.all(np.isfinite(C_got)),
                            f"grouped {dtype} group {gi} output has NaN/Inf")
            mre = _max_rel_err(C_got, C_ref)
            worst = max(worst, mre)
            self.assertLessEqual(
                mre, _TOL[dtype],
                f"grouped {dtype} group {gi} max_rel={mre:.4f} "
                f"> tol={_TOL[dtype]} (M={M} N={N} K={K})",
            )
        print(f"[grouped/{dtype}] worst max_rel={worst:.4e}, "
              f"time_ms={result.time_ms:.3f}")

    def test_fp16(self):
        self._run_dtype("fp16")

    def test_bf16(self):
        self._run_dtype("bf16")


if __name__ == "__main__":
    unittest.main(verbosity=2)
