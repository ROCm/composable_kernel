#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU correctness test for the MX (microscaling) GEMM dispatcher bridge (PR #9329).

MX GEMM computes a block-scaled low-precision GEMM: A/B are fp8 (e4m3) or fp4
(e2m1) with a per-32-K e8m0 block scale on each of A and B, accumulated in fp32
to fp16. This test builds a real mx_gemm dispatcher .so, runs it on-device via
``GpuMxGemmRunner``, and compares C to the block-scaled fp32 numpy reference
``mx_gemm_reference`` within a low-precision (5e-2) tolerance.

Supported devices: gfx950 and gfx1250. gfx1250 uses the non-cluster
16x16x128 WMMA scale32 path, including revision 0.

Run from the CK root:
  CK_TILE_BENCH_WARMUP=1 CK_TILE_BENCH_REPEAT=2 \
    python3 -m unittest discover -s dispatcher/tests -p test_mx_gemm_gpu_correctness.py -v
"""

import shutil
import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from mx_gemm_utils import (  # noqa: E402
    GpuMxGemmRunner,
    MxGemmProblem,
    MxGemmKernelConfig,
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
        # CPU-only and unrelated GPU jobs do not compile MX kernels.
        if self.ARCH is None:
            self.skipTest("no GPU / rocminfo not available")
        if self.ARCH not in ("gfx950", "gfx1250"):
            self.skipTest(f"mx_gemm requires gfx950 or gfx1250; detected {self.ARCH}")
        if shutil.which("hipcc") is None and not Path("/opt/rocm/bin/hipcc").exists():
            self.skipTest("hipcc not found")

    def _run_dtype(self, dtype: str):
        build_dir = tempfile.TemporaryDirectory(prefix="mx_gemm_gpu_test_")
        self.addCleanup(build_dir.cleanup)
        configs = [
            (default_fp4_config if dtype == "fp4" else default_fp8_config)(
                self.ARCH, pipeline
            )
            for pipeline in (
                "comp_async",
                "comp_async_eight_waves",
                "weight_preshuffle",
            )
        ]
        if self.ARCH == "gfx1250":
            configs += [
                MxGemmKernelConfig(
                    datatype=dtype,
                    gpu_target=self.ARCH,
                    pipeline=pipeline,
                    tile_m=m,
                    tile_n=n,
                )
                for pipeline in ("comp_tdm", "comp_tdm_v2")
                for m, n in ((64, 64), (64, 128), (128, 64), (128, 128))
            ]
            # Exceeds the old 64 KiB LDS fallback with both FP4 and FP8.
            configs += [
                MxGemmKernelConfig(
                    datatype=dtype, gpu_target=self.ARCH, pipeline=pipeline,
                    tile_m=256, tile_n=256, tile_k=256,
                )
                for pipeline in ("comp_tdm", "comp_tdm_v2")
            ]
            configs += [
                MxGemmKernelConfig(
                    datatype=dtype, gpu_target=self.ARCH, pipeline=pipeline,
                    warp_tile_m=32, warp_tile_n=32,
                )
                for pipeline in ("comp_tdm", "comp_tdm_v2")
            ]
        so_paths = setup_multiple_mx_gemm_dispatchers(
            configs,
            output_dir=Path(build_dir.name),
            gfx_arch=self.ARCH,
            parallel=True,
            max_workers=2,
        )
        self.assertTrue(all(so_paths), f"mx_gemm {dtype} kernel failed to build")
        for config, so in zip(configs, so_paths):
            runner = GpuMxGemmRunner(so, dtype=dtype, arch=self.ARCH)
            bm, bn, bk = config.tile_m, config.tile_n, config.tile_k
            shapes = [
                (bm, bn, bk),
                (bm, 2 * bn, 2 * bk),
                (2 * bm, bn, 3 * bk),
                (2 * bm, 2 * bn, 4 * bk),
            ]
            if self.ARCH == "gfx1250":
                if config.pipeline == "weight_preshuffle":
                    shapes += [(1, 16, bk), (bm + 1, bn + 16, 2 * bk)]
                else:
                    shapes += [(1, 17, bk), (bm - 1, bn + 1, 2 * bk)]
                shapes += [(bm, bn, 5 * bk), (bm, bn, 8 * bk)]
                if config.pipeline in (
                    "comp_async", "comp_async_eight_waves", "weight_preshuffle"
                ):
                    # FP8 inputs occupy a full 1 MiB allocation here. Smaller
                    # allocations can hide a prefetch past the physical buffer.
                    shapes.append((512, 512, 2048))
            for M, N, K in shapes:
                for seed in (5, 19):
                    with self.subTest(
                        dtype=dtype,
                        pipeline=config.pipeline,
                        tile=(config.tile_m, config.tile_n),
                        shape=(M, N, K),
                        seed=seed,
                    ):
                        problem = MxGemmProblem(M, N, K)
                        A_deq, B_deq, A_bytes, B_bytes, sa, sb = runner.make_inputs(
                            problem, scale=1.0, seed=seed
                        )
                        # Vary scales independently across rows and K blocks. Uniform
                        # scales cannot detect a transposed or incorrectly packed layout.
                        rng = np.random.default_rng(seed)
                        sa[:] = rng.integers(124, 130, size=sa.shape, dtype=np.uint8)
                        sb[:] = rng.integers(124, 130, size=sb.shape, dtype=np.uint8)
                        result = runner.run(problem, A_bytes, B_bytes, sa, sb)
                        got = np.asarray(result.C, dtype=np.float32)
                        ref = runner.reference(A_deq, B_deq, sa, sb, problem).astype(
                            np.float32
                        )
                        self.assertGreater(result.time_ms, 0.0)
                        self.assertFalse(np.all(got == 0.0))
                        self.assertTrue(np.all(np.isfinite(got)))
                        error = _max_rel_err(got, ref)
                        self.assertLessEqual(error, _TOL)
                        if config.pipeline == "comp_async_eight_waves":
                            # Buffer reuse races can pass a single invocation.
                            for _ in range(4):
                                repeated = runner.run(problem, A_bytes, B_bytes, sa, sb)
                                np.testing.assert_array_equal(repeated.C, result.C)
                        print(
                            f"[mx_gemm/{dtype}/{config.pipeline}] tile={config.tile_m}x{config.tile_n} "
                            f"shape={M}x{N}x{K} seed={seed} max_rel={error:.4e}"
                        )
            # Unsupported K tails and split-K must be rejected before reshuffling.
            unsupported = [
                MxGemmProblem(bm, bn, 32),
                MxGemmProblem(bm, bn, bk, 2),
            ]
            if self.ARCH == "gfx1250" and config.pipeline == "weight_preshuffle":
                unsupported.append(MxGemmProblem(bm, 17, bk))
            for problem in unsupported:
                _, _, a, b, sa, sb = runner.make_inputs(problem, scale=1.0, seed=5)
                with self.assertRaisesRegex(RuntimeError, "unsupported"):
                    runner.run(problem, a, b, sa, sb)

    def test_fp8(self):
        self._run_dtype("fp8")

    def test_fp4(self):
        self._run_dtype("fp4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
