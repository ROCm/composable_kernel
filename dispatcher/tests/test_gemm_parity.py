#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GEMM bridge parity regression: Dispatcher GPU output vs NumPy reference.

This is the in-tree, reproducible version of the ad-hoc ``parity/`` sweep used to
validate the Tile Engine -> Dispatcher GEMM bridge. For each (dtype, layout) the
bridge supports it codegens + hipcc-compiles a kernel, runs it through
``GpuGemmRunner``, and compares the result to a NumPy reference across a square, a
rectangular, and an awkward (non-tile-aligned) problem shape.

Parity is checked as a GLOBAL relative error -- ``max|gpu - ref| / max|ref|`` --
not per-element: K-length accumulation of zero-mean inputs produces near-zero
entries whose per-element ratios explode and carry no signal.

The whole suite is GPU-gated: it skips cleanly (not fails) when hipcc, the
dispatcher static lib, or a GPU is unavailable, so CPU-only CI stays green while
GPU runners get real end-to-end coverage. The pure host-side helpers are covered
separately and cheaply by ``test_gemm_utils.py``.

Run:
    python3 -m pytest tests/test_gemm_parity.py -v      # discovery / CI
    python3 tests/test_gemm_parity.py                   # readable table
"""

import os
import sys
import shutil
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

import numpy as np  # noqa: E402

from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    GemmProblem,
    GpuGemmRunner,
    setup_multiple_gemm_dispatchers,
    _fp32_to_bf16_u16,
    _bf16_u16_to_fp32,
    _fp32_to_fp8_u8,
    _fp8_u8_to_fp32,
    _fp32_to_bf8_u8,
    _bf8_u8_to_fp32,
    _output_dtype,
)
from ctypes_utils import detect_gpu_arch, get_build_dir  # noqa: E402

# (dtype, layout) surface the regular bridge supports. Column-major C is rejected
# by ck_tile's universal GEMM at build, so every layout keeps row-major C, which
# leaves exactly the four A/B combinations below. Every dtype covers all four.
#
# fp16/bf16 are the PR #8479 surface; fp8 (E4M3), bf8 (E5M2) and int8 are the
# remaining dtypes TE's plain GEMM has MFMA warp tiles for (fp8/bf8 -> fp16 out,
# int8 -> int32 out). int8 only has warp tiles on gfx942; on other arches its
# kernels simply fail to build and the case skips (handled below).
_FLOAT_DTYPES = ("fp16", "bf16", "fp8", "bf8")
_INT_DTYPES = ("int8",)
_LAYOUTS = ("rcr", "rrr", "ccr", "crr")
_CASES = [
    (dt, lay) for dt in (*_FLOAT_DTYPES, *_INT_DTYPES) for lay in _LAYOUTS
]

# Padded default algorithm: pad_* all True so M/N need not divide the tile, which
# is what lets the awkward shape below pass. K must still be a multiple of 8 for
# the fp16/bf16 vectorized contiguous-reduction load, so every K here is divisible
# by 8.
_ALGO = dict(
    tile_m=128, tile_n=128, tile_k=32,
    wave_m=2, wave_n=2, wave_k=1,
    warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
    pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
    pad_m=True, pad_n=True, pad_k=True,
)

# (name, M, N, K). 'awkward' deliberately uses M, N that do not divide the 128
# tile to exercise padding; K stays divisible by 8.
_SHAPES = [
    ("square", 512, 512, 512),
    ("rectangular", 1024, 512, 256),
    ("awkward", 257, 129, 512),
]

# Global-relative-error gates. fp16 measured ~3-4e-4 and bf16 ~8e-3 on gfx942.
# fp8/bf8 are far coarser (3- and 2-bit mantissa) so their gates are looser; int8
# is an exact integer accumulation so it must match bit-for-bit. The fp8/bf8
# gates are first-cut headroom values and may want tightening once measured on a
# GPU.
_TOL = {
    "fp16": 2e-3,
    "bf16": 1.5e-2,
    "fp8": 1.5e-1,
    "bf8": 3.0e-1,
    "int8": 0.0,
}

_LAYOUT_WORD = {"r": "row", "c": "col"}


def _emulate_input(x: np.ndarray, dtype: str) -> np.ndarray:
    """Round an fp32 operand to the kernel's storage dtype so the CPU reference
    multiplies exactly what the GPU does. int8 inputs are already integral."""
    if dtype == "bf16":
        return _bf16_u16_to_fp32(_fp32_to_bf16_u16(x))
    if dtype == "fp8":
        return _fp8_u8_to_fp32(_fp32_to_fp8_u8(x))
    if dtype == "bf8":
        return _bf8_u8_to_fp32(_fp32_to_bf8_u8(x))
    if dtype == "int8":
        return x.astype(np.float64)  # exact; widened to avoid product overflow
    return x.astype(np.float16).astype(np.float32)


def _emulate_output(c: np.ndarray, out_dtype: str) -> np.ndarray:
    """Round the fp32 accumulator to the kernel's C storage dtype."""
    if out_dtype == "bf16":
        return _bf16_u16_to_fp32(_fp32_to_bf16_u16(c))
    if out_dtype == "int32":
        return c  # integer accumulation is exact
    return c.astype(np.float16).astype(np.float32)  # fp16


def _make_inputs(dtype, M, N, K, rng):
    """Random A (MxK), B (KxN) for a dtype: floats for the float dtypes, small
    integers for int8 (kept small so the int32 accumulation cannot overflow)."""
    if dtype == "int8":
        A = rng.integers(-4, 5, size=(M, K)).astype(np.float32)
        B = rng.integers(-4, 5, size=(K, N)).astype(np.float32)
        return A, B
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    return A, B


def _reference(A, B, dtype):
    """NumPy reference matching the kernel: round inputs to the storage dtype,
    accumulate (fp32 for floats / exact int for int8), then round to C dtype."""
    out_dtype = _output_dtype(dtype)
    acc = _emulate_input(A, dtype) @ _emulate_input(B, dtype)
    ref = _emulate_output(acc, out_dtype)
    return ref.astype(np.int32) if out_dtype == "int32" else ref


def _config(dtype: str, layout: str, arch: str) -> GemmKernelConfig:
    la, lb, lc = layout
    return GemmKernelConfig(
        dtype_a=dtype, dtype_b=dtype,
        dtype_c=_output_dtype(dtype),
        dtype_acc=("int32" if dtype == "int8" else "fp32"),
        layout_a=_LAYOUT_WORD[la], layout_b=_LAYOUT_WORD[lb], layout_c=_LAYOUT_WORD[lc],
        gfx_arch=arch, **_ALGO,
    )


def _max_rel(out: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref))) + 1e-12
    return float(np.max(np.abs(out - ref))) / denom


def _gpu_environment_reason():
    """Return None if the bridge can build+run here, else a human-readable reason
    to skip."""
    if not Path("/opt/rocm/bin/hipcc").exists():
        return "hipcc not found at /opt/rocm/bin/hipcc"
    if not (get_build_dir() / "libck_tile_dispatcher.a").exists():
        return "dispatcher static lib (libck_tile_dispatcher.a) not built"
    if shutil.which("rocminfo") is None:
        return "rocminfo not found (no ROCm runtime / GPU)"
    return None


class GemmBridgeParity(unittest.TestCase):
    """End-to-end GPU-vs-NumPy parity across the bridge's dtype/layout surface."""

    arch = None
    built = {}        # (dtype, layout) -> Path(.so)
    build_failures = {}

    @classmethod
    def setUpClass(cls):
        reason = _gpu_environment_reason()
        if reason:
            raise unittest.SkipTest(reason)
        cls.arch = detect_gpu_arch()

        configs = [_config(dt, lay, cls.arch) for dt, lay in _CASES]
        so_paths = setup_multiple_gemm_dispatchers(configs, verbose=False)
        for (dt, lay), so in zip(_CASES, so_paths):
            if so is None:
                cls.build_failures[(dt, lay)] = "codegen/hipcc returned no .so"
            else:
                cls.built[(dt, lay)] = so

        if not cls.built:
            raise unittest.SkipTest(
                f"no bridge kernels built on {cls.arch} "
                f"(failures: {cls.build_failures})"
            )

    def _run_case(self, dtype, layout, shape):
        so = self.built.get((dtype, layout))
        if so is None:
            self.skipTest(
                f"{dtype}/{layout} did not build on {self.arch}: "
                f"{self.build_failures.get((dtype, layout))}"
            )

        _, M, N, K = shape
        problem = GemmProblem(M=M, N=N, K=K)
        rng = np.random.default_rng(42)
        A, B = _make_inputs(dtype, M, N, K, rng)

        runner = GpuGemmRunner(lib_path=so)
        # The .so is the contract endpoint: the name it reports must be the config
        # name that drove codegen + the force-include build. The kernel name keys
        # off the input dtype (dtype_a), not the C/acc dtype.
        self.assertEqual(runner.kernel_name, _config(dtype, layout, self.arch).name)

        result = runner.run(A, B, problem)
        self.assertTrue(
            result.success,
            f"{dtype}/{layout} {shape[0]} run failed (status {result.status})",
        )

        ref = _reference(A, B, dtype)
        max_rel = _max_rel(result.output.astype(np.float64), ref.astype(np.float64))
        self.assertLessEqual(
            max_rel, _TOL[dtype],
            f"{dtype}/{layout} {shape[0]} max_rel={max_rel:.2e} > {_TOL[dtype]:.0e}",
        )


def _add_parity_tests():
    """Generate one test method per (case, shape) so failures pinpoint exactly
    which dtype/layout/shape regressed."""
    for dtype, layout in _CASES:
        for shape in _SHAPES:
            shape_name = shape[0]

            def _method(self, dtype=dtype, layout=layout, shape=shape):
                self._run_case(dtype, layout, shape)

            _method.__name__ = f"test_{dtype}_{layout}_{shape_name}"
            _method.__doc__ = f"{dtype} {layout} {shape_name} {shape[1:]} parity"
            setattr(GemmBridgeParity, _method.__name__, _method)


_add_parity_tests()


def _main() -> int:
    """Readable table run (mirrors test_fmha_parity.py's report style)."""
    reason = _gpu_environment_reason()
    if reason:
        print(f"SKIP: {reason}")
        return 0

    arch = detect_gpu_arch()
    print("=" * 78)
    print(f"GEMM Bridge Parity: Dispatcher (GPU {arch}) vs NumPy reference")
    print("=" * 78)

    configs = [_config(dt, lay, arch) for dt, lay in _CASES]
    print(f"  Building {len(configs)} bridge kernels (codegen + hipcc)...")
    so_paths = setup_multiple_gemm_dispatchers(configs, verbose=False)

    print(f"\n  {'case':<12} {'shape':<12} {'tflops':>9} {'max_rel':>10} {'tol':>8} {'':>6}")
    print("  " + "-" * 60)

    rng = np.random.default_rng(42)
    total = 0
    passed = 0
    for (dtype, layout), so in zip(_CASES, so_paths):
        tag = f"{dtype}/{layout}"
        if so is None:
            print(f"  {tag:<12} {'-':<12} {'BUILD FAILED':>35}")
            total += len(_SHAPES)
            continue
        runner = GpuGemmRunner(lib_path=so)
        for sname, M, N, K in _SHAPES:
            total += 1
            problem = GemmProblem(M=M, N=N, K=K)
            A, B = _make_inputs(dtype, M, N, K, rng)
            result = runner.run(A, B, problem)
            if not result.success:
                print(f"  {tag:<12} {sname:<12} {'RUN FAILED':>9} status={result.status}")
                continue
            ref = _reference(A, B, dtype)
            mr = _max_rel(result.output.astype(np.float64), ref.astype(np.float64))
            ok = mr <= _TOL[dtype]
            passed += ok
            print(f"  {tag:<12} {sname:<12} {result.tflops:>9.1f} "
                  f"{mr:>10.2e} {_TOL[dtype]:>8.0e} {'PASS' if ok else 'FAIL':>6}")

    print("\n" + "=" * 78)
    print(f"  {passed}/{total} parity checks passed")
    print("=" * 78)
    return 0 if passed == total else 1


if __name__ == "__main__":
    # Default to the readable table; `-m pytest` / `unittest` use the generated
    # test methods instead.
    if os.environ.get("GEMM_PARITY_UNITTEST"):
        unittest.main()
    else:
        sys.exit(_main())
