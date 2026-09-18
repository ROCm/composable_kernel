#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU correctness test for the MULTI-D GEMM dispatcher bridge (PR #9308).

Multi-D GEMM fuses extra D operands into the epilogue:
``E = elementwise_op(A @ B, D0, D1, ...)``. This test builds a real multi_d
dispatcher .so (elementwise_op = MultiDAdd, 2 D tensors), runs it on-device via
``GpuMultiDGemmRunner``, and compares E to an independent fp32 numpy reference
``E_ref = A @ B + D0 + D1``.

Real numeric check -- random A/B/D, element-wise validation vs. fp32 reference,
plus a non-zero / finite guard so a mis-launched (all-zero) kernel FAILS.
The multi_d TE op is fp16-only, so only fp16 is exercised, but all four A/B
layout combinations are (rcrr / rrrr / ccrr / crrr).

Runs green on gfx942 (MI300X). Exits 77 (= ctest "Skipped") with no GPU /
static lib / hipcc, rather than reporting a vacuous PASS.

Run:
  python3 tests/test_multi_d_gpu_correctness.py
  python3 tests/test_multi_d_gpu_correctness.py -v          # verbose hipcc output
  python3 tests/test_multi_d_gpu_correctness.py --gfx gfx942
"""

import argparse
import logging
import shutil
import sys
from functools import partial
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    GpuMultiDGemmRunner,
    MultiDGemmProblem,
    setup_multiple_gemm_dispatchers,
)

log = logging.getLogger(__name__)

_TOL = 1e-2  # fp16 GEMM + fp16 D-fuse precision band

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

# ctest reports this as "skipped" rather than passed. Without it, a CPU-only
# runner would report every multi_d case as PASSED -- `-m pytest` exits 0 when
# every test in a file skips -- hiding a missing GPU behind a green result.
SKIP_EXIT = 77

# 4-char (A, B, C/E, D) layout codes, matching _VARIANT_DEFAULTS["multi_d"] in
# test_gemm_search_space.py. C and E are row-major throughout: the TE multi_d
# epilogue writes row-major, so only the A and B chars actually vary.
_LAYOUTS = ("rcrr", "rrrr", "ccrr", "crrr")
_LAYOUT_WORD = {"r": "row", "c": "col"}


def _detect_arch():
    # subprocess.run with a timeout (rocminfo can hang on misconfigured ROCm
    # installs and would otherwise stall the run). Validate the parsed token is
    # a real gfx string before returning it. Mirrors
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


def _check_layout(layout: str, gfx_arch: str):
    """E = A@B + D0 + D1 for one A/B layout combination.

    The runner reads the layout back out of the kernel name and transposes the
    host operands itself, so A and B are handed over as logical (M, K) and
    (K, N) in every case and the reference below is layout-independent.

    Returns (status, detail) and never raises: each layout is a separate hipcc
    build, and a failure in one must name its layout without hiding the other
    three.
    """
    num_d = 2
    la, lb, lc, ld = (_LAYOUT_WORD[c] for c in layout)
    cfg = GemmKernelConfig(
        dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32",
        layout_a=la, layout_b=lb, layout_c=lc,
        tile_m=128, tile_n=128, tile_k=32,
        wave_m=2, wave_n=2, wave_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
        pad_m=True, pad_n=True, pad_k=True, persistent=False,
        variant="multi_d", elementwise_op="MultiDAdd",
        num_d_tensors=num_d, d_layout=ld,
        gfx_arch=gfx_arch,
    )
    so_paths = setup_multiple_gemm_dispatchers(
        [cfg], verbose=log.isEnabledFor(logging.DEBUG)
    )
    so = so_paths[0]
    if so is None:
        return FAIL, f"multi_d/fp16/{layout}: kernel failed to build"

    M, N, K = 512, 512, 512
    problem = MultiDGemmProblem(M=M, N=N, K=K, num_d=num_d)

    rng = np.random.default_rng(11)
    A = rng.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    B = rng.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
    Ds = [rng.uniform(-1.0, 1.0, (M, N)).astype(np.float32)
          for _ in range(num_d)]

    runner = GpuMultiDGemmRunner(so)
    if runner.num_d_tensors != num_d:
        return FAIL, (f"multi_d/fp16/{layout}: kernel D-count mismatch "
                      f"({runner.num_d_tensors} != {num_d})")
    result = runner.run(A.astype(np.float16), B.astype(np.float16),
                        [d.astype(np.float16) for d in Ds], problem)
    if result.status != 0:
        return FAIL, f"multi_d/fp16/{layout}: run status={result.status}"
    if result.time_ms <= 0.0:
        return FAIL, (f"multi_d/fp16/{layout}: time_ms={result.time_ms:.4f} "
                      f"is not positive")

    # Reference at fp16 input precision, accumulated + fused in fp32:
    #   E = A@B + D0 + D1  (MultiDAdd).
    A_q = A.astype(np.float16).astype(np.float32)
    B_q = B.astype(np.float16).astype(np.float32)
    E_ref = A_q @ B_q
    for d in Ds:
        E_ref = E_ref + d.astype(np.float16).astype(np.float32)

    E_got = np.asarray(result.output).astype(np.float32)
    if np.all(E_got == 0.0):
        return FAIL, f"multi_d/fp16/{layout}: output all-zero"
    if not np.all(np.isfinite(E_got)):
        return FAIL, f"multi_d/fp16/{layout}: output NaN/Inf"
    mre = _max_rel_err(E_got, E_ref)
    if mre > _TOL:
        return FAIL, (f"multi_d/fp16/{layout}: max_rel={mre:.4f} > tol={_TOL} "
                      f"(M={M} N={N} K={K})")
    return PASS, (f"multi_d/fp16/{layout}: max_rel={mre:.4e}, "
                  f"time_ms={result.time_ms:.3f}")


# One entry per layout rather than one test that loops: each is its own hipcc
# build, and a per-layout PASS/FAIL row is what makes a single bad layout
# legible in the summary.
TESTS = [(layout, partial(_check_layout, layout)) for layout in _LAYOUTS]


def main():
    parser = argparse.ArgumentParser(description="Multi-D GEMM GPU correctness tests")
    # No hardcoded default: it must stay possible to tell "user asked for
    # gfx942" from "we are on a GPU-less box", so the skip below can fire.
    parser.add_argument("--gfx", default=None,
                        help="GPU arch override (default: auto-detect)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    gfx = args.gfx or _detect_arch()
    if not gfx:
        print("SKIP: no GPU detected (rocminfo); multi_d GPU tests skipped")
        return SKIP_EXIT
    if not _static_lib_present():
        print("SKIP: dispatcher static lib (libck_tile_dispatcher.a) not built; "
              "multi_d is registry-routed and needs it")
        return SKIP_EXIT
    if shutil.which("hipcc") is None and not Path("/opt/rocm/bin/hipcc").exists():
        print("SKIP: hipcc not found; cannot build multi_d kernels")
        return SKIP_EXIT

    results = []
    for name, fn in TESTS:
        log.info("--- Running %s ---", name)
        try:
            status, detail = fn(gfx)
        except Exception as exc:
            status, detail = FAIL, f"{name}: exception: {exc}"
        results.append((name, status, detail))
        log.info("[%s] %s", status, detail)

    print("\n=== Summary ===")
    for name, status, detail in results:
        print(f"  [{status:4s}] {detail}")
    passed = sum(1 for _, s, _ in results if s == PASS)
    skipped = sum(1 for _, s, _ in results if s == SKIP)
    failed = len(results) - passed - skipped
    print(f"\n{passed}/{len(results) - skipped} passed"
          + (f", {skipped} skipped" if skipped else ""))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
