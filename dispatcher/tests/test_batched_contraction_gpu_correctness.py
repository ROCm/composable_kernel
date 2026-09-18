#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GPU correctness test for the batched-contraction dispatcher bridge.

Builds the batched-contraction dispatcher .so once per dtype (fp16 / bf16 /
fp32, all rcr / num_d_tensors=0 — the PassThrough v1 signature Old-TE's
batched_contraction argparse validates), runs a small contraction on-device via
GpuBatchedContractionRunner, and compares the GPU output to an fp32 numpy einsum
reference within a per-dtype tolerance (see TOLERANCE). Every dtype is attempted
even if an earlier one fails, so one broken dtype does not mask the others.
Skips cleanly (exit 77) when no GPU / hipcc is available.

The kernel computes:
    E[g,m,n] = sum_k A[g,m,k] * B[g,n,k]
i.e. a per-group GEMM where B is contracted on its trailing K axis (B is
[G,N,K], the "rcr" b-layout). The runner's built-in reference() computes the
same einsum in fp32, so this test drives run() + reference() and checks the two
agree.

Run:
  python3 test_batched_contraction_gpu_correctness.py
  python3 test_batched_contraction_gpu_correctness.py -v
  python3 test_batched_contraction_gpu_correctness.py --gfx gfx942
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from batched_contraction_utils import (  # noqa: E402
    BatchedContractionProblem,
    GpuBatchedContractionRunner,
    default_config,
    setup_multiple_batched_contraction_dispatchers,
    _get_arch,
    _validate_arch,
)

log = logging.getLogger(__name__)

# Per-dtype gates. All three accumulate in fp32, so the gate is set by the input
# precision: fp16 (10-bit mantissa) at 1e-2, bf16 (7-bit) proportionally looser,
# fp32 near exact -- it is the same arithmetic as the numpy reference, differing
# only in summation order. Small K keeps the worst case inside these.
TOLERANCE = {
    "fp16": 1e-2,
    "bf16": 3e-2,
    "fp32": 1e-5,
}

DTYPES = ("fp16", "bf16", "fp32")

PASS = "PASS"
FAIL = "FAIL"

# ctest reports this as "skipped" rather than passed; see SKIP_RETURN_CODE in
# dispatcher/tests/CMakeLists.txt. Returning 0 here would make a CPU-only runner
# report a green PASS for a test that never touched the GPU.
SKIP_EXIT = 77


def _has_gpu() -> bool:
    try:
        _get_arch()
        return True
    except Exception:
        return False


def _max_rel_err(E_gpu: np.ndarray, E_ref: np.ndarray) -> float:
    """Max absolute error normalized by the largest reference magnitude.

    A GEMM output has elements that partially cancel toward zero, so a naive
    per-element relative error is dominated by those near-zero entries and does
    NOT measure whether the kernel computed the right matrix. Normalizing the
    worst absolute error by the global reference scale (max |ref|) is the honest
    correctness bar: any structural error (wrong accumulation, mis-shuffled B,
    transposed operand) blows far past 1e-2, while correct fp16 math lands at
    ~1e-3-1e-4 here.
    """
    g = E_gpu.astype(np.float32)
    r = E_ref.astype(np.float32)
    ref_scale = max(float(np.abs(r).max()), 1e-6)
    return float(np.max(np.abs(g - r)) / ref_scale)


def check_contraction(dtype: str, gfx_arch: str) -> tuple[str, str]:
    tol = TOLERANCE[dtype]
    cfg = default_config(dtype, gfx_arch=gfx_arch)
    # A dtype whose warp tile is off the XDL allow-list is dropped by is_valid()
    # rather than rejected, so the build would just produce nothing. Catch it
    # here, where the cause is still obvious.
    if not cfg.is_valid():
        return FAIL, f"contraction/{dtype}: default config is not valid for this dtype"

    so_paths = setup_multiple_batched_contraction_dispatchers([cfg], gfx_arch=gfx_arch)
    if not so_paths or so_paths[0] is None:
        return FAIL, f"contraction/{dtype}: kernel build failed"

    runner = GpuBatchedContractionRunner(
        so_paths[0], dtype=dtype, num_d_tensors=0, elementwise="PassThrough"
    )

    # num_dim_g/m/n/k = 1 each (default_config): one axis per group.
    # K=128 gives 2 tile-K iterations (tile_k=64).
    G, M, N, K = 2, 128, 128, 128
    prob = BatchedContractionProblem(
        g_dims=[G], m_dims=[M], n_dims=[N], k_dims=[K], k_batch=1
    )

    rng = np.random.default_rng(11)
    A = rng.uniform(-1.0, 1.0, (G, M, K)).astype(np.float32)
    B = rng.uniform(-1.0, 1.0, (G, N, K)).astype(np.float32)

    result = runner.run(A, B, prob)
    E_gpu = result.E

    if E_gpu.shape != (G, M, N):
        return FAIL, f"contraction/{dtype}: output shape {E_gpu.shape} != {(G, M, N)}"
    if np.all(E_gpu == 0):
        return FAIL, f"contraction/{dtype}: GPU output is all-zero"
    if not np.all(np.isfinite(E_gpu.astype(np.float32))):
        return FAIL, f"contraction/{dtype}: GPU output contains NaN/Inf"

    # runner.reference computes E[g,m,n] = sum_k A[g,m,k]*B[g,n,k] in fp32.
    E_ref = runner.reference(A, B, prob)
    mre = _max_rel_err(E_gpu, E_ref)
    if mre > tol:
        return FAIL, (f"contraction/{dtype}: max_rel_err={mre:.4e} > tol={tol:.1e} "
                      f"(G={G} M={M} N={N} K={K})")
    if result.time_ms <= 0.0:
        return FAIL, f"contraction/{dtype}: time_ms={result.time_ms:.4f} not positive"

    return PASS, (f"contraction/{dtype}: max_rel_err={mre:.4e}, "
                  f"time_ms={result.time_ms:.3f}, G={G} MNK={M}/{N}/{K}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batched-contraction GPU correctness test"
    )
    parser.add_argument("--gfx", default=None, help="GPU arch (default: auto-detect)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not _has_gpu():
        print("SKIP: no supported GPU detected (rocminfo); contraction GPU test skipped")
        return SKIP_EXIT

    gfx = _validate_arch(args.gfx) if args.gfx else _get_arch()
    log.info("Running batched-contraction GPU correctness on %s", gfx)

    results = []
    for dtype in DTYPES:
        try:
            results.append(check_contraction(dtype, gfx))
        except Exception as exc:  # noqa: BLE001
            results.append((FAIL, f"contraction/{dtype}: exception: {exc}"))

    print("\n=== Summary ===")
    for status, detail in results:
        print(f"  [{status:4s}] {detail}")
    passed = sum(1 for status, _ in results if status == PASS)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
