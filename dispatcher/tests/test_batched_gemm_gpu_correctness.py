#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GPU correctness test for the batched GEMM dispatcher bridge.

Builds the batched GEMM dispatcher .so (fp16 / rcr — the only signature Old-TE's
batched_gemm_instance_builder validates), runs a small batched problem on-device
via GpuBatchedGemmRunner, and compares the GPU output to a per-batch fp32 numpy
reference within an fp16-appropriate tolerance. Skips cleanly (exit 0) when no
GPU / hipcc is available so it is safe in a CPU-only CI lane.

The kernel computes, for each batch b:
    C[b] = A[b] @ B[b]          A[b] is M x K, B[b] is K x N, C[b] is M x N
with the rcr layout: A row-major, B column-major, C row-major. The runner
handles the operand layout transform, so the test hands it logical row-major A
and logical (K, N) B per batch.

Run:
  python3 test_batched_gemm_gpu_correctness.py
  python3 test_batched_gemm_gpu_correctness.py -v
  python3 test_batched_gemm_gpu_correctness.py --gfx gfx942
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from batched_gemm_utils import (  # noqa: E402
    BatchedGemmKernelConfig,
    BatchedGemmProblem,
    GpuBatchedGemmRunner,
    setup_multiple_batched_gemm_dispatchers,
    _resolve_arch,
)

log = logging.getLogger(__name__)

# fp16 accumulate-in-fp32 GEMM: worst-case relative error is well under 1e-2 for
# the small K used here. Use the standard fp16 bar.
TOLERANCE = 1e-2

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def _has_gpu() -> bool:
    """True iff a supported GPU is visible to rocminfo (so the .so can run)."""
    try:
        _resolve_arch(None)
        return True
    except Exception:
        return False


def _max_rel_err(C_gpu: np.ndarray, C_ref: np.ndarray) -> float:
    """Max absolute error normalized by the largest reference magnitude.

    A GEMM output has elements that partially cancel toward zero, so a naive
    per-element relative error is dominated by those near-zero entries and does
    NOT measure whether the kernel computed the right matrix. Normalizing the
    worst absolute error by the global reference scale (max |ref|) is the honest
    correctness bar: any structural error (wrong accumulation, transposed
    operand, wrong batch stride) blows far past 1e-2, while correct fp16 math
    lands at ~1e-3-1e-4 here.
    """
    g = C_gpu.astype(np.float32)
    r = C_ref.astype(np.float32)
    ref_scale = max(float(np.abs(r).max()), 1e-6)
    return float(np.max(np.abs(g - r)) / ref_scale)


def _reference_batched(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-batch fp32 reference: C[b] = A[b] @ B[b] (A: b,M,K  B: b,K,N)."""
    return np.matmul(A.astype(np.float32), B.astype(np.float32))


def _make_fp16_config(gfx_arch: str) -> BatchedGemmKernelConfig:
    """A single small, valid fp16/rcr batched kernel.

    128x128x32 tile, 2x2x1 waves, 32x32x16 warp-tile, compv3/intrawave/cshuffle
    — a divisibility-valid combination on both gfx942 and gfx950. Padding is on
    so non-tile-multiple shapes still run.
    """
    return BatchedGemmKernelConfig(
        dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32",
        layout_a="row", layout_b="col", layout_c="row",
        tile_m=128, tile_n=128, tile_k=32,
        wave_m=2, wave_n=2, wave_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        pipeline="compv3", scheduler="intrawave", epilogue="cshuffle",
        pad_m=True, pad_n=True, pad_k=True, persistent=False,
        gfx_arch=gfx_arch,
    )


def test_batched_fp16(gfx_arch: str) -> tuple[str, str]:
    # Small multi-batch problem; K=128 gives 4 tile-K iterations (128/32).
    batch, M, N, K = 3, 128, 128, 128
    cfg = _make_fp16_config(gfx_arch)

    so_paths = setup_multiple_batched_gemm_dispatchers([cfg], verbose=False)
    if not so_paths or so_paths[0] is None:
        return FAIL, "batched/fp16: kernel build failed"

    runner = GpuBatchedGemmRunner(so_paths[0])

    rng = np.random.default_rng(7)
    A = rng.uniform(-1.0, 1.0, (batch, M, K)).astype(np.float32)
    B = rng.uniform(-1.0, 1.0, (batch, K, N)).astype(np.float32)

    problem = BatchedGemmProblem(batch_count=batch, M=M, N=N, K=K)
    result = runner.run(A, B, problem, warmup=5, repeat=10)

    if result.status != 0:
        return FAIL, f"batched/fp16: run status={result.status} (nonzero)"
    C_gpu = result.output
    if C_gpu.shape != (batch, M, N):
        return FAIL, f"batched/fp16: output shape {C_gpu.shape} != {(batch, M, N)}"
    if np.all(C_gpu == 0):
        return FAIL, "batched/fp16: GPU output is all-zero"
    if not np.all(np.isfinite(C_gpu.astype(np.float32))):
        return FAIL, "batched/fp16: GPU output contains NaN/Inf"

    # A[b]@B[b]: the runner already produced logical (batch, M, N). B is passed
    # logical (K, N); the runner transposes it for the col-major kernel, so the
    # reference multiplies the same logical operands.
    C_ref = _reference_batched(A, B)
    mre = _max_rel_err(C_gpu, C_ref)
    if mre > TOLERANCE:
        return FAIL, (f"batched/fp16: max_rel_err={mre:.4e} > tol={TOLERANCE:.1e} "
                      f"(batch={batch} M={M} N={N} K={K})")
    if result.time_ms <= 0.0:
        return FAIL, f"batched/fp16: time_ms={result.time_ms:.4f} not positive"

    return PASS, (f"batched/fp16: max_rel_err={mre:.4e}, "
                  f"time_ms={result.time_ms:.3f}, batch={batch} MNK={M}/{N}/{K}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batched GEMM GPU correctness test")
    parser.add_argument("--gfx", default=None, help="GPU arch (default: auto-detect)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not _has_gpu():
        print("SKIP: no supported GPU detected (rocminfo); batched GPU test skipped")
        return 0

    gfx = args.gfx or _resolve_arch(None)
    log.info("Running batched GEMM GPU correctness on %s", gfx)

    try:
        status, detail = test_batched_fp16(gfx)
    except Exception as exc:  # noqa: BLE001
        status, detail = FAIL, f"batched/fp16: exception: {exc}"

    print("\n=== Summary ===")
    print(f"  [{status:4s}] {detail}")
    print(f"\n{1 if status == PASS else 0}/1 passed")
    return 0 if status == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
