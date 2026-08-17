#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GPU correctness test for the batched-contraction dispatcher bridge.

Builds the batched-contraction dispatcher .so (fp16 / rcr / num_d_tensors=0 —
the PassThrough v1 signature Old-TE's batched_contraction argparse validates),
runs a small contraction on-device via GpuBatchedContractionRunner, and compares
the GPU output to an fp32 numpy einsum reference within an fp16-appropriate
tolerance. Skips cleanly (exit 0) when no GPU / hipcc is available.

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
    default_fp16_config,
    setup_multiple_batched_contraction_dispatchers,
    _get_arch,
    _validate_arch,
)

log = logging.getLogger(__name__)

# fp16 inputs, fp32 accumulate; small K keeps the worst-case error under 1e-2.
TOLERANCE = 1e-2

PASS = "PASS"
FAIL = "FAIL"


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


def test_contraction_fp16(gfx_arch: str) -> tuple[str, str]:
    cfg = default_fp16_config(gfx_arch=gfx_arch)

    so_paths = setup_multiple_batched_contraction_dispatchers([cfg], gfx_arch=gfx_arch)
    if not so_paths or so_paths[0] is None:
        return FAIL, "contraction/fp16: kernel build failed"

    runner = GpuBatchedContractionRunner(
        so_paths[0], dtype="fp16", num_d_tensors=0, elementwise="PassThrough"
    )

    # num_dim_g/m/n/k = 1 each (default_fp16_config): one axis per group.
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
        return FAIL, f"contraction/fp16: output shape {E_gpu.shape} != {(G, M, N)}"
    if np.all(E_gpu == 0):
        return FAIL, "contraction/fp16: GPU output is all-zero"
    if not np.all(np.isfinite(E_gpu.astype(np.float32))):
        return FAIL, "contraction/fp16: GPU output contains NaN/Inf"

    # runner.reference computes E[g,m,n] = sum_k A[g,m,k]*B[g,n,k] in fp32.
    E_ref = runner.reference(A, B, prob)
    mre = _max_rel_err(E_gpu, E_ref)
    if mre > TOLERANCE:
        return FAIL, (f"contraction/fp16: max_rel_err={mre:.4e} > tol={TOLERANCE:.1e} "
                      f"(G={G} M={M} N={N} K={K})")
    if result.time_ms <= 0.0:
        return FAIL, f"contraction/fp16: time_ms={result.time_ms:.4f} not positive"

    return PASS, (f"contraction/fp16: max_rel_err={mre:.4e}, "
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
        return 0

    gfx = _validate_arch(args.gfx) if args.gfx else _get_arch()
    log.info("Running batched-contraction GPU correctness on %s", gfx)

    try:
        status, detail = test_contraction_fp16(gfx)
    except Exception as exc:  # noqa: BLE001
        status, detail = FAIL, f"contraction/fp16: exception: {exc}"

    print("\n=== Summary ===")
    print(f"  [{status:4s}] {detail}")
    print(f"\n{1 if status == PASS else 0}/1 passed")
    return 0 if status == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
