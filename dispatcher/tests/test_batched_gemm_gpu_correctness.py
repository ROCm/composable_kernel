#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GPU correctness test for the batched GEMM dispatcher bridge.

Builds one small batched kernel per CASES entry -- every bridge dtype (fp16,
bf16, fp32, fp8, bf8) and every layout (rcr, rrr, crr, ccr) is covered -- runs
it on-device via GpuBatchedGemmRunner, and compares the GPU output to a
per-batch fp32 numpy reference. Each kernel is picked from the bridge's own
expand_sweep, so the warp tile follows the arch (MFMA on gfx90a/gfx942/gfx950,
WMMA on gfx1250). fp8/bf8 cases are skipped on gfx90a, which has no fp8 MFMA.
Skips cleanly (exit 77) when no GPU / hipcc is available so it is safe in a
CPU-only CI lane.

The kernel computes, for each batch b:
    C[b] = A[b] @ B[b]          A[b] is M x K, B[b] is K x N, C[b] is M x N
The runner handles the operand layout transform, so the test hands it logical
row-major A (batch, M, K) and logical B (batch, K, N) for every layout.

Inputs are quarter-integers in [-1, 1], exact in every input dtype down to bf8,
so the fp32 reference is exact and only C rounding and accumulation order
remain; a wrong fp8/bf8 host encoding (FNUZ vs OCP) still misses by ~2-4x.

Run:
  python3 test_batched_gemm_gpu_correctness.py
  python3 test_batched_gemm_gpu_correctness.py -v
  python3 test_batched_gemm_gpu_correctness.py --gfx gfx942
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from gemm_utils import _codegen_common  # noqa: E402
from batched_gemm_utils import (  # noqa: E402
    BATCHED_VERIFY_TOL,
    BatchedGemmKernelConfig,
    BatchedGemmProblem,
    GpuBatchedGemmRunner,
    expand_sweep,
    setup_multiple_batched_gemm_dispatchers,
    _resolve_arch,
)

log = logging.getLogger(__name__)

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

# ctest reports this as "skipped" rather than passed; see SKIP_RETURN_CODE in
# dispatcher/tests/CMakeLists.txt. Returning 0 here would make a CPU-only runner
# report a green PASS for a test that never touched the GPU.
SKIP_EXIT = 77

# Arches without fp8/bf8 matrix instructions.
_NO_FP8_ARCHES = ("gfx90a",)

# Small 128x128 tile, 2x2x1 waves, compv3/intrawave/cshuffle, padded so
# non-tile-multiple shapes still run. The warp-tile lists are a superset of
# every arch's fragments; expand_sweep keeps the ones legal for the arch/dtype.
_SWEEP = {
    "tile_config": {
        "tile_m": {"values": [128]},
        "tile_n": {"values": [128]},
        "tile_k": {"values": [32, 64, 128]},
        "warp_m": {"values": [2]},
        "warp_n": {"values": [2]},
        "warp_k": {"values": [1]},
        "warp_tile_m": {"values": [16, 32]},
        "warp_tile_n": {"values": [16, 32]},
        "warp_tile_k": {"values": [4, 8, 16, 32, 64, 128]},
    },
    "trait_config": {
        "pipeline": {"values": ["compv3"]},
        "scheduler": {"values": ["intrawave"]},
        "epilogue": {"values": ["cshuffle"]},
        "pad_m": {"values": [True]},
        "pad_n": {"values": [True]},
        "pad_k": {"values": [True]},
        "persistent": {"values": [False]},
    },
}

# (dtype, layout, batch, M, N, K). K=128 gives 4 tile-K iterations. K=257 is not
# a multiple of the native 8-wide fp16 A/B loads, so no native kernel accepts it;
# it runs on the narrowed fixed-width (_vec) kernel instead.
CASES = [
    ("fp16", "rcr", 3, 128, 128, 128),
    ("fp16", "rcr", 3, 128, 128, 257),
    ("bf16", "ccr", 3, 256, 128, 128),
    ("fp32", "rrr", 3, 256, 128, 128),
    ("fp8", "crr", 3, 256, 128, 128),
    ("bf8", "rcr", 3, 256, 128, 128),
]


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
    operand, wrong batch stride, wrong fp8 encoding) blows far past the
    tolerance, while the exact inputs here leave only C rounding (<= ~2e-3).
    """
    g = C_gpu.astype(np.float32)
    r = C_ref.astype(np.float32)
    ref_scale = max(float(np.abs(r).max()), 1e-6)
    return float(np.max(np.abs(g - r)) / ref_scale)


def _reference_batched(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-batch fp32 reference: C[b] = A[b] @ B[b] (A: b,M,K  B: b,K,N)."""
    return np.matmul(A.astype(np.float32), B.astype(np.float32))


def _case_config(
    gfx_arch: str, dtype: str, layout: str, M: int, N: int, K: int
) -> BatchedGemmKernelConfig:
    """Smallest-tile_k, widest-warp-tile kernel from _SWEEP that runs M/N/K.

    Native vector widths are preferred; the problem's widest legal widths are
    the fallback, as in the full benchmark.
    """
    cc = _codegen_common()
    out_dtype = cc.CommonTypeMappings.get_output_dtype(dtype)
    need = cc.gemm_problem_vector_sizes(M, N, K, layout, dtype, dtype, out_dtype)
    extents = [dict(m=M, n=N, k=K)[d] for d in cc.gemm_contiguous_dims(layout)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sweep.json"
        path.write_text(json.dumps(_SWEEP))
        configs = expand_sweep(
            str(path), arch=gfx_arch, dtype=dtype, layout=layout, vector_sizes=[(0, 0, 0), need]
        )
    fits = [
        c
        for c in configs
        if all(e % w == 0 for e, w in zip(extents, c.effective_vector_sizes))
    ]
    if not fits:
        raise RuntimeError(f"no {dtype}/{layout} kernel in the sweep runs M/N/K={M}/{N}/{K}")
    return min(
        fits,
        key=lambda c: (
            any(c.vector_sizes), c.tile_k, -c.warp_tile_m, -c.warp_tile_n, -c.warp_tile_k
        ),
    )


# NOTE: deliberately NOT named test_* -- this module is script-style and is
# run directly by ctest (see tests/CMakeLists.txt). Under the old name pytest
# collected it and failed with "fixture 'gfx_arch' not found" (conftest
# provides 'gpu_arch'), and it returns a (status, detail) tuple, which pytest
# also flags. main() below remains the supported entry point.
def check_batched(
    gfx_arch: str,
    cfg: BatchedGemmKernelConfig,
    so_path,
    dtype: str,
    layout: str,
    batch: int,
    M: int,
    N: int,
    K: int,
) -> tuple[str, str]:
    tag = f"batched/{dtype}/{layout} MNK={M}/{N}/{K}"
    if so_path is None:
        return FAIL, f"{tag}: kernel build failed ({cfg.name})"

    runner = GpuBatchedGemmRunner(so_path, arch=gfx_arch)

    rng = np.random.default_rng(7)
    A = rng.integers(-4, 5, (batch, M, K)).astype(np.float32) / 4
    B = rng.integers(-4, 5, (batch, K, N)).astype(np.float32) / 4

    problem = BatchedGemmProblem(batch_count=batch, M=M, N=N, K=K)
    result = runner.run(A, B, problem, warmup=5, repeat=10)

    if result.status != 0:
        return FAIL, f"{tag}: run status={result.status} (nonzero)"
    C_gpu = result.output
    if C_gpu.shape != (batch, M, N):
        return FAIL, f"{tag}: output shape {C_gpu.shape} != {(batch, M, N)}"
    if np.all(C_gpu == 0):
        return FAIL, f"{tag}: GPU output is all-zero"
    if not np.all(np.isfinite(C_gpu.astype(np.float32))):
        return FAIL, f"{tag}: GPU output contains NaN/Inf"

    # The runner already produced logical (batch, M, N) from logical A and B,
    # so the reference multiplies the same logical operands for every layout.
    mre = _max_rel_err(C_gpu, _reference_batched(A, B))
    tol = BATCHED_VERIFY_TOL[dtype]
    if mre > tol:
        return FAIL, f"{tag}: max_rel_err={mre:.4e} > tol={tol:.1e} (batch={batch})"
    if result.time_ms <= 0.0:
        return FAIL, f"{tag}: time_ms={result.time_ms:.4f} not positive"

    wt = (cfg.warp_tile_m, cfg.warp_tile_n, cfg.warp_tile_k)
    return PASS, (
        f"{tag} warp_tile{wt} vec{cfg.effective_vector_sizes}: max_rel_err={mre:.4e}, "
        f"time_ms={result.time_ms:.3f}, batch={batch}"
    )


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
        return SKIP_EXIT

    gfx = args.gfx or _resolve_arch(None)
    log.info("Running batched GEMM GPU correctness on %s", gfx)

    results = []
    runnable = []
    for case in CASES:
        if case[0] in ("fp8", "bf8") and gfx in _NO_FP8_ARCHES:
            results.append((SKIP, f"batched/{case[0]}/{case[1]}: no fp8 MFMA on {gfx}"))
            continue
        try:
            runnable.append((case, _case_config(gfx, *case[:2], *case[3:])))
        except Exception as exc:  # noqa: BLE001
            results.append((FAIL, f"batched {case}: no kernel: {exc}"))

    # One parallel build for every case.
    so_paths = setup_multiple_batched_gemm_dispatchers([c for _, c in runnable], verbose=False)
    for (case, cfg), so_path in zip(runnable, so_paths):
        try:
            results.append(check_batched(gfx, cfg, so_path, *case))
        except Exception as exc:  # noqa: BLE001
            results.append((FAIL, f"batched {case}: exception: {exc}"))

    print("\n=== Summary ===")
    for status, detail in results:
        print(f"  [{status:4s}] {detail}")
    n_pass = sum(status == PASS for status, _ in results)
    n_skip = sum(status == SKIP for status, _ in results)
    print(f"\n{n_pass}/{len(results)} passed, {n_skip} skipped")
    return 0 if n_pass + n_skip == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
