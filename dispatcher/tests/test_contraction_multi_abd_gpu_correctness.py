#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GPU correctness test for the batched-contraction-multi-ABD dispatcher bridge.

The CPU-only suite (test_contraction_multi_abd_bridge.py) validates plumbing:
kernel-name uniqueness, config round-trip, ABI marshalling against a mocked
ctypes lib. It does not validate arithmetic. The parts most likely to be wrong
are exactly the ones it cannot reach -- the per-tensor device staging, the flat
dim/stride marshalling, and the multi-axis G/M/N/K batch offset math -- so this
test builds a real .so and compares the on-device result to an fp32 numpy
reference.

Kernel computed here (num_a=1, num_b=1, num_d=2):
    E[g,m,n] = sum_k A[g,m,k] * B[g,n,k] + D0[g,m,n] + D1[g,m,n]
i.e. an "rcr" per-group contraction (B carries K on its trailing axis) followed
by the MultiDAdd epilogue.

Coverage beyond the CPU suite:
  * batch_count > 1 (G=2), so a wrong batch stride cannot pass
  * multi-axis dims (num_dim_m/n = 2), so the folded M/N descriptor math runs
  * multi-D (num_d=2), so the epilogue's D loop runs
  * both flat 1-D and N-D-shaped host buffers, which take different paths
    through the runner's stride construction

Skips cleanly (exit 0) when no supported GPU or no hipcc is available.

Run:
  python3 test_contraction_multi_abd_gpu_correctness.py
  python3 test_contraction_multi_abd_gpu_correctness.py -v
  python3 test_contraction_multi_abd_gpu_correctness.py --gfx gfx942
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from contraction_multi_abd_utils import (  # noqa: E402
    ContractionMultiABDKernelConfig,
    ContractionMultiABDProblem,
    ContractionMultiABDRunner,
    setup_multiple_contraction_multi_abd_dispatchers,
    _detect_gpu_arch,
    _validate_arch,
)

log = logging.getLogger(__name__)

# fp16 inputs with fp32 accumulate; K=64 keeps the worst-case error well under this.
TOLERANCE = 1e-2

PASS = "PASS"
FAIL = "FAIL"

# Must match the kernel's default tile (256x256x64) exactly: the default config
# pads nothing, so M and N have to be whole multiples of the tile or
# IsSupportedArguments rejects the launch.
G_DIMS = [2]
M_DIMS = [2, 128]   # M_total = 256
N_DIMS = [2, 128]   # N_total = 256
K_DIMS = [64]
NUM_D = 2


def _has_gpu() -> bool:
    try:
        _detect_gpu_arch()
    except Exception:
        return False
    return shutil.which("hipcc") is not None


def _max_rel_err(gpu: np.ndarray, ref: np.ndarray) -> float:
    """Max absolute error normalized by the largest reference magnitude.

    A contraction output has entries that partially cancel toward zero, so a
    per-element relative error is dominated by those near-zero entries and does
    not measure whether the kernel computed the right tensor. Normalizing the
    worst absolute error by the global reference scale is the honest bar: any
    structural error (wrong batch offset, transposed operand, dropped D) blows
    far past 1e-2, while correct fp16 math lands around 1e-3-1e-4 here.
    """
    g = gpu.astype(np.float32)
    r = ref.astype(np.float32)
    return float(np.max(np.abs(g - r)) / max(float(np.abs(r).max()), 1e-6))


def _reference(A: np.ndarray, B: np.ndarray, Ds: list) -> np.ndarray:
    """E[g,m,n] = sum_k A[g,m,k]*B[g,n,k] + sum_i Ds[i][g,m,n], all in fp32."""
    e = np.einsum("gmk,gnk->gmn", A.astype(np.float32), B.astype(np.float32))
    for d in Ds:
        e = e + d.astype(np.float32)
    return e


def _build(gfx_arch: str):
    cfg = ContractionMultiABDKernelConfig(
        dtype="fp16",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=256, tile_n=256, tile_k=64,
        warp_m=2, warp_n=2, warp_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        num_a_tensor=1,
        num_b_tensor=1,
        num_d_tensor=NUM_D,
        num_dim_g=len(G_DIMS),
        num_dim_m=len(M_DIMS),
        num_dim_n=len(N_DIMS),
        num_dim_k=len(K_DIMS),
        gfx_arch=gfx_arch,
    )
    so_paths = setup_multiple_contraction_multi_abd_dispatchers([cfg], gfx_arch=gfx_arch)
    return so_paths[0] if so_paths else None


def test_contraction_multi_abd_fp16(gfx_arch: str):
    so_path = _build(gfx_arch)
    if so_path is None:
        return FAIL, "contraction_multi_abd/fp16: kernel build failed"

    runner = ContractionMultiABDRunner(so_path)
    prob = ContractionMultiABDProblem(
        g_dims=G_DIMS, m_dims=M_DIMS, n_dims=N_DIMS, k_dims=K_DIMS
    )
    G, M, N, K = prob.G_total, prob.M_total, prob.N_total, prob.K_total

    rng = np.random.default_rng(7)
    A = rng.uniform(-1.0, 1.0, (G, M, K)).astype(np.float16)
    B = rng.uniform(-1.0, 1.0, (G, N, K)).astype(np.float16)
    Ds = [rng.uniform(-1.0, 1.0, (G, M, N)).astype(np.float16) for _ in range(NUM_D)]
    ref = _reference(A, B, Ds)

    details = []
    # Flat and N-D host buffers reach the ABI through different stride paths;
    # both must produce the same answer.
    for label, shape in (("flat", lambda a: a.reshape(-1)), ("nd", lambda a: a)):
        result = runner.run([shape(A)], [shape(B)], [shape(d) for d in Ds], prob)
        gpu = np.asarray(result.E).reshape(G, M, N)

        if np.all(gpu == 0):
            return FAIL, f"contraction_multi_abd/fp16 [{label}]: GPU output is all-zero"
        if not np.all(np.isfinite(gpu.astype(np.float32))):
            return FAIL, f"contraction_multi_abd/fp16 [{label}]: output has NaN/Inf"

        mre = _max_rel_err(gpu, ref)
        if mre > TOLERANCE:
            return FAIL, (
                f"contraction_multi_abd/fp16 [{label}]: max_rel_err={mre:.4e} > "
                f"tol={TOLERANCE:.1e} (G={G} M={M} N={N} K={K} num_d={NUM_D})"
            )
        if result.time_ms <= 0.0:
            return FAIL, (
                f"contraction_multi_abd/fp16 [{label}]: "
                f"time_ms={result.time_ms:.4f} not positive"
            )
        details.append(f"{label} max_rel_err={mre:.4e} time_ms={result.time_ms:.3f}")

    return PASS, (
        f"contraction_multi_abd/fp16: {', '.join(details)} "
        f"(G={G} M={M} N={N} K={K} num_d={NUM_D})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batched-contraction-multi-ABD GPU correctness test"
    )
    parser.add_argument("--gfx", default=None, help="GPU arch (default: auto-detect)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not _has_gpu():
        print("SKIP: no supported GPU or hipcc detected; "
              "contraction_multi_abd GPU test skipped")
        return 0

    gfx = _validate_arch(args.gfx) if args.gfx else _detect_gpu_arch()
    log.info("Running contraction_multi_abd GPU correctness on %s", gfx)

    try:
        status, detail = test_contraction_multi_abd_fp16(gfx)
    except Exception as exc:  # noqa: BLE001
        status, detail = FAIL, f"contraction_multi_abd/fp16: exception: {exc}"

    print("\n=== Summary ===")
    print(f"  [{status:4s}] {detail}")
    print(f"\n{1 if status == PASS else 0}/1 passed")
    return 0 if status == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
