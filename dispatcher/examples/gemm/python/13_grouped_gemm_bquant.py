#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 13 — GroupedGemm BQuant via the Dispatcher

Demonstrates the full three-layer path:
  1. Codegen  — unified_grouped_gemm_bquant_codegen.py → .hpp
  2. Compile  — hipcc → .so
  3. Run      — BQuantGpuGemmRunner → C = A @ dequant(B, BQ)

Verifies the GPU result against a NumPy fp32 reference.

Requirements:
  - gfx950 GPU (MI350X)
  - hipcc in PATH
  - CK include path discoverable relative to this repo

Usage:
  python3 13_grouped_gemm_bquant.py                     # fp8, 1x1x128 groups, M=16 N=64 K=256
  python3 13_grouped_gemm_bquant.py --dtype bf8
  python3 13_grouped_gemm_bquant.py --dtype fp8 --M 32 --N 128 --K 512 --quant-group-k 128
  python3 13_grouped_gemm_bquant.py --no-verify         # skip CPU reference check
"""

import argparse
import logging
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add dispatcher/python to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))

from grouped_gemm_bquant_utils import (
    BQuantKernelConfig,
    BQuantGemmProblem,
    BQuantGpuGemmRunner,
    setup_multiple_bquant_dispatchers,
    default_fp8_config,
    default_bf8_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# NumPy reference: C = A @ dequant(B, BQ)
# =============================================================================


def _float32_to_fp8(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Encode float32 values as fp8 bytes (uint8 view of the fp8 bit pattern).

    dtype: "fp8" -> float8_e4m3fn, "bf8" -> float8_e5m2.
    Falls back to clamping the float32 values to [-2.0, 2.0] and storing as
    uint8 when ml_dtypes is not installed; the bit patterns are not true fp8 in
    that case but the buffer has the correct element size (1 byte/element).
    """
    try:
        import ml_dtypes
        ml_t = ml_dtypes.float8_e4m3fn if dtype == "fp8" else ml_dtypes.float8_e5m2
        return arr.astype(ml_t).view(np.uint8)
    except ImportError:
        # Fallback: clamp to a small range so values fit in fp8, store as uint8.
        # The bit patterns are not genuine fp8, but the buffer size is correct.
        clamped = np.clip(arr, -2.0, 2.0)
        return (clamped * 64).astype(np.int8).view(np.uint8)


def _fp8_to_float32(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Decode fp8 bytes (uint8 view) back to float32.

    dtype: "fp8" -> float8_e4m3fn, "bf8" -> float8_e5m2.
    Must be called on the same array produced by _float32_to_fp8 to get the
    values the kernel actually computes on.
    """
    try:
        import ml_dtypes
        ml_t = ml_dtypes.float8_e4m3fn if dtype == "fp8" else ml_dtypes.float8_e5m2
        return arr.view(ml_t).astype(np.float32)
    except ImportError:
        return arr.view(np.int8).astype(np.float32) / 64.0


def reference_bquant_gemm(
    A: np.ndarray,
    B: np.ndarray,
    BQ: np.ndarray,
    problem: BQuantGemmProblem,
) -> np.ndarray:
    """
    CPU fp32 reference for C = A @ dequant(B, BQ).

    A   [M, K]  float32 — must be decoded from the same fp8 bytes sent to the GPU
    B   [K, N]  float32 — must be decoded from the same fp8 bytes sent to the GPU
    BQ  [QK_B, QN_B] float32 scale factors

    Dequant: B[k, n] *= BQ[k // gK, n // gN]
    """
    M, N, K = problem.M, problem.N, problem.K
    gK = problem.quant_group_k
    gN = problem.quant_group_n

    A_f32 = A.astype(np.float32)
    B_f32 = B.astype(np.float32)

    # Apply per-block scales to B
    B_dequant = B_f32.copy()
    for qi in range(problem.QK_B):
        for qj in range(problem.QN_B):
            k_start = qi * gK
            k_end   = min(k_start + gK, K)
            n_start = qj * gN
            n_end   = min(n_start + gN, N)
            scale   = float(BQ[qi, qj])
            B_dequant[k_start:k_end, n_start:n_end] *= scale

    C_ref = A_f32 @ B_dequant
    return C_ref.astype(np.float16)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="GroupedGemm BQuant dispatcher example")
    parser.add_argument("--dtype", choices=["fp8", "bf8"], default="fp8")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--quant-group-k", type=int, default=128)
    parser.add_argument("--quant-group-n", type=int, default=1)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gfx-arch", type=str, default="gfx950")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    gK = args.quant_group_k
    gN = args.quant_group_n

    # -------------------------------------------------------------------------
    # 1. Build kernel config
    # -------------------------------------------------------------------------
    if args.dtype == "fp8":
        config = default_fp8_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=args.gfx_arch)
    else:
        config = default_bf8_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=args.gfx_arch)

    log.info("Kernel: %s", config.name)

    # -------------------------------------------------------------------------
    # 2. Codegen + compile
    # -------------------------------------------------------------------------
    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="bquant_ex13_"))
    log.info("Output dir: %s", out_dir)

    so_paths = setup_multiple_bquant_dispatchers(
        configs=[config],
        output_dir=out_dir,
        gfx_arch=args.gfx_arch,
    )

    if not so_paths or so_paths[0] is None:
        log.error("Kernel build failed — see errors above")
        return 1

    so_path = so_paths[0]
    log.info("Built: %s", so_path)

    # -------------------------------------------------------------------------
    # 3. Generate inputs
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(42)
    QK_B = math.ceil(K / gK)
    QN_B = math.ceil(N / gN)

    # Generate float32 values in the fp8 representable range, then encode as
    # real fp8 bytes (1 byte/element) so the C layer receives the correct size.
    A_f32 = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
    B_f32 = rng.uniform(-2.0, 2.0, (K, N)).astype(np.float32)
    BQ_f32 = rng.uniform(0.5, 2.0, (QK_B, QN_B)).astype(np.float32)

    # Encode as fp8 bytes (uint8 view). _float32_to_fp8 uses ml_dtypes when
    # available for accurate fp8 bit patterns, with a fallback otherwise.
    A_raw = _float32_to_fp8(A_f32, args.dtype)   # shape (M, K), 1 byte/element
    B_raw = _float32_to_fp8(B_f32, args.dtype)   # shape (K, N), 1 byte/element

    # Decode back to float32 so the CPU reference sees the same values the
    # kernel will compute on (fp8 encoding introduces rounding).
    A_dec = _fp8_to_float32(A_raw, args.dtype)
    B_dec = _fp8_to_float32(B_raw, args.dtype)

    # -------------------------------------------------------------------------
    # 4. Run on GPU
    # -------------------------------------------------------------------------
    problem = BQuantGemmProblem(M=M, N=N, K=K,
                                quant_group_m=1,
                                quant_group_n=gN,
                                quant_group_k=gK)

    runner = BQuantGpuGemmRunner(so_path)
    log.info("Running kernel: %s", runner.kernel_name)

    result = runner.run(A=A_raw, B=B_raw, BQ=BQ_f32, problem=problem)
    log.info("Kernel time: %.3f ms", result.time_ms)

    # -------------------------------------------------------------------------
    # 5. Verify
    # -------------------------------------------------------------------------
    if not args.no_verify:
        # Reference uses decoded fp8 values — the same bit patterns the kernel sees.
        C_ref = reference_bquant_gemm(A_dec, B_dec, BQ_f32, problem)
        C_gpu = result.C

        max_rel = float(np.max(np.abs(C_gpu.astype(np.float32) - C_ref.astype(np.float32)))
                        / (np.max(np.abs(C_ref.astype(np.float32))) + 1e-6))

        tolerance = 0.05  # fp8 ~1e-2 to 5e-2
        if max_rel <= tolerance:
            log.info("PASSED (max_rel=%.4f, tol=%.4f)", max_rel, tolerance)
        else:
            log.error("FAILED (max_rel=%.4f > tol=%.4f)", max_rel, tolerance)
            return 1
    else:
        log.info("Verification skipped (--no-verify)")

    log.info("Example 13 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
