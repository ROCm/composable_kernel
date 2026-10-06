#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GPU correctness tests for AQuant GEMM dispatcher — A-side per-group scales.

Companion to test_bquant_gpu_correctness.py. Where BQuant scales the B operand
per (k, n) group, AQuant scales A per (m, k) group:

    C = dequant(A, AQ) @ B,   A[m, k] *= AQ[m // gM, k // gK]

Requires a gfx942 (MI300X) or gfx950 (MI350X / MI355X) GPU and hipcc in PATH.
Skips cleanly (exit 77) when no GPU is visible or the detected arch is neither.

The non-preshuffleaq tests (fp8, bf8, fp8/tiled) use standard fp8 MFMA
(warp_tile_k=32) which gfx942 has. The preshuffleaq tests are gated behind
PRESHUFFLEAQ_SUPPORTED_ARCHS = ("gfx950",): they use FlatMM (warp_tile_k=64
on gfx942), and that path has not yet been CI-verified on gfx942.

The host fp8 codec follows the target arch — OCP on gfx950, FNUZ on gfx942.

Tests:
  fp8, bf8 — GPU output is non-zero, non-constant, finite, and within 5%
             max-relative-error vs. a fp32 CPU reference; time_ms is positive.
  fp8/tiled — same checks on a shape spanning a 4x4 grid of output tiles
             (tile_m=16, tile_n=64), which the 1x1-tile cases cannot catch.

Run:
  python3 test_aquant_gpu_correctness.py
  python3 test_aquant_gpu_correctness.py -v          # verbose hipcc output
  python3 test_aquant_gpu_correctness.py --gfx gfx942
"""

import argparse
import logging
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from dispatcher_common import fp8_uses_ocp

from grouped_gemm_aquant_utils import (
    AQuantGemmProblem,
    AQuantGpuGemmRunner,
    setup_multiple_aquant_dispatchers,
    default_fp8_config,
    default_bf8_config,
)

log = logging.getLogger(__name__)

TOLERANCE = 0.05  # 5% max relative error — fp8/bf8 precision floor

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

# ctest reports this as "skipped" rather than passed. Without it, a CPU-only or
# non-supported runner would report every AQuant test as a hard FAIL for a reason
# that has nothing to do with the code under test.
SKIP_EXIT = 77

# fp8, bf8, fp8/tiled use standard MFMA (warp_tile_k=32), present on gfx942.
# The former gfx950-only restriction was the OCP-hardcoded host codec, now fixed.
SUPPORTED_ARCHS = ("gfx942", "gfx950")

# preshuffleaq uses FlatMM (warp_tile_k=64 on gfx942 per _preshuffleaq_warp_tile_k).
# Held back one CI run: FlatMM has not been empirically exercised on gfx942.
PRESHUFFLEAQ_SUPPORTED_ARCHS = ("gfx950",)


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

def _fp8_ml_dtype(dtype: str, gfx_arch: str):
    """The ml_dtypes fp8/bf8 type matching what the kernel for `gfx_arch` produces."""
    import ml_dtypes

    if fp8_uses_ocp(gfx_arch):
        return ml_dtypes.float8_e4m3fn if dtype == "fp8" else ml_dtypes.float8_e5m2
    return ml_dtypes.float8_e4m3fnuz if dtype == "fp8" else ml_dtypes.float8_e5m2fnuz


def _encode_fp8(arr: np.ndarray, dtype: str, gfx_arch: str) -> np.ndarray:
    """Encode float32 → fp8/bf8 bytes (uint8 view). Format follows target arch."""
    return arr.astype(_fp8_ml_dtype(dtype, gfx_arch)).view(np.uint8)


def _decode_fp8(arr: np.ndarray, dtype: str, gfx_arch: str) -> np.ndarray:
    """Decode fp8/bf8 bytes (uint8 view) → float32."""
    return arr.view(_fp8_ml_dtype(dtype, gfx_arch)).astype(np.float32)


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------

def _reference_gemm(A_f32: np.ndarray, B_f32: np.ndarray,
                    AQ: np.ndarray, problem: AQuantGemmProblem) -> np.ndarray:
    """C = dequant(A, AQ) @ B in fp32.

    A is scaled per (m, k) group — the transpose of BQuant's (k, n) grouping.
    Mirrors reference_aquant_gemm in examples/gemm/python/14_grouped_gemm_aquant.py.
    """
    gM, gK = problem.quant_group_m, problem.quant_group_k
    A_dq = A_f32.astype(np.float32).copy()
    for qi in range(problem.QM_A):
        for qk in range(problem.QK_A):
            m0, m1 = qi * gM, min((qi + 1) * gM, problem.M)
            k0, k1 = qk * gK, min((qk + 1) * gK, problem.K)
            A_dq[m0:m1, k0:k1] *= float(AQ[qi, qk])
    return (A_dq @ B_f32.astype(np.float32)).astype(np.float32)


def _max_rel_err(C_gpu: np.ndarray, C_ref: np.ndarray) -> float:
    C_gpu_f = C_gpu.astype(np.float32)
    C_ref_f = C_ref.astype(np.float32)
    num = np.abs(C_gpu_f - C_ref_f)
    # Use 1% of the global max magnitude as the denominator floor to avoid
    # inflating the relative error when individual elements are near zero
    # (a common occurrence with random inputs that partially cancel in GEMM).
    ref_max = float(np.abs(C_ref_f).max())
    den = np.abs(C_ref_f) + max(ref_max * 1e-2, 1e-6)
    return float(np.max(num / den))


# ---------------------------------------------------------------------------
# Core test helper
# ---------------------------------------------------------------------------

def _detect_arch() -> "str | None":
    """Detected gfx arch, or None when no GPU is visible.

    The empty fallback keeps absence detectable; detect_gpu_arch would otherwise
    invent a default and turn "no GPU" into a confusing build failure.
    """
    try:
        from dispatcher_common import detect_gpu_arch

        return detect_gpu_arch(fallback="") or None
    except Exception:
        return None


def _make_inputs(M, N, K, gM, gK, dtype, gfx_arch: str, seed=42):
    """Generate fp8-encoded A/B plus float32 AQ scales, and the decoded references.

    A and B are encoded to fp8 and decoded back so the CPU reference consumes the
    same rounded values the kernel does; otherwise the comparison would be
    dominated by quantisation error rather than kernel correctness.
    """
    rng = np.random.default_rng(seed)
    QM_A = math.ceil(M / gM)
    QK_A = math.ceil(K / gK)
    A_f32 = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
    B_f32 = rng.uniform(-2.0, 2.0, (K, N)).astype(np.float32)
    AQ    = rng.uniform(0.5, 2.0, (QM_A, QK_A)).astype(np.float32)
    A_raw = _encode_fp8(A_f32, dtype, gfx_arch)
    B_raw = _encode_fp8(B_f32, dtype, gfx_arch)
    return (A_raw, _decode_fp8(A_raw, dtype, gfx_arch),
            B_raw, _decode_fp8(B_raw, dtype, gfx_arch), AQ)


def _run_one(label: str, config, M: int, N: int, K: int,
             A_raw: np.ndarray, A_f32: np.ndarray,
             B_raw: np.ndarray, B_f32: np.ndarray,
             AQ: np.ndarray,
             out_dir: Path,
             gfx_arch: str) -> "tuple[str, str]":
    """Build, run, and verify one kernel. Returns (PASS|FAIL, detail_message)."""
    problem = AQuantGemmProblem(
        M=M, N=N, K=K,
        quant_group_m=config.quant_group_m,
        quant_group_n=config.quant_group_n,
        quant_group_k=config.quant_group_k,
    )

    so_paths = setup_multiple_aquant_dispatchers(
        configs=[config],
        output_dir=out_dir,
        gfx_arch=gfx_arch,
    )
    if not so_paths or so_paths[0] is None:
        return FAIL, f"{label}: kernel build failed"

    runner = AQuantGpuGemmRunner(so_paths[0])

    result = runner.run(A=A_raw, B=B_raw, AQ=AQ, problem=problem, c_dtype=np.float16)
    C_gpu = result.C.astype(np.float32)

    # A degenerate C is the signature of a kernel that launched but never did the
    # work — e.g. a wrong warp_tile_k (CK_GFX950_SUPPORT branch mismatch) returning
    # all zeros on gfx942. Check before the tolerance gate; see abquant for rationale.
    if np.all(C_gpu == 0):
        return FAIL, f"{label}: GPU output is all-zero"
    if float(C_gpu.std()) <= 0.0:
        return FAIL, (f"{label}: GPU output is constant "
                      f"(value={float(C_gpu.flat[0]):.6g}); kernel did not compute")
    nan_mask = ~np.isfinite(C_gpu)
    if np.any(nan_mask):
        nan_frac = nan_mask.mean()
        log.debug("%s: NaN/Inf fraction=%.3f, first 8 elements=%s",
                  label, nan_frac, C_gpu.flat[:8].tolist())
        return FAIL, f"{label}: GPU output contains NaN/Inf (frac={nan_frac:.3f})"

    C_ref = _reference_gemm(A_f32, B_f32, AQ, problem)
    mre = _max_rel_err(C_gpu, C_ref)
    if mre > TOLERANCE:
        return FAIL, (f"{label}: max_rel_err={mre:.4f} > tol={TOLERANCE:.4f} "
                      f"(shape M={M} N={N} K={K})")

    # Timing sanity: a zero time_ms means the HIP event pair never bracketed a launch.
    result_timed = runner.run(A=A_raw, B=B_raw, AQ=AQ, problem=problem,
                              c_dtype=np.float16)
    if result_timed.time_ms <= 0.0:
        return FAIL, f"{label}: time_ms={result_timed.time_ms:.4f} is not positive"

    return PASS, (f"{label}: max_rel_err={mre:.4f}, "
                  f"time_ms={result_timed.time_ms:.3f}")


# ---------------------------------------------------------------------------
# Individual test cases
# ---------------------------------------------------------------------------

def case_fp8(out_dir: Path, gfx_arch: str) -> "tuple[str, str]":
    # K=768 = 3*TileK(256): num_loop=3 gives the no-hot-loop/TailNumber::Odd path,
    # which exercises the AQ scale prefetch more thoroughly than an even tail.
    M, N, K, gM, gK = 16, 64, 768, 1, 128
    cfg = default_fp8_config(quant_group_k=gK, quant_group_m=gM, gfx_arch=gfx_arch)
    A_raw, A_dec, B_raw, B_dec, AQ = _make_inputs(M, N, K, gM, gK, "fp8", gfx_arch)
    return _run_one("fp8", cfg, M, N, K, A_raw, A_dec, B_raw, B_dec, AQ,
                    out_dir, gfx_arch=gfx_arch)


def case_bf8(out_dir: Path, gfx_arch: str) -> "tuple[str, str]":
    M, N, K, gM, gK = 16, 64, 768, 1, 128
    cfg = default_bf8_config(quant_group_k=gK, quant_group_m=gM, gfx_arch=gfx_arch)
    A_raw, A_dec, B_raw, B_dec, AQ = _make_inputs(M, N, K, gM, gK, "bf8", gfx_arch,
                                                    seed=43)
    return _run_one("bf8", cfg, M, N, K, A_raw, A_dec, B_raw, B_dec, AQ,
                    out_dir, gfx_arch=gfx_arch)


def case_fp8_tiled(out_dir: Path, gfx_arch: str) -> "tuple[str, str]":
    # M=64, N=256 spans 4x4 output tiles (tile_m=16, tile_n=64). The single-tile
    # cases above cannot distinguish a correct tile index from a constant one.
    M, N, K, gM, gK = 64, 256, 768, 1, 128
    cfg = default_fp8_config(quant_group_k=gK, quant_group_m=gM, gfx_arch=gfx_arch)
    A_raw, A_dec, B_raw, B_dec, AQ = _make_inputs(M, N, K, gM, gK, "fp8", gfx_arch,
                                                    seed=44)
    return _run_one("fp8/tiled", cfg, M, N, K, A_raw, A_dec, B_raw, B_dec, AQ,
                    out_dir, gfx_arch=gfx_arch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = [
    ("fp8",       case_fp8),
    ("bf8",       case_bf8),
    ("fp8/tiled", case_fp8_tiled),
]


def main():
    parser = argparse.ArgumentParser(description="AQuant GPU correctness tests")
    # No hardcoded default: it must stay possible to tell "user asked for
    # gfx950" from "we are on an unrelated box", so the skip below can fire.
    parser.add_argument("--gfx", default=None,
                        help="GPU arch override (default: auto-detect)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    gfx = args.gfx or _detect_arch()
    if not gfx:
        print("SKIP: no supported GPU detected (rocminfo); AQuant GPU tests skipped")
        return SKIP_EXIT
    if gfx not in SUPPORTED_ARCHS:
        print(f"SKIP: AQuant is {'/'.join(SUPPORTED_ARCHS)}-only; detected {gfx}")
        return SKIP_EXIT
    try:
        import ml_dtypes  # noqa: F401
    except ImportError:
        # Without ml_dtypes there is no trustworthy fp8 codec here, and a
        # stand-in would compare the kernel against the wrong values.
        print("SKIP: ml_dtypes not installed; AQuant fp8/bf8 encoding unavailable")
        return SKIP_EXIT

    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="aquant_gpu_test_"))
    log.info("Kernel output dir: %s", out_dir)

    results = []
    for name, fn in TESTS:
        log.info("--- Running %s ---", name)
        try:
            status, detail = fn(out_dir, gfx)
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
