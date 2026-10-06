#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GPU correctness tests for the single-GEMM ABQuant dispatcher bridge.

NOT the same operator as test_abquant_gpu_correctness.py, which imports
grouped_gemm_abquant_utils and drives the GROUPED bridge. Both modules export
the same class names (ABQuantGemmProblem, ABQuantGpuGemmRunner), so an import
line is the only thing distinguishing them -- and their problem descriptors are
genuinely different, see below. Neither test substitutes for the other.

Kernel computed here -- scales on *both* operands:

    C = dequant(A, AQ) @ dequant(B, BQ)
    A[m, k] *= AQ[m, k // aquant_group_k]
    B[k, n] *= BQ[k // bquant_group_k, n // bquant_group_n]

The single-GEMM ABQuantGemmProblem has no aquant_group_m and no QM_A -- the A
scale grid is (M, QK_A), one scale per row per K-group, where the grouped
bridge's is (QM_A, QK_A). Reusing the grouped reference's QM_A double loop here
gives a plausible but wrong answer, so the AQ shape is asserted before the
comparison.

Requires gfx942 (MI300X) or gfx950 (MI350X / MI355X) and hipcc in PATH. Skips
cleanly (exit 77) when no GPU is visible or the detected arch is neither.

The host fp8 codec follows the target arch -- OCP on gfx950, FNUZ on gfx942 --
via dispatcher_common.fp8_uses_ocp, the same predicate the compile flags use.
Encoding with the wrong one makes the device read NaN, which is why this is
sourced from one place rather than re-derived here.

BQ is handed to the runner in logical (QK_B, QN_B) indexing; the ctypes layer
does the col-major conversion itself (np.asfortranarray), so no transpose here.

Tests:
  fp8, bf8   -- GPU output is non-zero, non-constant, finite, and within 5%
                max-relative-error vs. an fp32 CPU reference; time_ms positive.
  fp8/tiled  -- same checks on a shape spanning a 2x2 grid of output tiles
                (tile 128x128), which the single-tile cases cannot catch.

Run:
  python3 test_gemm_abquant_gpu_correctness.py
  python3 test_gemm_abquant_gpu_correctness.py -v          # verbose hipcc output
  python3 test_gemm_abquant_gpu_correctness.py --gfx gfx942
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from dispatcher_common import fp8_uses_ocp  # noqa: E402

from gemm_abquant_utils import (  # noqa: E402
    ABQuantGemmProblem,
    ABQuantGpuGemmRunner,
    setup_multiple_abquant_dispatchers,
    default_fp8_config,
    default_bf8_config,
)

log = logging.getLogger(__name__)

TOLERANCE = 0.05  # 5% max relative error -- fp8/bf8 precision floor

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

# ctest SKIP_RETURN_CODE, and the Jenkins lane's run_ok helper maps it to 0.
# Without it a CPU-only runner reports every case as a hard FAIL for a reason
# that has nothing to do with the code under test.
SKIP_EXIT = 77

# Matches gemm_abquant_utils._SUPPORTED_ARCHS. ABQuant's compv3 pipeline uses
# standard fp8 MFMA (mfma_f32_16x16x32_fp8_fp8), which gfx942 has.
SUPPORTED_ARCHS = ("gfx942", "gfx950")


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

def _fp8_ml_dtype(dtype: str, gfx_arch: str):
    """The ml_dtypes fp8/bf8 type matching what the kernel for `gfx_arch` produces.

    gfx950 kernels are compiled OCP (e4m3fn / e5m2); gfx942 is native FNUZ
    (e4m3fnuz / e5m2fnuz), which differs by one in the exponent bias. Encoding
    with the wrong one shifts every value by a factor of two -- large enough to
    be real error, small enough for a 5% relative gate to sometimes swallow. The
    predicate is shared with the JIT flags (dispatcher_common.ocp_arch_defines)
    so the reference and the kernel cannot disagree.
    """
    import ml_dtypes

    if fp8_uses_ocp(gfx_arch):
        return ml_dtypes.float8_e4m3fn if dtype == "fp8" else ml_dtypes.float8_e5m2
    return ml_dtypes.float8_e4m3fnuz if dtype == "fp8" else ml_dtypes.float8_e5m2fnuz


def _encode_fp8(arr: np.ndarray, dtype: str, gfx_arch: str) -> np.ndarray:
    """Encode float32 -> fp8/bf8 bytes (uint8 view of the bit pattern)."""
    return arr.astype(_fp8_ml_dtype(dtype, gfx_arch)).view(np.uint8)


def _decode_fp8(arr: np.ndarray, dtype: str, gfx_arch: str) -> np.ndarray:
    """Decode fp8/bf8 bytes (uint8 view) back to float32."""
    return arr.view(_fp8_ml_dtype(dtype, gfx_arch)).astype(np.float32)


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------

def _reference_gemm(A_f32: np.ndarray, B_f32: np.ndarray,
                    AQ: np.ndarray, BQ: np.ndarray,
                    problem: ABQuantGemmProblem) -> np.ndarray:
    """C = dequant(A, AQ) @ dequant(B, BQ) in fp32.

    Both operands carry scales and their group axes differ: AQ is indexed
    (m, k-group), BQ is indexed (k-group, n-group). np.repeat expands each scale
    grid back to full operand shape; the trailing slices drop the padding when a
    dimension is not a whole multiple of its group size.
    """
    M, N, K = problem.M, problem.N, problem.K
    aK = problem.aquant_group_k
    bK, bN = problem.bquant_group_k, problem.bquant_group_n

    a_scales = np.repeat(AQ.astype(np.float32), aK, axis=1)[:, :K]
    A_dq = A_f32.astype(np.float32) * a_scales

    b_scales = np.repeat(np.repeat(BQ.astype(np.float32), bK, axis=0), bN, axis=1)
    B_dq = B_f32.astype(np.float32) * b_scales[:K, :N]

    return (A_dq @ B_dq).astype(np.float32)


def _max_rel_err(C_gpu: np.ndarray, C_ref: np.ndarray) -> float:
    """Max elementwise relative error with a floor on the denominator.

    A GEMM output has elements that partially cancel toward zero; dividing by a
    near-zero reference element inflates the error without saying anything about
    kernel correctness. Flooring the denominator at 1% of the global max keeps
    the metric sensitive to structural errors and insensitive to cancellation.
    """
    C_gpu_f = C_gpu.astype(np.float32)
    C_ref_f = C_ref.astype(np.float32)
    ref_max = float(np.abs(C_ref_f).max())
    den = np.abs(C_ref_f) + max(ref_max * 1e-2, 1e-6)
    return float(np.max(np.abs(C_gpu_f - C_ref_f) / den))


# ---------------------------------------------------------------------------
# Core test helper
# ---------------------------------------------------------------------------

def _detect_arch():
    """Detected gfx arch, or None when no GPU is visible.

    The empty fallback keeps absence detectable; detect_gpu_arch would otherwise
    invent a default and turn "no GPU" into a confusing build failure.
    """
    try:
        from dispatcher_common import detect_gpu_arch

        return detect_gpu_arch(fallback="") or None
    except Exception:
        return None


def _make_inputs(problem: ABQuantGemmProblem, dtype: str, gfx_arch: str, seed=42):
    """Generate fp8-encoded A/B plus float32 AQ/BQ scales, and the decoded references.

    A and B are encoded to fp8 and decoded back so the CPU reference consumes the
    same rounded values the kernel does; otherwise the comparison would be
    dominated by quantisation error rather than kernel correctness.

    AQ is row-major (M, QK_A) -- the single-GEMM bridge has no aquant_group_m.
    BQ is passed in logical (QK_B, QN_B); the runner handles the col-major
    conversion, so no transpose is applied here.
    """
    rng = np.random.default_rng(seed)
    M, N, K = problem.M, problem.N, problem.K

    A_f32 = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
    B_f32 = rng.uniform(-2.0, 2.0, (K, N)).astype(np.float32)
    AQ = rng.uniform(0.5, 2.0, (M, problem.QK_A)).astype(np.float32)
    BQ = rng.uniform(0.5, 2.0, (problem.QK_B, problem.QN_B)).astype(np.float32)

    A_raw = _encode_fp8(A_f32, dtype, gfx_arch)
    B_raw = _encode_fp8(B_f32, dtype, gfx_arch)
    return (A_raw, _decode_fp8(A_raw, dtype, gfx_arch),
            B_raw, _decode_fp8(B_raw, dtype, gfx_arch), AQ, BQ)


def _run_one(label, config, M, N, K, dtype, out_dir: Path, gfx_arch: str, seed=42):
    """Build, run, and verify one kernel. Returns (PASS|FAIL, detail_message)."""
    problem = ABQuantGemmProblem(
        M=M, N=N, K=K,
        aquant_group_k=config.aquant_group_k,
        bquant_group_n=config.bquant_group_n,
        bquant_group_k=config.bquant_group_k,
    )

    A_raw, A_f32, B_raw, B_f32, AQ, BQ = _make_inputs(problem, dtype, gfx_arch, seed)

    # Guards the (M, QK_A) vs the grouped bridge's (QM_A, QK_A) confusion: a
    # mismatch would otherwise surface as a tolerance failure whose cause is not
    # obvious from the message.
    if AQ.shape != (M, problem.QK_A):
        return FAIL, (f"{label}: AQ shape {AQ.shape} != {(M, problem.QK_A)}; "
                      f"single-GEMM abquant has no aquant_group_m")

    so_paths = setup_multiple_abquant_dispatchers(
        configs=[config],
        output_dir=out_dir,
        gfx_arch=gfx_arch,
    )
    if not so_paths or so_paths[0] is None:
        return FAIL, f"{label}: kernel build failed"

    runner = ABQuantGpuGemmRunner(so_paths[0])

    result = runner.run(A=A_raw, B=B_raw, AQ=AQ, BQ=BQ, problem=problem,
                        c_dtype=np.float16)
    C_gpu = result.C.astype(np.float32)

    # A degenerate C is the signature of a kernel that launched but never did the
    # work -- e.g. a wrong warp_tile_k (128 vs 32 across the gfx950/gfx942 split)
    # returning all zeros. Checked before the tolerance gate because a small
    # reference can let an all-zero output slide past a relative bound.
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
        # The usual cause is an fp8 codec that disagrees with the compiled arch.
        return FAIL, (f"{label}: GPU output contains NaN/Inf (frac={nan_frac:.3f}); "
                      f"check fp8 encoding vs {gfx_arch} "
                      f"({'OCP' if fp8_uses_ocp(gfx_arch) else 'FNUZ'})")

    C_ref = _reference_gemm(A_f32, B_f32, AQ, BQ, problem)
    mre = _max_rel_err(C_gpu, C_ref)
    if mre > TOLERANCE:
        return FAIL, (f"{label}: max_rel_err={mre:.4f} > tol={TOLERANCE:.4f} "
                      f"(shape M={M} N={N} K={K})")

    # Timing sanity: a zero time_ms means the HIP event pair never bracketed a launch.
    result_timed = runner.run(A=A_raw, B=B_raw, AQ=AQ, BQ=BQ, problem=problem,
                              c_dtype=np.float16)
    if result_timed.time_ms <= 0.0:
        return FAIL, f"{label}: time_ms={result_timed.time_ms:.4f} is not positive"

    return PASS, (f"{label}: max_rel_err={mre:.4f}, "
                  f"time_ms={result_timed.time_ms:.3f}, "
                  f"kernel={runner.kernel_name}")


# ---------------------------------------------------------------------------
# Individual test cases
# ---------------------------------------------------------------------------
#
# kPadM/kPadN/kPadK are all false for the ABQuant prefill config, so M, N and K
# must be whole multiples of the 128x128x128 tile or IsSupportedArguments
# rejects the launch. bquant_group_n stays 1, which is the hardcoded
# GemmConfigABQuantPrefill path (group_n>1 would route to eight_waves on gfx950
# and take a different, gfx950-only code path).

def case_fp8(out_dir: Path, gfx_arch: str):
    # K=512 = 4*TileK(128): four main-loop iterations, so the AQ/BQ scale
    # prefetch runs more than the degenerate single pass.
    M, N, K = 128, 128, 512
    cfg = default_fp8_config(bquant_group_n=1, gfx_arch=gfx_arch)
    return _run_one("fp8", cfg, M, N, K, "fp8", out_dir, gfx_arch, seed=42)


def case_bf8(out_dir: Path, gfx_arch: str):
    M, N, K = 128, 128, 512
    cfg = default_bf8_config(bquant_group_n=1, gfx_arch=gfx_arch)
    return _run_one("bf8", cfg, M, N, K, "bf8", out_dir, gfx_arch, seed=43)


def case_fp8_tiled(out_dir: Path, gfx_arch: str):
    # M=N=256 spans a 2x2 grid of 128x128 output tiles. The single-tile cases
    # above cannot distinguish a correct tile index from a constant one.
    M, N, K = 256, 256, 512
    cfg = default_fp8_config(bquant_group_n=1, gfx_arch=gfx_arch)
    return _run_one("fp8/tiled", cfg, M, N, K, "fp8", out_dir, gfx_arch, seed=44)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = [
    ("fp8",       case_fp8),
    ("bf8",       case_bf8),
    ("fp8/tiled", case_fp8_tiled),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-GEMM ABQuant GPU correctness tests"
    )
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
        print("SKIP: no supported GPU detected (rocminfo); "
              "gemm_abquant GPU tests skipped")
        return SKIP_EXIT
    if gfx not in SUPPORTED_ARCHS:
        print(f"SKIP: gemm_abquant needs native fp8 "
              f"({'/'.join(SUPPORTED_ARCHS)}); detected {gfx}")
        return SKIP_EXIT
    try:
        import ml_dtypes  # noqa: F401
    except ImportError:
        # Without ml_dtypes there is no trustworthy fp8 codec here, and a
        # stand-in would compare the kernel against the wrong values.
        print("SKIP: ml_dtypes not installed; fp8/bf8 encoding unavailable")
        return SKIP_EXIT

    log.info("Running gemm_abquant GPU correctness on %s (%s fp8)",
             gfx, "OCP" if fp8_uses_ocp(gfx) else "FNUZ")

    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="gemm_abquant_gputest_"))
    log.info("Kernel output dir: %s", out_dir)

    results = []
    for name, fn in TESTS:
        log.info("--- Running %s ---", name)
        try:
            status, detail = fn(out_dir, gfx)
        except Exception as exc:  # noqa: BLE001
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
