#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GPU correctness tests for ABQuant GEMM dispatcher — scales on *both* operands.

Completes the quant trio alongside test_aquant_gpu_correctness.py (A-side only)
and test_bquant_gpu_correctness.py (B-side only). ABQuant scales A per (m, k)
group and B per (k, n) group in the same kernel:

    C = dequant(A, AQ) @ dequant(B, BQ)
    A[m, k] *= AQ[m // aM, k // aK]
    B[k, n] *= BQ[k // bK, n // bN]

Requires a gfx942 (MI300X) or gfx950 (MI350X / MI355X) GPU and hipcc in PATH.
Skips cleanly (exit 77) when no GPU is visible or the detected arch is neither,
so it is safe to invoke unconditionally from a CI lane.

The fp8/bf8 host codec follows the target arch — OCP on gfx950, FNUZ on gfx942 —
via the same dispatcher_common.fp8_uses_ocp predicate that drives the JIT
defines, so the reference cannot end up encoding differently from the kernel.

Tests:
  fp8/compv3, bf8/compv3 — GPU output is non-zero, non-constant, finite, and
             within 5% max-relative-error vs. a fp32 CPU reference; time_ms is
             positive. M=N=256 spans a 2x2 grid of 128x128 output tiles, so a
             constant tile index cannot pass.
  fp8/eightwaves — the 8-wave pipeline (192x256 tiles, TransposeC=True,
             bquant_group_n=128), which exercises a different MFMA path and
             B-scale granularity than compv3. gfx950 only for now; see
             EIGHTWAVES_SUPPORTED_ARCHS.

Run:
  python3 test_abquant_gpu_correctness.py
  python3 test_abquant_gpu_correctness.py -v          # verbose hipcc output
  python3 test_abquant_gpu_correctness.py --gfx gfx950
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from dispatcher_common import fp8_uses_ocp

from grouped_gemm_abquant_utils import (
    ABQuantGemmProblem,
    ABQuantGpuGemmRunner,
    setup_multiple_abquant_dispatchers,
    default_fp8_compv3_config,
    default_bf8_compv3_config,
    default_fp8_eightwaves_config,
    _eightwaves_warp_tile_k,
    _preshuffleb_warp_tile_k,
)

log = logging.getLogger(__name__)

TOLERANCE = 0.05  # 5% max relative error — fp8/bf8 precision floor

PASS = "PASS"
FAIL = "FAIL"
# Per-case skip, for a pipeline not yet enabled on the detected arch. Distinct
# from SKIP_EXIT, which skips the whole file: here some cases did run, so the
# process still exits 0/1 on their result.
SKIP = "SKIP"

# ctest reports this as "skipped" rather than passed. Without it, a CPU-only or
# non-gfx950 runner would report every ABQuant test as a hard FAIL for a reason
# that has nothing to do with the code under test.
SKIP_EXIT = 77

# ABQuant's compv3 pipeline uses standard fp8 MFMA (mfma_f32_16x16x32_fp8_fp8),
# which gfx942 has; gemm_abquant/CMakeLists.txt lists gfx942 in DESIRED_TARGETS
# too. The former gfx950-only restriction was a conservative default, not a
# hardware limit -- what it was really working around was the fp8 encoding
# mismatch, now fixed by threading the arch through the host codec.
SUPPORTED_ARCHS = ("gfx942", "gfx950")

# eightwaves is held back one release. Its warp_tile_k is arch-aware
# (_eightwaves_warp_tile_k: 128 on gfx950, 32 elsewhere), so it *should* work,
# but 8-wave scheduling with 192x256 tiles has never been run on gfx942 and the
# failure mode if the tile is wrong is silent zeros rather than a build error.
# Widen this once one CI run has shown compv3 green on gfx942.
EIGHTWAVES_SUPPORTED_ARCHS = ("gfx950",)


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
    """Encode float32 → fp8/bf8 bytes (uint8 view of the bit pattern)."""
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

    Mirrors reference_abquant_gemm in examples/gemm/python/15_grouped_gemm_abquant.py.
    Unlike AQuant, both operands carry scales, and their group axes differ:
    AQ is indexed (m, k) and BQ is indexed (k, n).
    """
    M, N, K = problem.M, problem.N, problem.K
    aM, aK = problem.aquant_group_m, problem.aquant_group_k
    bK, bN = problem.bquant_group_k, problem.bquant_group_n

    A_dq = A_f32.astype(np.float32).copy()
    for qi in range(problem.QM_A):
        for qk in range(problem.QK_A):
            m0, m1 = qi * aM, min((qi + 1) * aM, M)
            k0, k1 = qk * aK, min((qk + 1) * aK, K)
            A_dq[m0:m1, k0:k1] *= float(AQ[qi, qk])

    B_dq = B_f32.astype(np.float32).copy()
    for qk in range(problem.QK_B):
        for qn in range(problem.QN_B):
            k0, k1 = qk * bK, min((qk + 1) * bK, K)
            n0, n1 = qn * bN, min((qn + 1) * bN, N)
            B_dq[k0:k1, n0:n1] *= float(BQ[qk, qn])

    return (A_dq @ B_dq).astype(np.float32)


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


def _make_inputs(problem: ABQuantGemmProblem, dtype: str, gfx_arch: str,
                 seed: int = 42):
    """Generate fp8-encoded A/B plus float32 AQ/BQ scales, and the decoded references.

    A and B are encoded to fp8 and decoded back so the CPU reference consumes the
    same rounded values the kernel does; otherwise the comparison would be
    dominated by quantisation error rather than kernel correctness.

    AQ is row-major [QM_A, QK_A]; BQ is column-major [QK_B, QN_B] — the runner
    applies the Fortran-order view, so the shape here stays logical.
    """
    rng = np.random.default_rng(seed)
    M, N, K = problem.M, problem.N, problem.K
    A_f32 = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
    B_f32 = rng.uniform(-2.0, 2.0, (K, N)).astype(np.float32)
    AQ    = rng.uniform(0.5, 2.0, (problem.QM_A, problem.QK_A)).astype(np.float32)
    BQ    = rng.uniform(0.5, 2.0, (problem.QK_B, problem.QN_B)).astype(np.float32)
    A_raw = _encode_fp8(A_f32, dtype, gfx_arch)
    B_raw = _encode_fp8(B_f32, dtype, gfx_arch)
    return (A_raw, _decode_fp8(A_raw, dtype, gfx_arch),
            B_raw, _decode_fp8(B_raw, dtype, gfx_arch), AQ, BQ)


def _expected_warp_tile_k(pipeline: str, gfx_arch: str) -> int:
    """The warp_tile_k get_k_warp_tile() will pick for `pipeline` on `gfx_arch`.

    compv3 is a standard compute pipeline (not FlatMM), so 32 -- the standard fp8
    MFMA -- is correct on both arches and is deliberately arch-independent.
    eightwaves and preshuffleb are FlatMM and do vary.
    """
    if pipeline == "compv3":
        return 32
    if pipeline == "preshuffleb":
        return _preshuffleb_warp_tile_k(gfx_arch)
    return _eightwaves_warp_tile_k(gfx_arch)


def _run_one(label: str, config, M: int, N: int, K: int, dtype: str,
             out_dir: Path, gfx_arch: str,
             seed: int = 42) -> "tuple[str, str]":
    """Build, run, and verify one kernel. Returns (PASS|FAIL, detail_message)."""
    # Arch trap, checked before the build: a warp_tile_k that disagrees with the
    # branch get_k_warp_tile() takes for this arch compiles cleanly and returns
    # zeros. The degenerate-C guards below do catch that, but only after a full
    # hipcc build, and they report it as "kernel did not compute" rather than
    # naming the config field that drifted.
    expected_wtk = _expected_warp_tile_k(config.pipeline, gfx_arch)
    if config.warp_tile_k != expected_wtk:
        return FAIL, (f"{label}: warp_tile_k arch trap: got {config.warp_tile_k}, "
                      f"expected {expected_wtk} for {config.pipeline} on {gfx_arch}")

    # The problem must repeat the config's quant grouping: the kernel bakes the
    # group sizes into the generated header, and the host strides are derived
    # from them, so a mismatch silently reads the wrong scale.
    problem = ABQuantGemmProblem(
        M=M, N=N, K=K,
        aquant_group_m=config.aquant_group_m,
        aquant_group_n=config.aquant_group_n,
        aquant_group_k=config.aquant_group_k,
        bquant_group_m=config.bquant_group_m,
        bquant_group_n=config.bquant_group_n,
        bquant_group_k=config.bquant_group_k,
    )

    so_paths = setup_multiple_abquant_dispatchers(
        configs=[config],
        output_dir=out_dir,
        gfx_arch=gfx_arch,
    )
    if not so_paths or so_paths[0] is None:
        return FAIL, f"{label}: kernel build failed"

    A_raw, A_dec, B_raw, B_dec, AQ, BQ = _make_inputs(
        problem, dtype, gfx_arch, seed=seed
    )

    runner = ABQuantGpuGemmRunner(so_paths[0])

    result = runner.run(A=A_raw, B=B_raw, AQ=AQ, BQ=BQ, problem=problem,
                        c_dtype=np.float16)
    C_gpu = result.C.astype(np.float32)

    # A degenerate C is the signature of a kernel that launched but never did the
    # work -- e.g. a warp_tile_k tuned for gfx950 built for gfx942, where
    # get_k_warp_tile() takes the other branch of CK_GFX950_SUPPORT and the
    # result is silently all zeros rather than a compile error. Random A/B/AQ/BQ
    # make any correct output wildly varying, so a zero standard deviation is
    # conclusive; check it before the tolerance gate, whose relative-error
    # denominator would otherwise report a confusing near-1.0 on a zero matrix.
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

    C_ref = _reference_gemm(A_dec, B_dec, AQ, BQ, problem)
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
                  f"time_ms={result_timed.time_ms:.3f}")


# ---------------------------------------------------------------------------
# Individual test cases
# ---------------------------------------------------------------------------
#
# kPadM/kPadN/kPadK are false for every ABQuant prefill config, so M, N and K
# must be exact multiples of TileM, TileN and TileK. Shapes below are the
# smallest multi-tile sizes that satisfy that for each pipeline.

def case_fp8_compv3(out_dir: Path, gfx_arch: str) -> "tuple[str, str]":
    # compv3 tiles are 128x128x128; M=N=256 spans a 2x2 tile grid.
    # bquant_group_n=1 gives one B scale per column, the finest granularity.
    M, N, K, gK, bN = 256, 256, 512, 128, 1
    cfg = default_fp8_compv3_config(quant_group_k=gK, bquant_group_n=bN,
                                    gfx_arch=gfx_arch)
    return _run_one("fp8/compv3", cfg, M, N, K, "fp8", out_dir,
                    gfx_arch=gfx_arch, seed=42)


def case_bf8_compv3(out_dir: Path, gfx_arch: str) -> "tuple[str, str]":
    M, N, K, gK, bN = 256, 256, 512, 128, 1
    cfg = default_bf8_compv3_config(quant_group_k=gK, bquant_group_n=bN,
                                    gfx_arch=gfx_arch)
    return _run_one("bf8/compv3", cfg, M, N, K, "bf8", out_dir,
                    gfx_arch=gfx_arch, seed=43)


def case_fp8_eightwaves(out_dir: Path, gfx_arch: str) -> "tuple[str, str]":
    if gfx_arch not in EIGHTWAVES_SUPPORTED_ARCHS:
        return SKIP, (f"fp8/eightwaves: not yet enabled on {gfx_arch} "
                      f"(see EIGHTWAVES_SUPPORTED_ARCHS)")
    # eightwaves tiles are 192x256x128, so M must be a multiple of 192 and N of
    # 256. bquant_group_n=128 is the only granularity the C++ tests validate for
    # this pipeline.
    M, N, K, gK, bN = 384, 512, 512, 128, 128
    cfg = default_fp8_eightwaves_config(quant_group_k=gK, bquant_group_n=bN,
                                        gfx_arch=gfx_arch)
    return _run_one("fp8/eightwaves", cfg, M, N, K, "fp8", out_dir,
                    gfx_arch=gfx_arch, seed=44)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = [
    ("fp8/compv3",     case_fp8_compv3),
    ("bf8/compv3",     case_bf8_compv3),
    ("fp8/eightwaves", case_fp8_eightwaves),
]


def main():
    parser = argparse.ArgumentParser(description="ABQuant GPU correctness tests")
    # No hardcoded default: it must stay possible to tell "user asked for
    # gfx950" from "we are on an unrelated box", so the skip below can fire.
    parser.add_argument("--gfx", default=None,
                        help="GPU arch override (default: auto-detect; gfx950 only)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    gfx = args.gfx or _detect_arch()
    if not gfx:
        print("SKIP: no supported GPU detected (rocminfo); ABQuant GPU tests skipped")
        return SKIP_EXIT
    if gfx not in SUPPORTED_ARCHS:
        print(f"SKIP: ABQuant is {'/'.join(SUPPORTED_ARCHS)}-only; detected {gfx}")
        return SKIP_EXIT
    try:
        import ml_dtypes  # noqa: F401
    except ImportError:
        # Without ml_dtypes there is no trustworthy fp8 codec here, and a
        # stand-in would compare the kernel against the wrong values.
        print("SKIP: ml_dtypes not installed; ABQuant fp8/bf8 encoding unavailable")
        return SKIP_EXIT

    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="abquant_gpu_test_"))
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
