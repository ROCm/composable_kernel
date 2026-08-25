#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GPU correctness tests for RowColQuant GEMM dispatcher.

Requires a gfx942 or gfx950 GPU and hipcc in PATH.  Skipped automatically when neither
is available (pytest.skip) so CI without a GPU still passes.

Tests:
  C4/fp8 -- GPU output is non-zero and within 5% max-relative-error vs. fp32 CPU ref
  C4/bf8 -- same for bf8 inputs
  M2     -- time_ms > 0 when timing is requested

Run:
  python3 -m pytest dispatcher/tests/test_rowcolquant_gpu_correctness.py -v
  python3 dispatcher/tests/test_rowcolquant_gpu_correctness.py --gfx gfx950
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from grouped_gemm_rowcolquant_utils import (
    RowColQuantGemmProblem,
    RowColQuantGpuGemmRunner,
    setup_multiple_rowcolquant_dispatchers,
    default_fp8_config,
    default_bf8_config,
)

log = logging.getLogger(__name__)

TOLERANCE = 0.05  # 5% max relative error — fp8/bf8 precision floor


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

def _has_hipcc() -> bool:
    return shutil.which("hipcc") is not None


def _detect_gfx_arch() -> str:
    """Return the first usable GPU arch, or empty string if none found."""
    try:
        r = subprocess.run(["rocm_agent_enumerator"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            arch = line.strip()
            if arch.startswith("gfx") and arch != "gfx000":
                return arch
    except Exception:
        pass
    return ""


_GFX_ARCH = _detect_gfx_arch()
# RowColQuant fp8/bf8 kernels use CK CompV3 pipelines that require native fp8 hardware.
# gfx90a (MI200 series) lacks native fp8 support and produces incorrect results.
# Only gfx942 (MI300X) and gfx950 (MI350X) are validated.
_SUPPORTED_ARCHES = ("gfx942", "gfx950")

requires_gpu = pytest.mark.skipif(
    not (_has_hipcc() and _GFX_ARCH in _SUPPORTED_ARCHES),
    reason=(
        f"GPU test: requires hipcc and native fp8 GPU ({', '.join(_SUPPORTED_ARCHES)}); "
        f"detected arch='{_GFX_ARCH}'"
    ),
)


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

try:
    import ml_dtypes as _ml_dtypes
except ImportError:  # pragma: no cover
    _ml_dtypes = None


def _require_ml_dtypes():
    if _ml_dtypes is None:
        pytest.skip(
            "ml_dtypes is required for valid fp8/bf8 encoding; "
            "install with: pip install ml-dtypes"
        )


def _fp8_uses_ocp(arch: str) -> bool:
    """Mirror the -DCK_USE_OCP_FP8 compile logic in grouped_gemm_rowcolquant_utils.py.

    gfx950 / gfx12 build the kernel with OCP fp8 (e4m3fn / e5m2); every other
    supported arch (notably gfx942) uses the native FNUZ format
    (e4m3fnuz / e5m2fnuz). The host-side encoding MUST match the format the kernel
    was compiled for, otherwise the reinterpreted bytes decode to NaN/Inf on device.
    """
    return "gfx950" in arch or "gfx12" in arch


def _fp8_ml_dtype(dtype: str):
    """Return the ml_dtypes fp8 type matching the compiled kernel's arch."""
    _require_ml_dtypes()
    if _fp8_uses_ocp(_GFX_ARCH):
        return _ml_dtypes.float8_e4m3fn if dtype == "fp8" else _ml_dtypes.float8_e5m2
    return _ml_dtypes.float8_e4m3fnuz if dtype == "fp8" else _ml_dtypes.float8_e5m2fnuz


def _encode_fp8(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Encode float32 → fp8/bf8 bytes (uint8 view). Requires ml_dtypes.

    The fp8 format (OCP vs FNUZ) follows the compiled kernel's arch; see _fp8_uses_ocp.
    """
    ml_t = _fp8_ml_dtype(dtype)
    return arr.astype(ml_t).view(np.uint8)


def _decode_fp8(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Decode fp8/bf8 bytes (uint8 view) → float32. Requires ml_dtypes."""
    ml_t = _fp8_ml_dtype(dtype)
    return arr.view(ml_t).astype(np.float32)


# ---------------------------------------------------------------------------
# CPU reference: C[M,N] = diag(AQ) @ A_dq @ B_dq @ diag(BQ)
# RowColQuant: AQ[M] per-row scale, BQ[N] per-col scale.
# ---------------------------------------------------------------------------

def _reference_rowcolquant(
    A_f32: np.ndarray,  # (M, K)
    B_f32: np.ndarray,  # (K, N)
    AQ: np.ndarray,     # (M,) float32
    BQ: np.ndarray,     # (N,) float32
) -> np.ndarray:
    """C[i,j] = AQ[i] * BQ[j] * sum_k(A[i,k] * B[k,j]) — computed in fp32."""
    C = A_f32.astype(np.float32) @ B_f32.astype(np.float32)  # (M, N)
    C *= AQ[:, np.newaxis]   # row scale
    C *= BQ[np.newaxis, :]   # col scale
    return C


def _max_rel_err(C_gpu: np.ndarray, C_ref: np.ndarray) -> float:
    C_gpu_f = C_gpu.astype(np.float32)
    C_ref_f = C_ref.astype(np.float32)
    num = np.abs(C_gpu_f - C_ref_f)
    ref_max = float(np.abs(C_ref_f).max())
    den = np.abs(C_ref_f) + max(ref_max * 1e-2, 1e-6)
    return float(np.max(num / den))


# ---------------------------------------------------------------------------
# Build + run helper
# ---------------------------------------------------------------------------

def _run_one(label: str, config, M: int, N: int, K: int,
             A_raw: np.ndarray, A_f32: np.ndarray,
             B_raw: np.ndarray, B_f32: np.ndarray,
             AQ: np.ndarray, BQ: np.ndarray,
             out_dir: Path,
             gfx_arch: str = "gfx950") -> tuple:
    problem = RowColQuantGemmProblem(M=M, N=N, K=K)

    so_paths = setup_multiple_rowcolquant_dispatchers(
        configs=[config],
        output_dir=out_dir,
        gfx_arch=gfx_arch,
    )
    if not so_paths or so_paths[0] is None:
        return "FAIL", f"{label}: kernel build failed"

    runner = RowColQuantGpuGemmRunner(so_paths[0])
    result = runner.run(A=A_raw, B=B_raw, AQ=AQ, BQ=BQ, problem=problem)
    C_gpu = result.C.astype(np.float32)

    if np.all(C_gpu == 0):
        return "FAIL", f"{label}: GPU output is all-zero"
    if not np.all(np.isfinite(C_gpu)):
        frac = (~np.isfinite(C_gpu)).mean()
        return "FAIL", f"{label}: GPU output contains NaN/Inf (frac={frac:.3f})"

    C_ref = _reference_rowcolquant(A_f32, B_f32, AQ, BQ)
    mre = _max_rel_err(C_gpu, C_ref)
    if mre > TOLERANCE:
        return "FAIL", (f"{label}: max_rel_err={mre:.4f} > tol={TOLERANCE:.4f} "
                        f"(M={M} N={N} K={K})")

    result_timed = runner.run(A=A_raw, B=B_raw, AQ=AQ, BQ=BQ, problem=problem)
    if result_timed.time_ms <= 0.0:
        return "FAIL", f"{label}: time_ms={result_timed.time_ms:.4f} is not positive"

    return "PASS", f"{label}: max_rel_err={mre:.4f}, time_ms={result_timed.time_ms:.3f}"


# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------

def _make_inputs(M, N, K, dtype="fp8", seed=42):
    rng = np.random.default_rng(seed)
    A_f32 = rng.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    B_f32 = rng.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
    AQ    = rng.uniform(0.5, 2.0, (M,)).astype(np.float32)  # per-row A scale
    BQ    = rng.uniform(0.5, 2.0, (N,)).astype(np.float32)  # per-col B scale
    A_raw = _encode_fp8(A_f32, dtype)
    B_raw = _encode_fp8(B_f32, dtype)
    A_dec = _decode_fp8(A_raw, dtype)
    B_dec = _decode_fp8(B_raw, dtype)
    return A_raw, A_dec, B_raw, B_dec, AQ, BQ


# ---------------------------------------------------------------------------
# pytest test cases
# ---------------------------------------------------------------------------

@requires_gpu
def test_rowcolquant_fp8_correctness(tmp_path):
    """GPU RowColQuant fp8: output is non-zero and within 5% of CPU reference."""
    M, N, K = 128, 128, 192  # K=3*TileK(64) for Odd tail coverage
    cfg = default_fp8_config(gfx_arch=_GFX_ARCH)
    A_raw, A_dec, B_raw, B_dec, AQ, BQ = _make_inputs(M, N, K, dtype="fp8")
    status, detail = _run_one(
        "C4/fp8", cfg, M, N, K, A_raw, A_dec, B_raw, B_dec, AQ, BQ,
        tmp_path, gfx_arch=_GFX_ARCH,
    )
    assert status == "PASS", detail


@requires_gpu
def test_rowcolquant_bf8_correctness(tmp_path):
    """GPU RowColQuant bf8: output is non-zero and within 5% of CPU reference."""
    M, N, K = 128, 128, 192
    cfg = default_bf8_config(gfx_arch=_GFX_ARCH)
    A_raw, A_dec, B_raw, B_dec, AQ, BQ = _make_inputs(M, N, K, dtype="bf8")
    status, detail = _run_one(
        "C4/bf8", cfg, M, N, K, A_raw, A_dec, B_raw, B_dec, AQ, BQ,
        tmp_path, gfx_arch=_GFX_ARCH,
    )
    assert status == "PASS", detail


@requires_gpu
def test_rowcolquant_fp8_rectangular(tmp_path):
    """GPU RowColQuant fp8: non-square M/N/K to stress stride math."""
    M, N, K = 64, 256, 128
    cfg = default_fp8_config(gfx_arch=_GFX_ARCH)
    A_raw, A_dec, B_raw, B_dec, AQ, BQ = _make_inputs(M, N, K, dtype="fp8")
    status, detail = _run_one(
        "rect/fp8", cfg, M, N, K, A_raw, A_dec, B_raw, B_dec, AQ, BQ,
        tmp_path, gfx_arch=_GFX_ARCH,
    )
    assert status == "PASS", detail


@requires_gpu
def test_rowcolquant_timing_positive(tmp_path):
    """GPU RowColQuant: time_ms > 0 when timing is collected."""
    M, N, K = 128, 128, 64
    cfg = default_fp8_config(gfx_arch=_GFX_ARCH)
    A_raw, _, B_raw, _, AQ, BQ = _make_inputs(M, N, K, dtype="fp8")

    so_paths = setup_multiple_rowcolquant_dispatchers(
        configs=[cfg], output_dir=tmp_path, gfx_arch=_GFX_ARCH,
    )
    assert so_paths and so_paths[0] is not None, "kernel build failed"

    runner = RowColQuantGpuGemmRunner(so_paths[0])
    problem = RowColQuantGemmProblem(M=M, N=N, K=K)
    result = runner.run(A=A_raw, B=B_raw, AQ=AQ, BQ=BQ, problem=problem)
    assert result.time_ms > 0.0, f"time_ms={result.time_ms} is not positive"


# ---------------------------------------------------------------------------
# Standalone runner (mirrors test_bquant_gpu_correctness.py style)
# ---------------------------------------------------------------------------

TESTS = [
    ("C4/fp8", lambda od, gfx: _run_one(
        "C4/fp8", default_fp8_config(gfx_arch=gfx), 128, 128, 192,
        *_make_inputs(128, 128, 192, "fp8"), Path(od), gfx_arch=gfx)),
    ("C4/bf8", lambda od, gfx: _run_one(
        "C4/bf8", default_bf8_config(gfx_arch=gfx), 128, 128, 192,
        *_make_inputs(128, 128, 192, "bf8"), Path(od), gfx_arch=gfx)),
]


def main():
    parser = argparse.ArgumentParser(
        description="RowColQuant GPU correctness tests")
    parser.add_argument("--gfx", default="gfx950")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="rowcolquant_gpu_test_"))
    log.info("Kernel output dir: %s", out_dir)

    results = []
    for name, fn in TESTS:
        log.info("--- Running %s ---", name)
        try:
            status, detail = fn(out_dir, args.gfx)
        except Exception as exc:
            status, detail = "FAIL", f"{name}: exception: {exc}"
        results.append((name, status, detail))
        log.info("[%s] %s", status, detail)

    print("\n=== Summary ===")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    for _, status, detail in results:
        print(f"  [{status:4s}] {detail}")
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
