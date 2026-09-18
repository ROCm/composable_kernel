#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
On-device GPU correctness test for the NON-GROUPED gemm_rowcolquant dispatcher
bridge (PR #9979), i.e. dispatcher/python/gemm_rowcolquant_utils.py.

NOTE: the GROUPED grouped_gemm_rowcolquant bridge has its own on-device test in
test_rowcolquant_gpu_correctness.py (which imports grouped_gemm_rowcolquant_utils).
The two utils modules export identically-named symbols (RowColQuantGemmProblem,
RowColQuantGpuGemmRunner, setup_multiple_rowcolquant_dispatchers,
default_{fp8,bf8}_config), so these tests must live in separate files -- folding
them together would make one family's cases silently exercise the other's bridge.

Mirrors test_bquant_gpu_correctness.py: build the op's block-scale dispatcher
.so, run it on the GPU via RowColQuantGpuGemmRunner, and compare against a
NumPy fp32 reference. RowColQuant applies a per-row scale to A and a per-col
scale to B (equivalently, a per-column scale of the output):

    C[m, n] = AQ[m] * BQ[n] * sum_k A[m, k] * B[k, n]

The reference rounds A/B through the SAME fp8/bf8 quantization the kernel sees,
so the tolerance only absorbs GEMM accumulation error.

Block-scale traps handled explicitly:
  1. fp8 flavour: FNUZ on gfx942, OCP on gfx950/gfx12 (wrong flavour NaNs the
     reference). Uses the utils' own encode_fp8_bytes / quantize_dequantize_fp8,
     which select the flavour from the arch.
  2. warp_tile_k=32 on gfx942 (128 silently all-zeros): default_fp8_config /
     default_bf8_config derive it from the arch; the test asserts it matches.

Run:
    python3 -m pytest test_gemm_rowcolquant_gpu_correctness.py -v
    python3 test_gemm_rowcolquant_gpu_correctness.py          # standalone
    python3 test_gemm_rowcolquant_gpu_correctness.py --gfx gfx942

The standalone path is what ctest and the Jenkins lane drive; it exits 77 (see
SKIP_EXIT) when the box cannot run the test at all, so an unrunnable runner is
recorded as Skipped rather than as a vacuous Passed or a spurious Failed.
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))  # for conftest helpers under standalone run

# Shared GPU build->run->verify harness. rowcolquant's fp8/bf8 codecs live in its
# own utils (encode_fp8_bytes / quantize_dequantize_fp8, used below), so only the
# GPU/ml_dtypes probes and the harness are pulled from conftest here.
from conftest import (  # noqa: E402
    gpu_available as _have_gpu,
    run_and_verify,
)


def _find_python_dir() -> Path:
    for c in (_HERE.parent.parent / "python", _HERE.parent / "python"):
        if (c / "gemm_rowcolquant_utils.py").is_file():
            return c
    for parent in _HERE.parents:
        cand = parent / "dispatcher" / "python"
        if (cand / "gemm_rowcolquant_utils.py").is_file():
            return cand
    raise RuntimeError("could not locate dispatcher/python/gemm_rowcolquant_utils.py")


sys.path.insert(0, str(_find_python_dir()))

import gemm_rowcolquant_utils as u  # noqa: E402
from gemm_rowcolquant_utils import (  # noqa: E402
    RowColQuantGemmProblem,
    RowColQuantGpuGemmRunner,
    setup_multiple_rowcolquant_dispatchers,
    default_fp8_config,
    default_bf8_config,
    _detect_gpu_arch,
    _warp_tile_k_for,
)


log = logging.getLogger(__name__)

PASS = "PASS"
FAIL = "FAIL"
SKIP_EXIT = 77    # ctest SKIP_RETURN_CODE; see main()

# RowColQuant is a block-scale fp8/bf8 op: it needs native fp8 MFMA. A GPU
# outside this set clears conftest's gpu_available() probe and then fails at
# hipcc, which reads as a correctness FAIL rather than "this box cannot run it".
# Matches SUPPORTED_ARCHS in the grouped tensorquant/rowcolquant GPU tests.
_SUPPORTED_ARCHS = ("gfx942", "gfx950")


def _have_ml_dtypes() -> bool:
    # rowcolquant's fp8 encoders come from ml_dtypes via its own utils.
    return u.fp8_encoding_available()


def _has_hipcc() -> bool:
    try:
        subprocess.run(["hipcc", "--version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def _safe_detect_arch() -> Optional[str]:
    """_detect_gpu_arch() raises on absence/unsupported; the gates want None."""
    try:
        return _detect_gpu_arch()
    except Exception:
        return None


_GFX_ARCH = _safe_detect_arch()


# =============================================================================
# NumPy reference: C[m,n] = AQ[m] * BQ[n] * sum_k A[m,k] B[k,n]
# =============================================================================


def reference_rowcolquant_gemm(A_dec, B_dec, AQ, BQ) -> np.ndarray:
    acc = A_dec.astype(np.float32) @ B_dec.astype(np.float32)   # [M, N]
    acc = acc * AQ.astype(np.float32)[:, None]                  # per-row scale
    acc = acc * BQ.astype(np.float32)[None, :]                  # per-col scale
    return acc.astype(np.float32)


# =============================================================================
# build + run + verify
# =============================================================================


def _run_case(dtype: str, M: int, N: int, K: int, out_dir: Path,
              arch: Optional[str] = None):
    # main() resolves the arch once (honouring --gfx) and passes it down; the
    # pytest entry points leave it None and take the detected one.
    arch = arch or _detect_gpu_arch()
    make_cfg = default_fp8_config if dtype == "fp8" else default_bf8_config
    config = make_cfg(gfx_arch=arch)

    expected_wtk = _warp_tile_k_for(dtype, arch)
    assert config.warp_tile_k == expected_wtk, (
        f"warp_tile_k arch trap: got {config.warp_tile_k}, expected {expected_wtk} for {arch}"
    )

    rng = np.random.default_rng(1234)
    A_f = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
    B_f = rng.uniform(-2.0, 2.0, (K, N)).astype(np.float32)
    AQ = rng.uniform(0.5, 1.5, (M,)).astype(np.float32)
    BQ = rng.uniform(0.5, 1.5, (N,)).astype(np.float32)

    # Encode real fp8/bf8 bytes (arch-correct flavour), round the reference the same way.
    A_raw = u.encode_fp8_bytes(A_f, dtype, arch)
    B_raw = u.encode_fp8_bytes(B_f, dtype, arch)
    A_dec = u.quantize_dequantize_fp8(A_f, dtype, arch)
    B_dec = u.quantize_dequantize_fp8(B_f, dtype, arch)

    problem = RowColQuantGemmProblem(M=M, N=N, K=K)

    def _build():
        so_paths = setup_multiple_rowcolquant_dispatchers(
            configs=[config], output_dir=out_dir, gfx_arch=arch
        )
        return so_paths[0] if so_paths else None

    def _run(so_path):
        runner = RowColQuantGpuGemmRunner(so_path)
        result = runner.run(A_raw, B_raw, AQ, BQ, problem)
        return result.C, result.time_ms

    res = run_and_verify(
        build_so=_build,
        run_kernel=_run,
        reference=lambda: reference_rowcolquant_gemm(A_dec, B_dec, AQ, BQ),
        tol=0.05,  # fp8/bf8 block-scale ~1e-2 .. 5e-2
        label=f"rowcolquant {dtype} {M}x{N}x{K}",
    )
    return res.max_rel, res.time_ms


_SKIP_NO_GPU = pytest.mark.skipif(not _have_gpu(), reason="no ROCm GPU detected")
_SKIP_NO_MLD = pytest.mark.skipif(not _have_ml_dtypes(), reason="ml_dtypes not installed")
# gpu_available() only says "a ROCm GPU is present". Without this second gate a
# gfx90a box runs the fp8 build and reports a red FAIL for hardware that was
# never in scope.
_SKIP_BAD_ARCH = pytest.mark.skipif(
    _GFX_ARCH not in _SUPPORTED_ARCHS,
    reason=f"RowColQuant needs native fp8 ({'/'.join(_SUPPORTED_ARCHS)}); "
           f"detected {_GFX_ARCH}",
)


@_SKIP_NO_GPU
@_SKIP_NO_MLD
@_SKIP_BAD_ARCH
@pytest.mark.parametrize("dtype", ["fp8", "bf8"])
def test_rowcolquant_gpu_matches_reference(dtype, tmp_path):
    max_rel, _ = _run_case(dtype, M=256, N=256, K=512, out_dir=tmp_path)
    assert max_rel <= 0.05


@_SKIP_NO_GPU
@_SKIP_NO_MLD
@_SKIP_BAD_ARCH
def test_rowcolquant_gpu_not_all_zeros(tmp_path):
    _run_case("fp8", M=256, N=256, K=512, out_dir=tmp_path)


# =============================================================================
# Standalone entry point (the one ctest and Jenkins drive)
# =============================================================================


def _case(dtype: str):
    """Build a never-raising case fn returning (status, detail).

    The TESTS-loop convention is that a case reports its own verdict; main()'s
    ``except Exception`` is a backstop for genuine bugs, not the normal path.
    """
    def run(out_dir: Path, arch: str):
        max_rel, time_ms = _run_case(dtype, 256, 256, 512, out_dir, arch)
        if max_rel > 0.05:
            return FAIL, f"rowcolquant {dtype}: max_rel={max_rel:.4f} > tol=0.05"
        return PASS, (f"rowcolquant {dtype}: max_rel={max_rel:.4f} "
                      f"time_ms={time_ms:.3f}")
    return run


TESTS = [(f"rowcolquant_{dt}", _case(dt)) for dt in ("fp8", "bf8")]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RowColQuant (non-grouped) GPU correctness tests")
    # No hardcoded default: an explicit --gfx is an override the caller is
    # trusted on, but falling back to a fixed arch would make a CPU-only or
    # gfx90a box compile for gfx950 and crash instead of skipping.
    parser.add_argument("--gfx", default=None,
                        help=f"GPU arch override (default: auto-detect; "
                             f"{'/'.join(_SUPPORTED_ARCHS)} only)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # The pytest path gates via the skipif marks above; this standalone path is
    # what CI drives, so it needs the same gates spelled out. SKIP_EXIT keeps a
    # clean skip distinguishable from a failure -- ctest maps it via
    # SKIP_RETURN_CODE and the Jenkins lane via its run_ok helper.
    gfx = args.gfx or _GFX_ARCH
    if not gfx:
        print("SKIP: no supported GPU detected (rocm_agent_enumerator)")
        return SKIP_EXIT
    if gfx not in _SUPPORTED_ARCHS:
        print(f"SKIP: RowColQuant needs native fp8 "
              f"({'/'.join(_SUPPORTED_ARCHS)}); detected {gfx}")
        return SKIP_EXIT
    if not _has_hipcc():
        print("SKIP: hipcc not found in PATH; cannot JIT-compile kernels")
        return SKIP_EXIT
    if not _have_ml_dtypes():
        print("SKIP: ml_dtypes not installed; fp8/bf8 encoding unavailable")
        return SKIP_EXIT

    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="rowcolquant_gputest_"))
    log.info("Kernel output dir: %s", out_dir)

    results = []
    for name, fn in TESTS:
        log.info("--- Running %s ---", name)
        try:
            status, detail = fn(out_dir, gfx)
        except Exception as exc:  # noqa: BLE001
            status, detail = FAIL, f"{name}: exception: {exc}"
        results.append((status, detail))
        log.info("[%s] %s", status, detail)

    print("\n=== Summary ===")
    passed = sum(1 for s, _ in results if s == PASS)
    for status, detail in results:
        print(f"  [{status:4s}] {detail}")
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
