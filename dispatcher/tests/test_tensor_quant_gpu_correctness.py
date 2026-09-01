#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
On-device GPU correctness test for the TensorQuant (per-tensor scale) GEMM
dispatcher bridge (PR #9978).

Mirrors test_bquant_gpu_correctness.py: build the op's block-scale dispatcher
.so, run it on the GPU via TensorQuantGpuGemmRunner, and compare against a
NumPy fp32 reference. TensorQuant applies a single scalar scale to A and to B:

    C[m, n] = (AQ * BQ) * sum_k A[m, k] * B[k, n]

The reference rounds A/B through the SAME fp8/bf8 quantization the kernel sees,
so the tolerance only absorbs GEMM accumulation error.

Block-scale traps handled explicitly:
  1. fp8 flavour: FNUZ on gfx942, OCP on gfx950/gfx12 (wrong flavour NaNs /
     decorrelates the reference). Selected from the detected arch.
  2. warp_tile_k=32 on gfx942 (128 silently all-zeros): default_fp8_config /
     default_bf8_config derive it from the arch via fp8_warp_tile_k_for_arch;
     the test asserts it matches.
The tensor_quant runner+kernel consume a C-contiguous [K, N] B buffer directly
(verified bit-accurate on gfx942), so no host-side transpose is applied here.

Run:
    python3 -m pytest test_tensor_quant_gpu_correctness.py -v
    python3 test_tensor_quant_gpu_correctness.py          # standalone
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))  # for conftest helpers under standalone run

# Shared GPU probes, arch-aware fp8/bf8 codecs and build->run->verify harness.
from conftest import (  # noqa: E402
    encode_fp8 as _encode_arch,
    qdq_fp8 as _qdq_arch,
    gpu_available as _have_gpu,
    ml_dtypes_available as _have_ml_dtypes,
    run_and_verify,
)


def _find_python_dir() -> Path:
    for c in (_HERE.parent.parent / "python", _HERE.parent / "python"):
        if (c / "gemm_tensor_quant_utils.py").is_file():
            return c
    for parent in _HERE.parents:
        cand = parent / "dispatcher" / "python"
        if (cand / "gemm_tensor_quant_utils.py").is_file():
            return cand
    raise RuntimeError("could not locate dispatcher/python/gemm_tensor_quant_utils.py")


sys.path.insert(0, str(_find_python_dir()))

from gemm_tensor_quant_utils import (  # noqa: E402
    TensorQuantGemmProblem,
    TensorQuantGpuGemmRunner,
    setup_multiple_tensor_quant_dispatchers,
    default_fp8_config,
    default_bf8_config,
    fp8_warp_tile_k_for_arch,
    _detect_gpu_arch,
)


# Thin local aliases so the case body reads the same as before (the codecs are
# the single canonical copies from conftest.py).
def _encode(arr, dtype: str, arch: str) -> np.ndarray:
    return _encode_arch(arr, dtype, arch)


def _qdq(arr, dtype: str, arch: str) -> np.ndarray:
    return _qdq_arch(arr, dtype, arch)


# =============================================================================
# NumPy reference: C = (AQ * BQ) * (A @ B)
# =============================================================================


def reference_tensor_quant_gemm(A_dec, B_dec, AQ: float, BQ: float) -> np.ndarray:
    acc = A_dec.astype(np.float32) @ B_dec.astype(np.float32)
    return (float(AQ) * float(BQ) * acc).astype(np.float32)


# =============================================================================
# build + run + verify
# =============================================================================


def _run_case(dtype: str, M: int, N: int, K: int, out_dir: Path):
    arch = _detect_gpu_arch()
    make_cfg = default_fp8_config if dtype == "fp8" else default_bf8_config
    config = make_cfg(arch)

    expected_wtk = fp8_warp_tile_k_for_arch(arch)
    assert config.warp_tile_k == expected_wtk, (
        f"warp_tile_k arch trap: got {config.warp_tile_k}, expected {expected_wtk} for {arch}"
    )

    rng = np.random.default_rng(1234)
    A_f = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
    B_f = rng.uniform(-2.0, 2.0, (K, N)).astype(np.float32)   # logical [K, N]
    AQ = 0.5
    BQ = 0.25

    A_raw = _encode(A_f, dtype, arch)                 # [M, K] row-major
    # tensor_quant's runner+kernel consume a C-contiguous [K, N] B buffer directly
    # (verified bit-accurate on gfx942); no host-side transpose is required.
    B_raw = _encode(B_f, dtype, arch)                 # [K, N]
    A_dec = _qdq(A_f, dtype, arch)
    B_dec = _qdq(B_f, dtype, arch)

    problem = TensorQuantGemmProblem(M=M, N=N, K=K)

    def _build():
        so_paths = setup_multiple_tensor_quant_dispatchers(
            [config], output_dir=out_dir, gfx_arch=arch
        )
        return so_paths[0] if so_paths else None

    def _run(so_path):
        runner = TensorQuantGpuGemmRunner(so_path)
        result = runner.run(A_raw, B_raw, AQ, BQ, problem)
        return result.C, result.time_ms

    res = run_and_verify(
        build_so=_build,
        run_kernel=_run,
        reference=lambda: reference_tensor_quant_gemm(A_dec, B_dec, AQ, BQ),
        tol=0.05,  # fp8/bf8 block-scale ~1e-2 .. 5e-2
        label=f"tensor_quant {dtype} {M}x{N}x{K}",
    )
    return res.max_rel, res.time_ms


_SKIP_NO_GPU = pytest.mark.skipif(not _have_gpu(), reason="no ROCm GPU detected")
_SKIP_NO_MLD = pytest.mark.skipif(not _have_ml_dtypes(), reason="ml_dtypes not installed")


@_SKIP_NO_GPU
@_SKIP_NO_MLD
@pytest.mark.parametrize("dtype", ["fp8", "bf8"])
def test_tensor_quant_gpu_matches_reference(dtype, tmp_path):
    max_rel, _ = _run_case(dtype, M=256, N=256, K=512, out_dir=tmp_path)
    assert max_rel <= 0.05


@_SKIP_NO_GPU
@_SKIP_NO_MLD
def test_tensor_quant_gpu_not_all_zeros(tmp_path):
    _run_case("fp8", M=256, N=256, K=512, out_dir=tmp_path)


if __name__ == "__main__":
    if not _have_gpu():
        print("SKIP: no GPU"); raise SystemExit(0)
    if not _have_ml_dtypes():
        print("SKIP: ml_dtypes not installed"); raise SystemExit(0)
    d = Path(tempfile.mkdtemp(prefix="tensor_quant_gputest_"))
    ok = True
    for dt in ("fp8", "bf8"):
        try:
            mr, t = _run_case(dt, 256, 256, 512, d)
            print(f"PASS tensor_quant {dt}: max_rel={mr:.4f} time_ms={t:.3f}")
        except Exception as e:
            ok = False
            print(f"FAIL tensor_quant {dt}: {e}")
    raise SystemExit(0 if ok else 1)
