#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
On-device GPU correctness test for the AQuant (A-scaled block-scale) GEMM
dispatcher bridge (PR #9980).

Mirrors the structure of test_bquant_gpu_correctness.py: build the op's
block-scale dispatcher .so, run it on the GPU via AQuantGpuGemmRunner, and
compare against a NumPy fp32 reference

    C = dequant(A, AQ) @ B

where AQ applies a per-(row, K-group) scale to A:

    A_deq[m, k] = A[m, k] * AQ[m, k // quant_group_k]

Two block-scale correctness traps are handled explicitly (both were needed to
get a bit-accurate match on gfx942):

  1. fp8 encoding flavour.  gfx942 uses FNUZ fp8/bf8 (e4m3fnuz / e5m2fnuz);
     gfx950/gfx12 use OCP (e4m3 / e5m2).  Encoding A/B with the wrong flavour
     makes the reference NaN or uncorrelated.  Selected from the detected arch.
  2. warp_tile_k.  fp8/bf8 needs warp_tile_k=32 on gfx942 (128 silently outputs
     all-zeros).  default_fp8_config/default_bf8_config derive this from the
     arch; the test asserts it matches to guard against regressions.

Also note the rcr layout uses ColMajor B, so the [K, N] logical B is stored
transposed (a C-contiguous [N, K] buffer == [K, N] col-major) before it is
handed to the kernel.

Run:
    python3 -m pytest test_aquant_gpu_correctness.py -v
    python3 test_aquant_gpu_correctness.py            # standalone
"""

import math
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
        if (c / "gemm_aquant_utils.py").is_file():
            return c
    for parent in _HERE.parents:
        cand = parent / "dispatcher" / "python"
        if (cand / "gemm_aquant_utils.py").is_file():
            return cand
    raise RuntimeError("could not locate dispatcher/python/gemm_aquant_utils.py")


sys.path.insert(0, str(_find_python_dir()))

from gemm_aquant_utils import (  # noqa: E402
    AQuantGemmProblem,
    AQuantGpuGemmRunner,
    setup_multiple_aquant_dispatchers,
    default_fp8_config,
    default_bf8_config,
    _detect_gpu_arch,
    _warp_tile_k_for,
)


# Thin local aliases so the case body reads the same as before (the codecs are
# the single canonical copies from conftest.py).
def _encode(arr, dtype: str, arch: str) -> np.ndarray:
    return _encode_arch(arr, dtype, arch)


def _qdq(arr, dtype: str, arch: str) -> np.ndarray:
    """Round through fp8/bf8 and back to fp32 (reference-side counterpart)."""
    return _qdq_arch(arr, dtype, arch)


# =============================================================================
# NumPy reference: C = dequant(A, AQ) @ B
# =============================================================================


def reference_aquant_gemm(A_dec, B_dec, AQ, problem: AQuantGemmProblem) -> np.ndarray:
    """A_dec [M,K], B_dec [K,N] are already fp8-rounded fp32 (kernel-identical)."""
    M, N, K = problem.M, problem.N, problem.K
    gK = problem.quant_group_k
    A_deq = A_dec.astype(np.float32).copy()
    for qi in range(problem.QK_A):
        k0 = qi * gK
        k1 = min(k0 + gK, K)
        A_deq[:, k0:k1] *= AQ.astype(np.float32)[:, qi][:, None]
    return (A_deq @ B_dec.astype(np.float32)).astype(np.float32)


# =============================================================================
# build + run + verify
# =============================================================================


def _run_case(dtype: str, M: int, N: int, K: int, gK: int, out_dir: Path):
    arch = _detect_gpu_arch()
    make_cfg = default_fp8_config if dtype == "fp8" else default_bf8_config
    config = make_cfg(quant_group_k=gK, quant_group_n=1, layout="rcr", gfx_arch=arch)

    expected_wtk = _warp_tile_k_for(arch, preshuffle_aquant=False)
    assert config.warp_tile_k == expected_wtk, (
        f"warp_tile_k arch trap: got {config.warp_tile_k}, expected {expected_wtk} for {arch}"
    )

    rng = np.random.default_rng(1234)
    QK_A = math.ceil(K / gK)
    A_f = rng.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    B_f = rng.uniform(-1.0, 1.0, (K, N)).astype(np.float32)   # logical [K, N]
    AQ_f = rng.uniform(0.5, 1.5, (M, QK_A)).astype(np.float32)

    A_raw = _encode(A_f, dtype, arch)                 # logical [M, K] fp8 bytes
    B_raw = _encode(B_f, dtype, arch)                 # logical [K, N]; runner materializes rcr col-major
    A_dec = _qdq(A_f, dtype, arch)
    B_dec = _qdq(B_f, dtype, arch)

    problem = AQuantGemmProblem(M=M, N=N, K=K, quant_group_k=gK)

    def _build():
        so_paths = setup_multiple_aquant_dispatchers(
            configs=[config], output_dir=out_dir, gfx_arch=arch
        )
        return so_paths[0] if so_paths else None

    def _run(so_path):
        runner = AQuantGpuGemmRunner(so_path, layout="rcr")
        result = runner.run(A=A_raw, AQ=AQ_f, B=B_raw, problem=problem)
        return result.C, result.time_ms

    res = run_and_verify(
        build_so=_build,
        run_kernel=_run,
        reference=lambda: reference_aquant_gemm(A_dec, B_dec, AQ_f, problem),
        tol=0.05,  # fp8/bf8 block-scale ~1e-2 .. 5e-2
        label=f"aquant {dtype} {M}x{N}x{K} gK={gK}",
    )
    return res.max_rel, res.time_ms


_SKIP_NO_GPU = pytest.mark.skipif(not _have_gpu(), reason="no ROCm GPU detected")
_SKIP_NO_MLD = pytest.mark.skipif(not _have_ml_dtypes(), reason="ml_dtypes not installed")


@_SKIP_NO_GPU
@_SKIP_NO_MLD
@pytest.mark.parametrize("dtype", ["fp8", "bf8"])
def test_aquant_gpu_matches_reference(dtype, tmp_path):
    max_rel, _ = _run_case(dtype, M=256, N=256, K=512, gK=128, out_dir=tmp_path)
    assert max_rel <= 0.05


@_SKIP_NO_GPU
@_SKIP_NO_MLD
def test_aquant_gpu_not_all_zeros(tmp_path):
    _run_case("fp8", M=256, N=256, K=512, gK=128, out_dir=tmp_path)


if __name__ == "__main__":
    if not _have_gpu():
        print("SKIP: no GPU"); raise SystemExit(0)
    if not _have_ml_dtypes():
        print("SKIP: ml_dtypes not installed"); raise SystemExit(0)
    d = Path(tempfile.mkdtemp(prefix="aquant_gputest_"))
    ok = True
    for dt in ("fp8", "bf8"):
        try:
            mr, t = _run_case(dt, 256, 256, 512, 128, d)
            print(f"PASS aquant {dt}: max_rel={mr:.4f} time_ms={t:.3f}")
        except Exception as e:
            ok = False
            print(f"FAIL aquant {dt}: {e}")
    raise SystemExit(0 if ok else 1)
