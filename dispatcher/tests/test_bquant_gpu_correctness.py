#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GPU correctness tests for BQuant GEMM dispatcher — C4 and H3 coverage.

Requires a gfx950 (MI350X / MI355X) GPU and hipcc in PATH.  Skips cleanly
(exit 77) when no GPU is visible, the detected arch is not gfx950, or
ml_dtypes is missing.

C4 (fp8/bf8 compv3) uses standard fp8 MFMA, which the gfx942 ISA also has, and
the host fp8 codec now follows the target arch via dispatcher_common.fp8_uses_ocp
rather than hardcoding OCP.  gfx942 is nevertheless not in SUPPORTED_ARCHS: CI
gates the whole bquant lane to gfx950 (ck.groovy), so a gfx942 run here would
exercise a path no lane covers.  Enabling it is a follow-up, not an oversight.
H3 (mx_bf16*) cannot be enabled on gfx942 at all: the scale-MFMA builtins it
needs (__builtin_amdgcn_mfma_scale_f32_*_f8f6f4) are absent from that ISA and
fail at compile time rather than silently.

Tests:
  C4 — fp8, bf8: GPU output is non-zero, non-constant, and within 5%
       max-relative-error vs. a fp32 CPU reference.
  C  — fp8i4, bf8i4: pk_int4 B with an fp8/bf8-encoded BQ scale.
  H3 — mx_bf16bf16, mx_bf16bf8, mx_bf16fp4: same non-zero / rel-error checks,
       plus verify QuantType::BQuantGrouped + e8m0 pipeline compiles and runs.
       gfx950 only (see MX_SUPPORTED_ARCHS).
  M2 — timing: time_ms is non-zero when timing is requested.

Every case sweeps N over _N_SWEEP to cover the per-N-tile de-permute.

Run:
  python3 test_bquant_gpu_correctness.py
  python3 test_bquant_gpu_correctness.py -v          # verbose hipcc output
  python3 test_bquant_gpu_correctness.py --gfx gfx942
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

from grouped_gemm_bquant_utils import (
    BQuantGemmProblem,
    BQuantGpuGemmRunner,
    setup_multiple_bquant_dispatchers,
    default_fp8_config,
    default_bf8_config,
    default_fp8i4_config,
    default_bf8i4_config,
    default_mx_bf16bf16_config,
    default_mx_bf16bf8_config,
    default_mx_bf16fp4_config,
)

log = logging.getLogger(__name__)

TOLERANCE = 0.05  # 5% max relative error — fp8/bf8 precision floor


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


def _encode_bf8(arr: np.ndarray, gfx_arch: str) -> np.ndarray:
    return _encode_fp8(arr, "bf8", gfx_arch)


def _decode_bf8(arr: np.ndarray, gfx_arch: str) -> np.ndarray:
    return _decode_fp8(arr, "bf8", gfx_arch)


def _encode_e8m0(arr: np.ndarray) -> np.ndarray:
    """Encode float32 scale values → e8m0 uint8 (MX block scale format).

    e8m0 stores a power-of-two exponent: byte b represents 2^(b - 127).
    Scales must be positive; zero maps to 0 (subnormal/zero in e8m0).
    """
    arr = np.asarray(arr, dtype=np.float32)
    # Clamp to representable range: 2^-127 … 2^127
    arr = np.clip(arr, 0.0, np.float32(2.0 ** 127))
    nonzero = arr > 0.0
    out = np.zeros(arr.shape, dtype=np.uint8)
    # biased exponent = floor(log2(s)) + 127, clamped to [0, 254]
    exp = np.floor(np.log2(arr[nonzero])).astype(np.int32) + 127
    out[nonzero] = np.clip(exp, 0, 254).astype(np.uint8)
    return out


def _decode_e8m0(arr: np.ndarray) -> np.ndarray:
    """Decode e8m0 uint8 → float32 scale values (2^(b - 127))."""
    arr = np.asarray(arr, dtype=np.uint8)
    return np.exp2(arr.astype(np.float32) - 127.0)


# OCP FP4 E2M1 lookup table (from pk_fp4.hpp e2m1_to_fp32_table).
# Index i (0-15) gives the float32 value for the 4-bit code i.
_FP4_E2M1_LUT: np.ndarray = np.array([
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


def _decode_fp4(packed: np.ndarray, K: int, N: int) -> np.ndarray:
    """Unpack K*N OCP FP4 E2M1 values from K*N/2 packed bytes.

    pk_fp4_t packing (from pk_fp4.hpp _pack/_unpack):
      byte = (element1 << 4) | (element0 & 0xF)
    Low nibble = element at flat index 2i; high nibble = element at flat index 2i+1.
    The flat layout is row-major [K, N], so each byte contains two consecutive N-elements.
    """
    flat = packed.flatten()
    lo = (flat & 0x0F).astype(np.uint8)
    hi = ((flat >> 4) & 0x0F).astype(np.uint8)
    out = np.empty(K * N, dtype=np.float32)
    out[0::2] = _FP4_E2M1_LUT[lo]
    out[1::2] = _FP4_E2M1_LUT[hi]
    return out.reshape(K, N)


def _bf16_raw_to_f32(arr: np.ndarray) -> np.ndarray:
    """Reinterpret a uint16 array of bf16 bit patterns as float32."""
    u16 = arr.flatten().astype(np.uint16)
    words = np.zeros(len(u16) * 2, dtype=np.uint16)
    words[1::2] = u16  # bf16 occupies upper 2 bytes of float32 (little-endian)
    return words.view(np.float32).reshape(arr.shape)


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------

def _reference_gemm(A_f32: np.ndarray, B_f32: np.ndarray,
                    BQ: np.ndarray, problem: BQuantGemmProblem,
                    c_dtype=np.float16) -> np.ndarray:
    """C = A @ dequant(B, BQ) in fp32, cast to c_dtype."""
    gK, gN = problem.quant_group_k, problem.quant_group_n
    B_dq = B_f32.copy()
    for qi in range(problem.QK_B):
        for qj in range(problem.QN_B):
            k0, k1 = qi * gK, min((qi + 1) * gK, problem.K)
            n0, n1 = qj * gN, min((qj + 1) * gN, problem.N)
            B_dq[k0:k1, n0:n1] *= float(BQ[qi, qj])
    return (A_f32.astype(np.float32) @ B_dq.astype(np.float32)).astype(c_dtype)


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

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

# ctest reports this as "skipped" rather than passed. Without it, a CPU-only or
# non-supported runner would report every BQuant test as a hard FAIL for a reason
# that has nothing to do with the code under test.
SKIP_EXIT = 77

# Kept in lockstep with the CI gate (`arch == "gfx950"` in ck.groovy): the two
# must not disagree, or the test claims coverage no lane produces. The blockers
# that made this gfx950-only -- the OCP-hardcoded host codec and the warp_tile_k
# mismatches in the utils -- are fixed, so widening this to include "gfx942" is
# a matter of flipping the lane and this tuple together, in that follow-up.
SUPPORTED_ARCHS = ("gfx950",)

# H3 (mx_*) is gfx950-only at the ISA level: __builtin_amdgcn_mfma_scale_f32_*
# builtins do not exist on gfx942 and cause a compile error there, not a silent
# zero. These cases are not merely deferred -- they cannot be enabled on gfx942.
MX_SUPPORTED_ARCHS = ("gfx950",)


def _detect_arch() -> str | None:
    """Detected gfx arch, or None when no GPU is visible.

    The empty fallback keeps absence detectable; detect_gpu_arch would otherwise
    invent a default and turn "no GPU" into a confusing build failure.
    """
    try:
        from dispatcher_common import detect_gpu_arch

        return detect_gpu_arch(fallback="") or None
    except Exception:
        return None


def _run_one(label: str, config, M: int, N: int, K: int,
             A_raw: np.ndarray, A_f32: np.ndarray,
             B_raw: np.ndarray, B_f32: np.ndarray,
             BQ: np.ndarray,
             out_dir: Path,
             c_dtype=np.float16,
             c_decode_fn=None,
             BQ_ref: np.ndarray = None,
             gfx_arch: str = "gfx950") -> tuple[str, str]:
    """
    Build, run, and verify one kernel.

    BQ       -- raw buffer passed to the GPU kernel (may be e8m0 uint8 for MX variants)
    BQ_ref   -- float32 scales used for the CPU reference; defaults to BQ if not given
                (caller should supply the decoded float32 version when BQ is e8m0)
    c_decode_fn -- optional fn(raw_C_array) -> float32 array, used when c_dtype is uint16
                   (bf16 raw output) to convert before comparison

    Returns (PASS|FAIL, detail_message).
    """
    if BQ_ref is None:
        BQ_ref = BQ.astype(np.float32)

    problem = BQuantGemmProblem(
        M=M, N=N, K=K,
        quant_group_m=config.quant_group_m,
        quant_group_n=config.quant_group_n,
        quant_group_k=config.quant_group_k,
    )

    so_paths = setup_multiple_bquant_dispatchers(
        configs=[config],
        output_dir=out_dir,
        gfx_arch=gfx_arch,
    )
    if not so_paths or so_paths[0] is None:
        return FAIL, f"{label}: kernel build failed"

    runner = BQuantGpuGemmRunner(so_paths[0])

    # Run once (timing is collected internally by the runner)
    result = runner.run(A=A_raw, B=B_raw, BQ=BQ, problem=problem, c_dtype=c_dtype)
    C_gpu_raw = result.C
    # Convert raw output to float32 for validation (needed for bf16 output via uint16 buffer)
    C_gpu = c_decode_fn(C_gpu_raw) if c_decode_fn is not None else C_gpu_raw.astype(np.float32)

    # Non-trivial check (C4 / H3 smoke). A wrong warp_tile_k (CK_GFX950_SUPPORT
    # branch mismatch) produces all-zero or all-constant output on gfx942; catch
    # it here before the tolerance gate, whose denominator near zero is unhelpful.
    if np.all(C_gpu == 0):
        return FAIL, f"{label}: GPU output is all-zero"
    if float(C_gpu.std()) <= 0.0:
        return FAIL, (f"{label}: GPU output is constant "
                      f"(value={float(C_gpu.flat[0]):.6g}); kernel did not compute")
    nan_mask = ~np.isfinite(C_gpu.astype(np.float32))
    if np.any(nan_mask):
        nan_frac = nan_mask.mean()
        sample = C_gpu.astype(np.float32).flat[:8].tolist()
        log.debug("%s: NaN/Inf fraction=%.3f, first 8 elements=%s", label, nan_frac, sample)
        return FAIL, f"{label}: GPU output contains NaN/Inf (frac={nan_frac:.3f})"

    # Correctness check vs. CPU reference (always in float32)
    C_ref = _reference_gemm(A_f32, B_f32, BQ_ref, problem, np.float32)
    mre = _max_rel_err(C_gpu, C_ref)
    if mre > TOLERANCE:
        return FAIL, (f"{label}: max_rel_err={mre:.4f} > tol={TOLERANCE:.4f} "
                      f"(shape M={M} N={N} K={K})")

    # Timing check (M2 sanity)
    result_timed = runner.run(A=A_raw, B=B_raw, BQ=BQ, problem=problem,
                              c_dtype=c_dtype)
    if result_timed.time_ms <= 0.0:
        return FAIL, f"{label}: time_ms={result_timed.time_ms:.4f} is not positive"

    return PASS, (f"{label}: max_rel_err={mre:.4f}, "
                  f"time_ms={result_timed.time_ms:.3f}")


# ---------------------------------------------------------------------------
# Individual test cases
# ---------------------------------------------------------------------------

def _make_fp8_inputs(M, N, K, gK, gN, dtype="fp8", gfx_arch="gfx950", seed=42):
    rng = np.random.default_rng(seed)
    QK_B = math.ceil(K / gK)
    QN_B = math.ceil(N / gN)
    A_f32 = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
    B_f32 = rng.uniform(-2.0, 2.0, (K, N)).astype(np.float32)
    BQ    = rng.uniform(0.5, 2.0, (QK_B, QN_B)).astype(np.float32)
    A_raw = _encode_fp8(A_f32, dtype, gfx_arch)
    B_raw = _encode_fp8(B_f32, dtype, gfx_arch)
    A_dec = _decode_fp8(A_raw, dtype, gfx_arch)
    B_dec = _decode_fp8(B_raw, dtype, gfx_arch)
    return A_raw, A_dec, B_raw, B_dec, BQ


def _to_bf16_raw(x: np.ndarray) -> np.ndarray:
    """Encode float32 array → uint16 array of bfloat16 bit patterns."""
    packed = np.frombuffer(x.astype(np.float32).tobytes(), dtype=np.uint16)
    # Little-endian: bf16 occupies the upper 2 bytes of each float32,
    # which are at odd indices (1, 3, 5, ...) in the uint16 view.
    return packed[1::2].reshape(x.shape)


def _make_bf16_inputs(M, N, K, gK, gN, seed=42):
    rng = np.random.default_rng(seed)
    QK_B = math.ceil(K / gK)
    QN_B = math.ceil(N / gN)
    A = rng.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    B = rng.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
    # BQ for MX: scales stored as e8m0 (uint8, one byte per group).
    # Generate float32 scales in [0.5, 1.5], encode to e8m0, then decode back
    # to float32 for the CPU reference (so reference uses the same quantised values).
    BQ_f32 = rng.uniform(0.5, 1.5, (QK_B, QN_B)).astype(np.float32)
    BQ_e8m0 = _encode_e8m0(BQ_f32)          # uint8, matches kernel's e8m0_t
    BQ_f32_dec = _decode_e8m0(BQ_e8m0)       # float32 after e8m0 round-trip
    # A and B as uint16 bfloat16; decode back for reference
    A_raw = _to_bf16_raw(A)
    B_raw = _to_bf16_raw(B)
    A_dec = _bf16_raw_to_f32(A_raw)
    B_dec = _bf16_raw_to_f32(B_raw)
    return A_raw, A_dec, B_raw, B_dec, BQ_e8m0, BQ_f32_dec


# N-tile sweep: N=128 is a single N-tile; N=256/512 span 2 and 4 TileN blocks and
# exercise the round-6 per-N-tile PermuteN de-permute (the round-5 global riffle
# scrambled columns at N>=256).  gN must divide N; TileN is 64 (decode) / 128 (MX).
#
# A single pinned N under TileN would leave the de-permute entirely unexercised,
# so this is coverage, not a knob: shrink it only with a replacement for that path.
_N_SWEEP = (128, 256, 512)


def case_c4_fp8(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    # K=768 = 3*TileK(256): use num_loop=3 (TailNumber::Odd) for better coverage.
    # num_loop=2 works but exercises only the no-hot-loop/Even tail path; 3 gives
    # the no-hot-loop/Odd tail path and exercises the BQ scale prefetch more robustly.
    M, K, gK, gN = 16, 768, 128, 1
    cfg = default_fp8_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    for N in _N_SWEEP:
        A_raw, A_dec, B_raw, B_dec, BQ = _make_fp8_inputs(M, N, K, gK, gN, "fp8",
                                                          gfx_arch=gfx_arch)
        status, detail = _run_one(f"C4/fp8/N{N}", cfg, M, N, K,
                                  A_raw, A_dec, B_raw, B_dec, BQ,
                                  out_dir, c_dtype=np.float16, gfx_arch=gfx_arch)
        if status != PASS:
            return status, detail
    return PASS, f"C4/fp8: PASS for N in {_N_SWEEP}"


def case_c4_bf8(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    # K=768 = 3*TileK(256): use num_loop=3 for the same reason as case_c4_fp8.
    M, K, gK, gN = 16, 768, 128, 1
    cfg = default_bf8_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    for N in _N_SWEEP:
        A_raw, A_dec, B_raw, B_dec, BQ = _make_fp8_inputs(M, N, K, gK, gN, "bf8",
                                                          gfx_arch=gfx_arch)
        status, detail = _run_one(f"C4/bf8/N{N}", cfg, M, N, K,
                                  A_raw, A_dec, B_raw, B_dec, BQ,
                                  out_dir, c_dtype=np.float16, gfx_arch=gfx_arch)
        if status != PASS:
            return status, detail
    return PASS, f"C4/bf8: PASS for N in {_N_SWEEP}"


def case_h3_mx_bf16bf16(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    if gfx_arch not in MX_SUPPORTED_ARCHS:
        return SKIP, (f"H3/mx_bf16bf16: not supported on {gfx_arch} "
                      f"(requires {'/'.join(MX_SUPPORTED_ARCHS)})")
    # K=256 = 2*TileK(128): MicroscaleCompV3 needs num_loop>=2 to avoid OOB second prefetch
    M, K, gK, gN = 128, 256, 32, 1
    cfg = default_mx_bf16bf16_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    for N in _N_SWEEP:
        # C is bf16_t (2 bytes); use uint16 buffer + decode fn so size matches exactly.
        A_raw, A_dec, B_raw, B_dec, BQ_e8m0, BQ_f32 = _make_bf16_inputs(M, N, K, gK, gN)
        status, detail = _run_one(f"H3/mx_bf16bf16/N{N}", cfg, M, N, K,
                                  A_raw, A_dec, B_raw, B_dec, BQ_e8m0,
                                  out_dir, c_dtype=np.uint16, c_decode_fn=_bf16_raw_to_f32,
                                  BQ_ref=BQ_f32, gfx_arch=gfx_arch)
        if status != PASS:
            return status, detail
    return PASS, f"H3/mx_bf16bf16: PASS for N in {_N_SWEEP}"


def case_h3_mx_bf16bf8(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    if gfx_arch not in MX_SUPPORTED_ARCHS:
        return SKIP, (f"H3/mx_bf16bf8: not supported on {gfx_arch} "
                      f"(requires {'/'.join(MX_SUPPORTED_ARCHS)})")
    # K=384 = 3*TileK(128): use num_loop=3 (TailNumber::Odd) for broader pipeline coverage.
    M, K, gK, gN = 128, 384, 128, 1
    cfg = default_mx_bf16bf8_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    for N in _N_SWEEP:
        A_raw, A_dec, _, _, BQ_e8m0, BQ_f32 = _make_bf16_inputs(M, N, K, gK, gN)
        # B is bf8: encode K*N float32 values as bf8 bytes
        B_f32 = np.random.default_rng(43).uniform(-1.0, 1.0, (K, N)).astype(np.float32)
        B_raw_bf8 = _encode_fp8(B_f32, "bf8", gfx_arch)
        B_dec_bf8 = _decode_fp8(B_raw_bf8, "bf8", gfx_arch)
        status, detail = _run_one(f"H3/mx_bf16bf8/N{N}", cfg, M, N, K,
                                  A_raw, A_dec, B_raw_bf8, B_dec_bf8, BQ_e8m0,
                                  out_dir, c_dtype=np.uint16, c_decode_fn=_bf16_raw_to_f32,
                                  BQ_ref=BQ_f32, gfx_arch=gfx_arch)
        if status != PASS:
            return status, detail
    return PASS, f"H3/mx_bf16bf8: PASS for N in {_N_SWEEP}"


def case_h3_mx_bf16fp4(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    if gfx_arch not in MX_SUPPORTED_ARCHS:
        return SKIP, (f"H3/mx_bf16fp4: not supported on {gfx_arch} "
                      f"(requires {'/'.join(MX_SUPPORTED_ARCHS)})")
    # pk_fp4: 2 fp4 values per byte → B buffer is K*N/2 bytes.
    # K=256 = 2*TileK(128): MicroscaleCompV3 needs num_loop>=2 to avoid OOB second prefetch.
    M, K, gK, gN = 128, 256, 32, 1
    cfg = default_mx_bf16fp4_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    for N in _N_SWEEP:
        A_raw, A_dec, _, _, BQ_e8m0, BQ_f32 = _make_bf16_inputs(M, N, K, gK, gN)
        rng = np.random.default_rng(44)
        # rcr kernel reads pk_fp4 B COLUMN-MAJOR: consecutive K-elements share a byte
        # (low nibble = k even, high nibble = k odd), per reference_mx_gemm_bquant's (k&1)
        # branch. Generate logical (K,N) fp4 codes, pack column-major, LUT-decode reference.
        codes = rng.integers(0, 16, size=(K, N), dtype=np.uint8)
        B_f32_approx = _FP4_E2M1_LUT[codes]                  # reference values (K,N)
        _flat = codes.flatten(order='F').astype(np.uint8)    # col-major: idx = n*K + k
        _lo = _flat[0::2] & 0x0F                             # k even -> low nibble
        _hi = _flat[1::2] & 0x0F                             # k odd  -> high nibble
        B_raw = (_lo | (_hi << 4)).astype(np.uint8)
        status, detail = _run_one(f"H3/mx_bf16fp4/N{N}", cfg, M, N, K,
                                  A_raw, A_dec, B_raw, B_f32_approx, BQ_e8m0,
                                  out_dir, c_dtype=np.uint16, c_decode_fn=_bf16_raw_to_f32,
                                  BQ_ref=BQ_f32, gfx_arch=gfx_arch)
        if status != PASS:
            return status, detail
    return PASS, f"H3/mx_bf16fp4: PASS for N in {_N_SWEEP}"


def case_c_i4(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    """Round-6: fp8i4 / bf8i4 must be exact once BQ is encoded to the kernel's
    QDataType (fp8/bf8, 1 byte).  The round-5 float32 BQ produced NaN.  Swept over
    N to also exercise the per-N-tile de-permute.  B is pk_int4 (2 per byte)."""
    M, K, gK, gN = 16, 256, 128, 1
    results = []
    for variant, ctor, dtype in (("fp8i4", default_fp8i4_config, "fp8"),
                                 ("bf8i4", default_bf8i4_config, "bf8")):
        cfg = ctor(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
        for N in _N_SWEEP:
            rng = np.random.default_rng(45)
            A_f32 = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
            A_raw = _encode_fp8(A_f32, dtype, gfx_arch)
            A_dec = _decode_fp8(A_raw, dtype, gfx_arch)
            # B is pk_int4: 2 K-elements per byte, packed COLUMN-MAJOR (rcr).
            #
            # Ground-truth nibble convention (Old-TE reference_gemm_quant +
            # pk_int4.hpp with CK_TILE_USE_PK4_LAYOUT_SHUFFLE=1):
            #   * value(code) = code - 8   (code in 0..15 -> value in [-8, +7]),
            #     NOT the two's-complement (codes<8?codes:codes-16) round-5 used.
            #   * pk_int4_t_to_fp32x2_t returns {x_h, x_l} (SHUFFLE swap), and
            #     load_b picks (k&1) ? .hi : .lo, so for the byte covering
            #     (k even, k odd) of a fixed n:
            #         k even -> HIGH nibble;  k odd -> LOW nibble.
            #   The device conversion consumes the same bytes after the
            #   UNCONDITIONAL permute_vectors_i4x4_b in the ctypes lib, so the
            #   host byte layout must match this reference exactly.
            codes = rng.integers(0, 16, size=(K, N), dtype=np.uint8)  # 0..15
            B_dec = codes.astype(np.float32) - 8.0                    # value = code-8
            _flat = codes.flatten(order='F').astype(np.uint8)  # col-major idx=n*K+k
            _even = _flat[0::2] & 0x0F   # k even -> HIGH nibble
            _odd = _flat[1::2] & 0x0F    # k odd  -> LOW nibble
            B_raw = (_odd | (_even << 4)).astype(np.uint8)
            QK_B = math.ceil(K / gK)
            QN_B = math.ceil(N / gN)
            # BQ supplied as float32 -> the runner encodes it to fp8/bf8 (QDataType,
            # arch-aware).  The kernel therefore sees the fp8/bf8-rounded scale, so
            # the CPU reference must use the same round-tripped value to stay fair.
            BQ = rng.uniform(0.5, 2.0, (QK_B, QN_B)).astype(np.float32)
            BQ_ref = _decode_fp8(_encode_fp8(BQ, dtype, gfx_arch), dtype, gfx_arch)
            BQ_ref = BQ_ref.reshape(QK_B, QN_B).astype(np.float32)
            status, detail = _run_one(f"C/{variant}/N{N}", cfg, M, N, K,
                                      A_raw, A_dec, B_raw, B_dec, BQ,
                                      out_dir, c_dtype=np.float16,
                                      BQ_ref=BQ_ref, gfx_arch=gfx_arch)
            if status != PASS:
                return status, detail
        results.append(variant)
    return PASS, f"C/i4: PASS ({', '.join(results)}) for N in {_N_SWEEP}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = [
    ("C4/fp8",          case_c4_fp8),
    ("C4/bf8",          case_c4_bf8),
    ("C/i4",            case_c_i4),
    ("H3/mx_bf16bf16",  case_h3_mx_bf16bf16),
    ("H3/mx_bf16bf8",   case_h3_mx_bf16bf8),
    ("H3/mx_bf16fp4",   case_h3_mx_bf16fp4),
]


def main():
    parser = argparse.ArgumentParser(description="BQuant GPU correctness tests (C4 + H3)")
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
        print("SKIP: no supported GPU detected (rocminfo); BQuant GPU tests skipped")
        return SKIP_EXIT
    if gfx not in SUPPORTED_ARCHS:
        print(f"SKIP: BQuant is {'/'.join(SUPPORTED_ARCHS)}-only; detected {gfx}")
        return SKIP_EXIT
    try:
        import ml_dtypes  # noqa: F401
    except ImportError:
        # Gate here rather than letting _encode_fp8 raise: an ImportError inside a
        # test function is swallowed by the loop's `except Exception` below and
        # reported as FAIL, blaming the kernel for a missing host dependency.
        print("SKIP: ml_dtypes not installed; BQuant fp8/bf8 encoding unavailable")
        return SKIP_EXIT

    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="bquant_gpu_test_"))
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
