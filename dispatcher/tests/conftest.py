#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures, numeric codecs, and a GPU build/run/verify harness for
the block-scale quant GEMM dispatcher bridge tests.

This module is the single canonical home for the machinery that the five
block-scale quant ``*_gpu_correctness.py`` files (aquant / abquant / bquant /
rowcolquant / tensor_quant) used to copy-paste verbatim:

  * ``hipcc_available`` / ``gpu_available`` -- probe hipcc + the running device so
    GPU tests ``pytest.skip`` cleanly on CPU-only boxes (they never build a .so
    where no GPU is present).  A ``skip_without_gpu`` autouse-friendly fixture is
    also exposed for convenience.
  * the arch-aware fp8/bf8 (OCP + FNUZ), e8m0, OCP-fp4-LUT and bf16<->f32 codec
    helpers -- ONE copy, used by every quant GPU test.
  * ``run_and_verify(...)`` -- the common build .so -> run -> finite -> non-zero ->
    fp32 reference -> max-rel-tolerance flow, parameterized by an op-specific
    reference function and tolerance.

Everything here imports without a GPU (all device access is lazy / probed), so the
CPU-only unit tests that merely ``import`` a sibling test module are unaffected.
"""

import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pytest


# =============================================================================
# GPU / hipcc probes
# =============================================================================


def _rocm_agent_archs():
    """Return the list of gfx arch strings the running box reports, or []."""
    for tool in ("rocm_agent_enumerator", "rocminfo"):
        if shutil.which(tool) is None:
            continue
        try:
            out = subprocess.run(
                [tool], capture_output=True, text=True, timeout=30
            ).stdout
        except Exception:
            continue
        archs = []
        if tool == "rocm_agent_enumerator":
            for line in out.splitlines():
                tok = line.strip()
                if tok.startswith("gfx") and tok != "gfx000":
                    archs.append(tok)
        else:  # rocminfo
            for line in out.splitlines():
                tok = line.strip()
                if tok.startswith("Name:") and "gfx" in tok:
                    name = tok.split(":", 1)[1].strip()
                    if name.startswith("gfx"):
                        archs.append(name)
        if archs:
            return archs
    return []


def hipcc_available() -> bool:
    """True if hipcc is on PATH (or at the canonical ROCm location)."""
    from pathlib import Path

    return shutil.which("hipcc") is not None or Path("/opt/rocm/bin/hipcc").exists()


def gpu_available() -> bool:
    """True only if BOTH hipcc and a usable ROCm GPU are present.

    These tests build a kernel .so at runtime via hipcc, so a box that reports a
    device but has no hipcc (or vice versa) must skip cleanly rather than run and
    then fail at build time.  ``rocminfo`` is used as the device gate (it fails
    when no runtime device is actually usable) to match the historical
    per-op ``_have_gpu`` behaviour; ``rocm_agent_enumerator`` is used elsewhere as
    the arch resolver.
    """
    if not hipcc_available():
        return False
    if shutil.which("rocminfo") is None:
        # Fall back to the enumerator when rocminfo is absent entirely.
        return bool(_rocm_agent_archs())
    try:
        out = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=30
        )
        return "gfx" in out.stdout
    except Exception:
        return False


def ml_dtypes_available() -> bool:
    try:
        import ml_dtypes  # noqa: F401

        return True
    except ImportError:
        return False


def detect_gpu_arch(default: str = "gfx950") -> str:
    """Best-effort running-device arch (via rocm_agent_enumerator/rocminfo)."""
    archs = _rocm_agent_archs()
    return archs[0] if archs else default


# --- pytest skip fixtures --------------------------------------------------


@pytest.fixture(scope="session")
def has_gpu() -> bool:
    """Session-scoped boolean: is a GPU+hipcc usable here?"""
    return gpu_available()


@pytest.fixture
def skip_without_gpu(has_gpu):
    """Skip the requesting test unless a GPU + hipcc are usable."""
    if not has_gpu:
        pytest.skip("no ROCm GPU / hipcc detected")


@pytest.fixture
def skip_without_ml_dtypes():
    """Skip the requesting test unless the ml_dtypes fp8/bf8 codecs are present."""
    if not ml_dtypes_available():
        pytest.skip("ml_dtypes not installed")


@pytest.fixture
def gpu_arch(skip_without_gpu) -> str:
    """The detected GPU arch (only resolved after the GPU skip-gate passes)."""
    return detect_gpu_arch()


# =============================================================================
# Arch-aware fp8 / bf8 codecs (OCP on gfx950/gfx12, FNUZ on gfx942 / gfx90a)
# =============================================================================


def uses_ocp_fp8(arch: str) -> bool:
    """True when ck_tile::fp8_t is OCP (e4m3/e5m2), not FNUZ, for ``arch``.

    gfx950 / gfx12* build with OCP fp8; everything else (gfx942 / gfx90a) uses
    FNUZ e4m3fnuz / e5m2fnuz.  Encoding with the wrong flavour NaNs / decorrelates
    the reference.  ``None`` defaults to OCP (historical gfx950 self-test default).
    """
    if arch is None:
        return True
    return ("gfx950" in arch) or ("gfx12" in arch)


def ml_fp8_dtype(dtype: str, arch: str):
    """Return the ml_dtypes fp8/bf8 type for ``dtype`` on ``arch``."""
    import ml_dtypes

    if uses_ocp_fp8(arch):
        return ml_dtypes.float8_e4m3fn if dtype == "fp8" else ml_dtypes.float8_e5m2
    return ml_dtypes.float8_e4m3fnuz if dtype == "fp8" else ml_dtypes.float8_e5m2fnuz


def encode_fp8(arr: np.ndarray, dtype: str, arch: str) -> np.ndarray:
    """Encode a float array -> fp8/bf8 bytes (uint8 view), arch-aware flavour."""
    t = ml_fp8_dtype(dtype, arch)
    return np.ascontiguousarray(arr.astype(np.float32).astype(t)).view(np.uint8)


def qdq_fp8(arr: np.ndarray, dtype: str, arch: str) -> np.ndarray:
    """Round a float array through fp8/bf8 and back to float32 (reference side)."""
    t = ml_fp8_dtype(dtype, arch)
    return arr.astype(np.float32).astype(t).astype(np.float32)


# =============================================================================
# e8m0 block-scale codec (MX)
# =============================================================================


def encode_e8m0(arr: np.ndarray) -> np.ndarray:
    """Encode float32 scale values -> e8m0 uint8 (MX block-scale format).

    e8m0 stores a power-of-two exponent: byte b represents 2^(b - 127).
    Scales must be positive; zero maps to 0 (subnormal/zero in e8m0).
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0.0, np.float32(2.0 ** 127))
    nonzero = arr > 0.0
    out = np.zeros(arr.shape, dtype=np.uint8)
    exp = np.floor(np.log2(arr[nonzero])).astype(np.int32) + 127
    out[nonzero] = np.clip(exp, 0, 254).astype(np.uint8)
    return out


def decode_e8m0(arr: np.ndarray) -> np.ndarray:
    """Decode e8m0 uint8 -> float32 scale values (2^(b - 127))."""
    arr = np.asarray(arr, dtype=np.uint8)
    return np.exp2(arr.astype(np.float32) - 127.0)


# =============================================================================
# OCP FP4 (E2M1) lookup table + unpack
# =============================================================================

# Index i (0-15) gives the float32 value for the 4-bit code i (from pk_fp4.hpp
# e2m1_to_fp32_table).
FP4_E2M1_LUT: np.ndarray = np.array([
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


def decode_fp4(packed: np.ndarray, K: int, N: int) -> np.ndarray:
    """Unpack K*N OCP FP4 E2M1 values from K*N/2 packed bytes (row-major [K,N]).

    pk_fp4_t packing: byte = (element1 << 4) | (element0 & 0xF).  Low nibble = the
    element at flat index 2i; high nibble = flat index 2i+1.
    """
    flat = packed.flatten()
    lo = (flat & 0x0F).astype(np.uint8)
    hi = ((flat >> 4) & 0x0F).astype(np.uint8)
    out = np.empty(K * N, dtype=np.float32)
    out[0::2] = FP4_E2M1_LUT[lo]
    out[1::2] = FP4_E2M1_LUT[hi]
    return out.reshape(K, N)


# =============================================================================
# bf16 <-> f32 bit reinterpretation
# =============================================================================


def to_bf16_raw(x: np.ndarray) -> np.ndarray:
    """Encode a float32 array -> uint16 array of bfloat16 bit patterns."""
    packed = np.frombuffer(x.astype(np.float32).tobytes(), dtype=np.uint16)
    # Little-endian: bf16 is the upper 2 bytes of each float32 (odd uint16 index).
    return packed[1::2].reshape(x.shape)


def bf16_raw_to_f32(arr: np.ndarray) -> np.ndarray:
    """Reinterpret a uint16 array of bf16 bit patterns as float32."""
    u16 = arr.flatten().astype(np.uint16)
    words = np.zeros(len(u16) * 2, dtype=np.uint16)
    words[1::2] = u16
    return words.view(np.float32).reshape(arr.shape)


# =============================================================================
# max-relative-error helper (global-max floor to avoid near-zero blow-up)
# =============================================================================


def max_rel_err(got: np.ndarray, ref: np.ndarray, floor_frac: float = 1e-2) -> float:
    """Max |got - ref| / (|ref| + floor), floor = max(floor_frac*max|ref|, 1e-6).

    Per-element relative error with a global-max denominator floor.  This is the
    bquant metric (its ``_max_rel_err`` was exactly ``floor_frac=1e-2``).  The
    other four quant ops historically used ``global_max_rel_err`` below (a scalar
    max-diff / max-ref ratio); do NOT reuse this per-element form for them -- when
    a GEMM produces near-zero elements (mixed-sign inputs partially cancel) the
    per-element ``|r|`` denominator explodes the ratio (the rowcolquant 27x
    blow-up came from running these four ops through this helper).
    """
    g = got.astype(np.float32)
    r = ref.astype(np.float32)
    ref_max = float(np.abs(r).max())
    den = np.abs(r) + max(ref_max * floor_frac, 1e-6)
    return float(np.max(np.abs(g - r) / den))


def global_max_rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    """max|got - ref| / (max|ref| + 1e-6) -- the global-normalized metric.

    Byte-for-byte the error metric the aquant / abquant / rowcolquant /
    tensor_quant GPU tests used before the shared harness existed:

        max_rel = float(np.max(np.abs(C_gpu - C_ref)) / (np.max(np.abs(C_ref)) + 1e-6))

    The numerator is the *global* max absolute difference (a scalar) and the
    denominator is the *global* max magnitude of the reference; both are scalars,
    so a large error on a near-zero element is normalized by the overall output
    scale rather than by that element's own (possibly ~0) magnitude.  Keeping this
    exact form is required to reproduce the historical pass/fail behaviour.
    """
    g = got.astype(np.float32)
    r = ref.astype(np.float32)
    return float(np.max(np.abs(g - r)) / (np.max(np.abs(r)) + 1e-6))


# =============================================================================
# Shared build -> run -> verify GPU harness
# =============================================================================


@dataclass
class VerifyResult:
    """Outcome of ``run_and_verify``: the max relative error and kernel time."""

    max_rel: float
    time_ms: float


def run_and_verify(
    *,
    build_so: Callable[[], Optional[str]],
    run_kernel: Callable[[str], "tuple"],
    reference: Callable[[], np.ndarray],
    tol: float,
    label: str = "",
    on_build_fail: Optional[Callable[[], None]] = None,
    rel_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> VerifyResult:
    """Common block-scale GPU correctness flow.

    Parameters
    ----------
    build_so
        Builds the op's dispatcher .so and returns its path, or ``None`` on
        failure.
    run_kernel
        Given the .so path, runs the kernel on-device and returns
        ``(C_gpu_array, time_ms)``.
    reference
        Returns the fp32 numpy reference ``C_ref`` (kernel-identical rounding).
    tol
        Max-relative-error tolerance.
    label
        Human-readable case label used in assertion messages.
    on_build_fail
        Optional callback invoked when the build returns ``None`` -- lets an op
        translate a known toolchain gate into ``pytest.skip`` before this helper
        raises.  If it does not raise/skip, an ``AssertionError`` is raised.
    rel_metric
        The error metric ``fn(C_gpu, C_ref) -> float``.  Defaults to
        ``global_max_rel_err`` -- the ``max|diff| / (max|ref| + 1e-6)`` form the
        aquant / abquant / rowcolquant / tensor_quant tests have always used.  Do
        NOT substitute a per-element relative error here: near-zero reference
        elements (common when mixed-sign GEMM inputs cancel) make it explode
        (this caused the rowcolquant 27x max_rel regression).

    Returns
    -------
    VerifyResult(max_rel, time_ms)
    """
    tag = f"{label}: " if label else ""
    if rel_metric is None:
        rel_metric = global_max_rel_err

    so_path = build_so()
    if so_path is None:
        if on_build_fail is not None:
            on_build_fail()  # may pytest.skip / pytest.fail
        raise AssertionError(f"{tag}kernel build failed")

    C_gpu, time_ms = run_kernel(so_path)
    C_gpu = np.asarray(C_gpu, dtype=np.float32)

    assert np.max(np.abs(C_gpu)) > 1e-3, (
        f"{tag}GPU output all-zeros (warp_tile_k arch trap?)"
    )
    assert np.all(np.isfinite(C_gpu)), f"{tag}GPU output contains NaN/Inf"

    C_ref = np.asarray(reference(), dtype=np.float32)
    mre = rel_metric(C_gpu, C_ref)
    assert mre <= tol, f"{tag}max_rel={mre:.4f} > tol={tol}"

    return VerifyResult(max_rel=mre, time_ms=float(time_ms))
