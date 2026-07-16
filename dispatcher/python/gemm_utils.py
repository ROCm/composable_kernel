# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GEMM Tile Engine <-> Dispatcher bridge.

This is the GEMM counterpart of ``grouped_conv_utils.py`` / ``fmha_utils.py``:
a single shared config dataclass (``GemmKernelConfig``) that Tile Engine imports
and hands back to the dispatcher. There is no translator between two
vocabularies -- both sides share the one object whose ``.name`` mirrors the
kernel identifier baked into the generated kernel header.

Public surface (mirrors the grouped_conv bridge):

    GemmKernelConfig                 -- the shared contract dataclass
        .name                        -- registry/runtime lookup key (byte-exact)
        .to_codegen_json()           -- feeds unified_gemm_codegen.py
    GemmProblem                      -- a single (M, N, K) problem
    setup_multiple_gemm_dispatchers  -- codegen + hipcc -> .so paths (NO GPU)
    GemmDispatcherLib                -- thin ctypes ABI wrapper
    GpuGemmRunner                    -- GPU memory + run + time (from a .so path)
    expand_sweep                     -- TE JSON sweep config -> [GemmKernelConfig]

The heavy lifting for codegen and compilation is reused from ``ctypes_utils``
so there is a single source of truth for how a kernel header is produced and
how it is compiled into a ``.so``.
"""

from __future__ import annotations

import ctypes
import functools
import itertools
import json
import multiprocessing
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reuse the proven codegen/compile leaf helpers from the dispatcher's own
# python layer. gemm_utils is a thin bridge on top of these.
import ctypes_utils as _cu


# ============================================================================
# Layout / dtype helpers
# ============================================================================

_LAYOUT_CHAR = {"row": "r", "col": "c", "r": "r", "c": "c"}
_LAYOUT_WORD = {"r": "row", "c": "col"}


def _cap(flag: bool) -> str:
    """Reproduce Python ``str(bool).capitalize()`` -> 'True' / 'False'."""
    return "True" if flag else "False"


# ---------------------------------------------------------------------------
# Dtype codecs: map a bridge dtype token -> numpy dtype for host operands.
#
# fp16 maps to plain numpy; bf16/fp8/bf8 need ml_dtypes. fp8/bf8 use the FNUZ
# encodings (E4M3FNUZ / E5M2FNUZ) that the gfx942 MFMA path expects -- matching
# the regular bridge's fp8/bf8 codec (PR #8887). ml_dtypes is imported lazily so
# the fp16-only path keeps working where ml_dtypes is unavailable.
# ---------------------------------------------------------------------------

# Canonicalize common spellings to a single token.
_DTYPE_ALIASES = {
    "fp16": "fp16",
    "f16": "fp16",
    "half": "fp16",
    "float16": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp8": "fp8",
    "fp8_e4m3": "fp8",
    "e4m3": "fp8",
    "bf8": "bf8",
    "fp8_e5m2": "bf8",
    "e5m2": "bf8",
}


def numpy_dtype_for(dtype: str):
    """Return the numpy dtype object used for host operands of ``dtype``.

    fp16 -> np.float16; bf16/fp8/bf8 require the ``ml_dtypes`` package (imported
    lazily) and use FNUZ fp8 encodings for gfx942 parity.
    """
    token = _DTYPE_ALIASES.get(str(dtype).lower())
    if token is None:
        raise ValueError(f"Unsupported grouped GEMM dtype: {dtype!r}")
    if token == "fp16":
        return np.float16
    try:
        import ml_dtypes  # noqa: WPS433 (lazy: optional dep)
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            f"dtype {dtype!r} requires the 'ml_dtypes' package (pip install ml_dtypes)"
        ) from exc
    if token == "bf16":
        return np.dtype(ml_dtypes.bfloat16)
    if token == "fp8":
        return np.dtype(ml_dtypes.float8_e4m3fnuz)
    if token == "bf8":
        return np.dtype(ml_dtypes.float8_e5m2fnuz)
    raise ValueError(f"Unsupported grouped GEMM dtype: {dtype!r}")  # pragma: no cover


def output_dtype_for(dtype: str) -> str:
    """Return the bridge dtype token of a kernel's OUTPUT for input ``dtype``.

    Mirrors ``codegen_common.CommonTypeMappings.get_output_dtype`` (fp8/bf8 ->
    fp16, else identity): the generated grouped kernel emits an fp16 ``CDataType``
    for fp8/bf8 inputs, so the host C buffer must be sized/typed by the OUTPUT
    dtype, not the INPUT dtype. ``codegen_common`` lives on the dispatcher
    ``codegen`` dir which ctypes_utils already puts on ``sys.path``; import it
    lazily so the fp16-only path has no extra dependency.
    """
    token = _DTYPE_ALIASES.get(str(dtype).lower())
    if token is None:
        raise ValueError(f"Unsupported grouped GEMM dtype: {dtype!r}")
    try:
        from codegen_common import CommonTypeMappings  # noqa: WPS433 (lazy)
    except ImportError:  # pragma: no cover - fall back to the documented mapping
        return "fp16" if token in ("fp8", "bf8") else token
    return CommonTypeMappings.get_output_dtype(token)


def output_numpy_dtype_for(dtype: str):
    """Numpy dtype of a kernel's OUTPUT buffer for input ``dtype``.

    Composition of :func:`output_dtype_for` + :func:`numpy_dtype_for`. For
    fp8/bf8 this resolves to ``np.float16`` (2 bytes) because the kernel's
    ``CDataType`` is fp16; for fp16/bf16 it equals the input dtype.
    """
    return numpy_dtype_for(output_dtype_for(dtype))


# ============================================================================
# The shared contract: GemmKernelConfig
# ============================================================================


@dataclass
class GemmKernelConfig:
    """The common config struct shared by Tile Engine and the Dispatcher.

    Naming convention (the "warp/wave trap" lives here, in ONE place):
      * ``wave_m/n/k``      -- warps per block (C++ ``wave_shape``; TE "warp").
      * ``warp_tile_m/n/k`` -- MFMA instruction shape (C++ ``warp_tile_shape``;
                               TE "warp_tile").
    """

    # --- Signature: what operation is computed -----------------------------
    dtype_a: str = "fp16"
    dtype_b: str = "fp16"
    dtype_c: str = "fp16"
    dtype_acc: str = "fp32"
    layout_a: str = "row"
    layout_b: str = "col"
    layout_c: str = "row"

    # --- Algorithm: how it is implemented ----------------------------------
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 32
    wave_m: int = 2
    wave_n: int = 2
    wave_k: int = 1
    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16

    pipeline: str = "compv4"
    scheduler: str = "intrawave"
    epilogue: str = "cshuffle"

    pad_m: bool = True
    pad_n: bool = True
    pad_k: bool = True
    persistent: bool = False

    gfx_arch: str = "gfx942"
    variant: str = "standard"

    # ------------------------------------------------------------------ #
    # Derived string fragments
    # ------------------------------------------------------------------ #
    @property
    def layout(self) -> str:
        """3-char layout string, e.g. 'rcr'."""
        return (
            _LAYOUT_CHAR[self.layout_a]
            + _LAYOUT_CHAR[self.layout_b]
            + _LAYOUT_CHAR[self.layout_c]
        )

    @property
    def tile_str(self) -> str:
        return f"{self.tile_m}x{self.tile_n}x{self.tile_k}"

    @property
    def wave_str(self) -> str:
        return f"{self.wave_m}x{self.wave_n}x{self.wave_k}"

    @property
    def warp_tile_str(self) -> str:
        return f"{self.warp_tile_m}x{self.warp_tile_n}x{self.warp_tile_k}"

    @property
    def name(self) -> str:
        """Registry / runtime lookup key.

        Reproduces, byte-for-byte, the ``KERNEL_NAME`` that
        ``unified_gemm_codegen.py::KernelNaming.generate`` bakes into the
        generated kernel header (and that the .so reports via
        ``dispatcher_get_kernel_name``). This is the single thread tying
        config -> codegen -> runtime together.
        """
        name = (
            f"gemm_{self.dtype_a}_{self.layout}"
            f"_{self.pipeline}_{self.epilogue}_{self.scheduler}"
            f"_{_cap(self.pad_m)}_{_cap(self.pad_n)}_{_cap(self.pad_k)}"
            f"_{_cap(self.persistent)}"
            f"_{self.tile_str}_{self.wave_str}_{self.warp_tile_str}"
        )
        if self.variant == "preshuffle":
            name += "_preshuffle"
        elif self.variant == "streamk":
            name += "_streamk"
        elif self.variant == "grouped":
            name += "_grouped"
        return name

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_codegen_json(self) -> Dict[str, Any]:
        """Single-config JSON consumed by unified_gemm_codegen.py.

        Note the warp/wave mapping: the codegen calls the warps-per-block
        triple ``warp_*`` and the MFMA triple ``warp_tile_*``. We translate
        from dispatcher semantics here so the mapping cannot drift.
        """
        return {
            "tile_config": {
                "tile_m": [self.tile_m],
                "tile_n": [self.tile_n],
                "tile_k": [self.tile_k],
                # dispatcher wave_* -> codegen warp_* (warps per block)
                "warp_m": [self.wave_m],
                "warp_n": [self.wave_n],
                "warp_k": [self.wave_k],
                # dispatcher warp_tile_* -> codegen warp_tile_* (MFMA shape)
                "warp_tile_m": [self.warp_tile_m],
                "warp_tile_n": [self.warp_tile_n],
                "warp_tile_k": [self.warp_tile_k],
            },
            "trait_config": {
                "pipeline": [self.pipeline],
                "epilogue": [self.epilogue],
                "scheduler": [self.scheduler],
                "pad_m": [self.pad_m],
                "pad_n": [self.pad_n],
                "pad_k": [self.pad_k],
                "persistent": [self.persistent],
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dtype_a": self.dtype_a,
            "dtype_b": self.dtype_b,
            "dtype_c": self.dtype_c,
            "dtype_acc": self.dtype_acc,
            "layout": self.layout,
            "tile": [self.tile_m, self.tile_n, self.tile_k],
            "wave": [self.wave_m, self.wave_n, self.wave_k],
            "warp_tile": [self.warp_tile_m, self.warp_tile_n, self.warp_tile_k],
            "pipeline": self.pipeline,
            "scheduler": self.scheduler,
            "epilogue": self.epilogue,
            "pad": [self.pad_m, self.pad_n, self.pad_k],
            "persistent": self.persistent,
            "gfx_arch": self.gfx_arch,
            "variant": self.variant,
            "name": self.name,
        }

    def to_ctypes_config(self) -> "_cu.KernelConfig":
        """Convert to the ctypes_utils.KernelConfig used by the codegen/validate
        helpers. ctypes_utils renames the MFMA triple ``warp_*`` (no _tile)."""
        return _cu.KernelConfig(
            dtype_a=self.dtype_a,
            dtype_b=self.dtype_b,
            dtype_c=self.dtype_c,
            dtype_acc=self.dtype_acc,
            layout_a=_LAYOUT_WORD[_LAYOUT_CHAR[self.layout_a]],
            layout_b=_LAYOUT_WORD[_LAYOUT_CHAR[self.layout_b]],
            layout_c=_LAYOUT_WORD[_LAYOUT_CHAR[self.layout_c]],
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            wave_m=self.wave_m,
            wave_n=self.wave_n,
            wave_k=self.wave_k,
            warp_m=self.warp_tile_m,
            warp_n=self.warp_tile_n,
            warp_k=self.warp_tile_k,
            pipeline=self.pipeline,
            scheduler=self.scheduler,
            epilogue=self.epilogue,
            pad_m=self.pad_m,
            pad_n=self.pad_n,
            pad_k=self.pad_k,
            gfx_arch=self.gfx_arch,
            variant=self.variant,
        )


# ============================================================================
# Problem
# ============================================================================


@dataclass
class GemmProblem:
    """A single GEMM problem: C[MxN] = A[MxK] @ B[KxN]."""

    M: int
    N: int
    K: int

    @property
    def flops(self) -> float:
        return 2.0 * self.M * self.N * self.K

    def to_dict(self) -> Dict[str, int]:
        return {"M": self.M, "N": self.N, "K": self.K}

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "GemmProblem":
        return cls(M=int(d["M"]), N=int(d["N"]), K=int(d["K"]))


@dataclass
class GroupedGemmProblem:
    """A grouped GEMM problem: a list of independent (M, N, K) sub-problems
    all run by a single grouped kernel launch.

    Each group g computes C_g[M_g x N_g] = A_g[M_g x K_g] @ B_g[K_g x N_g].
    """

    groups: List[Tuple[int, int, int]]

    @classmethod
    def uniform(
        cls, group_count: int, M: int, N: int, K: int
    ) -> "GroupedGemmProblem":
        """All groups share the same (M, N, K) shape."""
        return cls(groups=[(int(M), int(N), int(K)) for _ in range(int(group_count))])

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def flops(self) -> float:
        return sum(2.0 * m * n * k for (m, n, k) in self.groups)

    def to_dict(self) -> Dict[str, Any]:
        return {"groups": [[int(m), int(n), int(k)] for (m, n, k) in self.groups]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GroupedGemmProblem":
        return cls(groups=[(int(m), int(n), int(k)) for (m, n, k) in d["groups"]])


@dataclass
class GemmResult:
    output: np.ndarray
    time_ms: float
    status: int
    tflops: float
    kernel_name: str

    @property
    def success(self) -> bool:
        return self.status == 0


@dataclass
class GroupedGemmResult:
    """Result of a grouped GEMM launch: one output per group plus aggregate
    timing/throughput across the whole batch."""

    outputs: List[np.ndarray]
    time_ms: float
    status: int
    tflops: float
    kernel_name: str

    @property
    def success(self) -> bool:
        return self.status == 0


# ============================================================================
# ctypes ABI wrapper
# ============================================================================


class GemmDispatcherLib:
    """Thin ctypes wrapper around a compiled GEMM dispatcher .so.

    Supports both the legacy single-kernel ABI (``dispatcher_get_kernel_name``)
    and the multi-kernel ABI (``dispatcher_get_kernel_name_at(index, buf, n)``)
    so one .so can report a whole batch and be selected by name.
    """

    def __init__(self, so_path: Path):
        self._path = Path(so_path)
        self._lib = ctypes.CDLL(str(self._path))
        self._has_indexed = hasattr(self._lib, "dispatcher_get_kernel_name_at")
        self._has_grouped = hasattr(self._lib, "dispatcher_run_grouped_gemm")
        self._has_single = hasattr(self._lib, "dispatcher_run_gemm")
        self._setup_functions()

    def _setup_functions(self) -> None:
        lib = self._lib

        lib.dispatcher_initialize.argtypes = []
        lib.dispatcher_initialize.restype = ctypes.c_int

        lib.dispatcher_get_kernel_count.argtypes = []
        lib.dispatcher_get_kernel_count.restype = ctypes.c_int

        lib.dispatcher_get_kernel_name.argtypes = []
        lib.dispatcher_get_kernel_name.restype = ctypes.c_char_p

        if self._has_indexed:
            lib.dispatcher_get_kernel_name_at.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            lib.dispatcher_get_kernel_name_at.restype = ctypes.c_int

        # Single-problem ABI (regular GEMM .so). Absent on grouped libs.
        if self._has_single:
            lib.dispatcher_run_gemm.argtypes = [
                ctypes.c_void_p,  # A (host)
                ctypes.c_void_p,  # B (host)
                ctypes.c_void_p,  # C (host)
                ctypes.c_int64,  # M
                ctypes.c_int64,  # N
                ctypes.c_int64,  # K
                ctypes.POINTER(ctypes.c_float),  # time_ms
            ]
            lib.dispatcher_run_gemm.restype = ctypes.c_int

        # Multi-problem ABI (grouped GEMM .so). Absent on regular libs.
        if self._has_grouped:
            lib.dispatcher_run_grouped_gemm.argtypes = [
                ctypes.c_int,  # group_count
                ctypes.POINTER(ctypes.c_int64),  # Ms[]
                ctypes.POINTER(ctypes.c_int64),  # Ns[]
                ctypes.POINTER(ctypes.c_int64),  # Ks[]
                ctypes.POINTER(ctypes.c_void_p),  # A_ptrs[]
                ctypes.POINTER(ctypes.c_void_p),  # B_ptrs[]
                ctypes.POINTER(ctypes.c_void_p),  # C_ptrs[]
                ctypes.POINTER(ctypes.c_float),  # time_ms
            ]
            lib.dispatcher_run_grouped_gemm.restype = ctypes.c_int

        lib.dispatcher_cleanup.argtypes = []
        lib.dispatcher_cleanup.restype = None

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> bool:
        return self._lib.dispatcher_initialize() == 0

    def get_kernel_count(self) -> int:
        return int(self._lib.dispatcher_get_kernel_count())

    @property
    def kernel_names(self) -> List[str]:
        """List every kernel the .so exposes, by index when available."""
        if self._has_indexed:
            names: List[str] = []
            count = self.get_kernel_count()
            buf = ctypes.create_string_buffer(256)
            for i in range(count):
                if self._lib.dispatcher_get_kernel_name_at(i, buf, 256) == 0:
                    names.append(buf.value.decode("utf-8"))
            if names:
                return names
        # Legacy single-kernel fallback.
        raw = self._lib.dispatcher_get_kernel_name()
        return [raw.decode("utf-8")] if raw else []

    def run(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, M: int, N: int, K: int
    ) -> Tuple[int, float]:
        time_ms = ctypes.c_float(0.0)
        status = self._lib.dispatcher_run_gemm(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            M,
            N,
            K,
            ctypes.byref(time_ms),
        )
        return status, time_ms.value

    def run_grouped(
        self,
        A_list: List[np.ndarray],
        B_list: List[np.ndarray],
        C_list: List[np.ndarray],
        Ms: List[int],
        Ns: List[int],
        Ks: List[int],
    ) -> Tuple[int, float]:
        """Launch the grouped kernel over a batch of (M, N, K) sub-problems.

        Each A/B/C entry is a host numpy array already laid out (dtype + row/col
        transpose) as the kernel expects for its compile-time layout; the caller
        (GpuGroupedGemmRunner) does that per-dtype/per-layout packing. Pointers
        are marshalled into ctypes pointer arrays.
        """
        if not self._has_grouped:
            raise RuntimeError(
                f"{self._path} does not expose dispatcher_run_grouped_gemm"
            )

        g = len(A_list)
        c_int64_arr = (ctypes.c_int64 * g)
        c_void_arr = (ctypes.c_void_p * g)

        ms = c_int64_arr(*[int(m) for m in Ms])
        ns = c_int64_arr(*[int(n) for n in Ns])
        ks = c_int64_arr(*[int(k) for k in Ks])

        a_ptrs = c_void_arr(*[A.ctypes.data_as(ctypes.c_void_p) for A in A_list])
        b_ptrs = c_void_arr(*[B.ctypes.data_as(ctypes.c_void_p) for B in B_list])
        c_ptrs = c_void_arr(*[C.ctypes.data_as(ctypes.c_void_p) for C in C_list])

        time_ms = ctypes.c_float(0.0)
        status = self._lib.dispatcher_run_grouped_gemm(
            g,
            ms,
            ns,
            ks,
            a_ptrs,
            b_ptrs,
            c_ptrs,
            ctypes.byref(time_ms),
        )
        return status, time_ms.value

    def cleanup(self) -> None:
        self._lib.dispatcher_cleanup()


# ============================================================================
# GPU runner (constructed from a .so path; loaded only inside a worker)
# ============================================================================


def _fp32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    """Encode fp32 -> bfloat16 bit pattern in a uint16 array (round-to-nearest-even).

    numpy has no native bf16, but the C ABI only cares about the 2-byte memory
    layout (sizeof(bf16_t) == 2 == sizeof(uint16)). Truncating the low 16 bits of
    the fp32 representation with round-to-nearest-even matches ck_tile's bf16.
    """
    u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    # round-to-nearest-even: add (lsb-of-kept-bits + 0x7FFF) before truncating
    rounding = ((u32 >> 16) & 1) + np.uint32(0x7FFF)
    return ((u32 + rounding) >> 16).astype(np.uint16)


def _bf16_u16_to_fp32(u16: np.ndarray) -> np.ndarray:
    """Decode a uint16 bf16 bit pattern back to fp32 (low 16 mantissa bits zero)."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------------
# fp8 (E4M3) / bf8 (E5M2) -- FNUZ ("NANOO") encoding used by gfx942/MI300.
#
# numpy has no native 8-bit float, and the C ABI only cares about the 1-byte
# memory layout (sizeof(fp8_t) == sizeof(bf8_t) == 1). We carry the value as a
# uint8 bit pattern. As with bf16, the DECODE is the load-bearing half: it must
# return the exact value the device's fp8_t/bf8_t represents for a byte, so the
# NumPy reference multiplies bit-for-bit what the GPU multiplies. The ENCODE only
# needs to land on the nearest representable byte.
#
# FNUZ format (gfx942): bias = 2^(exp_bits-1); the all-1s exponent is a normal
# number (no Inf), the sole NaN is the sign=1/exp=0/mant=0 byte (0x80), and there
# is no negative zero. gfx950/MI350 uses the OCP fp8 format instead; this codec
# targets the gfx942 default and the OCP path needs separate handling.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _fnuz_decode_table(exp_bits: int, mant_bits: int) -> np.ndarray:
    """Build the 256-entry byte -> fp32 value table for an 8-bit FNUZ float.

    The table is a pure function of (exp_bits, mant_bits), so it is cached; the
    returned array is marked read-only because callers share the one instance.
    """
    bias = (1 << (exp_bits - 1))
    mant_max = 1 << mant_bits
    sign_shift = exp_bits + mant_bits
    exp_mask = (1 << exp_bits) - 1
    table = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = (b >> sign_shift) & 1
        exp = (b >> mant_bits) & exp_mask
        mant = b & (mant_max - 1)
        if exp == 0 and mant == 0:
            # +0 (0x00); the negative-zero slot (0x80) is the lone NaN.
            table[b] = np.float32(np.nan) if sign else np.float32(0.0)
            continue
        if exp == 0:
            val = (mant / mant_max) * (2.0 ** (1 - bias))  # subnormal
        else:
            val = (1.0 + mant / mant_max) * (2.0 ** (exp - bias))  # normal
        table[b] = np.float32(-val if sign else val)
    table.flags.writeable = False  # shared cached instance -- do not mutate
    return table


def _fnuz_encode(x: np.ndarray, exp_bits: int, mant_bits: int) -> np.ndarray:
    """Encode fp32 -> nearest 8-bit FNUZ float, returned as a uint8 bit pattern."""
    table = _fnuz_decode_table(exp_bits, mant_bits)
    sign_byte = np.uint8(1 << (exp_bits + mant_bits))  # 0x80

    # Positive half (bytes 0..127) holds every non-negative magnitude, sorted.
    # Compare in float64: for very large inputs the gap between the two top
    # magnitudes is below fp32 resolution, which would tie and mis-saturate.
    pos_mag = table[: int(sign_byte)].astype(np.float64)
    order = np.argsort(pos_mag)
    sorted_mag = pos_mag[order]
    sorted_byte = order.astype(np.uint8)

    xf = np.ascontiguousarray(x, dtype=np.float32)
    ax = np.abs(xf).astype(np.float64)
    # Both neighbours come from the raw insertion point: raw==size saturates to
    # the top magnitude (lo==hi), raw==0 pins to zero, otherwise compare the two.
    raw = np.searchsorted(sorted_mag, ax)
    hi = np.clip(raw, 0, sorted_mag.size - 1)
    lo = np.clip(raw - 1, 0, sorted_mag.size - 1)
    pick_lo = np.abs(sorted_mag[lo] - ax) <= np.abs(sorted_mag[hi] - ax)
    chosen = np.where(pick_lo, lo, hi)
    out = sorted_byte[chosen]

    # Apply sign, but never the 0x80 (-0 == NaN) slot: zeros stay +0.
    is_zero = sorted_mag[chosen] == 0
    out = np.where((xf < 0) & ~is_zero, out | sign_byte, out)
    out = np.where(np.isnan(xf), sign_byte, out)  # NaN inputs -> NaN byte
    return out.astype(np.uint8).reshape(np.shape(x))


def _fp32_to_fp8_u8(x: np.ndarray) -> np.ndarray:
    """Encode fp32 -> fp8 E4M3 (FNUZ) bit pattern in a uint8 array."""
    return _fnuz_encode(x, exp_bits=4, mant_bits=3)


def _fp8_u8_to_fp32(u8: np.ndarray) -> np.ndarray:
    """Decode an fp8 E4M3 (FNUZ) bit pattern back to fp32."""
    return _fnuz_decode_table(4, 3)[u8.astype(np.intp)]


def _fp32_to_bf8_u8(x: np.ndarray) -> np.ndarray:
    """Encode fp32 -> bf8 E5M2 (FNUZ) bit pattern in a uint8 array."""
    return _fnuz_encode(x, exp_bits=5, mant_bits=2)


def _bf8_u8_to_fp32(u8: np.ndarray) -> np.ndarray:
    """Decode a bf8 E5M2 (FNUZ) bit pattern back to fp32."""
    return _fnuz_decode_table(5, 2)[u8.astype(np.intp)]


# Output (C) element dtype for an A/B element dtype, mirroring the codegen's
# CommonTypeMappings.get_output_dtype: fp8/bf8 accumulate into fp16, int8 into
# int32, everything else stores in its own dtype.
_OUTPUT_DTYPE = {"fp8": "fp16", "bf8": "fp16", "int8": "int32"}


def _output_dtype(dtype: str) -> str:
    return _OUTPUT_DTYPE.get(dtype, dtype)


def _dtype_from_kernel_name(name: str) -> str:
    """Extract the dtype token from a kernel name like ``gemm_<dtype>_<layout>_...``."""
    parts = name.split("_")
    return parts[1] if len(parts) > 1 else "fp16"


def _layout_from_kernel_name(name: str) -> str:
    """Extract the 3-char layout token (e.g. 'rcr') from a kernel name.

    Name format is ``gemm_<dtype>_<layout>_...``; each char is 'r' (row-major)
    or 'c' (column-major) for operands A, B, C respectively.
    """
    parts = name.split("_")
    if len(parts) > 2 and len(parts[2]) == 3 and set(parts[2]) <= {"r", "c"}:
        return parts[2]
    return "rcr"


class GpuGemmRunner:
    """High-level runner: construct from a .so path, call run(A, B, problem).

    The GEMM ctypes ABI takes HOST pointers and manages GPU memory internally
    (hipMalloc/hipMemcpy/hipFree), so this runner stays simple -- it hands
    numpy arrays straight to the .so.
    """

    def __init__(self, lib_path: Path):
        self.lib = GemmDispatcherLib(lib_path)
        if not self.lib.initialize():
            raise RuntimeError(f"Failed to initialize dispatcher .so: {lib_path}")
        names = self.lib.kernel_names
        self._kernel_name = names[0] if names else "unknown"

    @property
    def kernel_name(self) -> str:
        return self._kernel_name

    def run(
        self, A: np.ndarray, B: np.ndarray, problem: GemmProblem
    ) -> GemmResult:
        M, N, K = problem.M, problem.N, problem.K

        # Caller passes logical A (MxK) and B (KxN) row-major. The compiled
        # kernel dictates both the element dtype and the memory layout of each
        # operand (encoded in its name, e.g. gemm_bf16_rcr_...). The C ABI sizes
        # its device buffers from sizeof(ADataType) and the kernel computes
        # strides from its compiled layout + M,N,K -- so the host buffers must
        # be laid out byte-for-byte in the order the kernel expects.
        #
        # For a 'c' (column-major) operand we transpose so the contiguous host
        # buffer's flat memory matches column-major order:
        #   col-major A (MxK)  <=>  ascontiguousarray(A.T)  (KxM row-major)
        # Likewise column-major C (MxN) lands in memory as NxM row-major, so we
        # allocate (N,M) and transpose the result back to logical MxN.
        dtype = _dtype_from_kernel_name(self._kernel_name)
        la, lb, lc = _layout_from_kernel_name(self._kernel_name)

        A_lay = A if la == "r" else A.T
        B_lay = B if lb == "r" else B.T
        C_shape = (M, N) if lc == "r" else (N, M)

        # Build A/B host buffers in the kernel's element dtype. The encode
        # helpers (bf16/fp8/bf8) already force a contiguous float32 source, so an
        # outer ascontiguousarray would only add a redundant copy; the native
        # numpy dtypes (fp16/int8) still need it.
        if dtype == "bf16":
            A_h = _fp32_to_bf16_u16(A_lay)
            B_h = _fp32_to_bf16_u16(B_lay)
        elif dtype == "fp8":
            A_h = _fp32_to_fp8_u8(A_lay)
            B_h = _fp32_to_fp8_u8(B_lay)
        elif dtype == "bf8":
            A_h = _fp32_to_bf8_u8(A_lay)
            B_h = _fp32_to_bf8_u8(B_lay)
        elif dtype == "int8":
            A_h = np.ascontiguousarray(A_lay, dtype=np.int8)
            B_h = np.ascontiguousarray(B_lay, dtype=np.int8)
        else:  # fp16 (default)
            A_h = np.ascontiguousarray(A_lay, dtype=np.float16)
            B_h = np.ascontiguousarray(B_lay, dtype=np.float16)

        # The C buffer's element size must equal sizeof(CDataType): fp8/bf8
        # accumulate into fp16, int8 into int32, otherwise the input dtype.
        out_dtype = _output_dtype(dtype)
        _C_NP = {"fp16": np.float16, "bf16": np.uint16, "int32": np.int32}
        if out_dtype not in _C_NP:
            # A silent fp16 fallback would size the host C buffer wrong for an
            # unrecognized dtype (sizeof(CDataType) mismatch -> corrupt results
            # across the C ABI). Fail loudly so a new dtype is added here.
            raise ValueError(
                f"unsupported C dtype {out_dtype!r} (from input dtype {dtype!r}); "
                "add it to _C_NP so the host buffer matches sizeof(CDataType)"
            )
        C_h = np.zeros(C_shape, dtype=_C_NP[out_dtype])

        status, time_ms = self.lib.run(A_h, B_h, C_h, M, N, K)

        # Decode the output back to a comparable numeric array.
        if out_dtype == "bf16":
            C_dec = _bf16_u16_to_fp32(C_h)
        else:  # fp16 / int32 are already directly comparable
            C_dec = C_h
        C_out = C_dec if lc == "r" else C_dec.T

        tflops = (problem.flops / (time_ms * 1e-3)) / 1e12 if time_ms > 0 else 0.0
        return GemmResult(
            output=C_out,
            time_ms=time_ms,
            status=status,
            tflops=tflops,
            kernel_name=self._kernel_name,
        )


class GpuGroupedGemmRunner:
    """High-level runner for the GROUPED variant: construct from a grouped .so
    path, call run(A_list, B_list, problem).

    Like GpuGemmRunner, the ctypes ABI takes HOST pointers and manages GPU
    memory internally (per group), so this runner only marshals the host operand
    arrays. The runner is parameterized by ``(dtype, layout)`` (mirroring
    ``GpuGemmRunner``/``GemmProblem``): the A/B operands are cast to the per-dtype
    INPUT numpy codec (fp16/bf16/fp8-E4M3FNUZ/bf8-E5M2FNUZ) and transposed per the
    A/B/C layout so the contiguous host buffer matches the layout the kernel was
    generated with (the ctypes lib derives strides from the same layouts).

    The C/output buffer is sized/typed by the kernel's OUTPUT dtype, not the input
    dtype: for fp8/bf8 inputs the generated kernel's ``CDataType`` is fp16, so the
    host C buffer is fp16 (2 bytes) even though A/B are 1-byte fp8/bf8. Sizing C by
    the input dtype would under-allocate by 2x and the ctypes copy-back would
    overrun the host buffer (heap corruption). See :func:`output_numpy_dtype_for`.
    """

    def __init__(self, lib_path: Path, dtype: str = "fp16", layout: str = "rcr"):
        self.lib = GemmDispatcherLib(lib_path)
        if not self.lib.initialize():
            raise RuntimeError(
                f"Failed to initialize grouped dispatcher .so: {lib_path}"
            )
        names = self.lib.kernel_names
        self._kernel_name = names[0] if names else "unknown"
        self._dtype = dtype
        # A/B (input) codec vs C (output) codec: they differ for fp8/bf8
        # (output is fp16), so keep them distinct to size the C buffer correctly.
        self._np_dtype = numpy_dtype_for(dtype)
        self._c_np_dtype = output_numpy_dtype_for(dtype)
        if len(layout) != 3 or any(ch not in ("r", "c") for ch in layout):
            raise ValueError(f"layout must be a 3-char r/c string, got {layout!r}")
        self._layout = layout

    @property
    def kernel_name(self) -> str:
        return self._kernel_name

    def run(
        self,
        A_list: List[np.ndarray],
        B_list: List[np.ndarray],
        problem: GroupedGemmProblem,
    ) -> GroupedGemmResult:
        groups = problem.groups
        if len(A_list) != len(groups) or len(B_list) != len(groups):
            raise ValueError(
                "A_list/B_list length must match the number of groups "
                f"({len(A_list)}/{len(B_list)} vs {len(groups)})"
            )

        Ms = [g[0] for g in groups]
        Ns = [g[1] for g in groups]
        Ks = [g[2] for g in groups]

        la, lb, _lc = self._layout[0], self._layout[1], self._layout[2]
        nd = self._np_dtype
        c_nd = self._c_np_dtype  # OUTPUT dtype (fp16 for fp8/bf8); see __init__.

        A_h: List[np.ndarray] = []
        B_h: List[np.ndarray] = []
        C_h: List[np.ndarray] = []
        for A, B, (M, N, _K) in zip(A_list, B_list, groups):
            # A logically MxK, B logically KxN, C row-major MxN (CLayout is always
            # RowMajor for grouped). Store each operand so its contiguous buffer
            # matches its layout: row-major -> as-is, col-major -> transpose.
            A_buf = A if la == "r" else A.T
            B_buf = B if lb == "r" else B.T
            A_h.append(np.ascontiguousarray(A_buf, dtype=nd))
            B_h.append(np.ascontiguousarray(B_buf, dtype=nd))
            # Size C by the kernel's CDataType (output dtype), NOT the input dtype:
            # fp8/bf8 inputs produce fp16 output, so a 1-byte C would be overrun.
            C_h.append(np.zeros((M, N), dtype=c_nd))

        status, time_ms = self.lib.run_grouped(A_h, B_h, C_h, Ms, Ns, Ks)

        tflops = (problem.flops / (time_ms * 1e-3)) / 1e12 if time_ms > 0 else 0.0
        return GroupedGemmResult(
            outputs=C_h,
            time_ms=time_ms,
            status=status,
            tflops=tflops,
            kernel_name=self._kernel_name,
        )


# ============================================================================
# Build API: codegen + hipcc -> .so paths (no GPU)
# ============================================================================

# AMDGPU codegen flags Tile Engine passes to hipcc for GEMM kernels. These MUST
# match, flag-for-flag, the set the Tile Engine gemm_universal benchmark TU is
# compiled with (projects/composablekernel/CMakeLists.txt) -- they steer inlining
# and register allocation, and because persistent kernels size their grid by
# occupancy, any mismatch produces large perf gaps vs Tile Engine and makes the
# parity comparison no longer apples-to-apples.
#
# Tile Engine's actual GEMM benchmark flags (verified from its compile_commands):
#     -fno-offload-uniform-block
#     -mllvm --lsr-drop-solution=1
#     -mllvm -enable-post-misched=0
#     -mllvm -amdgpu-early-inline-all=true
#     -mllvm -amdgpu-function-calls=false
#     -mllvm -amdgpu-coerce-illegal-types=1   (CMake adds this only when the
#                                              compiler accepts it; see below)
# NOTE: -enable-noalias-to-md-conversion=0 is NOT a Tile Engine GEMM flag (it only
# appears in the standalone CK examples/tests), so it deliberately is NOT here.
_TILE_ENGINE_CODEGEN_FLAGS = (
    "-mllvm", "--lsr-drop-solution=1",
    "-mllvm", "-enable-post-misched=0",
    "-mllvm", "-amdgpu-early-inline-all=true",
    "-mllvm", "-amdgpu-function-calls=false",
    "-fno-offload-uniform-block",
)

# Flags Tile Engine's CMake only adds when ``check_cxx_compiler_flag`` passes
# (newer -mllvm options that some clang builds reject). We mirror that probe so
# the bridge stays matched to Tile Engine on every toolchain: the flag is present
# exactly where TE would have it, and absent where TE's CMake would also skip it.
_PROBED_CODEGEN_FLAGS = (
    ("-mllvm", "-amdgpu-coerce-illegal-types=1"),
)

# The single hipcc used for BOTH the flag-acceptance probe and the actual
# compile/link. Pinned to match Old-TE, which builds via CMake's
# CMAKE_CXX_COMPILER (== /opt/rocm/bin/hipcc in CK CI) and never reads $HIPCC;
# ctypes_utils uses the same path. Keeping probe == compiler guarantees the
# -mllvm flag decision reflects the compiler that actually builds the kernel.
_HIPCC = "/opt/rocm/bin/hipcc"


def _resolve_hipcc() -> str:
    return _HIPCC


@functools.lru_cache(maxsize=None)
def _hipcc_accepts(flag_tuple: Tuple[str, ...]) -> bool:
    """Mirror CMake check_cxx_compiler_flag: does hipcc compile a trivial TU with
    these flags? Cached so the probe runs at most once per distinct flag set."""
    hipcc = _resolve_hipcc()
    try:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "probe.cpp"
            src.write_text("int main(){}\n")
            r = subprocess.run(
                [hipcc, *flag_tuple, "-c", str(src), "-o", str(Path(d) / "probe.o")],
                capture_output=True, timeout=120,
            )
            return r.returncode == 0
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _tile_engine_codegen_flags() -> Tuple[str, ...]:
    """Tile Engine's GEMM codegen flags plus any probe-gated flags the compiler
    accepts -- the exact backend flag set the TE benchmark is built with."""
    flags = list(_TILE_ENGINE_CODEGEN_FLAGS)
    for pair in _PROBED_CODEGEN_FLAGS:
        if _hipcc_accepts(pair):
            flags = list(pair) + flags
    return tuple(flags)


def _ctypes_source_name(config: GemmKernelConfig) -> str:
    """Pick the ctypes ABI source for a config's variant.

    The grouped kernel has a multi-problem launch signature that the
    single-problem ``gemm_ctypes_lib.cpp`` cannot express, so grouped configs
    compile against the dedicated ``grouped_gemm_ctypes_lib.cpp``.
    """
    if config.variant == "grouped":
        return "grouped_gemm_ctypes_lib.cpp"
    return "gemm_ctypes_lib.cpp"


def _build_compile_jobs(
    config: GemmKernelConfig, header: Path
) -> Tuple[Dict[str, Any], Path]:
    """Replicate the (validated) compile+link commands from ctypes_utils."""
    root = _cu.get_dispatcher_root()
    ck_root = root.parent
    build_dir = _cu.get_build_dir()
    output_dir = _cu.get_generated_kernels_dir()
    ctypes_source = root / "bindings" / "ctypes" / _ctypes_source_name(config)
    static_lib = build_dir / "libck_tile_dispatcher.a"

    lib_path = build_dir / "examples" / f"lib{config.name}.so"
    obj_file = lib_path.with_suffix(".o")

    compile_cmd = [
        _resolve_hipcc(),
        "-c",
        "-fPIC",
        "-O3",
        f"-I{root / 'include'}",
        f"-I{ck_root / 'include'}",
        f"-I{ck_root}",
        f"-I{str(output_dir)}",
        "-DCK_TILE_SINGLE_KERNEL_INCLUDE",
        f"-include{header}",
        "-D__HIP_PLATFORM_AMD__",
        f"--offload-arch={config.gfx_arch}",
        f'-DGFX_ARCH="{config.gfx_arch}"',
        # Match Tile Engine's AMDGPU codegen flags exactly (see
        # _tile_engine_codegen_flags). Without them the kernel is compiled with
        # different inlining/register allocation, which changes occupancy;
        # persistent kernels size their grid by occupancy
        # (UniversalGemmKernel::MaxOccupancyGridSize = #CUs x occupancy), so a
        # mismatch shows up as large perf gaps vs Tile Engine on persistent tiles.
        *_tile_engine_codegen_flags(),
        "-Wno-undefined-func-template",
        "-Wno-float-equal",
        str(ctypes_source),
        "-o",
        str(obj_file),
    ]
    link_cmd = [
        _resolve_hipcc(),
        "-shared",
        "-fPIC",
        f"--offload-arch={config.gfx_arch}",
        "--hip-link",
        str(obj_file),
        str(static_lib),
        "-o",
        str(lib_path),
    ]
    job = {"compile_cmd": compile_cmd, "link_cmd": link_cmd, "lib_path": str(lib_path)}
    return job, lib_path


def setup_multiple_gemm_dispatchers(
    configs: List[GemmKernelConfig],
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """Codegen + compile each config into its own .so. Returns .so paths.

    This is the build half of the bridge. It touches NO GPU -- pure CPU
    codegen + hipcc, run massively in parallel -- and returns only ``Path``
    objects (``None`` for configs that failed to generate/compile), aligned to
    the input order. Benchmarking happens later, in an isolated worker.
    """
    import sys

    n = len(configs)
    results: List[Optional[Path]] = [None] * n
    if n == 0:
        return results

    max_workers = max_workers or min(multiprocessing.cpu_count(), 8)

    # Dedupe identical configs by name; compile once, share the path.
    first_index: Dict[str, int] = {}
    unique: List[int] = []
    for i, c in enumerate(configs):
        key = c.name
        if key not in first_index:
            first_index[key] = i
            unique.append(i)

    codegen_script = _cu.get_codegen_path()
    output_dir = _cu.get_generated_kernels_dir()
    static_lib = _cu.get_build_dir() / "libck_tile_dispatcher.a"
    ctypes_dir = _cu.get_dispatcher_root() / "bindings" / "ctypes"
    needed_sources = {ctypes_dir / _ctypes_source_name(c) for c in configs}
    missing = [str(p) for p in needed_sources if not p.exists()]
    if not static_lib.exists() or missing:
        raise FileNotFoundError(
            "Missing static lib or ctypes source required for compilation:\n"
            f"  {static_lib}\n  " + "\n  ".join(missing) + "\n"
            "Build the dispatcher first (cmake + make)."
        )

    # -- Step 1: parallel codegen (one header per unique config) -----------
    codegen_args = []
    for i in unique:
        c = configs[i]
        codegen_args.append(
            {
                "index": i,
                "python": sys.executable,
                "codegen_script": str(codegen_script),
                "output_dir": str(output_dir),
                "dtype": c.dtype_a,
                "layout": c.layout,
                "gpu_target": c.gfx_arch,
                "tile_config_json": c.to_codegen_json(),
                "hpp_glob_pattern": f"{c.name}.hpp",
                # Honor the config's variant so non-standard kernels are codegen'd
                # as themselves; the kernel name (and thus hpp_glob_pattern) already
                # carries the variant suffix, so a missing/standard value here would
                # produce a header whose name never matches the requested pattern.
                "variant": c.variant,
            }
        )

    if verbose:
        print(
            f"[gemm-bridge] codegen: {len(codegen_args)} headers "
            f"(workers={max_workers})..."
        )

    headers: Dict[int, Path] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_cu._generate_single_kernel_subprocess, a): a["index"]
            for a in codegen_args
        }
        for fut in as_completed(futs):
            i = futs[fut]
            ok, hdr, err = fut.result()
            if ok and hdr:
                headers[i] = Path(hdr)
                if verbose:
                    print(f"  OK  codegen [{i}] {configs[i].name}")
            elif verbose:
                print(f"  FAIL codegen [{i}] {configs[i].name}: {err}")

    # -- Step 2: parallel compile + link -----------------------------------
    compile_jobs = []
    job_index: List[int] = []
    for i in unique:
        hdr = headers.get(i)
        if hdr is None:
            continue
        job, _ = _build_compile_jobs(configs[i], hdr)
        compile_jobs.append(job)
        job_index.append(i)

    if verbose and compile_jobs:
        print(
            f"[gemm-bridge] compile: {len(compile_jobs)} .so "
            f"(workers={max_workers})..."
        )

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_cu._run_hipcc_subprocess, job): job_index[j]
            for j, job in enumerate(compile_jobs)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            ok, lp, err = fut.result()
            if ok and lp:
                results[i] = Path(lp)
                if verbose:
                    print(f"  OK  compile [{i}] {Path(lp).name}")
            elif verbose:
                print(f"  FAIL compile [{i}] {configs[i].name}: {err}")

    # -- Fan the deduped result back out to every input index --------------
    for i, c in enumerate(configs):
        if results[i] is None:
            results[i] = results[first_index[c.name]]

    if verbose:
        ok_count = sum(1 for r in results if r is not None)
        print(f"[gemm-bridge] setup complete: {ok_count}/{n} configs -> .so")

    return results


# ============================================================================
# TE sweep config expansion
# ============================================================================


def _expand_range(entry: Dict[str, Any]) -> List[int]:
    """Expand a tile_config entry: either {min,max,step} or {values:[...]}."""
    if "values" in entry:
        return list(entry["values"])
    lo = int(entry["min"])
    hi = int(entry["max"])
    step = int(entry.get("step", 1))
    return list(range(lo, hi + 1, step))


def _expand_values(entry: Optional[Dict[str, Any]], default: List[Any]) -> List[Any]:
    if entry is None:
        return list(default)
    return list(entry.get("values", default))


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def expand_sweep(
    config_path: str,
    arch: str,
    dtype: str = "fp16",
    layout: str = "rcr",
    variant: str = "standard",
) -> List[GemmKernelConfig]:
    """Expand a Tile Engine GEMM JSON sweep config into GemmKernelConfig list.

    The TE config uses ``tile_config`` (ranges/value-lists for tile, warp and
    warp_tile triples) and ``trait_config`` (value-lists for pipeline,
    scheduler, epilogue, pad_*, persistent). Every valid combination becomes
    one GemmKernelConfig. Invalid combinations are dropped via the dispatcher's
    own validator, and duplicates (by .name) are collapsed.

    The operand signature (``dtype``, ``layout``) is applied to every emitted
    GemmKernelConfig, so the same sweep expands across any supported dtype/layout.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    tc = cfg.get("tile_config", {})
    tr = cfg.get("trait_config", {})

    tile_ms = _expand_range(tc["tile_m"])
    tile_ns = _expand_range(tc["tile_n"])
    tile_ks = _expand_range(tc["tile_k"])
    wave_ms = _expand_range(tc["warp_m"])  # TE "warp" == wave count
    wave_ns = _expand_range(tc["warp_n"])
    wave_ks = _expand_range(tc["warp_k"])
    wt_ms = _expand_range(tc["warp_tile_m"])
    wt_ns = _expand_range(tc["warp_tile_n"])
    wt_ks = _expand_range(tc["warp_tile_k"])

    pipelines = _expand_values(tr.get("pipeline"), ["compv3"])
    schedulers = _expand_values(tr.get("scheduler"), ["intrawave"])
    epilogues = _expand_values(tr.get("epilogue"), ["cshuffle"])
    pad_ms = _expand_values(tr.get("pad_m"), [False])
    pad_ns = _expand_values(tr.get("pad_n"), [False])
    pad_ks = _expand_values(tr.get("pad_k"), [False])
    persistents = _expand_values(tr.get("persistent"), [False])

    la, lb, lc = layout[0], layout[1], layout[2]

    configs: List[GemmKernelConfig] = []
    seen: set = set()
    for (
        tm,
        tn,
        tk,
        wm,
        wn,
        wk,
        wtm,
        wtn,
        wtk,
        pipe,
        sched,
        epi,
        pm,
        pn,
        pk,
        persist,
    ) in itertools.product(
        tile_ms,
        tile_ns,
        tile_ks,
        wave_ms,
        wave_ns,
        wave_ks,
        wt_ms,
        wt_ns,
        wt_ks,
        pipelines,
        schedulers,
        epilogues,
        pad_ms,
        pad_ns,
        pad_ks,
        persistents,
    ):
        c = GemmKernelConfig(
            dtype_a=dtype,
            dtype_b=dtype,
            dtype_c=_output_dtype(dtype),
            dtype_acc=("int32" if dtype == "int8" else "fp32"),
            layout_a=_LAYOUT_WORD[la],
            layout_b=_LAYOUT_WORD[lb],
            layout_c=_LAYOUT_WORD[lc],
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            wave_m=wm,
            wave_n=wn,
            wave_k=wk,
            warp_tile_m=wtm,
            warp_tile_n=wtn,
            warp_tile_k=wtk,
            pipeline=pipe,
            scheduler=sched,
            epilogue=epi,
            pad_m=bool(pm),
            pad_n=bool(pn),
            pad_k=bool(pk),
            persistent=bool(persist),
            gfx_arch=arch,
            variant=variant,
        )
        if c.name in seen:
            continue
        val = _cu.validate_kernel_config(c.to_ctypes_config())
        if not val.is_valid:
            continue
        # Tile/CShuffle correctness gate (mirrors unified_gemm_codegen's
        # TileConfig.is_valid + the power-of-two repeat rule; the ctypes
        # validate_kernel_config above does NOT enforce either). A block tile must
        # split evenly across its waves -- tile % (wave * warp_tile) == 0 -- and
        # the CShuffle epilogue stores the accumulator through LDS in power-of-two
        # MRepeat/NRepeat chunks, so the per-wave repeat must be a power of two.
        # Tiles that violate either still compile but produce numerically WRONG
        # results at runtime. Observed on MI350 for tile_m=192 (MRepeat=3) and
        # tile_n=192 (e.g. 64x192x64_1x4x1, 192 not divisible by 4*32) -- both
        # verified incorrect on the bridge and Tile Engine. Power-of-two tiles
        # (64/128/256) are unaffected.
        m_div = wm * wtm
        n_div = wn * wtn
        if m_div <= 0 or n_div <= 0 or tm % m_div != 0 or tn % n_div != 0:
            continue
        if epi == "cshuffle":
            if not _is_power_of_two(tm // m_div) or not _is_power_of_two(tn // n_div):
                continue
        seen.add(c.name)
        configs.append(c)

    return configs
