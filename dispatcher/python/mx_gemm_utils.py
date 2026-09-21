#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
MX-GEMM dispatcher utilities (TileEngine -> Dispatcher bridge).

Three-layer Python bridge for the dispatcher's microscaling-GEMM path
(fp4/fp8 A.B with per-32-K e8m0 block scales, gfx950/MI350 only):

  MxGemmKernelConfig       -- describes one kernel; .name is byte-exact with the
                              codegen KERNEL_NAME (obtained by shelling the codegen
                              CLI with --list-name so utils and codegen never drift)
  MxGemmDispatcherLib      -- thin ctypes wrapper around a compiled .so
  GpuMxGemmRunner          -- high-level runner: generates quantized inputs, calls the
                              kernel, and provides a numpy microscaled reference

Build helper (self-contained):
  setup_multiple_mx_gemm_dispatchers(configs, ...) : codegen -> hipcc -> .so paths

Data types (verified against ck_tile headers / example/ck_tile/42_mx_gemm):
  fp8 : ck_tile::fp8_t  == float8_e4m3_t (OCP e4m3, bias 7, on gfx950 device)
  fp4 : ck_tile::pk_fp4_t == pk_float4_e2m1_t (two e2m1 values packed per byte;
        low nibble = even-K element, high nibble = odd-K element)
  C   : ck_tile::fp16_t
  scale: ck_tile::e8m0_t (biased exponent, byte e decodes to 2^(e-127); 127 == 1.0)
"""

import concurrent.futures
import ctypes
import functools
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_mx_gemm_codegen.py"
_CTYPES_LIB_SRC = (
    Path(__file__).parent.parent / "bindings" / "ctypes" / "mx_gemm_ctypes_lib.cpp"
)
# ck root == the composablekernel dir (three levels up from dispatcher/python).
_CK_ROOT = Path(__file__).parent.parent.parent
_HIPCC = os.environ.get("CK_TILE_HIPCC", "/opt/rocm/bin/hipcc")


def _get_arch() -> str:
    """Detect GPU arch via rocminfo and validate; raise on failure. Never defaults."""
    import subprocess
    arch = ""
    try:
        out = subprocess.check_output(["rocminfo"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "Name:" in line and "gfx" in line:
                arch = line.split()[-1].strip(); break
    except Exception:
        arch = ""
    if not arch:
        raise RuntimeError("Could not detect GPU architecture from rocminfo; refusing to default. Pass gfx_arch explicitly.")
    # mx_gemm is gfx950-ONLY: the C++ bridge (mx_gemm_ctypes_lib.cpp) has a
    # static_assert(GFX_ARCH == "gfx950") because it uses the gfx950-only
    # preShuffleScaleBuffer_gfx950 host helper. Validating a broader set here
    # would let a gfx942/gfx90a caller past detection only to fail later at
    # build/runtime with a less actionable error, so restrict it to gfx950.
    _supported = ("gfx950",)
    if arch not in _supported:
        raise ValueError(
            f"mx_gemm is gfx950-only; detected {arch!r} is not supported "
            f"(supported: {list(_supported)})"
        )
    return arch

# MX GEMM scales every 32 K-elements with one e8m0 byte.
SCALE_BLOCK = 32

# e8m0 byte for scale == 1.0 (2^(127-127)).
E8M0_ONE = 127

# =============================================================================
# fp4 (e2m1) exact value grid, byte codes match pk_fp4 e2m1_to_fp32_table.
# index -> value ;  index is the 4-bit e2m1 code.
# =============================================================================
_FP4_E2M1_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
# value -> 4-bit code (skip -0.0; +0.0 code 0 covers zero).
_FP4_VALUE_TO_CODE = {
    0.0: 0, 0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4, 3.0: 5, 4.0: 6, 6.0: 7,
    -0.5: 9, -1.0: 10, -1.5: 11, -2.0: 12, -3.0: 13, -4.0: 14, -6.0: 15,
}

# =============================================================================
# fp8 (e4m3, OCP) raw-byte codec for the exact grid we test on.
#
# PRECONDITION (hard): the byte codes below are OCP e4m3 (bias 7). They are ONLY
# a valid reference for a kernel compiled with CK_TILE_USE_OCP_FP8 == 1. This is
# the gfx950 default (see include/ck_tile/core/config.hpp: OCP is selected for
# __gfx950__ / __gfx12__ and the bridge compiles the lib with
# --offload-arch=gfx950 without overriding the flag). On any arch where CK
# defaults to (or is built for) FNUZ e4m3 (bias 8), the DEVICE bytes for the same
# float value differ, so this numpy reference would SILENTLY disagree with the
# kernel. Callers that target a non-OCP arch must not use this codec.
#
# The grid values are exactly representable in BOTH OCP and FNUZ, but their
# RAW BYTES are not the same between the two formats, which is why the codec is
# OCP-specific. Bytes verified against pk_fp4 e2m1_to_fp8_table (OCP branch) +
# hand-derived grid entries (sign|exp4|mant3, bias 7).
# =============================================================================
_FP8_OCP_VALUE_TO_BYTE = {
    0.0: 0x00, 0.5: 0x30, 1.0: 0x38, 1.5: 0x3C, 2.0: 0x40, 2.5: 0x42,
    3.0: 0x44, 4.0: 0x48, 6.0: 0x4C,
    -0.5: 0xB0, -1.0: 0xB8, -1.5: 0xBC, -2.0: 0xC0, -2.5: 0xC2,
    -3.0: 0xC4, -4.0: 0xC8, -6.0: 0xCC,
}
_FP8_OCP_BYTE_TO_VALUE = {b: v for v, b in _FP8_OCP_VALUE_TO_BYTE.items()}
# 256-entry byte -> value LUT for vectorized dequantize_fp8. Untested bytes stay
# NaN (they cannot come from quantize_fp8, so a NaN result flags a foreign byte).
_FP8_OCP_BYTE_LUT = np.full(256, np.nan, dtype=np.float32)
for _b, _v in _FP8_OCP_BYTE_TO_VALUE.items():
    _FP8_OCP_BYTE_LUT[_b] = np.float32(_v)


def fp8_ocp_is_default_for_arch(arch: str) -> bool:
    """True iff CK_TILE_USE_OCP_FP8 defaults to 1 for `arch`.

    Mirrors include/ck_tile/core/config.hpp: OCP e4m3 is the device default only
    for gfx950 and gfx12; every other arch defaults to FNUZ e4m3. The bridge
    compiles the mx_gemm lib without an explicit -DCK_TILE_USE_OCP_FP8, so the
    effective fp8 format is exactly this arch default. Use this to guard the fp8
    numpy reference (see assert_fp8_ocp_supported / quantize_fp8).
    """
    a = (arch or "").lower()
    return a.startswith("gfx950") or a.startswith("gfx12")


def assert_fp8_ocp_supported(arch: Optional[str] = None) -> None:
    """Fail loudly if the fp8 OCP codec would silently disagree with the device.

    The fp8 quantize/dequantize helpers emit/decode OCP e4m3 bytes; that is only
    correct when the kernel is compiled with CK_TILE_USE_OCP_FP8 == 1 (see the
    codec precondition above). Raise instead of producing bytes that mismatch a
    FNUZ-built kernel.
    """
    arch = arch or _get_arch()
    if not fp8_ocp_is_default_for_arch(arch):
        raise ValueError(
            f"fp8 mx_gemm reference is OCP e4m3 only, but arch '{arch}' defaults to "
            "FNUZ e4m3 (CK_TILE_USE_OCP_FP8 == 0). The bridge lib is compiled without "
            "an explicit -DCK_TILE_USE_OCP_FP8, so its device fp8 bytes would not match "
            "this numpy reference. OCP fp8 is supported for gfx950/gfx12 only."
        )


def e8m0_to_float(byte) -> np.ndarray:
    """Decode e8m0 bytes to float: 2^(e-127); e==255 is NaN."""
    b = np.asarray(byte, dtype=np.int32)
    out = np.exp2((b - 127).astype(np.float32))
    out = np.where(b == 255, np.nan, out)
    return out.astype(np.float32)


def float_to_e8m0(scale) -> np.ndarray:
    """Encode a strictly-positive power-of-two float scale to an e8m0 byte.

    The e8m0 format only encodes 2^(e-127) for e in [0, 254] (255 == NaN), so
    the contract is a positive, finite input. A non-positive or non-finite value
    is a caller bug (e.g. a zeroed/garbage scale); fail loudly rather than
    silently substituting 1.0, which would emit a wrong scale byte.
    """
    s = np.asarray(scale, dtype=np.float32)
    if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError(
            "float_to_e8m0 requires strictly positive, finite scales "
            "(power-of-two); got a non-positive or non-finite value"
        )
    e = np.rint(np.log2(s)).astype(np.int32) + 127
    e = np.clip(e, 0, 254).astype(np.uint8)
    return e


# =============================================================================
# Config
# =============================================================================


@dataclass
class MxGemmKernelConfig:
    datatype: str = "fp8"  # fp8 | fp4
    layout: str = "rcr"    # a/b/c ; only rcr supported by mx_gemm
    gpu_target: Optional[str] = None
    pipeline: str = "comp_async"
    epilogue: str = "cshuffle"
    scheduler: str = "intrawave"
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False
    persistent: bool = False

    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 128
    warp_m: int = 2
    warp_n: int = 2
    warp_k: int = 1
    warp_tile_m: int = 16
    warp_tile_n: int = 16
    warp_tile_k: int = 128

    k_block_per_cu: int = 1

    # cached name so we do not re-shell the codegen repeatedly.
    _name_cache: Optional[str] = field(default=None, repr=False, compare=False)

    def to_codegen_config(self) -> dict:
        return {
            "datatype": self.datatype,
            "layout": self.layout,
            "gpu_target": self.gpu_target or _get_arch(),
            "pipeline": self.pipeline,
            "epilogue": self.epilogue,
            "scheduler": self.scheduler,
            "pad_m": self.pad_m,
            "pad_n": self.pad_n,
            "pad_k": self.pad_k,
            "persistent": self.persistent,
            "tile_config": {
                "tile_m": self.tile_m, "tile_n": self.tile_n, "tile_k": self.tile_k,
                "warp_m": self.warp_m, "warp_n": self.warp_n, "warp_k": self.warp_k,
                "warp_tile_m": self.warp_tile_m, "warp_tile_n": self.warp_tile_n,
                "warp_tile_k": self.warp_tile_k,
            },
            "k_block_per_cu": self.k_block_per_cu,
        }

    @property
    def name(self) -> str:
        """Byte-exact kernel name from the codegen CLI (--list-name).

        The codegen owns the canonical name format; we never reconstruct it here
        to avoid drift. Falls back to a locally computed name only if the codegen
        script is not yet present (so utils stays import/unit testable before the
        codegen component lands).
        """
        if self._name_cache is not None:
            return self._name_cache
        if _CODEGEN_SCRIPT.exists():
            try:
                r = subprocess.run(
                    [sys.executable, str(_CODEGEN_SCRIPT),
                     "--output-dir", tempfile.gettempdir(),
                     "--config-json", json.dumps(self.to_codegen_config()),
                     "--list-name"],
                    capture_output=True, text=True, timeout=120,
                )
                if r.returncode == 0 and r.stdout.strip():
                    self._name_cache = r.stdout.strip().splitlines()[-1].strip()
                    return self._name_cache
                log.warning("codegen --list-name failed for %s:\n%s",
                            self._fallback_name(), r.stderr[-800:])
            except Exception as exc:  # noqa: BLE001
                log.warning("codegen --list-name error: %s", exc)
        self._name_cache = self._fallback_name()
        return self._name_cache

    def _fallback_name(self) -> str:
        """Local reconstruction mirroring the documented codegen name format.

        Format (per contract):
          mx_gemm_{dtype}_{layout}_comp_async_cshuffle_intrawave_
          {PadM}_{PadN}_{PadK}[_{Persistent}]_
          {tileM}x{tileN}x{tileK}_{warpM}x{warpN}x{warpK}_{wtM}x{wtN}x{wtK}
        Only used when the codegen script is absent (never at real build time).
        """
        parts = [
            "mx_gemm", self.datatype, self.layout,
            self.pipeline, self.epilogue, self.scheduler,
            str(self.pad_m), str(self.pad_n), str(self.pad_k),
        ]
        if self.persistent:
            parts.append(str(self.persistent))
        parts.append(f"{self.tile_m}x{self.tile_n}x{self.tile_k}")
        parts.append(f"{self.warp_m}x{self.warp_n}x{self.warp_k}")
        parts.append(f"{self.warp_tile_m}x{self.warp_tile_n}x{self.warp_tile_k}")
        return "_".join(parts)

    def is_valid(self) -> bool:
        if self.layout != "rcr":
            return False
        if self.datatype not in ("fp8", "fp4"):
            return False
        if not (
            self.tile_m % (self.warp_m * self.warp_tile_m) == 0
            and self.tile_n % (self.warp_n * self.warp_tile_n) == 0
            and self.tile_k % (self.warp_k * self.warp_tile_k) == 0
        ):
            return False
        # MX XDL uses 16x16x128 warp tiles on gfx950.
        return (self.warp_tile_m, self.warp_tile_n, self.warp_tile_k) == (16, 16, 128)


# =============================================================================
# Problem / result
# =============================================================================


@dataclass
class MxGemmProblem:
    M: int
    N: int
    K: int
    k_batch: int = 1

    def __post_init__(self):
        if self.K % SCALE_BLOCK != 0:
            raise ValueError(f"MX GEMM requires K % {SCALE_BLOCK} == 0, got K={self.K}")

    @property
    def scale_k(self) -> int:
        return self.K // SCALE_BLOCK

    @property
    def flops(self) -> int:
        return 2 * self.M * self.N * self.K


@dataclass
class MxGemmResult:
    C: object
    time_ms: float
    kernel_name: str


# =============================================================================
# ctypes wrapper
# =============================================================================


class MxGemmDispatcherLib:
    def __init__(self, so_path: Path, dtype: Optional[str] = None, arch: Optional[str] = None):
        self.so_path = Path(so_path)
        if not self.so_path.exists():
            raise FileNotFoundError(f"mx_gemm .so not found: {self.so_path}")
        self._lib = ctypes.CDLL(str(self.so_path))
        self._dtype = dtype
        self._arch = arch
        self._setup()
        # Contract exposes both dispatcher_initialize() and dispatcher_init().
        init = getattr(self._lib, "dispatcher_initialize", None)
        if init is None:
            init = self._lib.dispatcher_init
        if init() != 0:
            raise RuntimeError("dispatcher_initialize failed")

    def _setup(self):
        lib = self._lib
        for fn in ("dispatcher_initialize", "dispatcher_init"):
            f = getattr(lib, fn, None)
            if f is not None:
                f.restype = ctypes.c_int
        lib.dispatcher_get_kernel_name.restype = ctypes.c_char_p
        if hasattr(lib, "dispatcher_get_kernel_count"):
            lib.dispatcher_get_kernel_count.restype = ctypes.c_int
        lib.dispatcher_cleanup.restype = None
        lib.dispatcher_run_mx_gemm.restype = ctypes.c_int
        lib.dispatcher_run_mx_gemm.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # B
            ctypes.c_void_p,  # C
            ctypes.POINTER(ctypes.c_uint8),  # scale_a
            ctypes.POINTER(ctypes.c_uint8),  # scale_b
            ctypes.c_int,  # M
            ctypes.c_int,  # N
            ctypes.c_int,  # K
            ctypes.c_int,  # k_batch
            ctypes.POINTER(ctypes.c_float),  # time_ms
        ]

    def kernel_name(self) -> str:
        raw = self._lib.dispatcher_get_kernel_name()
        return raw.decode() if raw else ""

    def kernel_count(self) -> int:
        if hasattr(self._lib, "dispatcher_get_kernel_count"):
            return int(self._lib.dispatcher_get_kernel_count())
        return 1

    def run(self, A, B, C, scale_a, scale_b, prob: MxGemmProblem) -> float:
        # Guard: fp8 OCP bytes only match the device on gfx950/gfx12.  If the lib
        # was constructed with dtype="fp8", fail loudly now rather than silently
        # diverging from the numpy reference on an FNUZ arch.
        if self._dtype == "fp8":
            assert_fp8_ocp_supported(self._arch or _get_arch())
        A = np.ascontiguousarray(A)
        B = np.ascontiguousarray(B)
        # C is the OUTPUT buffer: the kernel writes results into it in place via a
        # raw pointer. ascontiguousarray() would silently redirect the write into a
        # throwaway copy and leave the caller's array unchanged -- a surprising bug
        # for an output API. Require a contiguous, writeable ndarray instead.
        if not isinstance(C, np.ndarray):
            raise TypeError(f"C output must be a numpy.ndarray, got {type(C).__name__}")
        if not C.flags["C_CONTIGUOUS"]:
            raise ValueError("C output must be C-contiguous (results are written in place)")
        if not C.flags["WRITEABLE"]:
            raise ValueError("C output must be writeable (results are written in place)")
        sa = np.ascontiguousarray(scale_a, dtype=np.uint8)
        sb = np.ascontiguousarray(scale_b, dtype=np.uint8)
        tms = ctypes.c_float(0.0)
        rc = self._lib.dispatcher_run_mx_gemm(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            sa.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            sb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(prob.M), int(prob.N), int(prob.K), int(prob.k_batch),
            ctypes.byref(tms),
        )
        if rc != 0:
            raise RuntimeError(f"dispatcher_run_mx_gemm rc={rc} "
                               f"({'unsupported' if rc == -2 else 'error'})")
        return tms.value

    def cleanup(self):
        try:
            self._lib.dispatcher_cleanup()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self):
        self.cleanup()


# =============================================================================
# Quantization helpers (host-side; produce the exact bytes the kernel expects)
# =============================================================================


# Tolerance for on-grid membership in _map_grid_to_bytes. The quantization grid
# entries are all exactly representable in float32 (0, +/-0.5, +/-1.0, ...), so
# grid-exact inputs match to the bit; this epsilon only absorbs the last few ULPs
# of float-repr noise (e.g. a value stored via float64 -> float32). Any input
# farther than this from the nearest grid point is a genuine off-grid value and
# is rejected. Values like 0.5004 (0.0004 off the 0.5 grid point) exceed this and
# now raise, matching the documented "rejects off-grid values" contract.
_GRID_MATCH_EPS = np.float32(1e-4)


def _map_grid_to_bytes(vals: np.ndarray, value_to_byte: dict, grid_name: str) -> np.ndarray:
    """Vectorized exact-grid float -> byte code lookup (uint8, flattened).

    The inputs are drawn from a small fixed grid (see make_inputs), so instead of
    a per-element Python loop -- which dominates runtime for realistic M*K -- we
    snap each input to the nearest grid point via a single searchsorted and index
    a sorted LUT. An input is accepted only if it lies within _GRID_MATCH_EPS
    (1e-4) of that nearest grid point; anything farther off (e.g. 0.5004) is a
    caller contract violation and raises (mirrors the old dict-KeyError) rather
    than silently snapping to a neighbour.
    """
    keys = np.array(sorted(value_to_byte), dtype=np.float32)
    byts = np.array([value_to_byte[float(k)] for k in keys], dtype=np.uint8)

    v = np.asarray(vals, dtype=np.float32) + np.float32(0.0)  # collapse -0.0 -> +0.0
    flat = v.reshape(-1)

    # Nearest grid key for each value: searchsorted gives the insertion point;
    # the closer of the two straddling keys is the nearest grid point.
    pos = np.clip(np.searchsorted(keys, flat), 1, keys.size - 1)
    left = keys[pos - 1]
    right = keys[pos]
    nearest = np.where(np.abs(flat - left) <= np.abs(flat - right), left, right)

    if not np.all(np.abs(flat - nearest) <= _GRID_MATCH_EPS):
        off = np.abs(flat - nearest) > _GRID_MATCH_EPS
        bad = flat[off]
        raise KeyError(
            f"{grid_name}: value(s) not on the exact quantization grid "
            f"(tolerance {float(_GRID_MATCH_EPS):g}): {np.unique(bad)[:8].tolist()}"
        )
    idx = np.searchsorted(keys, nearest)
    return byts[idx]


def quantize_fp8(vals: np.ndarray) -> np.ndarray:
    """Grid float values -> raw OCP e4m3 bytes (uint8), one per element.

    Values MUST come from the exact e4m3 grid (see _FP8_OCP_VALUE_TO_BYTE) so the
    mapping is lossless; that keeps the numpy reference unambiguous.

    PRECONDITION: OCP e4m3 only. These bytes match the device only for a kernel
    compiled with CK_TILE_USE_OCP_FP8 == 1 (gfx950/gfx12 default). See the codec
    comment above and assert_fp8_ocp_supported(); GpuMxGemmRunner enforces this.
    """
    return _map_grid_to_bytes(vals, _FP8_OCP_VALUE_TO_BYTE, "fp8 e4m3").reshape(vals.shape)


def dequantize_fp8(bytes_arr: np.ndarray) -> np.ndarray:
    """Raw OCP e4m3 bytes (uint8) -> float grid values, preserving shape.

    Vectorized LUT-gather (consistent with dequantize_fp4_packed) instead of a
    per-element Python loop, so this stays fast if ever applied to large buffers.
    Bytes outside the tested grid map to NaN (they cannot appear from quantize_fp8,
    which only emits grid codes, so a NaN here flags a corrupt/foreign buffer).
    """
    b = np.asarray(bytes_arr, dtype=np.uint8)
    return _FP8_OCP_BYTE_LUT[b]


def quantize_fp4_packed(vals: np.ndarray) -> np.ndarray:
    """[M,K] grid floats -> packed pk_fp4 bytes [M,K//2].

    Low nibble holds the even-K element, high nibble the odd-K element, matching
    pk_fp4 _pack(x0,x1) = (x1<<4)|(x0&0xF) and the reference unpack<0>=lo.

    Returns the physically-packed pk_fp4 buffer ([M, K//2] bytes, one byte == two
    fp4). mx_gemm_ctypes_lib sizes the A/B device buffer via
    HostTensor::get_element_space_size_in_bytes(), which divides the logical
    element count by numeric_traits<pk_fp4_t>::PackedSize==2, so this packed layout
    is exactly what it expects (fp8 with PackedSize==1 is the 1 byte/element case).
    """
    M, K = vals.shape
    assert K % 2 == 0, "fp4 packs two K elements per byte"
    codes = _map_grid_to_bytes(vals, _FP4_VALUE_TO_CODE, "fp4 e2m1").reshape(M, K)
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    return ((hi << 4) | (lo & 0x0F)).astype(np.uint8)


def dequantize_fp4_packed(packed: np.ndarray, K: int) -> np.ndarray:
    """Packed pk_fp4 bytes [M,K//2] -> [M,K] float grid values."""
    p = np.asarray(packed, dtype=np.uint8)
    M = p.shape[0]
    lo = (p & 0x0F).astype(np.int32)
    hi = ((p >> 4) & 0x0F).astype(np.int32)
    out = np.empty((M, K), dtype=np.float32)
    out[:, 0::2] = _FP4_E2M1_VALUES[lo]
    out[:, 1::2] = _FP4_E2M1_VALUES[hi]
    return out


# =============================================================================
# Numpy microscaled reference
# =============================================================================


def mx_gemm_reference(A_deq, B_deq, scale_a_byte, scale_b_byte, prob: MxGemmProblem):
    """C[m,n] = sum_kb sa[m,kb]*sb[n,kb] * sum_{j in block} A[m,kb*32+j]*B[kb*32+j,n].

    A_deq: [M,K] dequantized floats (logical, unscaled A values).
    B_deq: [K,N] dequantized floats (logical, unscaled B values).
    scale_a_byte: e8m0 bytes [M, K/32] ; scale_b_byte: e8m0 bytes [N, K/32].
    Accumulated in fp32, returned as fp16 (kernel output dtype).
    """
    M, N, K = prob.M, prob.N, prob.K
    nkb = prob.scale_k
    A = np.asarray(A_deq, dtype=np.float32).reshape(M, K)
    B = np.asarray(B_deq, dtype=np.float32).reshape(K, N)
    sa = e8m0_to_float(scale_a_byte).reshape(M, nkb)     # [M, K/32]
    sb = e8m0_to_float(scale_b_byte).reshape(N, nkb)     # [N, K/32]

    # Scale A per (m, K-block) and B per (n, K-block). Expand block scales to K.
    sa_k = np.repeat(sa, SCALE_BLOCK, axis=1)            # [M, K]
    sb_k = np.repeat(sb, SCALE_BLOCK, axis=1)            # [N, K]
    A_scaled = A * sa_k                                  # [M, K]
    B_scaled = (B.T * sb_k).T                            # [K, N]
    C = A_scaled.astype(np.float32) @ B_scaled.astype(np.float32)  # [M, N]
    return C.astype(np.float16)


# =============================================================================
# Runner
# =============================================================================


class GpuMxGemmRunner:
    def __init__(self, so_path: Path, dtype: str = "fp8", arch: Optional[str] = None):
        # The fp8 numpy reference (quantize_fp8/dequantize_fp8) emits OCP e4m3
        # bytes; that only matches the device when the lib was compiled with
        # CK_TILE_USE_OCP_FP8 == 1 (the gfx950 default). Guard loudly so an
        # FNUZ arch can't silently diverge from the reference.
        arch = arch or _get_arch()
        if dtype == "fp8":
            assert_fp8_ocp_supported(arch)
        self._lib = MxGemmDispatcherLib(so_path)
        self.dtype = dtype
        self.arch = arch

    @property
    def kernel_name(self) -> str:
        return self._lib.kernel_name()

    def make_inputs(self, prob: MxGemmProblem, scale: float = 1.0, seed: int = 0):
        """Generate grid-exact A/B and e8m0 scales.

        Returns (A_deq[M,K], B_deq[K,N], A_bytes, B_bytes, sa_bytes[M,K/32],
        sb_bytes[N,K/32]). A_bytes/B_bytes are the raw device buffers.
        scale is a single power-of-two applied uniformly (1.0 == e8m0 byte 127).
        """
        M, N, K = prob.M, prob.N, prob.K
        rng = np.random.default_rng(seed)
        # Draw from a small grid that is exact in both fp8 e4m3 and fp4 e2m1.
        grid = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
                        dtype=np.float32)
        A_deq = rng.choice(grid, size=(M, K)).astype(np.float32)
        B_deq = rng.choice(grid, size=(K, N)).astype(np.float32)

        if self.dtype == "fp8":
            A_bytes = quantize_fp8(A_deq)                       # [M,K] uint8
            # B stored col-major (rcr) => [N,K] row-major bytes.
            B_bytes = quantize_fp8(B_deq.T)                     # [N,K] uint8
        elif self.dtype == "fp4":
            A_bytes = quantize_fp4_packed(A_deq)                # [M,K//2] uint8
            B_bytes = quantize_fp4_packed(B_deq.T)              # [N,K//2] uint8
        else:
            raise ValueError(f"unsupported dtype {self.dtype}")

        sbyte = int(float_to_e8m0(np.float32(scale)))
        sa_bytes = np.full((M, prob.scale_k), sbyte, dtype=np.uint8)
        sb_bytes = np.full((N, prob.scale_k), sbyte, dtype=np.uint8)
        return A_deq, B_deq, A_bytes, B_bytes, sa_bytes, sb_bytes

    def run(self, prob: MxGemmProblem, A_bytes, B_bytes, sa_bytes, sb_bytes):
        C = np.zeros((prob.M, prob.N), dtype=np.float16)
        t = self._lib.run(A_bytes, B_bytes, C, sa_bytes, sb_bytes, prob)
        return MxGemmResult(C=C, time_ms=t, kernel_name=self.kernel_name)

    @staticmethod
    def reference(A_deq, B_deq, sa_bytes, sb_bytes, prob: MxGemmProblem):
        return mx_gemm_reference(A_deq, B_deq, sa_bytes, sb_bytes, prob)


# =============================================================================
# Build pipeline
# =============================================================================


def _generate_kernel(cfg: MxGemmKernelConfig, headers_dir: Path) -> Optional[Path]:
    cmd = [
        sys.executable, str(_CODEGEN_SCRIPT), "--output-dir", str(headers_dir),
        "--config-json", json.dumps(cfg.to_codegen_config()),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        log.error("codegen failed for %s:\n%s", cfg.name, r.stderr[-1500:])
        return None
    hpp = headers_dir / f"{cfg.name}.hpp"
    return hpp if hpp.exists() else None


# AMDGPU codegen / optimization flags that Old-TE compiles the mx_gemm device
# kernel with. They MUST match Old-TE flag-for-flag: the generated device-kernel
# header is byte-identical, so any flag difference (inlining, register
# allocation, occupancy) shows up as a large perf gap. Compiling with only
# "-O3 -std=c++17" made bridge fp8 kernels run ~30% slower than Old-TE.
#
# Source of truth: projects/composablekernel/CMakeLists.txt add_compile_options()
# (inherited by tile_engine/ops/gemm/mx_gemm) + mx_gemm/CMakeLists.txt's
# --offload-compress. This mirrors gemm_utils._tile_engine_codegen_flags so the
# two bridges stay consistent.
#
# Unconditional set (CK CMake adds these on every supported toolchain):
_MX_CODEGEN_FLAGS = (
    "-mllvm", "-amdgpu-early-inline-all=true",
    "-mllvm", "-amdgpu-function-calls=false",
    "-mllvm", "--lsr-drop-solution=1",
    "-mllvm", "-enable-post-misched=0",
    "-fno-offload-uniform-block",
    "--offload-compress",
)
# Probe-gated set: CK's CMake only adds these when check_cxx_compiler_flag
# passes (newer -mllvm options some clang builds reject, e.g. ROCm 7.2 does not
# know -amdgpu-coerce-illegal-types). Mirror that probe so the bridge matches
# Old-TE wherever the compiler accepts it and stays buildable where it does not.
_MX_PROBED_CODEGEN_FLAGS = (
    ("-mllvm", "-amdgpu-coerce-illegal-types=1"),
)


@functools.lru_cache(maxsize=None)
def _hipcc_accepts(flag_tuple: Tuple[str, ...]) -> bool:
    """Mirror CMake check_cxx_compiler_flag: does hipcc compile a trivial TU with
    these flags? Cached so the probe runs at most once per distinct flag set."""
    try:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "probe.cpp"
            src.write_text("int main(){}\n")
            r = subprocess.run(
                [_HIPCC, *flag_tuple, "-c", str(src), "-o", str(Path(d) / "probe.o")],
                capture_output=True, timeout=120,
            )
            return r.returncode == 0
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _mx_codegen_flags() -> Tuple[str, ...]:
    """Old-TE's mx_gemm codegen flags plus any probe-gated flags the compiler
    accepts -- the exact backend flag set the TE benchmark TU is built with."""
    flags = list(_MX_CODEGEN_FLAGS)
    for pair in _MX_PROBED_CODEGEN_FLAGS:
        if _hipcc_accepts(pair):
            flags = list(pair) + flags
    return tuple(flags)


def _compile_kernel(hpp: Path, so: Path, arch: str) -> bool:
    inc = [
        f"-I{_CK_ROOT}/include", f"-I{_CK_ROOT}",
        f"-I{_CK_ROOT}/tile_engine/ops",
        f"-I{_CK_ROOT}/tile_engine/ops/gemm",
        f"-I{_CK_ROOT}/tile_engine/ops/gemm/mx_gemm",
    ]
    cmd = [
        # -std=c++20 matches CK_CXX_STANDARD; codegen flags match Old-TE (see
        # _mx_codegen_flags above). Without them the byte-identical fp8 device
        # kernel ran ~30% slower than Old-TE.
        _HIPCC, "-shared", "-fPIC", "-O3", "-std=c++20",
        *_mx_codegen_flags(),
        *inc,
        "-DCK_TILE_SINGLE_KERNEL_INCLUDE", f"-include{hpp}",
        "-D__HIP_PLATFORM_AMD__", f"--offload-arch={arch}", f'-DGFX_ARCH="{arch}"',
        "-Wno-undefined-func-template", "-Wno-float-equal",
        str(_CTYPES_LIB_SRC), "-o", str(so),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        log.error("compile failed for %s:\n%s", so.name, r.stderr[-3000:])
        return False
    return True


def setup_multiple_mx_gemm_dispatchers(
    configs: List[MxGemmKernelConfig],
    output_dir: Optional[Path] = None,
    gfx_arch: Optional[str] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """codegen -> hipcc -> .so for each config. Returns paths aligned with
    `configs` (None on failure). Dedups by .name; per-arch .so cache."""
    if not configs:
        return []
    arch = gfx_arch or _get_arch()
    # When an explicit gfx_arch is passed, pin every config to it BEFORE computing
    # cfg.name / running codegen. Otherwise cfg.to_codegen_config() falls back to
    # `self.gpu_target or _get_arch()`, so codegen (and the cached name) would use
    # the host-detected arch (or fail if rocminfo is missing) while the .so is
    # compiled for `arch` -- an arch mismatch between the header and the binary.
    # Resetting _name_cache forces the name to be recomputed for the chosen arch.
    if gfx_arch is not None:
        # mx_gemm is gfx950-only (the C++ bridge static_asserts GFX_ARCH==gfx950);
        # reject any other explicit arch here so a build that can never succeed
        # fails early with a clear message instead of at compile/runtime.
        _supported = ("gfx950",)
        if gfx_arch not in _supported:
            raise ValueError(
                f"mx_gemm is gfx950-only; requested {gfx_arch!r} is not supported "
                f"(supported: {list(_supported)})"
            )
        for c in configs:
            if c.gpu_target != gfx_arch:
                c.gpu_target = gfx_arch
                c._name_cache = None
    base = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="mx_gemm_bridge_"))
    headers = base / "generated_kernels"
    libs = base / "libs"
    headers.mkdir(parents=True, exist_ok=True)
    libs.mkdir(parents=True, exist_ok=True)

    seen: Dict[str, int] = {}
    deduped: List[Tuple[int, MxGemmKernelConfig]] = []
    for i, c in enumerate(configs):
        if c.name not in seen:
            seen[c.name] = i
            deduped.append((i, c))
    results: List[Optional[Path]] = [None] * len(configs)

    def build_one(idx: int, cfg: MxGemmKernelConfig):
        so = libs / f"lib_{cfg.name}_{arch}.so"
        if so.exists():
            return idx, so
        hpp = _generate_kernel(cfg, headers)
        if hpp is None:
            return idx, None
        return idx, (so if _compile_kernel(hpp, so, arch) else None)

    if parallel and len(deduped) > 1:
        workers = max_workers or min(len(deduped), os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(build_one, i, c): i for i, c in deduped}
            for f in concurrent.futures.as_completed(futs):
                idx, so = f.result()
                results[idx] = so
    else:
        for i, c in deduped:
            _, so = build_one(i, c)
            results[i] = so

    # Propagate deduped results to duplicate slots.
    for i, c in enumerate(configs):
        if results[i] is None and seen.get(c.name, i) != i:
            results[i] = results[seen[c.name]]
    built = sum(1 for r in results if r)
    log.info("built %d/%d mx_gemm kernels for %s", built, len(configs), arch)
    return results


def default_fp8_config(gfx_arch: Optional[str] = None) -> MxGemmKernelConfig:
    return MxGemmKernelConfig(
        datatype="fp8", layout="rcr", gpu_target=gfx_arch,
        pipeline="comp_async", epilogue="cshuffle", scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128, warp_m=2, warp_n=2, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
    )


def default_fp4_config(gfx_arch: Optional[str] = None) -> MxGemmKernelConfig:
    cfg = default_fp8_config(gfx_arch)
    cfg.datatype = "fp4"
    cfg._name_cache = None
    return cfg
