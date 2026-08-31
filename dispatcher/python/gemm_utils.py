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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reuse the proven codegen/compile leaf helpers from the dispatcher's own
# python layer. gemm_utils is a thin bridge on top of these.
import ctypes_utils as _cu

_LAYOUT_CHAR = {"row": "r", "col": "c", "r": "r", "c": "c"}
_LAYOUT_WORD = {"r": "row", "c": "col"}

# --- Bridge shared helpers (canonical superset; byte-identical across bridges) ---
# Supported GPU architectures for the bridge (single source of truth).
_SUPPORTED_ARCHES = ("gfx90a", "gfx942", "gfx950", "gfx1250")

# Single source of truth for the preshuffle B-shuffle permutation used by the
# bridge. The bridge codegen only emits the NON-permuteN preshuffle pipeline
# (WeightPreshufflePipelineAGmemBGmemCRegV2), whose device-side B packing matches
# ck_tile::shuffle_b (permute_n=False). Old-TE's default_config.json /
# default_ci_config.json set permute_n=true, but that is a HOST-marker that
# selects a distinct (permuteN) TE pipeline the bridge does not generate -- it
# does NOT map to a separate bridged device kernel. Honoring true here would
# mis-shuffle B (GPU-verified max_rel ~1.25 vs ~5e-4). So every bridge pin reads
# this one constant. TODO: to support permute_n=True, emit the permuteN pipeline
# in unified_gemm_codegen and set this to a swept/config-driven value.
BRIDGE_PERMUTE_N = False


try:
    # Reuse the single canonical amd-smi bridge instead of re-implementing it.
    from dispatcher_common import _detect_gpu_arch_via_amd_smi
except Exception:  # noqa: BLE001 - standalone use without dispatcher_common on path
    def _detect_gpu_arch_via_amd_smi() -> Optional[str]:
        return None


@functools.lru_cache(maxsize=1)
def _get_arch() -> str:
    """Detect the GPU architecture from rocminfo and validate it.

    Returns the detected ``gfxNNN`` string. Raises ``RuntimeError`` when no arch
    can be detected (no GPU / rocminfo unavailable) -- we refuse to silently
    default to a specific architecture -- and ``ValueError`` when the detected
    arch is not one this bridge supports.
    """
    detected: Optional[str] = _detect_gpu_arch_via_amd_smi()
    if detected is None:
        try:
            out = subprocess.check_output(
                ["rocminfo"], stderr=subprocess.DEVNULL, text=True
            )
            for line in out.splitlines():
                stripped = line.strip()
                if stripped.startswith("Name:") and "gfx" in stripped:
                    name = stripped.split(":", 1)[1].strip()
                    if name.startswith("gfx"):
                        detected = name
                        break
        except Exception:  # noqa: BLE001 - rocminfo missing / no GPU / timeout
            detected = None

    if detected is None:
        raise RuntimeError(
            "Could not detect GPU architecture from rocminfo; refusing to "
            "default to a specific GPU architecture. Pass an explicit --arch / "
            "gfx_arch (one of "
            f"{', '.join(_SUPPORTED_ARCHES)})."
        )
    if detected not in _SUPPORTED_ARCHES:
        raise ValueError(
            f"Unsupported GPU architecture {detected!r}; supported: "
            f"{', '.join(_SUPPORTED_ARCHES)}."
        )
    return detected


def _resolve_arch(arch: Optional[str]) -> str:
    """Resolve a possibly-``None`` arch to a validated, supported ``gfxNNN``.

    ``None``/empty -> detect via :func:`_get_arch`. An explicit value is
    validated against ``_SUPPORTED_ARCHES`` (raising ``ValueError`` if unknown)
    so a typo can never silently reach the compiler.
    """
    if not arch:
        return _get_arch()
    if arch not in _SUPPORTED_ARCHES:
        raise ValueError(
            f"Unsupported GPU architecture {arch!r}; supported: "
            f"{', '.join(_SUPPORTED_ARCHES)}."
        )
    return arch


def _cshuffle_store_ok(
    m_repeat: int, n_repeat: int, warp_tile_m: int, warp_tile_n: int
) -> bool:
    """Return False for the one CShuffle-store combination that is numerically
    wrong (issue #9684): an ODD per-wave repeat (>1) paired with a 32-wide warp
    tile in that dimension. GPU-verified on gfx942 -- e.g. tile_m=192 / wave_m=2
    / warp_tile_m=32 (MRepeat=3) returns garbage, while every other non-power-of-
    two repeat (incl. MRepeat=3 with warp_tile_m=16, and even repeats like 6/12)
    is correct. Only relevant for the CShuffle epilogue; the default epilogue is
    exempt."""

    def _dim_bad(repeat: int, warp_tile: int) -> bool:
        return repeat > 1 and repeat % 2 == 1 and warp_tile == 32

    return not (_dim_bad(m_repeat, warp_tile_m) or _dim_bad(n_repeat, warp_tile_n))
# --- end bridge shared helpers ---


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

    # No silent default: the arch must be resolved (rocminfo-detected or passed
    # explicitly) before this config feeds the compiler. expand_sweep /
    # setup_multiple_gemm_dispatchers guarantee a non-None value; a stray None
    # reaching -DGFX_ARCH / --offload-arch would build for the wrong device.
    gfx_arch: Optional[str] = None
    variant: str = "standard"
    # Stream-K reduction strategy: "atomic" (default), "linear", or "tree".
    # Only meaningful when variant == "stream_k".
    reduction_strategy: str = "atomic"

    # --- Preshuffle only ---------------------------------------------------
    # Selects the B-preshuffle permutation (shuffle_b_permuteN vs shuffle_b).
    # Mirrors Old-TE's permute_n config knob; participates in the kernel name so
    # it must match unified_gemm_codegen.py::key_name. Ignored by other variants.
    permute_n: bool = False

    # --- Multi-ABD only ----------------------------------------------------
    # Arrays of A/B/D tensors and per-group element-wise ops. These are
    # behavior-affecting and appear in .name (and thus in the codegen kernel
    # name) so distinct tensor counts / ops never collapse to one kernel.
    # layout_d is the 4th ('D') char of the multi_abd rcrr layout code.
    num_a_tensors: int = 2
    num_b_tensors: int = 2
    num_d_tensors: int = 2
    a_elementwise_op: str = "PassThrough"
    b_elementwise_op: str = "PassThrough"
    cde_elementwise_op: str = "PassThrough"
    layout_d: str = "row"

    # --- Multi-D only (variant=="multi_d") ---------------------------------
    #   elementwise_op: "MultiDAdd" | "MultiDMultiply" | "PassThrough"
    #   d_layout      : row/col of every D tensor (row for the TE multi_d builder)
    # num_d_tensors (above) is reused as the fused-D operand count for multi_d.
    elementwise_op: str = "PassThrough"
    d_layout: str = "row"

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
    def layout4(self) -> str:
        """4-char multi_abd layout string (A,B,E,D), e.g. 'rcrr'."""
        return self.layout + _LAYOUT_CHAR[self.layout_d]

    @property
    def codegen_layout(self) -> str:
        """Layout string passed to unified_gemm_codegen.py --layout.

        Multi-D takes a 4-char layout (A,B,C + D); the codegen splits off the
        4th char as the D-tensor layout. Every other variant uses the 3-char
        A,B,C layout.
        """
        if self.variant == "multi_d":
            return self.layout + _LAYOUT_CHAR[self.d_layout]
        return self.layout

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
        # Multi-ABD uses the 4-char layout (A,B,E,D); multi_d likewise appends
        # its D-tensor layout char; every other variant uses the 3-char (A,B,C).
        # This mirrors KernelNaming.generate in the codegen.
        if self.variant == "multi_abd":
            layout_str = self.layout4
        elif self.variant == "multi_d":
            layout_str = self.layout + _LAYOUT_CHAR[self.d_layout]
        else:
            layout_str = self.layout
        name = (
            f"gemm_{self.dtype_a}_{layout_str}"
            f"_{self.pipeline}_{self.epilogue}_{self.scheduler}"
            f"_{_cap(self.pad_m)}_{_cap(self.pad_n)}_{_cap(self.pad_k)}"
            f"_{_cap(self.persistent)}"
            f"_{self.tile_str}_{self.wave_str}_{self.warp_tile_str}"
        )
        if self.variant == "preshuffle":
            name += "_preshuffle"
            if self.permute_n:
                name += "_permuteN"
        elif self.variant == "stream_k":
            name += "_streamk"
            # Atomic keeps the bare "_streamk" suffix (original parity); linear
            # and tree are disambiguated, matching KernelNaming.generate.
            if self.reduction_strategy != "atomic":
                name += f"_{self.reduction_strategy}"
        elif self.variant == "multi_abd":
            # Byte-for-byte match to codegen KernelNaming.generate's multiabd
            # suffix: tensor counts then the three element-wise ops.
            name += (
                f"_multiabd_a{self.num_a_tensors}_b{self.num_b_tensors}"
                f"_d{self.num_d_tensors}"
                f"_{self.a_elementwise_op}_{self.b_elementwise_op}"
                f"_{self.cde_elementwise_op}"
            )
        elif self.variant == "multi_d":
            # Mirror KernelNaming.generate: "_multid_{elementwise_op}_d{num_d}".
            name += f"_multid_{self.elementwise_op}_d{self.num_d_tensors}"
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
        cfg = {
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
            # Top-level knob read by unified_gemm_codegen for the preshuffle
            # variant (selects shuffle_b_permuteN vs shuffle_b). Harmless for
            # other variants, which ignore it.
            "permute_n": self.permute_n,
        }
        # Pin the single reduction strategy so stream-K codegen emits exactly this
        # kernel (the generator otherwise expands all strategies in its default).
        if self.variant == "stream_k":
            cfg["streamk_config"] = {"reduction_strategy": [self.reduction_strategy]}
        # Multi-ABD codegen reads its tensor counts / element-wise ops from a
        # dedicated ``multi_abd_config`` block. These are scalars (one kernel per
        # config), matching the codegen's _get_configs_for_variant reader.
        if self.variant == "multi_abd":
            cfg["multi_abd_config"] = {
                "num_a_tensors": self.num_a_tensors,
                "num_b_tensors": self.num_b_tensors,
                "num_d_tensors": self.num_d_tensors,
                "a_elementwise_op": self.a_elementwise_op,
                "b_elementwise_op": self.b_elementwise_op,
                "cde_elementwise_op": self.cde_elementwise_op,
            }
        # Multi-D signature: the codegen expands its multi_d variant over the
        # (elementwise_op x num_d_tensors) product, so pin both to this config's
        # single values. Only emitted for multi_d (ignored elsewhere).
        if self.variant == "multi_d":
            cfg["multi_d_config"] = {
                "elementwise_ops": [self.elementwise_op],
                "num_d_tensors": [self.num_d_tensors],
            }
        return cfg

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
    # Optional numeric-verification metric: global relative error of the kernel
    # output vs a numpy reference (max|out-ref|/max|ref|). None when the runner
    # did not compute a reference. Multi-ABD populates this in-runner because it
    # generates its own A/B/D operands internally (see GpuMultiABDRunner.run).
    max_rel: Optional[float] = None

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
        self._has_single = hasattr(self._lib, "dispatcher_run_gemm")
        self._has_grouped = hasattr(self._lib, "dispatcher_run_grouped_gemm")
        self._has_multi_d = hasattr(self._lib, "dispatcher_run_multi_d_gemm")
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

        # Regular single-problem GEMM ABI (gemm_ctypes_lib.cpp). Absent on the
        # grouped and multi_d libs, which expose dispatcher_run_grouped_gemm /
        # dispatcher_run_multi_d_gemm instead.
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

        # Multi-D ABI: extra D-pointer array + count (multi_d_gemm_ctypes_lib.cpp).
        if self._has_multi_d:
            lib.dispatcher_run_multi_d_gemm.argtypes = [
                ctypes.c_void_p,  # A (host)
                ctypes.c_void_p,  # B (host)
                ctypes.POINTER(ctypes.c_void_p),  # D_ptrs[] (host)
                ctypes.c_int,  # num_d
                ctypes.c_void_p,  # C (host)
                ctypes.c_int64,  # M
                ctypes.c_int64,  # N
                ctypes.c_int64,  # K
                ctypes.POINTER(ctypes.c_float),  # time_ms
            ]
            lib.dispatcher_run_multi_d_gemm.restype = ctypes.c_int
            if hasattr(lib, "dispatcher_get_num_d_tensors"):
                lib.dispatcher_get_num_d_tensors.argtypes = []
                lib.dispatcher_get_num_d_tensors.restype = ctypes.c_int

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
        if not self._has_single:
            raise RuntimeError(
                f"{self._path} does not expose dispatcher_run_gemm; this is not a "
                f"regular GEMM .so (grouped/multi_d libs use run_grouped/run_multi_d)"
            )
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

    @property
    def has_multi_d(self) -> bool:
        return self._has_multi_d

    def num_d_tensors(self) -> int:
        """Number of D tensors baked into this multi_d .so (0 if not multi_d)."""
        if self._has_multi_d and hasattr(self._lib, "dispatcher_get_num_d_tensors"):
            return int(self._lib.dispatcher_get_num_d_tensors())
        return 0

    def run_multi_d(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Ds: List[np.ndarray],
        C: np.ndarray,
        M: int,
        N: int,
        K: int,
    ) -> Tuple[int, float]:
        """Run a multi_d GEMM: E = elementwise_op(A@B, D0, D1, ...).

        ``Ds`` is a list of ``num_d_tensors`` host arrays (each MxN, same element
        type as C). Pointers are collected into a ctypes ``c_void_p`` array; the
        .so owns all device memory.
        """
        if not self._has_multi_d:
            raise RuntimeError(
                f"{self._path} does not expose dispatcher_run_multi_d_gemm"
            )
        num_d = len(Ds)
        d_arr_t = ctypes.c_void_p * max(num_d, 1)
        d_arr = d_arr_t(*[d.ctypes.data_as(ctypes.c_void_p) for d in Ds])
        time_ms = ctypes.c_float(0.0)
        status = self._lib.dispatcher_run_multi_d_gemm(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            d_arr,
            num_d,
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


def _fp32_to_fp8_ocp_u8(x):
    """Encode fp32 -> fp8 E4M3 **OCP** (gfx950/gfx12xx) bit pattern (uint8)."""
    import ml_dtypes
    return np.ascontiguousarray(x, dtype=np.float32).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)


def _fp32_to_bf8_ocp_u8(x):
    """Encode fp32 -> bf8 E5M2 **OCP** (gfx950/gfx12xx) bit pattern (uint8)."""
    import ml_dtypes
    return np.ascontiguousarray(x, dtype=np.float32).astype(ml_dtypes.float8_e5m2).view(np.uint8)


_OCP_FP8_CACHE = {}


def _use_ocp_fp8():
    """True on archs whose device fp8_t is OCP (gfx950/MI350, gfx12xx/RDNA) rather
    than the FNUZ encoding used by gfx942/MI300. Detected once via rocminfo."""
    if "v" not in _OCP_FP8_CACHE:
        try:
            a = _get_arch()
        except Exception:
            a = ""
        _OCP_FP8_CACHE["v"] = a.startswith("gfx12") or a == "gfx950"
    return _OCP_FP8_CACHE["v"]


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
            _enc = _fp32_to_fp8_ocp_u8 if _use_ocp_fp8() else _fp32_to_fp8_u8
            A_h = _enc(A_lay)
            B_h = _enc(B_lay)
        elif dtype == "bf8":
            _enc = _fp32_to_bf8_ocp_u8 if _use_ocp_fp8() else _fp32_to_bf8_u8
            A_h = _enc(A_lay)
            B_h = _enc(B_lay)
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
# Multi-D GEMM problem / result / runner
# ============================================================================


@dataclass
class MultiDGemmProblem:
    """A multi_d GEMM problem: E[MxN] = op(A[MxK] @ B[KxN], D0, D1, ...).

    ``num_d`` D tensors, each MxN and stored in the output (C) element dtype.
    """

    M: int
    N: int
    K: int
    num_d: int = 2

    @property
    def flops(self) -> float:
        # 2*M*N*K for the GEMM; the element-wise D fuse is negligible and matches
        # how Old-TE reports multi_d TFLOPs.
        return 2.0 * self.M * self.N * self.K


@dataclass
class MultiDGemmResult:
    output: np.ndarray
    time_ms: float
    status: int
    tflops: float
    kernel_name: str

    @property
    def success(self) -> bool:
        return self.status == 0


def _multi_d_layout_from_kernel_name(name: str) -> str:
    """Extract the 4-char layout (e.g. 'rcrr') from a multi_d kernel name.

    Name format is ``gemm_<dtype>_<layout4>_...``; each char is 'r'/'c' for
    operands A, B, C, D. Falls back to 'rcrr' if not found.
    """
    parts = name.split("_")
    if len(parts) > 2 and len(parts[2]) == 4 and set(parts[2]) <= {"r", "c"}:
        return parts[2]
    return "rcrr"


class GpuMultiDGemmRunner:
    """High-level runner for the multi_d bridge .so.

    Constructed from a .so path; call ``run(A, B, Ds, problem)`` with logical
    row-major A (MxK), B (KxN) and a list of row-major D tensors (each MxN). The
    kernel's compiled dtype/layout (from its name) dictates operand memory
    layout; the C ABI owns all device memory. fp16-only for now (the TE multi_d
    op supports only fp16).
    """

    def __init__(self, lib_path: Path):
        self.lib = GemmDispatcherLib(lib_path)
        if not self.lib.initialize():
            raise RuntimeError(f"Failed to initialize multi_d .so: {lib_path}")
        if not self.lib.has_multi_d:
            raise RuntimeError(
                f"{lib_path} is not a multi_d .so (no dispatcher_run_multi_d_gemm)"
            )
        names = self.lib.kernel_names
        self._kernel_name = names[0] if names else "unknown"
        self._num_d = self.lib.num_d_tensors()

    @property
    def kernel_name(self) -> str:
        return self._kernel_name

    @property
    def num_d_tensors(self) -> int:
        return self._num_d

    def run(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Ds: List[np.ndarray],
        problem: MultiDGemmProblem,
    ) -> MultiDGemmResult:
        M, N, K = problem.M, problem.N, problem.K
        dtype = _dtype_from_kernel_name(self._kernel_name)
        layout4 = _multi_d_layout_from_kernel_name(self._kernel_name)
        la, lb, lc, ld = layout4[0], layout4[1], layout4[2], layout4[3]

        if dtype != "fp16":
            raise ValueError(f"multi_d bridge currently supports fp16 only, got {dtype}")
        if len(Ds) != self._num_d:
            raise ValueError(
                f"kernel expects {self._num_d} D tensors, got {len(Ds)}"
            )

        # A/B host buffers, transposed for column-major operands (see GpuGemmRunner).
        A_lay = A if la == "r" else A.T
        B_lay = B if lb == "r" else B.T
        A_h = np.ascontiguousarray(A_lay, dtype=np.float16)
        B_h = np.ascontiguousarray(B_lay, dtype=np.float16)

        # C and D are row-major (last two layout chars are 'r' for the TE
        # multi_d builder); keep them MxN contiguous.
        C_shape = (M, N) if lc == "r" else (N, M)
        C_h = np.zeros(C_shape, dtype=np.float16)
        D_h = []
        for d in Ds:
            d_lay = d if ld == "r" else d.T
            D_h.append(np.ascontiguousarray(d_lay, dtype=np.float16))

        status, time_ms = self.lib.run_multi_d(A_h, B_h, D_h, C_h, M, N, K)

        C_out = C_h if lc == "r" else C_h.T
        tflops = (problem.flops / (time_ms * 1e-3)) / 1e12 if time_ms > 0 else 0.0
        return MultiDGemmResult(
            output=C_out,
            time_ms=time_ms,
            status=status,
            tflops=tflops,
            kernel_name=self._kernel_name,
        )


# ============================================================================
# Multi-ABD ctypes ABI wrapper + runner (divergent, array-pointer ABI)
# ============================================================================

# Element size (bytes) per CK dtype -- mirrors the codegen's ELEMENT_SIZE_MAP and
# lets the ctypes shim size its device buffers without knowing the CK type.
_ELEM_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8": 1, "bf8": 1, "int8": 1, "int32": 4}


class MultiABDDispatcherLib:
    """Thin ctypes wrapper around a compiled gemm_multi_abd dispatcher .so.

    Multi-ABD is registry-bypass with a divergent ABI: ``dispatcher_run_multi_abd``
    takes ARRAYS of host pointers (one per A/B/D tensor) plus per-group element
    sizes, and the .so owns all GPU memory (hipMalloc/Memcpy/Free) internally.
    """

    def __init__(self, so_path: Path):
        self._path = Path(so_path)
        self._lib = ctypes.CDLL(str(self._path))
        self._setup_functions()

    def _setup_functions(self) -> None:
        lib = self._lib
        lib.dispatcher_initialize.argtypes = []
        lib.dispatcher_initialize.restype = ctypes.c_int
        lib.dispatcher_get_kernel_name.argtypes = []
        lib.dispatcher_get_kernel_name.restype = ctypes.c_char_p
        for fn in (
            "dispatcher_get_num_a_tensors",
            "dispatcher_get_num_b_tensors",
            "dispatcher_get_num_d_tensors",
        ):
            getattr(lib, fn).argtypes = []
            getattr(lib, fn).restype = ctypes.c_int
        lib.dispatcher_run_multi_abd.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # as_hosts
            ctypes.POINTER(ctypes.c_void_p),  # bs_hosts
            ctypes.POINTER(ctypes.c_void_p),  # ds_hosts
            ctypes.c_void_p,  # e_host
            ctypes.POINTER(ctypes.c_int64),  # stride_as
            ctypes.POINTER(ctypes.c_int64),  # stride_bs
            ctypes.POINTER(ctypes.c_int64),  # stride_ds
            ctypes.c_int64,  # stride_e
            ctypes.c_int,  # elem_a
            ctypes.c_int,  # elem_b
            ctypes.c_int,  # elem_d
            ctypes.c_int,  # elem_e
            ctypes.c_int,  # num_a
            ctypes.c_int,  # num_b
            ctypes.c_int,  # num_d
            ctypes.c_int64,  # M
            ctypes.c_int64,  # N
            ctypes.c_int64,  # K
            ctypes.POINTER(ctypes.c_float),  # time_ms
        ]
        lib.dispatcher_run_multi_abd.restype = ctypes.c_int
        lib.dispatcher_cleanup.argtypes = []
        lib.dispatcher_cleanup.restype = None

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> bool:
        return self._lib.dispatcher_initialize() == 0

    @property
    def kernel_name(self) -> str:
        raw = self._lib.dispatcher_get_kernel_name()
        return raw.decode("utf-8") if raw else "unknown"

    @property
    def tensor_counts(self) -> Tuple[int, int, int]:
        return (
            int(self._lib.dispatcher_get_num_a_tensors()),
            int(self._lib.dispatcher_get_num_b_tensors()),
            int(self._lib.dispatcher_get_num_d_tensors()),
        )

    def run(
        self,
        as_arrays: List[np.ndarray],
        bs_arrays: List[np.ndarray],
        ds_arrays: List[np.ndarray],
        e_array: np.ndarray,
        M: int,
        N: int,
        K: int,
        elem_a: int,
        elem_b: int,
        elem_d: int,
        elem_e: int,
        stride_as: Optional[List[int]] = None,
        stride_bs: Optional[List[int]] = None,
        stride_ds: Optional[List[int]] = None,
        stride_e: int = 0,
    ) -> Tuple[int, float]:
        def _ptr_array(arrays):
            arr = (ctypes.c_void_p * max(len(arrays), 1))()
            for i, a in enumerate(arrays):
                arr[i] = a.ctypes.data_as(ctypes.c_void_p)
            return arr

        def _i64_array(vals):
            # The GemmMultiABDKernel does NOT derive strides from a 0 sentinel
            # (unlike some CK host helpers); it passes them straight to the
            # UniversalGemm kernel args. We must therefore supply the explicit
            # leading strides (see GpuMultiABDRunner.run, which mirrors the
            # Old-TE profiler's get_default_stride).
            if not vals:
                return ctypes.POINTER(ctypes.c_int64)()
            arr = (ctypes.c_int64 * len(vals))()
            for i, v in enumerate(vals):
                arr[i] = int(v)
            return arr

        as_ptrs = _ptr_array(as_arrays)
        bs_ptrs = _ptr_array(bs_arrays)
        ds_ptrs = _ptr_array(ds_arrays)
        stride_as_arr = _i64_array(stride_as)
        stride_bs_arr = _i64_array(stride_bs)
        stride_ds_arr = _i64_array(stride_ds)
        time_ms = ctypes.c_float(0.0)
        status = self._lib.dispatcher_run_multi_abd(
            as_ptrs,
            bs_ptrs,
            ds_ptrs,
            e_array.ctypes.data_as(ctypes.c_void_p),
            stride_as_arr,
            stride_bs_arr,
            stride_ds_arr,
            int(stride_e),
            elem_a,
            elem_b,
            elem_d,
            elem_e,
            len(as_arrays),
            len(bs_arrays),
            len(ds_arrays),
            M,
            N,
            K,
            ctypes.byref(time_ms),
        )
        return status, time_ms.value

    def cleanup(self) -> None:
        self._lib.dispatcher_cleanup()


# Multi-ABD per-group element-wise CDE ops, as numpy reductions over
# (acc, D0, D1, ...). These mirror ck_tile::element_wise (see
# unary_element_wise_operation.hpp) EXACTLY so the numpy reference matches the
# device epilogue:
#   PassThrough    : E = C                       (D tensors ignored)
#   MultiDAdd      : E = C + D0 + D1 + ...
#   MultiDMultiply : E = C * D0 * D1 * ...
#   AddScale       : E = scale * (C + D0 + D1 + ...)   (default scale = 1.0;
#                    note AddScale folds *all* its arguments including C, so C
#                    participates in the sum -- see struct AddScale)
def _cde_reference(op: str, acc: np.ndarray, ds: List[np.ndarray]) -> np.ndarray:
    """Apply the CDE element-wise op to the fp32 accumulator + D tensors."""
    acc = acc.astype(np.float32)
    ds32 = [d.astype(np.float32) for d in ds]
    if op == "PassThrough":
        return acc
    if op == "MultiDAdd":
        out = acc.copy()
        for d in ds32:
            out = out + d
        return out
    if op == "MultiDMultiply":
        out = acc.copy()
        for d in ds32:
            out = out * d
        return out
    if op == "AddScale":
        # AddScale starts at 0 and folds every argument (C included).
        out = acc.copy()
        for d in ds32:
            out = out + d
        return out  # scale defaults to 1.0
    raise ValueError(f"Unsupported CDE element-wise op for reference: {op}")


def _ab_reference(op: str, group: List[np.ndarray]) -> np.ndarray:
    """Combine a group of A (or B) tensors into a single matrix via the op.

    Mirrors reference_gemm_multiple_abd's A/B pre-pass, which applies the group
    element-wise op across the tuple of tensors element-by-element:
      PassThrough    : first tensor only (op(y, x0, x1...) assigns y = x0)
      MultiDAdd      : sum of all tensors
      MultiDMultiply : product of all tensors
      AddScale       : scale * sum of all tensors (scale defaults to 1.0)
    """
    g32 = [t.astype(np.float32) for t in group]
    if op == "PassThrough":
        return g32[0]
    if op == "MultiDAdd":
        out = g32[0].copy()
        for t in g32[1:]:
            out = out + t
        return out
    if op == "AddScale":
        out = np.zeros_like(g32[0])
        for t in g32:
            out = out + t
        return out  # scale defaults to 1.0
    if op == "MultiDMultiply":
        out = g32[0].copy()
        for t in g32[1:]:
            out = out * t
        return out
    raise ValueError(f"Unsupported A/B element-wise op for reference: {op}")


class GpuMultiABDRunner:
    """High-level multi-ABD runner: construct from a .so, call run(problem).

    All A/B/D operands share the group dtype/layout (matching the Tile Engine
    gemm_multi_abd op). The runner builds the host operand buffers in the
    kernel's element dtype + layout and hands raw pointer arrays to the .so,
    which owns GPU memory. fp16 is the only supported multi-abd dtype.

    Numeric verification (B1): the runner generates all A/B/D operands itself,
    so it also owns the numpy reference. When ``verify`` is set on ``run`` it
    computes ``E = CDE( AB(As) @ BB(Bs), {Ds} )`` -- byte-for-byte mirroring
    ck_tile::reference_gemm_multiple_abd (A/B groups combined element-wise via
    their ops into one matrix, single GEMM, then the CDE op folds the D tensors)
    -- and reports max_rel = max|E_gpu - E_ref| / max|E_ref| on the result.

    Layout and per-group element-wise ops / tensor counts are taken from the
    supplied config object when available (N2); the kernel name is used only as
    a fallback so the runner never silently guesses a wrong layout.
    """

    def __init__(
        self,
        lib_path: Path,
        layout4: Optional[str] = None,
        a_elementwise_op: Optional[str] = None,
        b_elementwise_op: Optional[str] = None,
        cde_elementwise_op: Optional[str] = None,
    ):
        self.lib = MultiABDDispatcherLib(lib_path)
        if not self.lib.initialize():
            raise RuntimeError(f"Failed to initialize multi_abd .so: {lib_path}")
        self._kernel_name = self.lib.kernel_name
        self._num_a, self._num_b, self._num_d = self.lib.tensor_counts

        # N2: prefer the layout / ops derived from the config object. Only fall
        # back to parsing the kernel name (which is deterministic from config)
        # when the caller did not supply them -- no silent "rcrr" default.
        self._layout4 = layout4 or self._parse_layout4()
        a_op, b_op, cde_op = self._parse_ops()
        self._a_op = a_elementwise_op or a_op
        self._b_op = b_elementwise_op or b_op
        self._cde_op = cde_elementwise_op or cde_op

    @property
    def kernel_name(self) -> str:
        return self._kernel_name

    def _parse_layout4(self) -> str:
        """Fallback: 4-char (A,B,E,D) layout from ``gemm_<dtype>_<layout>_...``."""
        parts = self._kernel_name.split("_")
        if len(parts) > 2 and len(parts[2]) == 4 and set(parts[2]) <= {"r", "c"}:
            return parts[2]
        raise ValueError(
            f"Cannot derive multi_abd layout from kernel name {self._kernel_name!r}; "
            "pass layout4 from the config object instead"
        )

    def _parse_ops(self) -> Tuple[str, str, str]:
        """Fallback: parse the three element-wise ops from the kernel name.

        Name suffix is ``..._multiabd_a<NA>_b<NB>_d<ND>_<Aop>_<Bop>_<CDEop>``.
        """
        marker = "_multiabd_"
        idx = self._kernel_name.find(marker)
        if idx >= 0:
            tail = self._kernel_name[idx + len(marker) :].split("_")
            # tail == [aNA, bNB, dND, Aop, Bop, CDEop]
            if len(tail) >= 6:
                return tail[3], tail[4], tail[5]
        # Raise rather than silently defaulting to PassThrough (which would yield
        # a wrong numeric reference): the config is the source of truth and the
        # name is deterministic from it, so an unparseable name is a real error.
        # Mirrors _parse_layout4, which also raises.
        raise ValueError(
            f"cannot parse multi_abd element-wise ops from kernel name "
            f"{self._kernel_name!r}; expected a "
            f"'..._multiabd_a<NA>_b<NB>_d<ND>_<Aop>_<Bop>_<CDEop>' suffix"
        )

    def run(
        self,
        problem: GemmProblem,
        seed: int = 0,
        verify: bool = False,
        verify_tol: float = 2e-2,
    ) -> GemmResult:
        M, N, K = problem.M, problem.N, problem.K
        dtype = _dtype_from_kernel_name(self._kernel_name)
        if dtype != "fp16":
            raise ValueError(
                f"multi_abd runner supports fp16 only, got {dtype!r} "
                f"(kernel {self._kernel_name!r})"
            )
        layout4 = self._layout4
        la, lb = layout4[0], layout4[1]
        ld = layout4[3] if len(layout4) >= 4 else "r"

        rng = np.random.default_rng(seed)

        # Logical (row-major, M-major) operands used for the numpy reference, and
        # the physically-laid-out contiguous buffers handed to the .so. The .so
        # interprets each buffer per the compiled layout, so a column-major
        # operand is stored transposed but represents the same logical matrix.
        def _mk(rows, cols, layout_char, n_tensors, lo, hi):
            logical, physical = [], []
            for _ in range(n_tensors):
                x = rng.uniform(lo, hi, size=(rows, cols)).astype(np.float32)
                if dtype != "fp16":
                    raise ValueError(f"multi_abd runner supports fp16 only, got {dtype}")
                x16 = x.astype(np.float16)
                logical.append(x16)
                x_lay = x16 if layout_char == "r" else x16.T
                physical.append(np.ascontiguousarray(x_lay, dtype=np.float16))
            return logical, physical

        as_logical, as_arrays = _mk(M, K, la, self._num_a, -5.0, 5.0)
        bs_logical, bs_arrays = _mk(K, N, lb, self._num_b, -5.0, 5.0)
        ds_logical, ds_arrays = _mk(M, N, ld, self._num_d, -1.0, 1.0)
        elem = _ELEM_BYTES.get(dtype, 2)
        e_array = np.zeros((M, N), dtype=np.float16)

        # Explicit leading strides -- the GemmMultiABDKernel does NOT derive them
        # from a zero sentinel (it forwards them straight to the UniversalGemm
        # kernel args), so a 0 stride collapses the whole output onto row 0.
        # Mirror the Old-TE profiler's get_default_stride(rows, cols, is_row):
        # row-major -> #cols, col-major -> #rows.
        def _lead_stride(rows, cols, layout_char):
            return cols if layout_char == "r" else rows

        stride_as = [_lead_stride(M, K, la)] * self._num_a
        stride_bs = [_lead_stride(K, N, lb)] * self._num_b
        stride_ds = [_lead_stride(M, N, ld)] * self._num_d
        # E is the C position (index 2) of the 4-char layout.
        le = layout4[2] if len(layout4) >= 3 else "r"
        stride_e = _lead_stride(M, N, le)

        status, time_ms = self.lib.run(
            as_arrays,
            bs_arrays,
            ds_arrays,
            e_array,
            M,
            N,
            K,
            elem,
            elem,
            elem,
            elem,
            stride_as=stride_as,
            stride_bs=stride_bs,
            stride_ds=stride_ds,
            stride_e=stride_e,
        )
        tflops = (problem.flops / (time_ms * 1e-3)) / 1e12 if time_ms > 0 else 0.0

        max_rel = None
        if verify and status == 0:
            # A/B groups are combined element-wise into a single matrix, then a
            # single GEMM, then the CDE op folds the D tensors -- exactly
            # reference_gemm_multiple_abd. Compute in fp32.
            a_m_k = _ab_reference(self._a_op, as_logical)
            b_k_n = _ab_reference(self._b_op, bs_logical)
            acc = a_m_k @ b_k_n
            ref = _cde_reference(self._cde_op, acc, ds_logical).astype(np.float32)
            got = e_array.astype(np.float32)
            denom = float(np.max(np.abs(ref))) or 1.0
            max_rel = float(np.max(np.abs(got - ref)) / denom)

        return GemmResult(
            output=e_array,
            time_ms=time_ms,
            status=status,
            tflops=tflops,
            kernel_name=self._kernel_name,
            max_rel=max_rel,
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

    Variants whose launch ABI differs from the single-problem
    ``dispatcher_run_gemm`` path need their own lib:
      * stream_k keeps the single-problem C ABI (single A/B/C, M/N/K) but its
        lib builds a ``StreamKHostArgs`` and calls ``SelectedKernel::launch``
        directly instead of routing through the registry.
      * grouped has a multi-problem launch signature the single-problem
        ``gemm_ctypes_lib.cpp`` cannot express.
      * multi_d fuses extra D operands and exposes a GemmMultiDArgs launch
        signature, so it compiles against its dedicated ctypes source.
      * multi_abd has a divergent (array-pointer) ABI
        (dispatcher_run_multi_abd) that the single-problem
        ``gemm_ctypes_lib.cpp`` cannot express.
    """
    if config.variant == "stream_k":
        return "streamk_gemm_ctypes_lib.cpp"
    if config.variant == "grouped":
        return "grouped_gemm_ctypes_lib.cpp"
    if config.variant == "multi_d":
        return "multi_d_gemm_ctypes_lib.cpp"
    if config.variant == "multi_abd":
        return "gemm_multi_abd_ctypes_lib.cpp"
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
    # Multi-ABD is self-contained (registry-bypass, no static lib) so it can be
    # built without a prior full CMake configure of the dispatcher; that CMake
    # step is what normally creates build/examples. Ensure the output directory
    # exists so the hipcc -o path is always writable (harmless if it already is).
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    obj_file = lib_path.with_suffix(".o")
    # The Stream-K path skips the cmake build that would normally create this
    # directory, so ensure it exists before hipcc writes the object/.so here.
    lib_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-variant AMDGPU codegen flags. The regular path matches Tile Engine's
    # gemm_universal build via _tile_engine_codegen_flags(). Stream-K must instead
    # match TE's gemm_streamk build EXACTLY for a fair A/B: -enable-post-misched=0
    # is applied unconditionally (not persistent-gated) and it does NOT use
    # -enable-noalias-to-md-conversion=0.
    is_streamk = getattr(config, "variant", "") == "stream_k"
    variant_flags = (
        [
            "-std=c++20",
            "-fno-offload-uniform-block",
            "-mllvm", "--lsr-drop-solution=1",
            "-mllvm", "-enable-post-misched=0",
            "-mllvm", "-amdgpu-early-inline-all=true",
            "-mllvm", "-amdgpu-function-calls=false",
            "--offload-compress",
        ]
        if is_streamk
        else list(_tile_engine_codegen_flags())
    )

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
        # Match Tile Engine's AMDGPU codegen flags exactly (see variant_flags /
        # _tile_engine_codegen_flags). Without them the kernel is compiled with
        # different inlining/register allocation, which changes occupancy;
        # persistent kernels size their grid by occupancy
        # (UniversalGemmKernel::MaxOccupancyGridSize = #CUs x occupancy), so a
        # mismatch shows up as large perf gaps vs Tile Engine on persistent tiles.
        *variant_flags,
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
    ]
    # The regular GEMM ABI goes through the dispatcher registry and must link the
    # dispatcher static lib. Both Stream-K and Multi-ABD are registry-bypass
    # (their ctypes libs launch the force-included kernel directly and reference
    # no registry/dispatcher symbols), so their .so needs only the force-included
    # kernel -- no static lib -- keeping it self-contained.
    registry_bypass = is_streamk or config.variant == "multi_abd"
    if not registry_bypass:
        link_cmd.append(str(static_lib))
    link_cmd += ["-o", str(lib_path)]
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

    # Guard the compile path: every config's gfx_arch must be a concrete,
    # supported arch before it reaches -DGFX_ARCH / --offload-arch / gpu_target.
    # expand_sweep already resolves this, but a config built directly (gfx_arch
    # left as None) would otherwise emit a literal "None" arch. Resolve/validate
    # here too, defaulting a None to the rocminfo-detected arch (never gfx942).
    _shared_arch: Optional[str] = None
    resolved_configs: List[GemmKernelConfig] = []
    for c in configs:
        if c.gfx_arch:
            resolved_configs.append(replace(c, gfx_arch=_resolve_arch(c.gfx_arch)))
        else:
            if _shared_arch is None:
                _shared_arch = _get_arch()
            resolved_configs.append(replace(c, gfx_arch=_shared_arch))
    configs = resolved_configs

    # Hard-fail rather than build a runnable but WRONG kernel: a preshuffle config
    # with permute_n=True would compile a "_permuteN" kernel whose device pipeline
    # is not yet bridged (it mis-shuffles B -> wrong results; see BRIDGE_PERMUTE_N).
    # expand_sweep never yields such a config (it pins permute_n to BRIDGE_PERMUTE_N),
    # so this only catches a hand-constructed / misused config before it becomes a
    # .so that could silently produce incorrect output.
    if not BRIDGE_PERMUTE_N:
        for c in configs:
            if getattr(c, "variant", "") == "preshuffle" and getattr(
                c, "permute_n", False
            ):
                raise ValueError(
                    "permute_n=True is not supported by the bridge yet "
                    "(BRIDGE_PERMUTE_N=False): refusing to build a permuteN kernel "
                    f"that would mis-shuffle B ({c.name}). Flip BRIDGE_PERMUTE_N once "
                    "the permuteN pipeline is emitted in unified_gemm_codegen."
                )

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
    # Multi-ABD is registry-bypass: it links only the force-included kernel, so it
    # needs its own ctypes source but NOT the dispatcher static lib. Every other
    # variant goes through the registry and requires the static lib too.
    needed_sources = {ctypes_dir / _ctypes_source_name(c) for c in configs}
    missing = [str(p) for p in needed_sources if not p.exists()]
    # Stream-K and Multi-ABD .so files are registry-bypass: they link only the
    # force-included kernel (no registry/dispatcher symbols), so they do not need
    # the dispatcher static lib. Only a build in which every config is one of
    # these can skip the static lib; any other variant requires it.
    all_registry_bypass = {c.variant for c in configs} <= {"stream_k", "multi_abd"}
    need_static_lib = not all_registry_bypass
    if (need_static_lib and not static_lib.exists()) or missing:
        parts = []
        if need_static_lib and not static_lib.exists():
            parts.append(str(static_lib))
        parts.extend(missing)
        raise FileNotFoundError(
            "Missing static lib or ctypes source required for compilation:\n  "
            + "\n  ".join(parts)
            + "\n"
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
                "layout": c.codegen_layout,
                # Multi-ABD codegen expects the 4-char (A,B,E,D) layout so it can
                # split off the D layout; every other variant uses 3-char.
                "layout": c.layout4 if c.variant == "multi_abd" else c.layout,
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


def _cshuffle_store_ok(
    m_repeat: int, n_repeat: int, warp_tile_m: int, warp_tile_n: int
) -> bool:
    """Return False for the one CShuffle-store combination that is numerically
    wrong (issue #9684): an ODD per-wave repeat (>1) paired with a 32-wide warp
    tile in that dimension. GPU-verified on gfx942 -- e.g. tile_m=192 / wave_m=2
    / warp_tile_m=32 (MRepeat=3) returns garbage, while every other non-power-of-
    two repeat (incl. MRepeat=3 with warp_tile_m=16, and even repeats like 6/12)
    is correct. Only relevant for the CShuffle epilogue; the default epilogue is
    exempt."""

    def _dim_bad(repeat: int, warp_tile: int) -> bool:
        return repeat > 1 and repeat % 2 == 1 and warp_tile == 32

    return not (_dim_bad(m_repeat, warp_tile_m) or _dim_bad(n_repeat, warp_tile_n))


# --- Warp-configuration gate (parity with Old-TE) --------------------------
# Old-TE's gemm_validation_utils.validate_warp_configuration restricts the
# warps-per-block triple (wave_m/n/k) to WARP_SUPPORTED_COMBINATIONS[arch].
# expand_sweep must apply the SAME gate or the bridge emits wave configs Old-TE
# never generates (product != 4 on CDNA), diverging the two instance sets.
_WARP_SUPPORTED_COMBINATIONS_FALLBACK = {
    "gfx90a": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
    "gfx942": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
    "gfx950": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
    "gfx1250": [[2, 4, 1], [1, 8, 1], [8, 1, 1], [4, 2, 1], [2, 1, 1], [1, 2, 2], [4, 1, 1], [1, 4, 1], [2, 2, 1]],
    "gfx1201": [[2, 4, 1], [1, 8, 1], [8, 1, 1], [4, 2, 1]],
}


def _warp_supported_table():
    """Canonical WARP_SUPPORTED_COMBINATIONS from gemm_validation_utils, with a
    hardcoded fallback so the bridge never silently skips the gate."""
    try:
        from gemm_validation_utils import WARP_SUPPORTED_COMBINATIONS as _t
        return _t
    except Exception:  # pragma: no cover - fallback keeps the gate active
        return _WARP_SUPPORTED_COMBINATIONS_FALLBACK


def _warp_config_supported(wave_m: int, wave_n: int, wave_k: int, arch: str) -> bool:
    """True iff [wave_m, wave_n, wave_k] is an allowed warps-per-block triple
    for ``arch`` (mirrors Old-TE validate_warp_configuration). Unknown arch =>
    permissive (matches Old-TE's log-and-allow behavior)."""
    allowed = _warp_supported_table().get(arch)
    if not allowed:
        return True
    return [wave_m, wave_n, wave_k] in allowed


def expand_sweep(
    config_path: str,
    arch: Optional[str] = None,
    dtype: str = "fp16",
    layout: str = "rcr",
    variant: str = "standard",
    num_a_tensors: int = 2,
    num_b_tensors: int = 2,
    num_d_tensors: int = 2,
    a_elementwise_op: str = "PassThrough",
    b_elementwise_op: str = "PassThrough",
    cde_elementwise_op: str = "PassThrough",
    mabd_cli_overrides: Optional[Dict[str, Any]] = None,
) -> List[GemmKernelConfig]:
    """Expand a Tile Engine GEMM JSON sweep config into GemmKernelConfig list.

    The TE config uses ``tile_config`` (ranges/value-lists for tile, warp and
    warp_tile triples) and ``trait_config`` (value-lists for pipeline,
    scheduler, epilogue, pad_*, persistent). Every valid combination becomes
    one GemmKernelConfig. Invalid combinations are dropped via the dispatcher's
    own validator, and duplicates (by .name) are collapsed.

    The operand signature (``dtype``, ``layout``) is applied to every emitted
    GemmKernelConfig, so the same sweep expands across any supported dtype/layout.

    For ``variant='multi_abd'`` the ``layout`` is the 4-char (A,B,E,D) code
    (e.g. ``rcrr``); the tensor counts and per-group element-wise ops are carried
    onto every produced config so they participate in the kernel name. For
    ``variant='multi_d'`` the ``layout`` may be 4-char (4th char = D-tensor
    layout) and each base config is further expanded over ``multi_d_config``
    (elementwise_ops x num_d_tensors), mirroring the codegen's multi_d expansion.

    ``arch`` may be ``None`` (or omitted): it is resolved once here via
    :func:`_resolve_arch` (rocminfo-detect + validate) so every produced config
    carries a concrete, supported ``gfx_arch`` -- the compile command's
    ``-DGFX_ARCH`` / ``--offload-arch`` never see ``None``. An explicit,
    unsupported arch raises ``ValueError``.
    """
    # Multi-ABD is fp16-only end-to-end (codegen, ctypes lib, and GpuMultiABDRunner
    # all assume fp16). Reject other dtypes here -- before any codegen/build -- so
    # callers get a clear error instead of a runtime failure after kernels compile.
    if variant == "multi_abd" and dtype != "fp16":
        raise ValueError(
            f"multi_abd bridge supports fp16 only, got {dtype!r}; "
            "codegen, ctypes lib and the runner are all fp16-only for this variant"
        )

    # Resolve the arch up front so it cannot silently default: this is the single
    # value stamped onto every emitted config's .gfx_arch below.
    arch = _resolve_arch(arch)

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

    # Preshuffle B-shuffle permutation knob -- pinned to the single source of
    # truth BRIDGE_PERMUTE_N (see its definition for the full rationale). We
    # deliberately ignore cfg.get("permute_n"): both default_config.json and
    # default_ci_config.json ship permute_n=true, but that TE host-marker selects
    # a permuteN pipeline the bridge does not codegen (it is NOT a distinct
    # bridged device kernel), so honoring it would mis-shuffle B. Do NOT "fix" this
    # to read the config until the permuteN pipeline is bridged.
    # TODO: support permute_n=True by emitting the permuteN pipeline in
    # unified_gemm_codegen and flipping BRIDGE_PERMUTE_N to a config-driven value.
    permute_n = BRIDGE_PERMUTE_N

    # Stream-K only: sweep reduction strategies (atomic/linear/tree). Other
    # variants keep a single dummy value so the product is unaffected.
    if variant == "stream_k":
        sk = cfg.get("streamk_config", {})
        reductions = _expand_values(sk.get("reduction_strategy"), ["atomic"])
    else:
        reductions = ["atomic"]

    la, lb, lc = layout[0], layout[1], layout[2]
    # Multi-ABD carries a 4th (D) layout char; default D to C's layout otherwise.
    ld = layout[3] if (variant == "multi_abd" and len(layout) >= 4) else lc

    # Multi-D: 4th layout char (if present) is the D-tensor layout; default row.
    d_layout_char = layout[3] if (variant == "multi_d" and len(layout) >= 4) else "r"
    d_layout_word = _LAYOUT_WORD[d_layout_char]

    # Multi-ABD (B2): the tensor counts and per-group element-wise ops are a real
    # part of the swept configuration -- distinct ops produce distinct kernels
    # (different epilogue math). Read them from an optional ``multi_abd_config``
    # block in the TE config JSON (lists of values), falling back to the scalar
    # kwargs (which default to the Old-TE 2/2/2 all-PassThrough combo). This is
    # what lets the driver actually generate + verify a non-PassThrough kernel.
    #
    # Allowed ops mirror the Old-TE gemm_multi_abd instance builder:
    #   {PassThrough, AddScale, MultiDMultiply, MultiDAdd}.
    if variant == "multi_abd":
        mabd = dict(cfg.get("multi_abd_config", {}) or {})
        # CLI overrides win over both the config block and the scalar defaults.
        if mabd_cli_overrides:
            mabd.update(mabd_cli_overrides)

        def _as_list(v, default):
            if v is None:
                return list(default)
            return list(v) if isinstance(v, (list, tuple)) else [v]

        na_list = _as_list(mabd.get("num_a_tensors"), [num_a_tensors])
        nb_list = _as_list(mabd.get("num_b_tensors"), [num_b_tensors])
        nd_list = _as_list(mabd.get("num_d_tensors"), [num_d_tensors])
        # CK's GemmKernelMultiABD requires >=1 A and B tensors and
        # DsLayout::size() > 0 (num_d_tensors >= 1); a 0 or non-integer count
        # otherwise fails later with a cryptic tuple-size-0 compile error, so
        # reject it here with a clear message.
        for _label, _vals in (
            ("num_a_tensors", na_list),
            ("num_b_tensors", nb_list),
            ("num_d_tensors", nd_list),
        ):
            for _v in _vals:
                if not isinstance(_v, int) or _v < 1:
                    raise ValueError(
                        f"multi_abd {_label} must be a positive integer (>= 1), "
                        f"got {_v!r}"
                    )
        a_ops = _as_list(mabd.get("a_elementwise_op"), [a_elementwise_op])
        b_ops = _as_list(mabd.get("b_elementwise_op"), [b_elementwise_op])
        cde_ops = _as_list(mabd.get("cde_elementwise_op"), [cde_elementwise_op])
        _ALLOWED_MABD_OPS = {"PassThrough", "AddScale", "MultiDMultiply", "MultiDAdd"}
        for op in (*a_ops, *b_ops, *cde_ops):
            if op not in _ALLOWED_MABD_OPS:
                raise ValueError(
                    f"Invalid multi_abd element-wise op {op!r}; "
                    f"valid: {sorted(_ALLOWED_MABD_OPS)}"
                )
        mabd_combos = list(
            itertools.product(na_list, nb_list, nd_list, a_ops, b_ops, cde_ops)
        )
    else:
        # Non-multi_abd variants: a single, inert combo carrying the scalar
        # kwargs (unused by those code paths' names).
        mabd_combos = [
            (
                num_a_tensors,
                num_b_tensors,
                num_d_tensors,
                a_elementwise_op,
                b_elementwise_op,
                cde_elementwise_op,
            )
        ]

    # Multi-D expansion combos (elementwise_op, num_d); a single ("PassThrough",0)
    # entry for non-multi_d variants keeps the loop below variant-agnostic.
    if variant == "multi_d":
        mdc = cfg.get("multi_d_config", {})
        md_ops = _expand_values(mdc.get("elementwise_ops"), ["MultiDAdd"])
        md_nds = _expand_values(mdc.get("num_d_tensors"), [2])
        md_combos = list(itertools.product(md_ops, md_nds))
    else:
        md_combos = [("PassThrough", 0)]

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
        red,
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
        reductions,
    ):
        # Tile/CShuffle correctness gate. A block tile must split evenly across
        # its waves -- tile % (wave * warp_tile) == 0 -- else the kernel is
        # genuinely invalid.
        #
        # Narrowed CShuffle-store gate (issue #9684): the CShuffle epilogue only
        # mis-stores the accumulator for one specific combination -- an ODD
        # per-wave repeat (>1) paired with a 32-wide warp tile in that dimension.
        # GPU-verified on gfx942: the tile_m=192 / wave_m=2 / warp_tile_m=32
        # configs (MRepeat = 192/(2*32) = 3) return garbage, while EVERY other
        # non-power-of-two repeat is numerically correct -- including MRepeat=3
        # with warp_tile_m=16 (192/(4*16)) and even non-pow2 repeats like 6 and
        # 12. The previous "per-wave repeat must be a power of two" rule was too
        # broad and needlessly dropped 90 valid configs. The "default" epilogue
        # stores directly and is exempt.
        m_div = wm * wtm
        n_div = wn * wtn
        if m_div <= 0 or n_div <= 0 or tm % m_div != 0 or tn % n_div != 0:
            continue
        # Parity gate: only emit warps-per-block triples Old-TE allows
        # (WARP_SUPPORTED_COMBINATIONS[arch]); see _warp_config_supported.
        if not _warp_config_supported(wm, wn, wk, arch):
            continue
        # gfx1250 correctness gate (ROCm/rocm-libraries#11161): the compv3
        # intrawave pipeline is hand-scheduled for MFMA/CDNA (wave64) and
        # miscompiles for 8-warp blocks (2x4x1 / 4x2x1) on gfx1250 (WMMA/wave32),
        # producing wrong results (on-device max_rel 0.14-0.87 vs an fp32 CPU
        # reference; <=4-warp compv3 and all compv4/mem/interwave kernels are
        # bit-accurate). Gate it off until the pipeline is ported to wave32.
        if arch == "gfx1250" and pipe == "compv3" and sched == "intrawave" and wm * wn == 8:
            continue
        if epi == "cshuffle" and not _cshuffle_store_ok(
            tm // m_div, tn // n_div, wtm, wtn
        ):
            continue

        for (m_na, m_nb, m_nd, m_aop, m_bop, m_cdeop) in mabd_combos:
            for ew_op, md_nd in md_combos:
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
                    reduction_strategy=red,
                    permute_n=permute_n,
                    num_a_tensors=m_na,
                    num_b_tensors=m_nb,
                    num_d_tensors=(md_nd if variant == "multi_d" else m_nd),
                    a_elementwise_op=m_aop,
                    b_elementwise_op=m_bop,
                    cde_elementwise_op=m_cdeop,
                    layout_d=_LAYOUT_WORD[ld],
                    elementwise_op=ew_op,
                    d_layout=d_layout_word,
                )
                if c.name in seen:
                    continue
                val = _cu.validate_kernel_config(c.to_ctypes_config())
                if not val.is_valid:
                    continue
                seen.add(c.name)
                configs.append(c)

    return configs