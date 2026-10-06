#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GroupedGemm TensorQuant dispatcher utilities.

Three-layer Python bridge for the dispatcher's TensorQuant Grouped GEMM path:

  TensorQuantKernelConfig  — describes one kernel; .name is byte-exact with codegen KERNEL_NAME
  TensorQuantDispatcherLib — thin ctypes wrapper around a compiled .so
  TensorQuantGpuGemmRunner — high-level runner that accepts numpy arrays

Build helpers:
  setup_multiple_tensorquant_dispatchers(configs, ...)
       codegen → hipcc → list of .so paths, all in parallel

TensorQuant: A and B each have a single per-tensor scalar scale.
ADataType=BDataType=fp8/bf8; AQDataType=BQDataType=float; CDataType=half.
"""

from dispatcher_common import unified_framework_flags
import ctypes
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures

log = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_grouped_gemm_tensorquant_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "grouped_gemm_tensorquant_ctypes_lib.cpp"

_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
# Import from codegen_common, not from the codegen script: the runtime path should not
# depend on the generator. This matches how the aquant/bquant/abquant utils resolve
# their name builders, and keeps the shared tile/trait defaults in one place so this
# module's default_*_config() cannot drift from the codegen's _default_config().
from codegen_common import (  # noqa: E402
    ROWCOL_TENSOR_QUANT_DEFAULT_TRAITS,
    rowcol_tensor_quant_default_tile,
    normalize_gfx_arch,
    validate_rowcol_tensor_quant_gfx_arch,
    make_tensorquant_kernel_name,
)

_python_dir = str(Path(__file__).parent)
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)
from dispatcher_common import arch_feature_defines  # noqa: E402

_DEFAULT_HIPCC    = "hipcc"
_DEFAULT_GFX_ARCH = "gfx950"

# ABI revision of the compiled .so, folded into the artifact filename.
#
# setup_multiple_tensorquant_dispatchers() reuses an existing .so when the name
# matches, and callers that pass a persistent output_dir (the validation
# harnesses do, precisely to get cache hits) can therefore be handed an artifact
# built by an older revision of this module. The name used to be keyed on
# (kernel name, arch) only, which does not describe the exported symbol set, so
# a .so predating the dispatcher_get_tile_n()/dispatcher_get_pad_n() exports
# would be selected and then fail at attribute-lookup time with a bare
# "undefined symbol". Bump this whenever the exported C ABI of
# bindings/ctypes/grouped_gemm_tensorquant_ctypes_lib.cpp changes; the new name
# simply cannot collide with the stale artifact, which is then ignored rather
# than papered over with a runtime fallback.
#   1 -> original export set
#   2 -> added dispatcher_get_tile_n() / dispatcher_get_pad_n()
#   3 -> require regenerated headers with build-target validation
_SO_ABI = 3


# =============================================================================
# TensorQuantKernelConfig — byte-exact naming with codegen
# =============================================================================


@dataclass
class TensorQuantKernelConfig:
    """
    Complete description of one TensorQuant Grouped GEMM kernel.

    The .name property produces the exact string that unified_grouped_gemm_tensorquant_codegen.py
    emits as KERNEL_NAME, ensuring the Python side and compiled .so always agree.
    """

    dtype: str       # "fp8" or "bf8"
    layout: str      # "rcr"
    pipeline: str    # "compv3"
    epilogue: str    # "cshuffle"
    scheduler: str   # "intrawave"

    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = True
    persistent: bool = False
    block_size: int = 256
    k_block_per_cu: int = 1

    gfx_arch: str = _DEFAULT_GFX_ARCH

    @property
    def name(self) -> str:
        """Byte-exact match to codegen KERNEL_NAME."""
        return make_tensorquant_kernel_name(
            dtype=self.dtype,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            pad_m=self.pad_m,
            pad_n=self.pad_n,
            pad_k=self.pad_k,
            persistent=self.persistent,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            warp_m=self.warp_m, warp_n=self.warp_n, warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m, warp_tile_n=self.warp_tile_n, warp_tile_k=self.warp_tile_k,
        )

    def to_codegen_config(self) -> dict:
        """Produce the JSON config dict consumed by unified_grouped_gemm_tensorquant_codegen.py."""
        return {
            "dtypes": [self.dtype],
            "layouts": [self.layout],
            "pipeline": self.pipeline,
            "epilogue": self.epilogue,
            "scheduler": self.scheduler,
            "pad_m": self.pad_m,
            "pad_n": self.pad_n,
            "pad_k": self.pad_k,
            "persistent": self.persistent,
            "block_size": self.block_size,
            "k_block_per_cu": self.k_block_per_cu,
            "tile_configs": [{
                "tile_m": self.tile_m, "tile_n": self.tile_n, "tile_k": self.tile_k,
                "warp_m": self.warp_m, "warp_n": self.warp_n, "warp_k": self.warp_k,
                "warp_tile_m": self.warp_tile_m, "warp_tile_n": self.warp_tile_n, "warp_tile_k": self.warp_tile_k,
            }],
        }


# =============================================================================
# TensorQuantGemmProblem
# =============================================================================


@dataclass
class TensorQuantGemmProblem:
    M: int
    N: int
    K: int
    k_batch: int = 1


# =============================================================================
# TensorQuantGemmResult
# =============================================================================


@dataclass
class TensorQuantGemmResult:
    C: object
    time_ms: float
    kernel_name: str


# =============================================================================
# TensorQuantDispatcherLib — thin ctypes wrapper
# =============================================================================


class TensorQuantDispatcherLib:
    """
    Loads a compiled tensorquant_gemm .so and wraps its C API.

    Expected .so exports:
      int  dispatcher_initialize()
      int  dispatcher_run_gemm(A, B, AQ, BQ, C,
                               M, N, K,
                               stride_A, stride_B, stride_AQ, stride_BQ, stride_C,
                               QK_A, QK_B, k_batch, *time_ms)
      char* dispatcher_get_kernel_name()
      int   dispatcher_get_kernel_count()
      int   dispatcher_get_tile_n()
      int   dispatcher_get_pad_n()
      void  dispatcher_cleanup()
    """

    def __init__(self, so_path: Path):
        self.so_path = Path(so_path)
        self._cleaned_up = True
        if not self.so_path.exists():
            raise FileNotFoundError(f"TensorQuant .so not found: {self.so_path}")
        self._lib = ctypes.CDLL(str(self.so_path))
        self._setup()
        rc = self._lib.dispatcher_initialize()
        if rc != 0:
            raise RuntimeError(f"dispatcher_initialize() returned {rc}")
        self._cleaned_up = False

    def _setup(self):
        lib = self._lib

        lib.dispatcher_initialize.restype  = ctypes.c_int
        lib.dispatcher_initialize.argtypes = []

        lib.dispatcher_run_gemm.restype  = ctypes.c_int
        lib.dispatcher_run_gemm.argtypes = [
            ctypes.c_void_p,   # A
            ctypes.c_void_p,   # B
            ctypes.c_void_p,   # AQ
            ctypes.c_void_p,   # BQ
            ctypes.c_void_p,   # C
            ctypes.c_int64,    # M
            ctypes.c_int64,    # N
            ctypes.c_int64,    # K
            ctypes.c_int64,    # stride_A
            ctypes.c_int64,    # stride_B
            ctypes.c_int64,    # stride_AQ
            ctypes.c_int64,    # stride_BQ
            ctypes.c_int64,    # stride_C
            ctypes.c_int64,    # QK_A
            ctypes.c_int64,    # QK_B
            ctypes.c_int,      # k_batch
            ctypes.POINTER(ctypes.c_float),  # time_ms
        ]

        lib.dispatcher_get_kernel_name.restype  = ctypes.c_char_p
        lib.dispatcher_get_kernel_name.argtypes = []

        lib.dispatcher_get_kernel_count.restype  = ctypes.c_int
        lib.dispatcher_get_kernel_count.argtypes = []

        # Direct callers can supply libraries outside the versioned cache.
        try:
            lib.dispatcher_get_tile_n.restype = ctypes.c_int
            lib.dispatcher_get_tile_n.argtypes = []
            lib.dispatcher_get_pad_n.restype = ctypes.c_int
            lib.dispatcher_get_pad_n.argtypes = []
        except AttributeError as exc:
            raise RuntimeError(
                f"Incompatible quant bridge ABI in {self.so_path}: missing TileN/PadN "
                "exports (ABI 2 or newer required). Rebuild the library."
            ) from exc

        lib.dispatcher_cleanup.restype  = None
        lib.dispatcher_cleanup.argtypes = []

    def run(
        self,
        A, B, AQ, BQ, C,
        M: int, N: int, K: int,
        stride_A: int, stride_B: int,
        stride_AQ: int, stride_BQ: int, stride_C: int,
        QK_A: int, QK_B: int,
        k_batch: int = 1,
    ) -> Tuple[int, float]:
        """Call dispatcher_run_gemm with ctypes-wrapped pointers.

        B must already be F-contiguous (column-major) — the caller (GpuGemmRunner)
        converts it with asfortranarray before passing it here.  Using
        ascontiguousarray on a 2-D F-contiguous array would silently copy it back
        to C order, making the declared stride_B=K incorrect.

        C is the output buffer and is written in place by the C library, so it must
        already be C-contiguous; a non-contiguous C raises rather than being copied.
        """
        import numpy as np
        A  = np.ascontiguousarray(A)
        # Preserve F-contiguous layout for B (rcr: column-major B, stride_B = K).
        B  = np.asfortranarray(B) if B.ndim == 2 else np.ascontiguousarray(B)
        AQ = np.ascontiguousarray(AQ)
        BQ = np.ascontiguousarray(BQ)

        # Inputs may be copied into a contiguous temporary because the copy is what
        # gets uploaded. C may not: the library memcpys the device result back into
        # whatever buffer this pointer names. Copying C would send the results into a
        # temporary that is discarded on return, and the caller's array would silently
        # keep its pre-call contents.
        if not C.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "C must be a C-contiguous array; it is written in place. "
                "Pass np.ascontiguousarray(C) and copy the result back yourself, "
                "or allocate C with np.empty/np.zeros."
            )

        time_ms = ctypes.c_float(0.0)
        rc = self._lib.dispatcher_run_gemm(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            AQ.ctypes.data_as(ctypes.c_void_p),
            BQ.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(M),
            ctypes.c_int64(N),
            ctypes.c_int64(K),
            ctypes.c_int64(stride_A),
            ctypes.c_int64(stride_B),
            ctypes.c_int64(stride_AQ),
            ctypes.c_int64(stride_BQ),
            ctypes.c_int64(stride_C),
            ctypes.c_int64(QK_A),
            ctypes.c_int64(QK_B),
            ctypes.c_int(k_batch),
            ctypes.byref(time_ms),
        )
        return rc, time_ms.value

    def get_kernel_name(self) -> str:
        raw = self._lib.dispatcher_get_kernel_name()
        return raw.decode("utf-8") if raw else ""

    def get_kernel_count(self) -> int:
        return self._lib.dispatcher_get_kernel_count()

    def get_tile_n(self) -> int:
        """N-tile of the compiled kernel (SelectedKernel::TileN)."""
        return self._lib.dispatcher_get_tile_n()

    def get_pad_n(self) -> bool:
        """True when the compiled kernel was generated with pad_n=true."""
        return bool(self._lib.dispatcher_get_pad_n())

    def cleanup(self):
        if not self._cleaned_up:
            self._lib.dispatcher_cleanup()
            self._cleaned_up = True

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


# =============================================================================
# TensorQuantGpuGemmRunner — high-level runner
# =============================================================================


class TensorQuantGpuGemmRunner:
    """
    High-level runner that loads a TensorQuant .so and executes GEMM on the GPU.

    Accepts numpy arrays for A, B, AQ (scalar), BQ (scalar); allocates C; returns TensorQuantGemmResult.
    """

    def __init__(self, so_path: Path):
        self._lib = TensorQuantDispatcherLib(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    @property
    def tile_n(self) -> int:
        """N-tile of the compiled kernel, read from the .so (not hardcoded)."""
        return self._lib.get_tile_n()

    @property
    def pad_n(self) -> bool:
        """True when the compiled kernel pads N (the N % tile_n rule then lifts)."""
        return self._lib.get_pad_n()

    def run(self, A, B, AQ, BQ, problem: TensorQuantGemmProblem, c_dtype=None) -> TensorQuantGemmResult:
        """
        Run TensorQuant Grouped GEMM.

        A    shape: (M, K)   dtype: fp8/bf8  (row-major)
        B    shape: (K, N)   dtype: fp8/bf8  (col-major)
        AQ   shape: (1,)     dtype: float    (per-tensor A scale)
        BQ   shape: (1,)     dtype: float    (per-tensor B scale)
        c_dtype numpy dtype for the output C buffer. Defaults to np.float16.
        Returns TensorQuantGemmResult with C shape (M, N).

        Shape constraints enforced by the bridge (validated on gfx1250 with the
        default config). Violations raise RuntimeError; no output is written:

          K % 16 == 0     Required even though the default config sets pad_k=True --
                          pad_k covers the K-loop tail, not the global-load vector
                          width.
          N % tile_n == 0 Required whenever the kernel was generated with pad_n=False,
                          so N must be a whole number of N-tiles. tile_n is a property
                          of the compiled kernel -- read `self.tile_n` rather than
                          assuming the value of today's default config.

        There is no M constraint: TensorQuant applies scalar A/B scales rather than a
        per-row scale vector, so it is unaffected by the unpadded per-row AQ tail that
        restricts the RowColQuant bridge to M % 4 == 0 on gfx12.
        """
        import numpy as np

        M, N, K = problem.M, problem.N, problem.K

        if A.ndim != 2 or A.shape != (M, K):
            raise ValueError(f"A shape mismatch: expected ({M}, {K}), got {A.shape}")
        if B.ndim != 2 or B.shape != (K, N):
            raise ValueError(f"B shape mismatch: expected ({K}, {N}), got {B.shape}")
        if AQ.ndim != 1 or AQ.shape[0] != 1:
            raise ValueError(f"AQ shape mismatch: expected (1,), got {AQ.shape}")
        if BQ.ndim != 1 or BQ.shape[0] != 1:
            raise ValueError(f"BQ shape mismatch: expected (1,), got {BQ.shape}")
        # fp8/bf8 have no native numpy dtype; both are 1-byte elements.
        if A.itemsize != 1:
            raise ValueError(f"A dtype must be a 1-byte fp8/bf8 type, got {A.dtype} (itemsize={A.itemsize})")
        if B.itemsize != 1:
            raise ValueError(f"B dtype must be a 1-byte fp8/bf8 type, got {B.dtype} (itemsize={B.itemsize})")
        if AQ.dtype != np.float32:
            raise ValueError(f"AQ dtype must be float32, got {AQ.dtype}")
        if BQ.dtype != np.float32:
            raise ValueError(f"BQ dtype must be float32, got {BQ.dtype}")

        if c_dtype is None:
            c_dtype = np.float16
        if c_dtype != np.float16:
            raise ValueError(
                f"c_dtype must be float16 (the compiled ABI always writes CDataType=half); "
                f"got {c_dtype}"
            )

        C = np.zeros((M, N), dtype=c_dtype)

        # B is column-major (rcr layout): the kernel expects leading dim = K (stride_B = K),
        # which means elements are stored column-first in memory (Fortran order).
        # Reorder here so the raw pointer passed to C++ matches the stride we declare below.
        # (self._lib.run also calls asfortranarray; on an already-F-contiguous array that
        # is a no-op, so the conversion happens exactly once.)
        B = np.asfortranarray(B)

        # TensorQuant: single scalar scale per tensor → AQ/BQ strides are 1
        stride_A  = K
        stride_B  = K
        stride_AQ = 1
        stride_BQ = 1
        stride_C  = N

        # TensorQuant: single scalar scale per tensor → QK_A=1, QK_B=1
        rc, time_ms = self._lib.run(
            A=A, B=B, AQ=AQ, BQ=BQ, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A, stride_B=stride_B,
            stride_AQ=stride_AQ, stride_BQ=stride_BQ, stride_C=stride_C,
            QK_A=1, QK_B=1,
            k_batch=problem.k_batch,
        )

        if rc != 0:
            # rc alone is not actionable, and the C++ explanation goes to stderr, which
            # a caller capturing only the exception never sees. Restate the constraints
            # here so the traceback is self-contained. tile_n/pad_n come from the .so,
            # so this message stays true if the compiled tile changes.
            n_rule = (
                "N is unconstrained (pad_n=True)"
                if self.pad_n
                else f"N % {self.tile_n} == 0 when pad_n=False"
            )
            raise RuntimeError(
                f"dispatcher_run_gemm failed with code {rc} "
                f"for kernel {self.kernel_name} at M={M} N={N} K={K}. "
                f"(-1 = rejected by the bridge, -2 = rejected by the kernel, "
                f"-3 = launch threw.) Shape constraints: K % 16 == 0 (required even "
                f"with pad_k=True); {n_rule}. "
                f"See stderr for the exact reason."
            )

        return TensorQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


# =============================================================================
# Subprocess helpers
# =============================================================================


def _detect_gpu_arch() -> str:
    """Detect current GPU arch via rocm_agent_enumerator. Falls back to gfx950."""
    try:
        result = subprocess.run(
            ["rocm_agent_enumerator"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("gfx") and line != "gfx000":
                # Strip feature suffixes ("gfx1250:xnack-") here so they never reach
                # --offload-arch / -DGFX_ARCH. The C++ side prefix-matches the runtime
                # device name against the compile-time GFX_ARCH, so the bare target
                # still matches a device that reports suffixes.
                return normalize_gfx_arch(line)
    except Exception as e:
        log.warning("rocm_agent_enumerator failed (%s); defaulting to %s", e, _DEFAULT_GFX_ARCH)
        return _DEFAULT_GFX_ARCH
    log.warning("rocm_agent_enumerator returned no usable arch; defaulting to %s", _DEFAULT_GFX_ARCH)
    return _DEFAULT_GFX_ARCH


def _get_ck_include_dir() -> Optional[Path]:
    """Attempt to locate the CK include directory relative to this file."""
    here = Path(__file__).resolve().parent
    for parent in [here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "include"
        if (candidate / "ck_tile").is_dir():
            return candidate
    return None


def _generate_tensorquant_kernel(
    config: TensorQuantKernelConfig,
    output_dir: Path,
) -> Optional[Path]:
    """Run codegen for one config; return the .hpp path or None."""
    config_dict = config.to_codegen_config()
    config_json = json.dumps(config_dict)

    cmd = [
        sys.executable,
        str(_CODEGEN_SCRIPT),
        "--output-dir", str(output_dir),
        "--config-json", config_json,
        "--gfx-arch", validate_rowcol_tensor_quant_gfx_arch(config.gfx_arch),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log.error("Codegen failed for %s:\n%s", config.name, result.stderr)
            return None
    except subprocess.TimeoutExpired:
        log.error("Codegen timed out for %s", config.name)
        return None

    hpp = output_dir / f"{config.name}.hpp"
    if not hpp.exists():
        log.error("Codegen succeeded but %s not found", hpp)
        return None

    return hpp


def _get_dispatcher_static_lib() -> Optional[Path]:
    """Return libck_tile_dispatcher.a from the CMake build directory, or None."""
    dispatcher_root = _CTYPES_LIB_SRC.parent.parent.parent
    static_lib = dispatcher_root / "build" / "libck_tile_dispatcher.a"
    return static_lib if static_lib.exists() else None


def _compile_tensorquant_kernel(
    hpp_path: Path,
    so_path: Path,
    gfx_arch: str,
    hipcc: str = _DEFAULT_HIPCC,
    extra_include_dirs: Optional[List[str]] = None,
) -> bool:
    """Compile a generated .hpp into a .so via hipcc (compile then link)."""
    # Normalize once, here at the boundary, so every downstream use -- the arch
    # defines below and the --offload-arch/-DGFX_ARCH we hand to hipcc -- sees the
    # bare target. A caller-supplied "gfx1250:xnack-" must not reach the compiler
    # flags.
    gfx_arch = validate_rowcol_tensor_quant_gfx_arch(gfx_arch)

    ck_include = _get_ck_include_dir()
    static_lib = _get_dispatcher_static_lib()

    obj_path = so_path.with_suffix(".o")

    arch_defines = arch_feature_defines(gfx_arch)
    # Match top-level CK policy in both host and device compilation.
    if gfx_arch == "gfx1250":
        arch_defines.append("-DUSE_NEW_UNIFIED_FRAMEWORK=0")

    compile_cmd = [hipcc, "-c", "-fPIC", "-O3", "-std=c++17",
                   "-DCK_TILE_SINGLE_KERNEL_INCLUDE", "-w",
                   f"--offload-arch={gfx_arch}",
                   f"-DCK_CMAKE_GPU_TARGET_IDS=0x{gfx_arch[3:]}",
                   f"-DGFX_ARCH=\"{gfx_arch}\"",
                   *unified_framework_flags(gfx_arch),
                   *arch_defines,
                   "-include", str(hpp_path),
                   str(_CTYPES_LIB_SRC),
                   "-o", str(obj_path)]

    if ck_include:
        compile_cmd += [f"-I{ck_include}"]

    if extra_include_dirs:
        for d in extra_include_dirs:
            compile_cmd += [f"-I{d}"]

    log.debug("Compiling %s:\n  %s", so_path.name, " ".join(compile_cmd))

    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log.error("Compile failed for %s:\n%s", so_path.name, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        log.error("Compile timed out for %s", so_path.name)
        obj_path.unlink(missing_ok=True)
        return False

    link_cmd = [hipcc, "-shared", "-fPIC",
                f"--offload-arch={gfx_arch}", "--hip-link",
                str(obj_path)]

    if static_lib:
        link_cmd += [str(static_lib)]

    link_cmd += ["-o", str(so_path)]

    log.debug("Linking %s:\n  %s", so_path.name, " ".join(link_cmd))

    try:
        result = subprocess.run(link_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log.error("Link failed for %s:\n%s", so_path.name, result.stderr[-2000:])
            obj_path.unlink(missing_ok=True)
            return False
    except subprocess.TimeoutExpired:
        log.error("Link timed out for %s", so_path.name)
        obj_path.unlink(missing_ok=True)
        return False

    obj_path.unlink(missing_ok=True)
    return True


# =============================================================================
# setup_multiple_tensorquant_dispatchers — build pipeline
# =============================================================================


def setup_multiple_tensorquant_dispatchers(
    configs: List[TensorQuantKernelConfig],
    output_dir: Optional[Path] = None,
    hipcc: str = _DEFAULT_HIPCC,
    gfx_arch: Optional[str] = None,
    extra_include_dirs: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """
    For each TensorQuantKernelConfig: codegen → hipcc compile → .so path.

    Returns a list parallel to `configs` — each entry is the Path to the
    compiled .so, or None if that config failed.
    """
    if not configs:
        return []

    # Normalize the explicit branch too, not just detection: an explicitly passed
    # "gfx1250:xnack-" would otherwise flow into --offload-arch, -DGFX_ARCH and the
    # .so cache name. _detect_gpu_arch() already normalizes.
    arch = validate_rowcol_tensor_quant_gfx_arch(gfx_arch or _detect_gpu_arch())
    for cfg in configs:
        config_arch = validate_rowcol_tensor_quant_gfx_arch(cfg.gfx_arch)
        if config_arch != arch:
            raise ValueError(
                f"Config architecture {cfg.gfx_arch!r} does not match build target {arch!r}. "
                "Create configs for the selected build target."
            )
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix="tensorquant_dispatcher_"))
    base_dir.mkdir(parents=True, exist_ok=True)

    headers_dir = base_dir / "generated_kernels"
    so_dir      = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    log.info("Building %d TensorQuant kernel(s) for %s into %s", len(configs), arch, base_dir)

    seen: Dict[str, int] = {}
    deduped: List[Tuple[int, TensorQuantKernelConfig]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx: int, cfg: TensorQuantKernelConfig) -> Tuple[int, Optional[Path]]:
        hpp = _generate_tensorquant_kernel(cfg, headers_dir)
        if hpp is None:
            return idx, None

        # Name carries the ABI revision: a pre-_SO_ABI artifact in a persistent
        # output_dir can never be selected here (see _SO_ABI).
        so = so_dir / f"lib{cfg.name}_{arch}_abi{_SO_ABI}.so"
        if so.exists():
            log.info("  [cached] %s", so.name)
            return idx, so

        ok = _compile_tensorquant_kernel(
            hpp_path=hpp, so_path=so, gfx_arch=arch,
            hipcc=hipcc, extra_include_dirs=extra_include_dirs,
        )
        return idx, so if ok else None

    if parallel and len(deduped) > 1:
        workers = max_workers or min(len(deduped), os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_build_one, idx, cfg): (idx, cfg) for idx, cfg in deduped}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    idx, so_path = fut.result()
                    results[idx] = so_path
                    if so_path:
                        log.info("  built %s", so_path.name)
                    else:
                        _, cfg = futures[fut]
                        log.error("  FAILED %s", cfg.name)
                except Exception as e:
                    _, cfg = futures[fut]
                    log.error("  EXCEPTION for %s: %s", cfg.name, e)
    else:
        for idx, cfg in deduped:
            _, so_path = _build_one(idx, cfg)
            results[idx] = so_path

    for i, cfg in enumerate(configs):
        if results[i] is None:
            first_idx = seen.get(cfg.name)
            if first_idx is not None and first_idx != i:
                results[i] = results[first_idx]
                if results[i] is None:
                    log.debug("  dedup: %s (index %d) inherits failed build from index %d",
                              cfg.name, i, first_idx)

    built = sum(1 for r in results if r is not None)
    log.info("Built %d / %d TensorQuant kernels", built, len(configs))
    return results


# =============================================================================
# Convenience: default fp8 and bf8 configs
# =============================================================================


def _default_config(dtype: str, gfx_arch: str) -> TensorQuantKernelConfig:
    """Build the default config for `dtype` from the shared codegen defaults.

    Sourcing tile and traits from codegen_common means this runtime default and the
    codegen's _default_config() cannot drift: a tile change in one place changes the
    kernel name produced by both, so the .so the runner looks for is the .so codegen
    emits. Every trait that feeds the kernel name (pipeline/epilogue/scheduler and all
    four pad/persistent flags) is forwarded from the shared dict; only block_size and
    k_block_per_cu are codegen-only and are left to the dataclass defaults.
    """
    # Normalize at this boundary too: default_fp8_config()/default_bf8_config() are
    # public entry points, so a caller-supplied "gfx1250:xnack-" must not be stored
    # on the config and observed by later consumers.
    gfx_arch = validate_rowcol_tensor_quant_gfx_arch(gfx_arch)
    traits = ROWCOL_TENSOR_QUANT_DEFAULT_TRAITS
    return TensorQuantKernelConfig(
        dtype=dtype,
        layout="rcr",
        pipeline=traits["pipeline"],
        epilogue=traits["epilogue"],
        scheduler=traits["scheduler"],
        **rowcol_tensor_quant_default_tile(gfx_arch),
        pad_m=traits["pad_m"],
        pad_n=traits["pad_n"],
        pad_k=traits["pad_k"],
        persistent=traits["persistent"],
        gfx_arch=gfx_arch,
    )


def default_fp8_config(gfx_arch: str = _DEFAULT_GFX_ARCH) -> TensorQuantKernelConfig:
    """Return the default fp8 TensorQuant config."""
    return _default_config("fp8", gfx_arch)


def default_bf8_config(gfx_arch: str = _DEFAULT_GFX_ARCH) -> TensorQuantKernelConfig:
    """Return the default bf8 TensorQuant config."""
    return _default_config("bf8", gfx_arch)
