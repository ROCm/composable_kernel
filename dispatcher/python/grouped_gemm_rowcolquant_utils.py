#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GroupedGemm RowColQuant dispatcher utilities.

Three-layer Python bridge for the dispatcher's RowColQuant Grouped GEMM path:

  RowColQuantKernelConfig  — describes one kernel; .name is byte-exact with codegen KERNEL_NAME
  RowColQuantDispatcherLib — thin ctypes wrapper around a compiled .so
  RowColQuantGpuGemmRunner — high-level runner that accepts numpy arrays

Build helpers:
  setup_multiple_rowcolquant_dispatchers(configs, ...)
       codegen → hipcc → list of .so paths, all in parallel

RowColQuant: A has per-row scales [M, 1], B has per-column scales [1, N].
ADataType=BDataType=fp8/bf8; AQDataType=BQDataType=float; CDataType=half.
"""

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

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_grouped_gemm_rowcolquant_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "grouped_gemm_rowcolquant_ctypes_lib.cpp"

_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
# Import from codegen_common, not from the codegen script: the runtime path should not
# depend on the generator. This matches how the aquant/bquant/abquant utils resolve
# their name builders, and keeps the shared tile/trait defaults in one place so this
# module's default_*_config() cannot drift from the codegen's _default_config().
from codegen_common import (  # noqa: E402
    ROWCOL_TENSOR_QUANT_DEFAULT_TILE,
    ROWCOL_TENSOR_QUANT_DEFAULT_TRAITS,
    make_rowcolquant_kernel_name,
)

_DEFAULT_HIPCC    = "hipcc"
_DEFAULT_GFX_ARCH = "gfx950"


# =============================================================================
# RowColQuantKernelConfig — byte-exact naming with codegen
# =============================================================================


@dataclass
class RowColQuantKernelConfig:
    """
    Complete description of one RowColQuant Grouped GEMM kernel.

    The .name property produces the exact string that unified_grouped_gemm_rowcolquant_codegen.py
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
        return make_rowcolquant_kernel_name(
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
        """Produce the JSON config dict consumed by unified_grouped_gemm_rowcolquant_codegen.py."""
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
# RowColQuantGemmProblem
# =============================================================================


@dataclass
class RowColQuantGemmProblem:
    M: int
    N: int
    K: int
    k_batch: int = 1

    @property
    def QK_A(self) -> int:
        """Number of AQ elements (one per row). Used for buffer sizing only; the kernel uses broadcast strides."""
        return self.M

    @property
    def QK_B(self) -> int:
        """Number of BQ elements (one per column). Used for buffer sizing only; the kernel uses broadcast strides."""
        return self.N


# =============================================================================
# RowColQuantGemmResult
# =============================================================================


@dataclass
class RowColQuantGemmResult:
    C: object
    time_ms: float
    kernel_name: str


# =============================================================================
# RowColQuantDispatcherLib — thin ctypes wrapper
# =============================================================================


class RowColQuantDispatcherLib:
    """
    Loads a compiled rowcolquant_gemm .so and wraps its C API.

    Expected .so exports:
      int  dispatcher_initialize()
      int  dispatcher_run_gemm(A, B, AQ, BQ, C,
                               M, N, K,
                               stride_A, stride_B, stride_AQ, stride_BQ, stride_C,
                               QK_A, QK_B, k_batch, *time_ms)
      char* dispatcher_get_kernel_name()
      int   dispatcher_get_kernel_count()
      void  dispatcher_cleanup()
    """

    def __init__(self, so_path: Path):
        self.so_path = Path(so_path)
        self._cleaned_up = False
        if not self.so_path.exists():
            raise FileNotFoundError(f"RowColQuant .so not found: {self.so_path}")
        self._lib = ctypes.CDLL(str(self.so_path))
        self._setup()
        rc = self._lib.dispatcher_initialize()
        if rc != 0:
            raise RuntimeError(f"dispatcher_initialize() returned {rc}")

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
# RowColQuantGpuGemmRunner — high-level runner
# =============================================================================


class RowColQuantGpuGemmRunner:
    """
    High-level runner that loads a RowColQuant .so and executes GEMM on the GPU.

    Accepts numpy arrays for A, B, AQ, BQ; allocates C; returns RowColQuantGemmResult.
    """

    def __init__(self, so_path: Path):
        self._lib = RowColQuantDispatcherLib(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(self, A, B, AQ, BQ, problem: RowColQuantGemmProblem, c_dtype=None) -> RowColQuantGemmResult:
        """
        Run RowColQuant Grouped GEMM.

        A       shape: (M, K)     dtype: fp8/bf8  (row-major)
        B       shape: (K, N)     dtype: fp8/bf8  (col-major)
        AQ      shape: (M,)       dtype: float    (per-row A scale)
        BQ      shape: (N,)       dtype: float    (per-col B scale)
        c_dtype numpy dtype for the output C buffer. Defaults to np.float16.
        Returns RowColQuantGemmResult with C shape (M, N).
        """
        import numpy as np

        M, N, K = problem.M, problem.N, problem.K
        QK_A = problem.QK_A  # == M
        QK_B = problem.QK_B  # == N

        if A.ndim != 2 or A.shape != (M, K):
            raise ValueError(f"A shape mismatch: expected ({M}, {K}), got {A.shape}")
        if B.ndim != 2 or B.shape != (K, N):
            raise ValueError(f"B shape mismatch: expected ({K}, {N}), got {B.shape}")
        if AQ.ndim != 1 or AQ.shape[0] != M:
            raise ValueError(f"AQ shape mismatch: expected ({M},), got {AQ.shape}")
        if BQ.ndim != 1 or BQ.shape[0] != N:
            raise ValueError(f"BQ shape mismatch: expected ({N},), got {BQ.shape}")
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

        # Strides for A, B, C (standard packed layouts).
        stride_A = K   # A row-major [M, K]
        stride_B = K   # B col-major [K, N], leading dim = K
        stride_C = N   # C row-major [M, N]

        # stride_AQ=1, stride_BQ=1: the C++ lib requires exactly 1 and the RowColQuant
        # kernel never reads the scale strides at all -- it builds its AQ/BQ views with
        # literal broadcast strides derived from M and N.
        rc, time_ms = self._lib.run(
            A=A, B=B, AQ=AQ, BQ=BQ, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A, stride_B=stride_B,
            stride_AQ=1, stride_BQ=1, stride_C=stride_C,
            QK_A=QK_A, QK_B=QK_B,
            k_batch=problem.k_batch,
        )

        if rc != 0:
            raise RuntimeError(
                f"dispatcher_run_gemm failed with code {rc} "
                f"for kernel {self.kernel_name}"
            )

        return RowColQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


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
                return line
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


def _generate_rowcolquant_kernel(
    config: RowColQuantKernelConfig,
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


def _compile_rowcolquant_kernel(
    hpp_path: Path,
    so_path: Path,
    gfx_arch: str,
    hipcc: str = _DEFAULT_HIPCC,
    extra_include_dirs: Optional[List[str]] = None,
) -> bool:
    """Compile a generated .hpp into a .so via hipcc (compile then link)."""
    ck_include = _get_ck_include_dir()
    static_lib = _get_dispatcher_static_lib()

    obj_path = so_path.with_suffix(".o")

    arch_defines = []
    if "gfx12" in gfx_arch or "gfx950" in gfx_arch:
        arch_defines += ["-DCK_USE_OCP_FP8", "-DCK_TILE_USE_OCP_FP8"]
    if "gfx950" in gfx_arch:
        arch_defines += ["-DCK_USE_NATIVE_MX_SUPPORT", "-DCK_GFX950_SUPPORT"]

    compile_cmd = [hipcc, "-c", "-fPIC", "-O3", "-std=c++17",
                   "-DCK_TILE_SINGLE_KERNEL_INCLUDE", "-w",
                   f"--offload-arch={gfx_arch}",
                   f"-DGFX_ARCH=\"{gfx_arch}\"",
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
# setup_multiple_rowcolquant_dispatchers — build pipeline
# =============================================================================


def setup_multiple_rowcolquant_dispatchers(
    configs: List[RowColQuantKernelConfig],
    output_dir: Optional[Path] = None,
    hipcc: str = _DEFAULT_HIPCC,
    gfx_arch: Optional[str] = None,
    extra_include_dirs: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """
    For each RowColQuantKernelConfig: codegen → hipcc compile → .so path.

    Returns a list parallel to `configs` — each entry is the Path to the
    compiled .so, or None if that config failed.
    """
    if not configs:
        return []

    arch = gfx_arch or _detect_gpu_arch()
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix="rowcolquant_dispatcher_"))
    base_dir.mkdir(parents=True, exist_ok=True)

    headers_dir = base_dir / "generated_kernels"
    so_dir      = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    log.info("Building %d RowColQuant kernel(s) for %s into %s", len(configs), arch, base_dir)

    seen: Dict[str, int] = {}
    deduped: List[Tuple[int, RowColQuantKernelConfig]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx: int, cfg: RowColQuantKernelConfig) -> Tuple[int, Optional[Path]]:
        hpp = _generate_rowcolquant_kernel(cfg, headers_dir)
        if hpp is None:
            return idx, None

        so = so_dir / f"lib{cfg.name}_{arch}.so"
        if so.exists():
            log.info("  [cached] %s", so.name)
            return idx, so

        ok = _compile_rowcolquant_kernel(
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
    log.info("Built %d / %d RowColQuant kernels", built, len(configs))
    return results


# =============================================================================
# Convenience: default fp8 and bf8 configs
# =============================================================================


def _default_config(dtype: str, gfx_arch: str) -> RowColQuantKernelConfig:
    """Build the default config for `dtype` from the shared codegen defaults.

    Sourcing tile and traits from codegen_common means this runtime default and the
    codegen's _default_config() cannot drift: a tile change in one place changes the
    kernel name produced by both, so the .so the runner looks for is the .so codegen
    emits. Every trait that feeds the kernel name (pipeline/epilogue/scheduler and all
    four pad/persistent flags) is forwarded from the shared dict; only block_size and
    k_block_per_cu are codegen-only and are left to the dataclass defaults.
    """
    traits = ROWCOL_TENSOR_QUANT_DEFAULT_TRAITS
    return RowColQuantKernelConfig(
        dtype=dtype,
        layout="rcr",
        pipeline=traits["pipeline"],
        epilogue=traits["epilogue"],
        scheduler=traits["scheduler"],
        **ROWCOL_TENSOR_QUANT_DEFAULT_TILE,
        pad_m=traits["pad_m"],
        pad_n=traits["pad_n"],
        pad_k=traits["pad_k"],
        persistent=traits["persistent"],
        gfx_arch=gfx_arch,
    )


def default_fp8_config(gfx_arch: str = _DEFAULT_GFX_ARCH) -> RowColQuantKernelConfig:
    """Return the default fp8 RowColQuant config."""
    return _default_config("fp8", gfx_arch)


def default_bf8_config(gfx_arch: str = _DEFAULT_GFX_ARCH) -> RowColQuantKernelConfig:
    """Return the default bf8 RowColQuant config."""
    return _default_config("bf8", gfx_arch)
