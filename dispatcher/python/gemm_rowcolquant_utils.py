#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Gemm RowColQuant dispatcher utilities.

Three-layer Python bridge for the dispatcher's RowColQuant GEMM path:

  RowColQuantKernelConfig  -- describes one kernel; .name is byte-exact with
                              codegen KERNEL_NAME
  RowColQuantDispatcherLib -- thin ctypes wrapper around a compiled .so
  RowColQuantGpuGemmRunner -- high-level runner that accepts numpy arrays

RowColQuant = per-row scale of A (AQ, M floats) + per-column scale of B (BQ, N
floats). There is no quant-group size: the scales are global row/col vectors, so
the ctypes signature carries no QK_A/QK_B/stride_BQ arguments.

Build helpers (self-contained, do not import from gemm_utils.py):
  setup_multiple_rowcolquant_dispatchers(configs, ...)
       codegen -> hipcc -> list of .so paths, all in parallel

Usage (end-to-end):
  configs = [RowColQuantKernelConfig(variant_key="fp8", layout="rcr", ...)]
  so_paths = setup_multiple_rowcolquant_dispatchers(configs, output_dir=Path("/tmp/rc"))
  runner = RowColQuantGpuGemmRunner(so_paths[0])
  result = runner.run(A, B, AQ, BQ, RowColQuantGemmProblem(M=16, N=64, K=256))
"""

import ctypes
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

# Shared quant-bridge scaffolding (ctypes API install, codegen subprocess, build
# orchestration, CK include probe). Op-specific parts stay in this file.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from quant_bridge_base import (  # noqa: E402
    DispatcherLibBase,
    build_dispatchers,
    find_ck_include_dir,
    generate_kernel,
)

# =============================================================================
# Constants
# =============================================================================

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_gemm_rowcolquant_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "gemm_rowcolquant_ctypes_lib.cpp"

# Import the shared name-construction helper from codegen_common so both sides
# stay byte-exact without duplicating the logic.
_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from codegen_common import make_gemm_rowcolquant_kernel_name  # noqa: E402

_DEFAULT_HIPCC    = "hipcc"

# --- Tile-Engine perf flags: single source of truth (quant_bridge_flags.py) ---
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from quant_bridge_flags import te_perf_flags as _te_perf_flags  # noqa: E402
# --- end Tile-Engine perf flags ---

_DEFAULT_GFX_ARCH = "gfx950"


# =============================================================================
# RowColQuantKernelConfig -- byte-exact naming with codegen
# =============================================================================


@dataclass
class RowColQuantKernelConfig:
    """
    Complete description of one RowColQuant GEMM kernel.

    The .name property produces the exact string that
    unified_gemm_rowcolquant_codegen.py emits as KERNEL_NAME, ensuring the Python
    side and compiled .so always agree.
    """

    variant_key: str       # "fp8" or "bf8"
    layout: str            # "rcr" (A=RowMajor, B=ColMajor, C=RowMajor)
    pipeline: str          # "compv3"
    epilogue: str          # "cshuffle"
    scheduler: str         # "intrawave"

    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    double_smem_buffer: bool = False
    k_block_per_cu: int      = 1

    gfx_arch: str = _DEFAULT_GFX_ARCH

    @property
    def name(self) -> str:
        """Byte-exact match to codegen KERNEL_NAME (delegates to make_gemm_rowcolquant_kernel_name)."""
        return make_gemm_rowcolquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            warp_m=self.warp_m, warp_n=self.warp_n, warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m, warp_tile_n=self.warp_tile_n, warp_tile_k=self.warp_tile_k,
        )

    def to_codegen_config(self) -> dict:
        """Produce the JSON config dict consumed by unified_gemm_rowcolquant_codegen.py."""
        return {
            "variant_keys": [self.variant_key],
            "layouts": [self.layout],
            "pipeline": self.pipeline,
            "epilogue": self.epilogue,
            "scheduler": self.scheduler,
            "tile_configs": [{
                "tile_m": self.tile_m,
                "tile_n": self.tile_n,
                "tile_k": self.tile_k,
                "warp_m": self.warp_m,
                "warp_n": self.warp_n,
                "warp_k": self.warp_k,
                "warp_tile_m": self.warp_tile_m,
                "warp_tile_n": self.warp_tile_n,
                "warp_tile_k": self.warp_tile_k,
            }],
            "double_smem_buffer": self.double_smem_buffer,
            "k_block_per_cu": self.k_block_per_cu,
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


# =============================================================================
# RowColQuantGemmResult
# =============================================================================


@dataclass
class RowColQuantGemmResult:
    C: object          # numpy array
    time_ms: float
    kernel_name: str


# =============================================================================
# RowColQuantDispatcherLib -- thin ctypes wrapper
# =============================================================================


class RowColQuantDispatcherLib(DispatcherLibBase):
    """
    Loads a compiled rowcolquant_gemm .so and wraps its C API.

    Expected .so exports:
      int  dispatcher_initialize()
      int  dispatcher_run_rowcolquant_gemm(A, B, AQ, BQ, C, M, N, K,
                                           stride_A, stride_B, stride_C,
                                           k_batch, *time_ms)
      char* dispatcher_get_kernel_name()
      int   dispatcher_get_kernel_count()
      void  dispatcher_cleanup()

    The initialize / get_kernel_name / get_kernel_count / cleanup scaffold lives
    in DispatcherLibBase; only the op-specific dispatcher_run argtypes (below) and
    the run() marshalling stay here.
    """

    _NOT_FOUND_LABEL = "RowColQuant"
    _RUN_SYMBOL = "dispatcher_run_rowcolquant_gemm"
    _RUN_ARGTYPES = [
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
        ctypes.c_int64,    # stride_C
        ctypes.c_int,      # k_batch
        ctypes.POINTER(ctypes.c_float),  # time_ms
    ]

    def run(
        self,
        A,
        B,
        AQ,
        BQ,
        C,
        M: int,
        N: int,
        K: int,
        stride_A: int,
        stride_B: int,
        stride_C: int,
        k_batch: int = 1,
    ) -> Tuple[int, float]:
        """
        Call dispatcher_run_rowcolquant_gemm with ctypes-wrapped pointers.

        A, B, AQ, BQ, C must be numpy arrays (C-contiguous, packed).
        B should be a packed (K, N) array supplied column-major (stride_B=K).
        AQ is the per-row scale (M floats), BQ the per-column scale (N floats).
        Returns (status, time_ms).
        """
        import numpy as np

        # A/B are consumed as raw 1-byte fp8/bf8 (const fp8_t*/bf8_t*) on the
        # C++ side. Reject wider dtypes so float32/float16 can't be silently
        # reinterpreted as fp8 (which would read a fraction of the buffer as
        # garbage). Encode real fp8/bf8 bytes via encode_fp8_bytes() first.
        A = np.asarray(A)
        B = np.asarray(B)
        if A.dtype.itemsize != 1 or B.dtype.itemsize != 1:
            raise TypeError(
                f"A and B must be 1-byte fp8/bf8 arrays (got A={A.dtype}, "
                f"B={B.dtype}); encode via encode_fp8_bytes() before run()."
            )

        A  = np.ascontiguousarray(A)
        # Kernel BLayout is ColumnMajor (rcr): B[k,n] lives at offset n*K+k.
        # Supply column-major bytes for 2-D B; ascontiguousarray would force
        # row-major and silently transpose.
        B  = np.asfortranarray(B) if B.ndim == 2 else np.ascontiguousarray(B)
        # Scales are always float32 on the device side.
        AQ = np.ascontiguousarray(AQ, dtype=np.float32)
        BQ = np.ascontiguousarray(BQ, dtype=np.float32)
        C  = np.ascontiguousarray(C)

        time_ms = ctypes.c_float(0.0)

        rc = self._lib.dispatcher_run_rowcolquant_gemm(
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
            ctypes.c_int64(stride_C),
            ctypes.c_int(k_batch),
            ctypes.byref(time_ms),
        )
        return rc, time_ms.value


# =============================================================================
# RowColQuantGpuGemmRunner -- high-level runner
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
        Run RowColQuant GEMM.

        A       shape: (M, K)            dtype: fp8/bf8
        B       shape: (K, N) col-major  dtype: fp8/bf8
        AQ      shape: (M,)              dtype: float32 (per-row scale)
        BQ      shape: (N,)              dtype: float32 (per-column scale)
        c_dtype numpy dtype for the output C buffer. Defaults to np.float16
                (correct for fp8/bf8 variants whose CDataType is half_t).
        Returns RowColQuantGemmResult with C shape (M, N).
        """
        import numpy as np

        M, N, K = problem.M, problem.N, problem.K

        if c_dtype is None:
            c_dtype = np.float16

        # Output buffer -- dtype must match the compiled kernel's CDataType.
        C = np.zeros((M, N), dtype=c_dtype)

        # Strides (in elements): row-major A and C; col-major B has leading dim K.
        stride_A = K   # A is row-major [M, K]
        stride_B = K   # B is col-major [K, N] -> leading dim = K
        stride_C = N   # C is row-major [M, N]

        rc, time_ms = self._lib.run(
            A=A, B=B, AQ=AQ, BQ=BQ, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A,
            stride_B=stride_B,
            stride_C=stride_C,
            k_batch=problem.k_batch,
        )

        if rc != 0:
            raise RuntimeError(
                f"dispatcher_run_rowcolquant_gemm failed with code {rc} "
                f"for kernel {self.kernel_name}"
            )

        return RowColQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


# =============================================================================
# fp8 / bf8 host-side encode helpers (for genuine numeric self-tests)
# =============================================================================
#
# The kernel's ADataType/BDataType are 1-byte fp8 / bf8. The *encoding* is
# arch-dependent (mirrors CK_USE_OCP_FP8): gfx950 uses OCP fp8 (e4m3) / bf8
# (e5m2), while gfx942 uses FNUZ fp8 (e4m3fnuz) / bf8 (e5m2fnuz). Getting the
# encoding wrong makes the numpy reference silently NaN on gfx942 (OCP e4m3 has
# no inf/nan-free NUZ layout, so many gfx942-valid bit patterns decode to NaN).
# The ctypes lib reinterprets the raw A/B pointers as const fp8_t*, so the host
# MUST hand it actual 1-byte-per-element encoded buffers -- NOT float32 arrays
# (that would make it read a quarter of the buffer as garbage). These helpers
# encode float32 -> fp8/bf8 bytes and decode back, so a self-test can both feed
# real bytes to the kernel and round its numpy reference identically.


# Map (variant_key, encoding) -> the ml_dtypes fp8 dtype name that matches the
# kernel's on-device encoding. OCP is used on gfx950 (fp8 = e4m3, bf8 = e5m2);
# FNUZ is used on gfx942/other (fp8 = e4m3fnuz, bf8 = e5m2fnuz), mirroring the
# CK_USE_OCP_FP8 compile-time switch.
_ML_DTYPE_FOR_VARIANT_OCP  = {"fp8": "float8_e4m3", "bf8": "float8_e5m2"}
_ML_DTYPE_FOR_VARIANT_FNUZ = {"fp8": "float8_e4m3fnuz", "bf8": "float8_e5m2fnuz"}


def _uses_ocp_fp8(gfx_arch: Optional[str]) -> bool:
    """True if the arch uses OCP fp8/bf8 (gfx950/gfx12); False -> FNUZ (gfx942).

    Mirrors the CK_USE_OCP_FP8 compile-time switch. Defaults to OCP when the
    arch is unknown (None) to preserve the historical gfx950 self-test default.
    """
    if not gfx_arch:
        return True
    return "gfx950" in gfx_arch or "gfx12" in gfx_arch


def _ml_fp8_dtype(variant_key: str, gfx_arch: Optional[str] = None):
    """Return the numpy fp8 dtype (via ml_dtypes) for a variant+arch, or None.

    The dtype flavour (OCP vs FNUZ) is chosen from ``gfx_arch`` to match the
    kernel's on-device encoding; passing None keeps the OCP default.
    """
    try:
        import ml_dtypes  # noqa: F401
    except Exception:
        return None
    table = _ML_DTYPE_FOR_VARIANT_OCP if _uses_ocp_fp8(gfx_arch) else _ML_DTYPE_FOR_VARIANT_FNUZ
    name = table.get(variant_key)
    if name is None:
        return None
    return getattr(__import__("ml_dtypes"), name, None)


def encode_fp8_bytes(arr, variant_key: str, gfx_arch: Optional[str] = None):
    """Encode a float32/float array to packed 1-byte-per-element fp8/bf8.

    Returns a C-contiguous uint8 numpy array with the same shape as ``arr``
    whose raw bytes are exactly what the kernel expects to read as fp8_t*/bf8_t*.
    The fp8 flavour (OCP on gfx950, FNUZ on gfx942) is selected from ``gfx_arch``.
    Requires ml_dtypes; raises RuntimeError if unavailable.
    """
    import numpy as np

    dt = _ml_fp8_dtype(variant_key, gfx_arch)
    if dt is None:
        raise RuntimeError(
            f"ml_dtypes fp8 dtype unavailable for variant {variant_key!r}; "
            "cannot encode real fp8/bf8 bytes for a numeric self-test."
        )
    enc = np.asarray(arr, dtype=np.float32).astype(dt)
    # Reinterpret the 1-byte fp8 storage as uint8 without changing bit pattern.
    return np.ascontiguousarray(enc).view(np.uint8)


def quantize_dequantize_fp8(arr, variant_key: str, gfx_arch: Optional[str] = None):
    """Round a float array through fp8/bf8 and back to float32.

    This is the reference-side counterpart to encode_fp8_bytes: it applies the
    exact same fp8 rounding the kernel sees (OCP on gfx950, FNUZ on gfx942), so
    a numpy reference computed on the result is a fair comparison. Requires
    ml_dtypes.
    """
    import numpy as np

    dt = _ml_fp8_dtype(variant_key, gfx_arch)
    if dt is None:
        raise RuntimeError(
            f"ml_dtypes fp8 dtype unavailable for variant {variant_key!r}."
        )
    return np.asarray(arr, dtype=np.float32).astype(dt).astype(np.float32)


def fp8_encoding_available() -> bool:
    """True if ml_dtypes fp8/bf8 encoding is importable (for genuine verify)."""
    return _ml_fp8_dtype("fp8") is not None


# =============================================================================
# Subprocess helpers (self-contained, do not call ctypes_utils.py)
# =============================================================================


def _detect_gpu_arch() -> str:
    """Detect current GPU arch via rocm_agent_enumerator.

    Raises RuntimeError if the enumerator cannot be run or returns no usable
    arch. We deliberately do NOT silently fall back to a default arch: a flaky
    enumerator on (say) a gfx942 box would otherwise build gfx950 objects that
    fail to load at runtime. Callers who know their target arch should pass
    ``gfx_arch=`` explicitly to avoid detection entirely.
    """
    try:
        result = subprocess.run(
            ["rocm_agent_enumerator"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        raise RuntimeError(
            "rocm_agent_enumerator failed to run; cannot detect GPU arch. "
            "Pass gfx_arch= explicitly to build for a known target."
        ) from e

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("gfx") and line != "gfx000":
            return line

    raise RuntimeError(
        "rocm_agent_enumerator returned no usable gfx arch "
        f"(stdout={result.stdout!r}). Pass gfx_arch= explicitly."
    )


def _get_ck_include_dir() -> Optional[Path]:
    """Attempt to locate the CK include directory relative to this file."""
    return find_ck_include_dir()


def _generate_rowcolquant_kernel(
    config: RowColQuantKernelConfig,
    output_dir: Path,
) -> Optional[Path]:
    """
    Run unified_gemm_rowcolquant_codegen.py for one config; return the .hpp path or None.
    """
    return generate_kernel(config, output_dir, _CODEGEN_SCRIPT)


def _compile_rowcolquant_kernel(
    hpp_path: Path,
    so_path: Path,
    gfx_arch: str,
    hipcc: str = _DEFAULT_HIPCC,
    extra_include_dirs: Optional[List[str]] = None,
) -> bool:
    """
    Compile a generated .hpp into a .so via hipcc (compile then link).

    Two-step build:
      1. Compile to a .o object file.
      2. Link the .o into a shared .so (no dispatcher static lib needed;
         the RowColQuant ctypes lib does not use the registry or dispatcher).

    Returns True on success.
    """
    ck_include = _get_ck_include_dir()

    # -- Step 1: compile to object file --------------------------------------
    obj_path = so_path.with_suffix(".o")

    # Arch-specific defines: gfx950 uses OCP fp8 (not FNUZ). These mirror the
    # CMakeLists.txt definitions that are normally injected by CMake but are
    # absent in the standalone hipcc build path.
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
                   *_te_perf_flags(hipcc),
                   "-include", str(hpp_path),
                   str(_CTYPES_LIB_SRC),
                   "-o", str(obj_path)]

    if ck_include:
        compile_cmd += [f"-I{ck_include}"]

    # NOTE: dispatcher/include is intentionally excluded here (same reason as the
    # BQuant ctypes lib): it pulls in generated_tile_backend.hpp which instantiates
    # SelectedKernel::launch(GemmHostArgs&), conflicting with the RowColQuant
    # kernel's launch(QuantGemmHostArgs&).

    if extra_include_dirs:
        for d in extra_include_dirs:
            compile_cmd += [f"-I{d}"]

    log.debug("Compiling %s:\n  %s", so_path.name, " ".join(compile_cmd))

    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            log.error("Compile failed for %s:\n%s", so_path.name, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        log.error("Compile timed out for %s", so_path.name)
        return False

    # -- Step 2: link into shared library ------------------------------------
    link_cmd = [hipcc, "-shared", "-fPIC",
                f"--offload-arch={gfx_arch}", "--hip-link",
                str(obj_path), "-o", str(so_path)]

    log.debug("Linking %s:\n  %s", so_path.name, " ".join(link_cmd))

    try:
        result = subprocess.run(
            link_cmd,
            capture_output=True, text=True, timeout=120,
        )
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
# setup_multiple_rowcolquant_dispatchers -- build pipeline
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
    For each RowColQuantKernelConfig: codegen -> hipcc compile -> .so path.

    Returns a list parallel to `configs` -- each entry is the Path to the
    compiled .so, or None if that config failed.

    No GPU is required to call this function.
    """
    if not configs:
        return []

    arch = gfx_arch or _detect_gpu_arch()

    def _compile_fn(hpp: Path, so: Path, a: str) -> bool:
        return _compile_rowcolquant_kernel(
            hpp_path=hpp, so_path=so, gfx_arch=a,
            hipcc=hipcc, extra_include_dirs=extra_include_dirs,
        )

    return build_dispatchers(
        configs,
        arch=arch,
        tmp_prefix="rowcolquant_dispatcher_",
        log_label="RowColQuant",
        generate_fn=_generate_rowcolquant_kernel,
        compile_fn=_compile_fn,
        output_dir=output_dir,
        parallel=parallel,
        max_workers=max_workers,
    )


# =============================================================================
# Sweep expansion: JSON config -> list of RowColQuantKernelConfig
# =============================================================================


def expand_rowcolquant_sweep(
    config_path: str,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> List["RowColQuantKernelConfig"]:
    """Expand a RowColQuant JSON sweep config into RowColQuantKernelConfig objects.

    The JSON format mirrors unified_gemm_rowcolquant_codegen.py's _build_specs so
    the same config files work for both codegen and Python utils. Every valid
    (variant, layout, tile) combination produces one RowColQuantKernelConfig;
    duplicates (by .name) are collapsed.
    """
    import itertools

    with open(config_path) as f:
        cfg = json.load(f)

    pipeline           = cfg.get("pipeline", "compv3")
    epilogue           = cfg.get("epilogue", "cshuffle")
    scheduler          = cfg.get("scheduler", "intrawave")
    k_block_per_cu     = cfg.get("k_block_per_cu", 1)
    double_smem_buffer = cfg.get("double_smem_buffer", False)

    configs: List[RowColQuantKernelConfig] = []
    seen: set = set()

    for variant_key, layout, tile_dict in itertools.product(
        cfg.get("variant_keys", ["fp8"]),
        cfg.get("layouts", ["rcr"]),
        cfg.get("tile_configs", []),
    ):
        c = RowColQuantKernelConfig(
            variant_key=variant_key,
            layout=layout,
            pipeline=pipeline,
            epilogue=epilogue,
            scheduler=scheduler,
            tile_m=tile_dict["tile_m"],
            tile_n=tile_dict["tile_n"],
            tile_k=tile_dict["tile_k"],
            warp_m=tile_dict["warp_m"],
            warp_n=tile_dict["warp_n"],
            warp_k=tile_dict["warp_k"],
            warp_tile_m=tile_dict["warp_tile_m"],
            warp_tile_n=tile_dict["warp_tile_n"],
            warp_tile_k=tile_dict["warp_tile_k"],
            double_smem_buffer=double_smem_buffer,
            k_block_per_cu=k_block_per_cu,
            gfx_arch=gfx_arch,
        )
        if c.name not in seen:
            seen.add(c.name)
            configs.append(c)

    return configs


# =============================================================================
# Convenience: default fp8/bf8 configs (match GemmConfigRowColQuant<T>)
# =============================================================================


def _warp_tile_k_for(variant_key: str, gfx_arch: str) -> int:
    """Arch-derived K warp-tile, mirroring ck_tile::get_k_warp_tile<PrecType, 16>().

    (tile_gemm_shape.hpp, M_Warp_Tile=16, non-WMMA path)
      gfx950 (CK_GFX950_SUPPORT): fp8/bf8 -> 128
      gfx942/other              : fp8/bf8 ->  32   (no 16x16x128 fp8/bf8 warp-gemm)

    This is a BLOCKING correctness constraint, not just a naming detail: a
    warp_tile_k=128 fp8/bf8 kernel *compiles* on gfx942 but silently produces
    all-zeros output (confirmed on the sibling tensor_quant GPU tester). Old-TE
    uses 16x16x32 on gfx942 and is bit-exact there with warp_tile_k=32.
    """
    is_8bit_float = variant_key in ("fp8", "bf8")
    if "gfx950" in gfx_arch and is_8bit_float:
        return 128
    return 32


def default_fp8_config(gfx_arch: str = _DEFAULT_GFX_ARCH) -> RowColQuantKernelConfig:
    """Return the default fp8 RowColQuant config (tile = 16x64x256, warp = 1x4x1).

    Matches GemmConfigRowColQuant<fp8_t>. WarpTileK is arch-derived via
    get_k_warp_tile<fp8_t, M_Warp_Tile=16>(): 128 on gfx950, 32 on gfx942
    (128 silently outputs all-zeros on gfx942).
    """
    return RowColQuantKernelConfig(
        variant_key="fp8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16,
        warp_tile_k=_warp_tile_k_for("fp8", gfx_arch),
        gfx_arch=gfx_arch,
    )


def default_bf8_config(gfx_arch: str = _DEFAULT_GFX_ARCH) -> RowColQuantKernelConfig:
    """Return the default bf8 RowColQuant config (tile = 16x64x256, warp = 1x4x1).

    Matches GemmConfigRowColQuant<bf8_t>. WarpTileK is arch-derived via
    get_k_warp_tile<bf8_t, M_Warp_Tile=16>(): 128 on gfx950, 32 on gfx942
    (128 silently outputs all-zeros on gfx942).
    """
    return RowColQuantKernelConfig(
        variant_key="bf8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16,
        warp_tile_k=_warp_tile_k_for("bf8", gfx_arch),
        gfx_arch=gfx_arch,
    )
