#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GroupedGemm BQuant dispatcher utilities.

Three-layer Python bridge for the dispatcher's BQuantGrouped GEMM path:

  BQuantKernelConfig  — describes one kernel; .name is byte-exact with codegen KERNEL_NAME
  BQuantDispatcherLib — thin ctypes wrapper around a compiled .so
  BQuantGpuGemmRunner — high-level runner that accepts numpy arrays

Build helpers (self-contained, do not import from gemm_utils.py):
  setup_multiple_bquant_dispatchers(configs, ...)
       codegen → hipcc → list of .so paths, all in parallel

Usage (end-to-end):
  configs = [BQuantKernelConfig(variant_key="fp8", layout="rcr", ...)]
  so_paths = setup_multiple_bquant_dispatchers(configs, output_dir=Path("/tmp/bq"))
  runner = BQuantGpuGemmRunner(so_paths[0])
  result = runner.run(A, B, BQ, BQuantGemmProblem(M=16, N=64, K=256))
"""

import ctypes
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures

log = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_grouped_gemm_bquant_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "grouped_gemm_bquant_ctypes_lib.cpp"

# Import the shared name-construction helper from codegen_common so both sides
# stay byte-exact without duplicating the logic.
_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from codegen_common import make_bquant_kernel_name  # noqa: E402

_DEFAULT_HIPCC    = "hipcc"
_DEFAULT_GFX_ARCH = "gfx950"

# Flags that match the tile engine / dispatcher build flags for BQuant kernels
_HIPCC_BASE_FLAGS = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-shared",
    "-DCK_TILE_SINGLE_KERNEL_INCLUDE",
    "-w",  # suppress warnings during generated-code compilation
]


# =============================================================================
# BQuantKernelConfig — byte-exact naming with codegen
# =============================================================================


@dataclass
class BQuantKernelConfig:
    """
    Complete description of one BQuantGrouped GEMM kernel.

    The .name property produces the exact string that unified_grouped_gemm_bquant_codegen.py
    emits as KERNEL_NAME, ensuring the Python side and compiled .so always agree.
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

    quant_group_m: int = 1
    quant_group_n: int = 1
    quant_group_k: int = 128

    preshuffle_b: bool      = False
    preshuffle_bquant: bool  = False
    double_smem_buffer: bool = False
    k_block_per_cu: int      = 1

    gfx_arch: str = _DEFAULT_GFX_ARCH

    @property
    def name(self) -> str:
        """Byte-exact match to codegen KERNEL_NAME (delegates to make_bquant_kernel_name)."""
        return make_bquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            warp_m=self.warp_m, warp_n=self.warp_n, warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m, warp_tile_n=self.warp_tile_n, warp_tile_k=self.warp_tile_k,
            quant_group_m=self.quant_group_m,
            quant_group_n=self.quant_group_n,
            quant_group_k=self.quant_group_k,
            preshuffle_b=self.preshuffle_b,
            preshuffle_bquant=self.preshuffle_bquant,
        )

    def to_codegen_config(self) -> dict:
        """Produce the JSON config dict consumed by unified_grouped_gemm_bquant_codegen.py."""
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
            "quant_groups": [{
                "quant_group_m": self.quant_group_m,
                "quant_group_n": self.quant_group_n,
                "quant_group_k": self.quant_group_k,
            }],
            "preshuffle_b": self.preshuffle_b,
            "preshuffle_bquant": self.preshuffle_bquant,
            "double_smem_buffer": self.double_smem_buffer,
            "k_block_per_cu": self.k_block_per_cu,
        }


# =============================================================================
# BQuantGemmProblem
# =============================================================================


@dataclass
class BQuantGemmProblem:
    M: int
    N: int
    K: int
    quant_group_m: int = 1
    quant_group_n: int = 1
    quant_group_k: int = 128
    k_batch: int = 1

    @property
    def QK_B(self) -> int:
        """Number of K-groups: ceil(K / quant_group_k)."""
        return (self.K + self.quant_group_k - 1) // self.quant_group_k

    @property
    def QN_B(self) -> int:
        """Number of N-groups: ceil(N / quant_group_n)."""
        return (self.N + self.quant_group_n - 1) // self.quant_group_n


# =============================================================================
# BQuantGemmResult
# =============================================================================


@dataclass
class BQuantGemmResult:
    C: object          # numpy array
    time_ms: float
    kernel_name: str


# =============================================================================
# BQuantDispatcherLib — thin ctypes wrapper
# =============================================================================


class BQuantDispatcherLib:
    """
    Loads a compiled bquant_gemm .so and wraps its C API.

    Expected .so exports:
      int  dispatcher_initialize()
      int  dispatcher_run_bquant_gemm(A, B, BQ, C, M, N, K,
                                       stride_A, stride_B, stride_BQ, stride_C,
                                       QK_B, QN_B, k_batch, *time_ms)
      char* dispatcher_get_kernel_name()
      int   dispatcher_get_kernel_count()
      void  dispatcher_cleanup()
    """

    def __init__(self, so_path: Path):
        self.so_path = Path(so_path)
        if not self.so_path.exists():
            raise FileNotFoundError(f"BQuant .so not found: {self.so_path}")
        self._lib = ctypes.CDLL(str(self.so_path))
        self._setup()
        rc = self._lib.dispatcher_initialize()
        if rc != 0:
            raise RuntimeError(f"dispatcher_initialize() returned {rc}")

    def _setup(self):
        lib = self._lib

        lib.dispatcher_initialize.restype  = ctypes.c_int
        lib.dispatcher_initialize.argtypes = []

        lib.dispatcher_run_bquant_gemm.restype  = ctypes.c_int
        lib.dispatcher_run_bquant_gemm.argtypes = [
            ctypes.c_void_p,   # A
            ctypes.c_void_p,   # B
            ctypes.c_void_p,   # BQ
            ctypes.c_void_p,   # C
            ctypes.c_int64,    # M
            ctypes.c_int64,    # N
            ctypes.c_int64,    # K
            ctypes.c_int64,    # stride_A
            ctypes.c_int64,    # stride_B
            ctypes.c_int64,    # stride_BQ
            ctypes.c_int64,    # stride_C
            ctypes.c_int64,    # QK_B
            ctypes.c_int64,    # QN_B
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
        A,
        B,
        BQ,
        C,
        M: int,
        N: int,
        K: int,
        stride_A: int,
        stride_B: int,
        stride_BQ: int,
        stride_C: int,
        QK_B: int,
        QN_B: int,
        k_batch: int = 1,
    ) -> Tuple[int, float]:
        """
        Call dispatcher_run_bquant_gemm with ctypes-wrapped pointers.

        A, B, BQ, C must be numpy arrays (C-contiguous, packed).
        B should be a packed (K, N) C-contiguous array — the kernel interprets
        it as column-major via stride_B=K, not via numpy's Fortran-order flag.
        C must be the array that will receive output; a non-contiguous C would
        produce a temporary copy that is not returned to the caller.
        Returns (status, time_ms).
        """
        import numpy as np

        A   = np.ascontiguousarray(A)
        # Kernel BLayout is ColumnMajor (rcr): B[k,n] lives at offset n*K+k.
        # Supply column-major bytes for 2-D B; ascontiguousarray would force
        # row-major and silently transpose. Packed 1-D B (fp4) stays as-is.
        B   = np.asfortranarray(B) if B.ndim == 2 else np.ascontiguousarray(B)
        BQ  = np.ascontiguousarray(BQ)
        C   = np.ascontiguousarray(C)

        time_ms = ctypes.c_float(0.0)

        rc = self._lib.dispatcher_run_bquant_gemm(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            BQ.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(M),
            ctypes.c_int64(N),
            ctypes.c_int64(K),
            ctypes.c_int64(stride_A),
            ctypes.c_int64(stride_B),
            ctypes.c_int64(stride_BQ),
            ctypes.c_int64(stride_C),
            ctypes.c_int64(QK_B),
            ctypes.c_int64(QN_B),
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
        self._lib.dispatcher_cleanup()

    def __del__(self):
        try:
            self._lib.dispatcher_cleanup()
        except Exception:
            pass


# =============================================================================
# BQuantGpuGemmRunner — high-level runner
# =============================================================================


class BQuantGpuGemmRunner:
    """
    High-level runner that loads a BQuant .so and executes GEMM on the GPU.

    Accepts numpy arrays for A, B, BQ; allocates C; returns BQuantGemmResult.
    """

    def __init__(self, so_path: Path):
        self._lib = BQuantDispatcherLib(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(self, A, B, BQ, problem: BQuantGemmProblem, c_dtype=None) -> BQuantGemmResult:
        """
        Run BQuantGrouped GEMM.

        A       shape: (M, K)           dtype: fp8/bf8
        B       shape: (K, N) col-major  dtype: fp8/bf8
        BQ      shape: (QK_B, QN_B)     dtype: float/fp8
        c_dtype numpy dtype for the output C buffer.  Defaults to np.float16
                (correct for fp8/bf8/fp8i4/bf8i4 variants).  Pass np.bfloat16
                for MX variants (mx_bf16bf16, mx_bf16bf8, mx_bf16fp4) whose
                CDataType is bf16.
        Returns BQuantGemmResult with C shape (M, N).
        """
        import numpy as np

        M, N, K = problem.M, problem.N, problem.K
        QK_B    = problem.QK_B
        QN_B    = problem.QN_B

        if c_dtype is None:
            c_dtype = np.float16

        # Output buffer — dtype must match the compiled kernel's CDataType.
        C = np.zeros((M, N), dtype=c_dtype)

        # Strides (in elements, row-major for A and C; col-major for B means stride = K)
        stride_A  = K   # A is row-major [M, K]
        stride_B  = K   # B is col-major [K, N] → leading dim = K
        stride_BQ = QN_B
        stride_C  = N   # C is row-major [M, N]

        rc, time_ms = self._lib.run(
            A=A, B=B, BQ=BQ, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A,
            stride_B=stride_B,
            stride_BQ=stride_BQ,
            stride_C=stride_C,
            QK_B=QK_B,
            QN_B=QN_B,
            k_batch=problem.k_batch,
        )

        if rc != 0:
            raise RuntimeError(
                f"dispatcher_run_bquant_gemm failed with code {rc} "
                f"for kernel {self.kernel_name}"
            )

        # permute_n epilogue riffles N-columns within each tile of width tile_n.
        # Undo it per-tile so the caller gets logical (row-major) C.
        _name = self.kernel_name
        if 'permute_n' in _name:
            import re as _re
            _m = _re.search(r'_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_', _name)
            if _m:
                _tile_n = int(_m.group(2)); _warp_n = int(_m.group(5)); _wt_n = int(_m.group(8))
                _r = _tile_n // _wt_n // _warp_n
                if _r > 1 and (N % _tile_n) == 0:
                    _half = _tile_n // _r
                    _logical = [
                        (c // _tile_n) * _tile_n + (c % _tile_n % _r) * _half + (c % _tile_n // _r)
                        for c in range(N)
                    ]
                    _Cp = np.empty_like(C)
                    _Cp[:, _logical] = C
                    C = _Cp
        return BQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


# =============================================================================
# Subprocess helpers (self-contained, do not call ctypes_utils.py)
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
    except Exception:
        pass
    return _DEFAULT_GFX_ARCH


def _get_ck_include_dir() -> Optional[Path]:
    """Attempt to locate the CK include directory relative to this file."""
    # Walk up from dispatcher/python/ to find project root
    here = Path(__file__).resolve().parent
    for parent in [here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "include"
        if (candidate / "ck_tile").is_dir():
            return candidate
    return None


def _generate_bquant_kernel(
    config: BQuantKernelConfig,
    output_dir: Path,
) -> Optional[Path]:
    """
    Run unified_grouped_gemm_bquant_codegen.py for one config; return the .hpp path or None.
    """
    config_dict = config.to_codegen_config()
    config_json = json.dumps(config_dict)

    cmd = [
        sys.executable,
        str(_CODEGEN_SCRIPT),
        "--output-dir", str(output_dir),
        "--config-json", config_json,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=120,
        )
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


def _compile_bquant_kernel(
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
         the BQuant ctypes lib does not use the registry or dispatcher).

    Returns True on success.
    """
    ck_include = _get_ck_include_dir()
    static_lib = _get_dispatcher_static_lib()

    # -- Step 1: compile to object file --------------------------------------
    obj_path = so_path.with_suffix(".o")

    # Arch-specific defines: gfx950 uses OCP fp8 (not FNUZ) and native MX support.
    # These mirror the CMakeLists.txt definitions that are normally injected by CMake
    # but are absent in the standalone hipcc build path.
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

    # NOTE: dispatcher/include is intentionally excluded here.
    # It pulls in generated_tile_backend.hpp which instantiates
    # SelectedKernel::launch(GemmHostArgs&), conflicting with the BQuant
    # kernel's launch(QuantGemmHostArgs&). The BQuant ctypes lib only needs
    # the main CK include path (ck_tile/host/tensor_shuffle_utils.hpp lives there).

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
                str(obj_path)]

    if static_lib:
        link_cmd += [str(static_lib)]

    link_cmd += ["-o", str(so_path)]

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
# setup_multiple_bquant_dispatchers — build pipeline
# =============================================================================


def setup_multiple_bquant_dispatchers(
    configs: List[BQuantKernelConfig],
    output_dir: Optional[Path] = None,
    hipcc: str = _DEFAULT_HIPCC,
    gfx_arch: Optional[str] = None,
    extra_include_dirs: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """
    For each BQuantKernelConfig: codegen → hipcc compile → .so path.

    Returns a list parallel to `configs` — each entry is the Path to the
    compiled .so, or None if that config failed.

    No GPU is required to call this function.
    """
    if not configs:
        return []

    arch = gfx_arch or _detect_gpu_arch()
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix="bquant_dispatcher_"))
    base_dir.mkdir(parents=True, exist_ok=True)

    headers_dir = base_dir / "generated_kernels"
    so_dir      = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    log.info(
        "Building %d BQuant kernel(s) for %s into %s",
        len(configs), arch, base_dir,
    )

    # Deduplicate by name so we don't build the same kernel twice
    seen: Dict[str, int] = {}          # name → index of first occurrence
    deduped: List[Tuple[int, BQuantKernelConfig]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    # results[i] = Path or None, aligned with input configs
    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx: int, cfg: BQuantKernelConfig) -> Tuple[int, Optional[Path]]:
        hpp = _generate_bquant_kernel(cfg, headers_dir)
        if hpp is None:
            return idx, None

        so = so_dir / f"lib{cfg.name}_{arch}.so"
        if so.exists():
            log.info("  [cached] %s", so.name)
            return idx, so

        ok = _compile_bquant_kernel(
            hpp_path=hpp,
            so_path=so,
            gfx_arch=arch,
            hipcc=hipcc,
            extra_include_dirs=extra_include_dirs,
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

    # Fill in duplicates
    for i, cfg in enumerate(configs):
        if results[i] is None:
            first_idx = seen.get(cfg.name)
            if first_idx is not None and first_idx != i:
                results[i] = results[first_idx]

    built = sum(1 for r in results if r is not None)
    log.info("Built %d / %d BQuant kernels", built, len(configs))
    return results


# =============================================================================
# Sweep expansion: JSON config → list of BQuantKernelConfig
# =============================================================================


def expand_bquant_sweep(
    config_path: str,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> List["BQuantKernelConfig"]:
    """Expand a BQuant JSON sweep config into a list of BQuantKernelConfig objects.

    The JSON format mirrors unified_grouped_gemm_bquant_codegen.py's _build_specs
    so the same config files work for both codegen and Python utils. Every valid
    (variant, layout, tile, quant_group) combination produces one BQuantKernelConfig;
    duplicates (by .name) are collapsed.

    JSON schema:
      variant_keys:       list of dtype variants, e.g. ["fp8", "bf8"]
      layouts:            list of layout strings, e.g. ["rcr"]
      pipeline:           pipeline name, e.g. "compv3"
      epilogue:           epilogue name, e.g. "cshuffle"
      scheduler:          scheduler name, e.g. "intrawave"
      tile_configs:       list of {tile_m, tile_n, tile_k, warp_m, warp_n, warp_k,
                                   warp_tile_m, warp_tile_n, warp_tile_k}
      quant_groups:       list of {quant_group_m, quant_group_n, quant_group_k}
      pad_m/pad_n/pad_k:  bool
      block_size:         int (default 256)
      k_block_per_cu:     int (default 1)
      double_smem_buffer: bool (default false)
      preshuffle_b:       bool (default false)
      preshuffle_bquant:  bool (default false)
    """
    import itertools

    with open(config_path) as f:
        cfg = json.load(f)

    pipeline          = cfg.get("pipeline", "compv3")
    epilogue          = cfg.get("epilogue", "cshuffle")
    scheduler         = cfg.get("scheduler", "intrawave")
    pad_m             = cfg.get("pad_m", False)
    pad_n             = cfg.get("pad_n", False)
    pad_k             = cfg.get("pad_k", True)
    block_size        = cfg.get("block_size", 256)
    k_block_per_cu    = cfg.get("k_block_per_cu", 1)
    double_smem_buffer = cfg.get("double_smem_buffer", False)
    preshuffle_b      = cfg.get("preshuffle_b", False)
    preshuffle_bquant = cfg.get("preshuffle_bquant", False)

    configs: List[BQuantKernelConfig] = []
    seen: set = set()

    for variant_key, layout, tile_dict, qg in itertools.product(
        cfg.get("variant_keys", ["fp8"]),
        cfg.get("layouts", ["rcr"]),
        cfg.get("tile_configs", []),
        cfg.get("quant_groups", [{"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128}]),
    ):
        c = BQuantKernelConfig(
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
            quant_group_m=qg.get("quant_group_m", 1),
            quant_group_n=qg.get("quant_group_n", 1),
            quant_group_k=qg.get("quant_group_k", 128),
            preshuffle_b=preshuffle_b,
            preshuffle_bquant=preshuffle_bquant,
            double_smem_buffer=double_smem_buffer,
            k_block_per_cu=k_block_per_cu,
            gfx_arch=gfx_arch,
        )
        if c.name not in seen:
            seen.add(c.name)
            configs.append(c)

    return configs


# =============================================================================
# Convenience: default fp8 config (matches GemmConfigQuantDecode<fp8_t>)
# =============================================================================


def default_fp8_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """Return the default fp8 BQuant config (tile = 16x64x256, warp = 1x4x1).

    WarpTileK=128: on gfx950 get_k_warp_tile<fp8_t, M_Warp_Tile=16>() returns 128
    (is_8bit_float=true, M_Warp_Tile!=32 → K_warp=128). Using 16 causes zero output.
    """
    return BQuantKernelConfig(
        variant_key="fp8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


def default_bf8_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """Return the default bf8 BQuant config (tile = 16x64x256, warp = 1x4x1).

    WarpTileK=128: on gfx950 get_k_warp_tile<bf8_t, M_Warp_Tile=16>() returns 128.
    """
    return BQuantKernelConfig(
        variant_key="bf8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


def default_fp8i4_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """Return the default fp8i4 BQuant config (A=fp8, B=pk_int4, Q=fp8; tile = 16x64x256)."""
    return BQuantKernelConfig(
        variant_key="fp8i4",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


def default_bf8i4_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """Return the default bf8i4 BQuant config (A=bf8, B=pk_int4, Q=bf8; tile = 16x64x256)."""
    return BQuantKernelConfig(
        variant_key="bf8i4",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


# =============================================================================
# Phase 3a — preshuffle_b only (WPQuantBPipelineAgBgCrV2, prefill tile 128x128x128)
#
# Tile dims from GemmConfigPreshuffleB_BQuant_Prefill<PrecType>:
#   M=128, N=128, K=128/sizeof(PrecType)
#   Warp: 1x4x1, WarpTile: 16x16x K_warp
#   K_warp from get_k_warp_tile<PrecType, 16, IsFlatMM=true> on gfx950:
#     fp8/bf8  (8-bit float): K_warp = 128  → tile_k = 128
#     pk_int4  (not 8b float): K_warp = 32   → tile_k = 128 (sizeof=0.5 → 128/0.5=256 packed)
#   DoubleSmemBuffer=true, kBlockPerCu=2
# =============================================================================


def _preshuffleb_config(
    variant_key: str,
    warp_tile_k: int,
    quant_group_k: int,
    quant_group_n: int,
    gfx_arch: str,
) -> BQuantKernelConfig:
    return BQuantKernelConfig(
        variant_key=variant_key,
        layout="rcr",
        pipeline="preshuffleb",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=warp_tile_k,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        preshuffle_b=True,
        preshuffle_bquant=False,
        double_smem_buffer=True,
        k_block_per_cu=2,
        gfx_arch=gfx_arch,
    )


def default_fp8_preshuffleb_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """fp8 preshuffle_b prefill config (GemmConfigPreshuffleB_BQuant_Prefill<fp8_t>)."""
    return _preshuffleb_config("fp8", warp_tile_k=128, quant_group_k=quant_group_k,
                               quant_group_n=quant_group_n, gfx_arch=gfx_arch)


def default_bf8_preshuffleb_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """bf8 preshuffle_b prefill config (GemmConfigPreshuffleB_BQuant_Prefill<bf8_t>)."""
    return _preshuffleb_config("bf8", warp_tile_k=128, quant_group_k=quant_group_k,
                               quant_group_n=quant_group_n, gfx_arch=gfx_arch)


def default_fp8i4_preshuffleb_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """fp8i4 preshuffle_b prefill config (GemmConfigPreshuffleB_BQuant_Prefill<fp8_t>, B=pk_int4).

    K_warp_tile=32: pk_int4 is not an 8-bit float type so get_k_warp_tile returns 32 on gfx950.
    """
    return _preshuffleb_config("fp8i4", warp_tile_k=32, quant_group_k=quant_group_k,
                               quant_group_n=quant_group_n, gfx_arch=gfx_arch)


def default_bf8i4_preshuffleb_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """bf8i4 preshuffle_b prefill config (GemmConfigPreshuffleB_BQuant_Prefill<bf8_t>, B=pk_int4).

    K_warp_tile=32: pk_int4 is not an 8-bit float type so get_k_warp_tile returns 32 on gfx950.
    """
    return _preshuffleb_config("bf8i4", warp_tile_k=32, quant_group_k=quant_group_k,
                               quant_group_n=quant_group_n, gfx_arch=gfx_arch)


# =============================================================================
# Phase 3b — preshuffle_bquant only (BQuantGemmPipelineAgBgCrCompV3, prefill tile 128x128x128)
#
# Tile dims from GemmConfigPreshuffleBQuantPrefill<PrecType> (inherits GemmConfigQuantPrefill):
#   M=128, N=128, K=128, Warp: 1x4x1, WarpTile: 16x16xK_warp
#   K_warp from get_k_warp_tile<PrecType, 16, IsFlatMM=false> on gfx950:
#     fp8/bf8: K_warp = 128  (is_8bit_float=true, M_Warp_Tile != 32)
#     pk_int4: K_warp = 32
#   DoubleSmemBuffer=false (default), kBlockPerCu=1 (default)
#   Extra quant group: 1x16x128 (not in 3a)
# =============================================================================


def _preshufflequant_config(
    variant_key: str,
    warp_tile_k: int,
    quant_group_k: int,
    quant_group_n: int,
    gfx_arch: str,
) -> BQuantKernelConfig:
    return BQuantKernelConfig(
        variant_key=variant_key,
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=warp_tile_k,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        preshuffle_b=False,
        preshuffle_bquant=True,
        gfx_arch=gfx_arch,
    )


def default_fp8_preshufflequant_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """fp8 preshuffle_bquant prefill config (GemmConfigPreshuffleBQuantPrefill<fp8_t>)."""
    return _preshufflequant_config("fp8", warp_tile_k=128, quant_group_k=quant_group_k,
                                   quant_group_n=quant_group_n, gfx_arch=gfx_arch)


def default_bf8_preshufflequant_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """bf8 preshuffle_bquant prefill config (GemmConfigPreshuffleBQuantPrefill<bf8_t>)."""
    return _preshufflequant_config("bf8", warp_tile_k=128, quant_group_k=quant_group_k,
                                   quant_group_n=quant_group_n, gfx_arch=gfx_arch)


def default_fp8i4_preshufflequant_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """fp8i4 preshuffle_bquant prefill config."""
    return _preshufflequant_config("fp8i4", warp_tile_k=32, quant_group_k=quant_group_k,
                                   quant_group_n=quant_group_n, gfx_arch=gfx_arch)


def default_bf8i4_preshufflequant_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """bf8i4 preshuffle_bquant prefill config."""
    return _preshufflequant_config("bf8i4", warp_tile_k=32, quant_group_k=quant_group_k,
                                   quant_group_n=quant_group_n, gfx_arch=gfx_arch)


# =============================================================================
# Phase 3c — preshuffle_b + preshuffle_bquant (WPQuantBPipelineAgBgCrV2, prefill tile)
#
# Tile dims from GemmConfigPreshuffleB_PreshuffleBQuant_Prefill<PrecType>
#   (inherits GemmConfigPreshuffleB_BQuant_Prefill): identical tile/warp dims to 3a.
#   PreshuffleB=true, BPreshuffleQuant=true, DoubleSmemBuffer=true, kBlockPerCu=2
# =============================================================================


def _preshuffleb_bquant_config(
    variant_key: str,
    warp_tile_k: int,
    quant_group_k: int,
    quant_group_n: int,
    gfx_arch: str,
) -> BQuantKernelConfig:
    return BQuantKernelConfig(
        variant_key=variant_key,
        layout="rcr",
        pipeline="preshuffleb",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=warp_tile_k,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        preshuffle_b=True,
        preshuffle_bquant=True,
        double_smem_buffer=True,
        k_block_per_cu=2,
        gfx_arch=gfx_arch,
    )


def default_fp8_preshuffleb_bquant_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """fp8 preshuffle_b+preshuffle_bquant config (GemmConfigPreshuffleB_PreshuffleBQuant_Prefill<fp8_t>)."""
    return _preshuffleb_bquant_config("fp8", warp_tile_k=128, quant_group_k=quant_group_k,
                                      quant_group_n=quant_group_n, gfx_arch=gfx_arch)


def default_bf8_preshuffleb_bquant_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """bf8 preshuffle_b+preshuffle_bquant config."""
    return _preshuffleb_bquant_config("bf8", warp_tile_k=128, quant_group_k=quant_group_k,
                                      quant_group_n=quant_group_n, gfx_arch=gfx_arch)


def default_fp8i4_preshuffleb_bquant_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """fp8i4 preshuffle_b+preshuffle_bquant config."""
    return _preshuffleb_bquant_config("fp8i4", warp_tile_k=32, quant_group_k=quant_group_k,
                                      quant_group_n=quant_group_n, gfx_arch=gfx_arch)


def default_bf8i4_preshuffleb_bquant_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """bf8i4 preshuffle_b+preshuffle_bquant config."""
    return _preshuffleb_bquant_config("bf8i4", warp_tile_k=32, quant_group_k=quant_group_k,
                                      quant_group_n=quant_group_n, gfx_arch=gfx_arch)


# =============================================================================
# Phase 4 — MX microscale variants (A=bf16, Q=e8m0 block scale)
#
# Pipeline for all three: MicroscaleGemmPipelineAgBgCrCompV3
#   (QDataType=e8m0_t, PreshuffleB=false → microscale branch in BQuantPipeline selection)
# Base pipeline: BaseWeightPreshufflePipelineAGmemBGmemCRegV2
#   (BQuantGrouped, PreshuffleB=false, not AQuant/ABQuant → else branch)
#
# Tile dims:
#   mx_bf16bf16: GemmConfigQuantPrefill<bf16_t> → 128x128x128, warp 1x4x1, warp_tile_k=32
#   mx_bf16bf8:  GemmConfigMixedPrecision       → 128x128x128, warp 1x4x1, warp_tile_k=64
#   mx_bf16fp4:  GemmConfigQuantPrefill<bf16_t> → 128x128x128, warp 1x4x1, warp_tile_k=32
# =============================================================================


def default_mx_bf16bf16_config(
    quant_group_k: int = 32,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """MX bf16+bf16 config (A=bf16, B=bf16, Q=e8m0; GemmConfigQuantPrefill<bf16_t>).

    Default quant_group_k=32 matches the smallest MX block size for bf16+bf16.
    """
    return BQuantKernelConfig(
        variant_key="mx_bf16bf16",
        layout="rcr",
        pipeline="microscale",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


def default_mx_bf16bf8_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """MX bf16+bf8 config (A=bf16, B=bf8, Q=e8m0; GemmConfigMixedPrecision).

    warp_tile_k=64 is hardcoded in GemmConfigMixedPrecision (mixed-precision KPack=16).
    """
    return BQuantKernelConfig(
        variant_key="mx_bf16bf8",
        layout="rcr",
        pipeline="microscale",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=64,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


def default_mx_bf16fp4_config(
    quant_group_k: int = 32,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """MX bf16+fp4 config (A=bf16, B=pk_fp4, Q=e8m0; GemmConfigQuantPrefill<bf16_t>).

    Default quant_group_k=32 matches the smallest MX block size for fp4.
    """
    return BQuantKernelConfig(
        variant_key="mx_bf16fp4",
        layout="rcr",
        pipeline="microscale",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )
