#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GroupedGemm AQuant dispatcher utilities.

Three-layer Python bridge for the dispatcher's AQuantGrouped GEMM path:

  AQuantKernelConfig  — describes one kernel; .name is byte-exact with codegen KERNEL_NAME
  AQuantDispatcherLib — thin ctypes wrapper around a compiled .so
  AQuantGpuGemmRunner — high-level runner that accepts numpy arrays

Build helpers:
  setup_multiple_aquant_dispatchers(configs, ...)
       codegen -> hipcc -> list of .so paths, all in parallel

AQuant: A-side activation quantization.
  AQ[ceil(M/gM), ceil(K/gK)] is the A-side scale tensor (RowMajor).
  Non-preshuffle kernels use AQuantGemmPipelineAgBgCrMem (pipeline="mem").
  Preshuffle kernels (APreshuffleQuant=true) use AQuantGemmPipelineAgBgCrCompV3 (pipeline="compv3").

Usage:
  configs = [AQuantKernelConfig(variant_key="fp8", layout="rcr", pipeline="mem", ...)]
  so_paths = setup_multiple_aquant_dispatchers(configs, output_dir=Path("/tmp/aq"))
  runner = AQuantGpuGemmRunner(so_paths[0])
  result = runner.run(A, B, AQ, AQuantGemmProblem(M=16, N=64, K=256))
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

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_grouped_gemm_aquant_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "grouped_gemm_aquant_ctypes_lib.cpp"

_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from codegen_common import make_aquant_kernel_name  # noqa: E402

_DEFAULT_HIPCC    = "hipcc"
_DEFAULT_GFX_ARCH = "gfx950"

_HIPCC_BASE_FLAGS = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-shared",
    "-DCK_TILE_SINGLE_KERNEL_INCLUDE",
    "-w",
]


# =============================================================================
# AQuantKernelConfig — byte-exact naming with codegen
# =============================================================================


@dataclass
class AQuantKernelConfig:
    """
    Complete description of one AQuantGrouped GEMM kernel.

    The .name property produces the exact string that unified_grouped_gemm_aquant_codegen.py
    emits as KERNEL_NAME, ensuring the Python side and compiled .so always agree.

    pipeline: "mem"    — non-preshuffle (AQuantGemmPipelineAgBgCrMem)
              "compv3" — preshuffle, requires preshuffle_aq=True
    """

    variant_key: str       # "fp8", "bf8", "fp8i4", "bf8i4"
    layout: str            # "rcr"
    pipeline: str          # "mem" or "compv3"
    epilogue: str          # "cshuffle" (effective epilogue may differ per tile)
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

    preshuffle_aq: bool   = False  # APreshuffleQuant — requires pipeline="compv3"
    double_smem_buffer: bool = False
    k_block_per_cu: int   = 1
    transpose_c: bool     = False

    gfx_arch: str = _DEFAULT_GFX_ARCH

    @property
    def name(self) -> str:
        """Byte-exact match to codegen KERNEL_NAME (delegates to make_aquant_kernel_name)."""
        return make_aquant_kernel_name(
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
            preshuffle_aq=self.preshuffle_aq,
        )

    def to_codegen_config(self) -> dict:
        """Produce the JSON config dict consumed by unified_grouped_gemm_aquant_codegen.py."""
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
            "preshuffle_aq": self.preshuffle_aq,
            "double_smem_buffer": self.double_smem_buffer,
            "k_block_per_cu": self.k_block_per_cu,
            "transpose_c": self.transpose_c,
        }


# =============================================================================
# AQuantGemmProblem
# =============================================================================


@dataclass
class AQuantGemmProblem:
    M: int
    N: int
    K: int
    quant_group_m: int = 1
    quant_group_n: int = 1
    quant_group_k: int = 128
    k_batch: int = 1

    @property
    def QK_A(self) -> int:
        """Number of K-groups for A: ceil(K / quant_group_k)."""
        return (self.K + self.quant_group_k - 1) // self.quant_group_k

    @property
    def QM_A(self) -> int:
        """Number of M-groups for A: ceil(M / quant_group_m). Typically == M when gM=1."""
        return (self.M + self.quant_group_m - 1) // self.quant_group_m


# =============================================================================
# AQuantGemmResult
# =============================================================================


@dataclass
class AQuantGemmResult:
    C: object          # numpy array (M, N)
    time_ms: float
    kernel_name: str


# =============================================================================
# AQuantDispatcherLib — thin ctypes wrapper
# =============================================================================


class AQuantDispatcherLib:
    """
    Loads a compiled aquant_gemm .so and wraps its C API.

    Expected .so exports:
      int  dispatcher_initialize()
      int  dispatcher_run_aquant_gemm(A, B, AQ, C, M, N, K,
                                      stride_A, stride_B, stride_AQ, stride_C,
                                      QK_A, QM_A, k_batch, *time_ms)
      char* dispatcher_get_kernel_name()
      int   dispatcher_get_kernel_count()
      void  dispatcher_cleanup()
    """

    def __init__(self, so_path: Path):
        self.so_path = Path(so_path)
        if not self.so_path.exists():
            raise FileNotFoundError(f"AQuant .so not found: {self.so_path}")
        self._lib = ctypes.CDLL(str(self.so_path))
        self._setup()
        rc = self._lib.dispatcher_initialize()
        if rc != 0:
            raise RuntimeError(f"dispatcher_initialize() returned {rc}")

    def _setup(self):
        lib = self._lib

        lib.dispatcher_initialize.restype  = ctypes.c_int
        lib.dispatcher_initialize.argtypes = []

        lib.dispatcher_run_aquant_gemm.restype  = ctypes.c_int
        lib.dispatcher_run_aquant_gemm.argtypes = [
            ctypes.c_void_p,   # A
            ctypes.c_void_p,   # B
            ctypes.c_void_p,   # AQ
            ctypes.c_void_p,   # C
            ctypes.c_int64,    # M
            ctypes.c_int64,    # N
            ctypes.c_int64,    # K
            ctypes.c_int64,    # stride_A
            ctypes.c_int64,    # stride_B
            ctypes.c_int64,    # stride_AQ
            ctypes.c_int64,    # stride_C
            ctypes.c_int64,    # QK_A
            ctypes.c_int64,    # QM_A
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
        AQ,
        C,
        M: int,
        N: int,
        K: int,
        stride_A: int,
        stride_B: int,
        stride_AQ: int,
        stride_C: int,
        QK_A: int,
        QM_A: int,
        k_batch: int = 1,
    ) -> Tuple[int, float]:
        """
        Call dispatcher_run_aquant_gemm with ctypes-wrapped pointers.

        A, B, AQ, C must be numpy arrays (C-contiguous, packed).
        Returns (status, time_ms).
        """
        import numpy as np

        A  = np.ascontiguousarray(A)
        # B is col-major [K, N]: Fortran order makes the leading dim = K (stride_B = K).
        B  = np.asfortranarray(B)
        AQ = np.ascontiguousarray(AQ)
        C  = np.ascontiguousarray(C)

        time_ms = ctypes.c_float(0.0)

        rc = self._lib.dispatcher_run_aquant_gemm(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            AQ.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(M),
            ctypes.c_int64(N),
            ctypes.c_int64(K),
            ctypes.c_int64(stride_A),
            ctypes.c_int64(stride_B),
            ctypes.c_int64(stride_AQ),
            ctypes.c_int64(stride_C),
            ctypes.c_int64(QK_A),
            ctypes.c_int64(QM_A),
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
# AQuantGpuGemmRunner — high-level runner
# =============================================================================


class AQuantGpuGemmRunner:
    """
    High-level runner that loads an AQuant .so and executes GEMM on the GPU.

    Accepts numpy arrays for A, B, AQ; allocates C; returns AQuantGemmResult.
    """

    def __init__(self, so_path: Path):
        self._lib = AQuantDispatcherLib(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(self, A, B, AQ, problem: AQuantGemmProblem, c_dtype=None) -> AQuantGemmResult:
        """
        Run AQuantGrouped GEMM.

        A       shape: (M, K)           dtype: fp8/bf8
        B       shape: (K, N) col-major  dtype: fp8/bf8 or pk_int4
        AQ      shape: (QM_A, QK_A)     dtype: float (for fp8/bf8) or fp8/bf8 (for fp8i4/bf8i4)
        c_dtype numpy dtype for the output C buffer.  Defaults to np.float16.
                Pass np.bfloat16 for MX variants whose CDataType is bf16.
        Returns AQuantGemmResult with C shape (M, N).
        """
        import numpy as np

        M, N, K = problem.M, problem.N, problem.K
        QK_A    = problem.QK_A
        QM_A    = problem.QM_A

        if c_dtype is None:
            c_dtype = np.float16

        C = np.zeros((M, N), dtype=c_dtype)

        # AQ is RowMajor [QM_A, QK_A]: stride_AQ == QK_A (leading dim = number of K-groups)
        stride_A   = K
        stride_B   = K    # B col-major [K, N]: leading dim = K
        stride_AQ  = QK_A
        stride_C   = N

        rc, time_ms = self._lib.run(
            A=A, B=B, AQ=AQ, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A,
            stride_B=stride_B,
            stride_AQ=stride_AQ,
            stride_C=stride_C,
            QK_A=QK_A,
            QM_A=QM_A,
            k_batch=problem.k_batch,
        )

        if rc != 0:
            raise RuntimeError(
                f"dispatcher_run_aquant_gemm failed with code {rc} "
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

        return AQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


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
    except Exception:
        pass
    return _DEFAULT_GFX_ARCH


def _get_ck_include_dir() -> Optional[Path]:
    """Attempt to locate the CK include directory relative to this file."""
    here = Path(__file__).resolve().parent
    for parent in [here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "include"
        if (candidate / "ck_tile").is_dir():
            return candidate
    return None


def _get_dispatcher_include_dir() -> Optional[Path]:
    """Attempt to locate the dispatcher include directory relative to this file."""
    here = Path(__file__).resolve().parent
    candidate = here.parent / "include"
    if (candidate / "ck_tile" / "dispatcher").is_dir():
        return candidate
    return None


def _generate_aquant_kernel(
    config: AQuantKernelConfig,
    output_dir: Path,
) -> Optional[Path]:
    """Run unified_grouped_gemm_aquant_codegen.py for one config; return the .hpp path or None."""
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


def _compile_aquant_kernel(
    hpp_path: Path,
    so_path: Path,
    gfx_arch: str,
    hipcc: str = _DEFAULT_HIPCC,
    extra_include_dirs: Optional[List[str]] = None,
) -> bool:
    """Compile a generated .hpp into a .so via hipcc. Returns True on success."""
    ck_include = _get_ck_include_dir()

    cmd = [hipcc] + _HIPCC_BASE_FLAGS + [
        f"--offload-arch={gfx_arch}",
        f"-DGFX_ARCH=\"{gfx_arch}\"",
        "-include", str(hpp_path),
        str(_CTYPES_LIB_SRC),
        "-o", str(so_path),
    ]

    if ck_include:
        cmd += [f"-I{ck_include}"]

    dispatcher_include = _get_dispatcher_include_dir()
    if dispatcher_include:
        cmd += [f"-I{dispatcher_include}"]

    if extra_include_dirs:
        for d in extra_include_dirs:
            cmd += [f"-I{d}"]

    log.debug("Compiling %s:\n  %s", so_path.name, " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log.error("Compile failed for %s:\n%s", so_path.name, result.stderr[-2000:])
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error("Compile timed out for %s", so_path.name)
        return False


# =============================================================================
# setup_multiple_aquant_dispatchers — build pipeline
# =============================================================================


def setup_multiple_aquant_dispatchers(
    configs: List[AQuantKernelConfig],
    output_dir: Optional[Path] = None,
    hipcc: str = _DEFAULT_HIPCC,
    gfx_arch: Optional[str] = None,
    extra_include_dirs: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """
    For each AQuantKernelConfig: codegen -> hipcc compile -> .so path.

    Returns a list parallel to `configs` — each entry is the Path to the
    compiled .so, or None if that config failed.
    """
    if not configs:
        return []

    arch = gfx_arch or _detect_gpu_arch()
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix="aquant_dispatcher_"))
    base_dir.mkdir(parents=True, exist_ok=True)

    headers_dir = base_dir / "generated_kernels"
    so_dir      = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    log.info("Building %d AQuant kernel(s) for %s into %s", len(configs), arch, base_dir)

    seen: Dict[str, int] = {}
    deduped: List[Tuple[int, AQuantKernelConfig]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx: int, cfg: AQuantKernelConfig) -> Tuple[int, Optional[Path]]:
        hpp = _generate_aquant_kernel(cfg, headers_dir)
        if hpp is None:
            return idx, None

        so = so_dir / f"lib{cfg.name}_{arch}.so"
        if so.exists():
            log.info("  [cached] %s", so.name)
            return idx, so

        ok = _compile_aquant_kernel(
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

    for i, cfg in enumerate(configs):
        if results[i] is None:
            first_idx = seen.get(cfg.name)
            if first_idx is not None and first_idx != i:
                results[i] = results[first_idx]

    built = sum(1 for r in results if r is not None)
    log.info("Built %d / %d AQuant kernels", built, len(configs))
    return results


# =============================================================================
# Default configs (mapped from reference examples)
# =============================================================================


def default_fp8_config(
    quant_group_k: int = 128,
    quant_group_m: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> AQuantKernelConfig:
    """fp8 AQuant decode config (GemmConfigQuantDecodeInterwave<fp8_t>, Mem pipeline).

    warp_tile_k=32 maps to the MFMA mfma_f32_16x16x32_fp8_fp8 instruction which is
    valid on gfx9 (gfx942, gfx950). warp_tile_k=16 would select the gfx12-only WMMA
    instruction and silently returns zeros on gfx9 hardware.
    """
    return AQuantKernelConfig(
        variant_key="fp8",
        layout="rcr",
        pipeline="mem",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
        quant_group_m=quant_group_m,
        quant_group_n=1,
        quant_group_k=quant_group_k,
        preshuffle_aq=False,
        gfx_arch=gfx_arch,
    )


def default_bf8_config(
    quant_group_k: int = 128,
    quant_group_m: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> AQuantKernelConfig:
    """bf8 AQuant decode config (GemmConfigQuantDecodeInterwave<bf8_t>, Mem pipeline).

    warp_tile_k=32 maps to the MFMA mfma_f32_16x16x32_bf8_bf8 instruction which is
    valid on gfx9 (gfx942, gfx950). warp_tile_k=16 would select the gfx12-only WMMA
    instruction and silently returns zeros on gfx9 hardware.
    """
    return AQuantKernelConfig(
        variant_key="bf8",
        layout="rcr",
        pipeline="mem",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
        quant_group_m=quant_group_m,
        quant_group_n=1,
        quant_group_k=quant_group_k,
        preshuffle_aq=False,
        gfx_arch=gfx_arch,
    )


def default_fp8i4_config(
    quant_group_k: int = 128,
    quant_group_m: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> AQuantKernelConfig:
    """fp8i4 AQuant decode config (A=fp8, B=pk_int4, AQ=fp8; Mem pipeline).

    warp_tile_k=32 maps to valid MFMA instructions on gfx9 (gfx942, gfx950).
    """
    return AQuantKernelConfig(
        variant_key="fp8i4",
        layout="rcr",
        pipeline="mem",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
        quant_group_m=quant_group_m,
        quant_group_n=1,
        quant_group_k=quant_group_k,
        preshuffle_aq=False,
        gfx_arch=gfx_arch,
    )


def default_bf8i4_config(
    quant_group_k: int = 128,
    quant_group_m: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> AQuantKernelConfig:
    """bf8i4 AQuant decode config (A=bf8, B=pk_int4, AQ=bf8; Mem pipeline).

    warp_tile_k=32 maps to valid MFMA instructions on gfx9 (gfx942, gfx950).
    """
    return AQuantKernelConfig(
        variant_key="bf8i4",
        layout="rcr",
        pipeline="mem",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
        quant_group_m=quant_group_m,
        quant_group_n=1,
        quant_group_k=quant_group_k,
        preshuffle_aq=False,
        gfx_arch=gfx_arch,
    )


def default_fp8_preshuffleaq_config(
    quant_group_k: int = 128,
    quant_group_m: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> AQuantKernelConfig:
    """fp8 preshuffle-AQ config (GemmConfigPreshuffleQuantDecode<fp8_t>, CompV3 pipeline).

    APreshuffleQuant=true, BPreshuffleQuant=true — both scale tensors are preshuffled.
    Tile: 16x64x256, warp_tile_k=128 (gfx950 FlatMM, 8-bit float).
    """
    return AQuantKernelConfig(
        variant_key="fp8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
        quant_group_m=quant_group_m,
        quant_group_n=1,
        quant_group_k=quant_group_k,
        preshuffle_aq=True,
        gfx_arch=gfx_arch,
    )


def default_bf8_preshuffleaq_config(
    quant_group_k: int = 128,
    quant_group_m: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> AQuantKernelConfig:
    """bf8 preshuffle-AQ config (GemmConfigPreshuffleQuantDecode<bf8_t>, CompV3 pipeline)."""
    return AQuantKernelConfig(
        variant_key="bf8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
        quant_group_m=quant_group_m,
        quant_group_n=1,
        quant_group_k=quant_group_k,
        preshuffle_aq=True,
        gfx_arch=gfx_arch,
    )


# =============================================================================
# gfx1250 (MI400) default configs
# =============================================================================
# Empirically (on-GPU, MI400/gfx1250, HIP_VISIBLE_DEVICES=0, fp32-dequant
# reference), the AQuant fp8/bf8 kernels are correct on gfx1250 only with
# warp_tile_k=128 (the FlatMM path, same tile the gfx9 fp8/bf8 FlatMM decode
# uses). The gfx9 stock AQuant default uses warp_tile_k=32 (plain MFMA), which
# produces a WRONG result on gfx1250 (max_rel_err=1.0); warp_tile_k=16 (gfx12
# WMMA) returns all-zeros. So the only correct gfx1250 override is 16x16x128.
#
#   fp8: warp_tile 16x16x128 -> nonzero, max_rel_err ~3e-4  (PASS)
#   bf8: warp_tile 16x16x128 -> nonzero, max_rel_err ~1e-4  (PASS)
#
# The compile path injects -DCK_USE_OCP_FP8 / -DCK_TILE_USE_OCP_FP8 for gfx12
# archs, so fp8/bf8 use the OCP encoding on gfx1250.
#
# fp8i4 / bf8i4 (pk_int4 weights) do NOT compile on gfx1250 at any warp_tile_k
# (no gfx12 WMMA/FlatMM instruction for the packed-int4 quant path). Those
# variants are therefore not enabled on gfx1250: the C++ ctypes arch-gate is
# opened, but on-GPU correctness is deferred pending kernel work. No gfx1250
# config helper is provided for i4 because it would emit a non-buildable kernel.

_GFX1250_ARCH = "gfx1250"


def default_fp8_config_gfx1250(
    quant_group_k: int = 128,
    quant_group_m: int = 1,
    gfx_arch: str = _GFX1250_ARCH,
) -> AQuantKernelConfig:
    """fp8 AQuant decode config for gfx1250 (FlatMM, warp_tile_k=128).

    GPU-verified on MI400/gfx1250: max_rel_err ~3e-4 vs. fp32 dequant reference.
    warp_tile_k=32 (gfx9 stock) is WRONG on gfx1250; warp_tile_k=16 zeros out.
    """
    return AQuantKernelConfig(
        variant_key="fp8",
        layout="rcr",
        pipeline="mem",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
        quant_group_m=quant_group_m,
        quant_group_n=1,
        quant_group_k=quant_group_k,
        preshuffle_aq=False,
        gfx_arch=gfx_arch,
    )


def default_bf8_config_gfx1250(
    quant_group_k: int = 128,
    quant_group_m: int = 1,
    gfx_arch: str = _GFX1250_ARCH,
) -> AQuantKernelConfig:
    """bf8 AQuant decode config for gfx1250 (FlatMM, warp_tile_k=128).

    GPU-verified on MI400/gfx1250: max_rel_err ~1e-4 vs. fp32 dequant reference.
    """
    return AQuantKernelConfig(
        variant_key="bf8",
        layout="rcr",
        pipeline="mem",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
        quant_group_m=quant_group_m,
        quant_group_n=1,
        quant_group_k=quant_group_k,
        preshuffle_aq=False,
        gfx_arch=gfx_arch,
    )
