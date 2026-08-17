#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Batched-contraction dispatcher utilities (TileEngine -> Dispatcher bridge).

Three-layer Python bridge for the dispatcher's batched-contraction path:

  BatchedContractionKernelConfig  -- describes one kernel; .name is byte-exact with
                                     codegen KERNEL_NAME (both delegate to
                                     make_batched_contraction_kernel_name)
  BatchedContractionDispatcherLib -- thin ctypes wrapper around a compiled .so
  GpuBatchedContractionRunner     -- high-level runner that accepts numpy arrays and
                                     computes E[G..,M..,N..] = sum_K A[G..,M..,K..] * B[G..,N..,K..]

Build helper (self-contained):
  setup_multiple_batched_contraction_dispatchers(configs, ...) : codegen -> hipcc -> .so paths

v1 scope matches Old-TE gemm/batched_contraction argparse: dtype {fp16,bf16,fp32},
3-char a/b/e layout, num_dim_g/m/n/k, num_d_tensors == 0 (PassThrough).
"""

import concurrent.futures
import ctypes
import functools
import itertools
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Import the shared name helper so codegen and utils never drift.
_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from unified_batched_contraction_codegen import (  # noqa: E402
    make_batched_contraction_kernel_name,
)

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_batched_contraction_codegen.py"
_CTYPES_LIB_SRC = (
    Path(__file__).parent.parent / "bindings" / "ctypes" / "batched_contraction_ctypes_lib.cpp"
)
_HIPCC = os.environ.get("CK_TILE_HIPCC", "/opt/rocm/bin/hipcc")

# ---------------------------------------------------------------------------
# Tile Engine codegen flags -- must match exactly what Old-TE CMake applies to
# its GEMM/contraction benchmarks so the bridge .so produces the same machine
# code (inlining, register allocation, occupancy).  These are the same flags
# used by the sibling gemm_utils._tile_engine_codegen_flags(), kept inline here
# so batched_contraction_utils stays self-contained.
# ---------------------------------------------------------------------------
_TILE_ENGINE_CODEGEN_FLAGS = (
    "-mllvm", "--lsr-drop-solution=1",
    "-mllvm", "-enable-post-misched=0",
    "-mllvm", "-amdgpu-early-inline-all=true",
    "-mllvm", "-amdgpu-function-calls=false",
    "-fno-offload-uniform-block",
)

# Flags TE's CMake only adds when ``check_cxx_compiler_flag`` passes (newer
# -mllvm options that some clang builds reject).
_PROBED_CODEGEN_FLAGS = (
    ("-mllvm", "-amdgpu-coerce-illegal-types=1"),
)


@functools.lru_cache(maxsize=None)
def _hipcc_accepts(flag_tuple: Tuple[str, ...]) -> bool:
    """Mirror CMake check_cxx_compiler_flag: does hipcc compile a trivial TU
    with these flags?  Cached so the probe runs at most once per flag set."""
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
def _tile_engine_codegen_flags() -> Tuple[str, ...]:
    """TE's GEMM/contraction codegen flags plus any probe-gated flags the
    compiler accepts -- the exact backend flag set the TE benchmark is built
    with."""
    flags = list(_TILE_ENGINE_CODEGEN_FLAGS)
    for pair in _PROBED_CODEGEN_FLAGS:
        if _hipcc_accepts(pair):
            flags = list(pair) + flags
    return tuple(flags)


_SUPPORTED_ARCHS = ("gfx90a", "gfx942", "gfx950")

_NP_DTYPE = {"fp16": np.float16, "bf16": None, "fp32": np.float32}  # bf16 filled lazily


def _get_arch() -> str:
    """Detect GPU arch via rocminfo and validate; raise on failure. Never defaults."""
    import subprocess

    arch = ""
    try:
        out = subprocess.check_output(["rocminfo"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "Name:" in line and "gfx" in line:
                arch = line.split()[-1].strip()
                break
    except Exception:
        arch = ""
    if not arch:
        raise RuntimeError(
            "Could not detect GPU architecture from rocminfo; refusing to default "
            "to gfx942. Pass gfx_arch explicitly."
        )
    if arch not in _SUPPORTED_ARCHS:
        raise ValueError(
            f"Unsupported GPU architecture {arch!r}; supported: {list(_SUPPORTED_ARCHS)}"
        )
    return arch


def _validate_arch(arch: str) -> str:
    """Validate an explicitly supplied arch against the supported set."""
    if arch not in _SUPPORTED_ARCHS:
        raise ValueError(
            f"Unsupported GPU architecture {arch!r}; supported: {list(_SUPPORTED_ARCHS)}"
        )
    return arch


def _dtype_from_kernel_name(name: str) -> str:
    """Extract the dtype token from a batched-contraction kernel name.

    Name format is ``batched_contraction_<dtype>_<layout>_...`` (see
    make_batched_contraction_kernel_name), so the dtype is the token after the
    ``batched_contraction`` prefix. Mirrors gemm_utils._dtype_from_kernel_name."""
    parts = name.split("_")
    # prefix is two tokens ("batched", "contraction"); dtype follows.
    return parts[2] if len(parts) > 2 else "fp16"


def _np_dtype(dtype: str):
    if dtype == "bf16":
        try:
            import ml_dtypes

            return ml_dtypes.bfloat16
        except ImportError:
            raise RuntimeError("bf16 requires ml_dtypes")
    return _NP_DTYPE[dtype]


# =============================================================================
# Config
# =============================================================================


@dataclass
class BatchedContractionKernelConfig:
    dtype: str = "fp16"
    layout: str = "rcr"  # a/b/e
    pipeline: str = "compv3"
    epilogue: str = "cshuffle"
    scheduler: str = "intrawave"
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False
    persistent: bool = False

    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 64
    warp_m: int = 2
    warp_n: int = 2
    warp_k: int = 1
    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16

    num_dim_g: int = 1
    num_dim_m: int = 1
    num_dim_n: int = 1
    num_dim_k: int = 1
    num_d_tensors: int = 0
    elementwise: str = "PassThrough"

    block_size: int = 256
    k_block_per_cu: int = 1
    gfx_arch: Optional[str] = None

    @property
    def name(self) -> str:
        return make_batched_contraction_kernel_name(
            dtype=self.dtype, layout=self.layout, pipeline=self.pipeline,
            epilogue=self.epilogue, scheduler=self.scheduler,
            pad_m=self.pad_m, pad_n=self.pad_n, pad_k=self.pad_k, persistent=self.persistent,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            warp_m=self.warp_m, warp_n=self.warp_n, warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m, warp_tile_n=self.warp_tile_n, warp_tile_k=self.warp_tile_k,
            num_dim_g=self.num_dim_g, num_dim_m=self.num_dim_m,
            num_dim_n=self.num_dim_n, num_dim_k=self.num_dim_k,
            num_d_tensors=self.num_d_tensors, elementwise=self.elementwise,
            k_block_per_cu=self.k_block_per_cu,
        )

    def to_codegen_config(self) -> dict:
        return {
            "datatype": self.dtype, "layout": self.layout,
            "pipeline": self.pipeline, "epilogue": self.epilogue, "scheduler": self.scheduler,
            "pad_m": self.pad_m, "pad_n": self.pad_n, "pad_k": self.pad_k,
            "persistent": self.persistent,
            "tile_config": {
                "tile_m": self.tile_m, "tile_n": self.tile_n, "tile_k": self.tile_k,
                "warp_m": self.warp_m, "warp_n": self.warp_n, "warp_k": self.warp_k,
                "warp_tile_m": self.warp_tile_m, "warp_tile_n": self.warp_tile_n,
                "warp_tile_k": self.warp_tile_k,
            },
            "num_dim_g": self.num_dim_g, "num_dim_m": self.num_dim_m,
            "num_dim_n": self.num_dim_n, "num_dim_k": self.num_dim_k,
            "num_d_tensors": self.num_d_tensors, "elementwise": self.elementwise,
            "block_size": self.block_size, "k_block_per_cu": self.k_block_per_cu,
        }

    def is_valid(self) -> bool:
        if not (
            self.tile_m % (self.warp_m * self.warp_tile_m) == 0
            and self.tile_n % (self.warp_n * self.warp_tile_n) == 0
            and self.tile_k % (self.warp_k * self.warp_tile_k) == 0
        ):
            return False
        # Only the rcr layout compiles (col-major A/B trip kernel static_asserts).
        if self.layout != "rcr":
            return False
        # Old-TE argparse places no upper bound on num_d_tensors (default 0); the
        # DsDataType/DsLayout tuples and the epilogue support an arbitrary count.
        # 0..8 comfortably spans the MultiDAdd/MultiDMultiply usage while keeping
        # the ctypes marshalling bounded.
        if self.num_d_tensors < 0 or self.num_d_tensors > 8:
            return False
        # Epilogue op must agree with the D-tensor count. num_d==0 => PassThrough
        # (plain contraction); num_d>0 => a MultiD* op that consumes the D tensors.
        if self.num_d_tensors == 0:
            if self.elementwise != "PassThrough":
                return False
        else:
            if self.elementwise not in ("MultiDAdd", "MultiDMultiply"):
                return False
        # dtype -> valid MFMA warp tile (per gfx942/gfx950 XDL allow-list).
        wt = (self.warp_tile_m, self.warp_tile_n, self.warp_tile_k)
        allowed = {
            "fp16": {(32, 32, 16), (16, 16, 16), (16, 16, 32)},
            "bf16": {(32, 32, 16), (16, 16, 16), (16, 16, 32)},
            "fp32": {(16, 16, 4), (16, 16, 16), (32, 32, 8)},
        }
        return wt in allowed.get(self.dtype, set())


# =============================================================================
# Problem
# =============================================================================


@dataclass
class BatchedContractionProblem:
    g_dims: List[int]
    m_dims: List[int]
    n_dims: List[int]
    k_dims: List[int]
    k_batch: int = 1

    @property
    def G(self) -> int:
        return int(np.prod(self.g_dims)) if self.g_dims else 1

    @property
    def M(self) -> int:
        return int(np.prod(self.m_dims)) if self.m_dims else 1

    @property
    def N(self) -> int:
        return int(np.prod(self.n_dims)) if self.n_dims else 1

    @property
    def K(self) -> int:
        return int(np.prod(self.k_dims)) if self.k_dims else 1

    @property
    def flops(self) -> int:
        return 2 * self.G * self.M * self.N * self.K

    def to_dict(self) -> dict:
        return {
            "g_dims": self.g_dims, "m_dims": self.m_dims,
            "n_dims": self.n_dims, "k_dims": self.k_dims, "k_batch": self.k_batch,
        }

    @staticmethod
    def from_dict(d: dict) -> "BatchedContractionProblem":
        return BatchedContractionProblem(
            g_dims=list(d["g_dims"]), m_dims=list(d["m_dims"]),
            n_dims=list(d["n_dims"]), k_dims=list(d["k_dims"]),
            k_batch=int(d.get("k_batch", 1)),
        )


@dataclass
class BatchedContractionResult:
    E: object
    time_ms: float
    kernel_name: str
    Ds: Optional[List[object]] = None  # D tensors actually fed to the kernel (num_d>0)


# =============================================================================
# ctypes wrapper
# =============================================================================


class BatchedContractionDispatcherLib:
    def __init__(self, so_path: Path):
        self.so_path = Path(so_path)
        if not self.so_path.exists():
            raise FileNotFoundError(f"batched-contraction .so not found: {self.so_path}")
        self._lib = ctypes.CDLL(str(self.so_path))
        self._setup()
        if self._lib.dispatcher_initialize() != 0:
            raise RuntimeError("dispatcher_initialize failed")

    def _setup(self):
        lib = self._lib
        lib.dispatcher_initialize.restype = ctypes.c_int
        lib.dispatcher_get_kernel_name.restype = ctypes.c_char_p
        lib.dispatcher_get_num_dim_g.restype = ctypes.c_int
        lib.dispatcher_get_num_dim_m.restype = ctypes.c_int
        lib.dispatcher_get_num_dim_n.restype = ctypes.c_int
        lib.dispatcher_get_num_dim_k.restype = ctypes.c_int
        lib.dispatcher_get_num_d_tensors.restype = ctypes.c_int
        lib.dispatcher_cleanup.restype = None
        lib.dispatcher_run_batched_contraction.restype = ctypes.c_int
        lib.dispatcher_run_batched_contraction.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,  # d_ptrs, num_d
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_float),
        ]

    def kernel_name(self) -> str:
        raw = self._lib.dispatcher_get_kernel_name()
        return raw.decode() if raw else ""

    def num_dims(self) -> Tuple[int, int, int, int]:
        return (
            self._lib.dispatcher_get_num_dim_g(), self._lib.dispatcher_get_num_dim_m(),
            self._lib.dispatcher_get_num_dim_n(), self._lib.dispatcher_get_num_dim_k(),
        )

    def run(self, A, B, E, prob: BatchedContractionProblem, Ds=None) -> float:
        def i64(vals):
            return (ctypes.c_int64 * len(vals))(*[int(v) for v in vals])

        A = np.ascontiguousarray(A)
        B = np.ascontiguousarray(B)
        # E is the output buffer. ascontiguousarray() may return a *copy* when the
        # caller's E is non-contiguous; the kernel would then fill that copy and the
        # caller's original E would be left unchanged. Keep a handle to the original
        # so we can copy the result back into it after the kernel runs.
        original_E = E
        E = np.ascontiguousarray(E)
        Ds = Ds or []
        Ds = [np.ascontiguousarray(d) for d in Ds]
        num_d = len(Ds)
        if num_d:
            d_arr = (ctypes.c_void_p * num_d)(*[d.ctypes.data_as(ctypes.c_void_p) for d in Ds])
            d_ptr = ctypes.cast(d_arr, ctypes.POINTER(ctypes.c_void_p))
        else:
            d_ptr = None
        tms = ctypes.c_float(0.0)
        rc = self._lib.dispatcher_run_batched_contraction(
            A.ctypes.data_as(ctypes.c_void_p), B.ctypes.data_as(ctypes.c_void_p),
            E.ctypes.data_as(ctypes.c_void_p),
            d_ptr, num_d,
            i64(prob.g_dims), i64(prob.m_dims), i64(prob.n_dims), i64(prob.k_dims),
            len(prob.g_dims), len(prob.m_dims), len(prob.n_dims), len(prob.k_dims),
            int(prob.k_batch), ctypes.byref(tms),
        )
        if rc != 0:
            raise RuntimeError(f"dispatcher_run_batched_contraction rc={rc}")
        # If ascontiguousarray copied E, the kernel filled the copy, not the
        # caller's buffer. Copy the result back so the caller's E holds the output.
        if E is not original_E:
            np.copyto(original_E, E.reshape(original_E.shape))
        return tms.value

    def cleanup(self):
        try:
            self._lib.dispatcher_cleanup()
        except Exception:
            pass

    def __del__(self):
        self.cleanup()


# =============================================================================
# Runner
# =============================================================================


class GpuBatchedContractionRunner:
    def __init__(
        self,
        so_path: Path,
        dtype: str = "fp16",
        num_d_tensors: int = 0,
        elementwise: str = "PassThrough",
    ):
        self._lib = BatchedContractionDispatcherLib(so_path)
        # The compiled .so is the source of truth for the element dtype: the C++
        # bridge sizes its host<->device copies from sizeof(ADataType). Trusting a
        # mismatched caller dtype (e.g. dtype='fp16' on an fp32 .so) would allocate
        # host buffers of the wrong byte-width and cause OOB reads/writes. Derive
        # the compiled dtype from the exported kernel name and reject a mismatch.
        compiled_dtype = _dtype_from_kernel_name(self._lib.kernel_name())
        if compiled_dtype in _NP_DTYPE and dtype != compiled_dtype:
            raise ValueError(
                f"dtype mismatch: caller requested {dtype!r} but the compiled "
                f".so is {compiled_dtype!r} (kernel {self._lib.kernel_name()!r})"
            )
        self.dtype = compiled_dtype if compiled_dtype in _NP_DTYPE else dtype
        self.np_dtype = _np_dtype(self.dtype)
        # num_d / elementwise describe the D-tensor epilogue the compiled .so wired in.
        # We trust the .so's compiled-in count first (it is the source of truth), then
        # fall back to the caller-provided value.
        so_num_d = self._lib._lib.dispatcher_get_num_d_tensors()
        self.num_d_tensors = int(so_num_d if so_num_d >= 0 else num_d_tensors)
        self.elementwise = elementwise if self.num_d_tensors > 0 else "PassThrough"

    @property
    def kernel_name(self) -> str:
        return self._lib.kernel_name()

    def _make_ds(self, prob: BatchedContractionProblem, seed: Optional[int]) -> List[object]:
        """Build num_d D tensors of shape [G,M,N], dtype == A/B element type.

        Mirrors Old-TE's profiler which fills each D with FillUniformDistribution
        in [-1, 1] (see batched_contraction_profiler.hpp)."""
        rng = np.random.default_rng(seed)
        Ds: List[object] = []
        for _ in range(self.num_d_tensors):
            d = rng.uniform(-1.0, 1.0, size=(prob.G, prob.M, prob.N)).astype(self.np_dtype)
            Ds.append(d)
        return Ds

    def run(
        self,
        A,
        B,
        prob: BatchedContractionProblem,
        Ds=None,
        seed: Optional[int] = 0,
    ) -> BatchedContractionResult:
        # Coerce inputs to the kernel's element type so host byte-width matches device.
        A2 = np.asarray(A).astype(self.np_dtype).reshape(prob.G, prob.M, prob.K)
        B2 = np.asarray(B).astype(self.np_dtype).reshape(prob.G, prob.N, prob.K)
        E = np.zeros((prob.G, prob.M, prob.N), dtype=self.np_dtype)

        if self.num_d_tensors > 0:
            if Ds is None:
                Ds = self._make_ds(prob, seed)
            Ds = [np.asarray(d).astype(self.np_dtype).reshape(prob.G, prob.M, prob.N) for d in Ds]
            if len(Ds) != self.num_d_tensors:
                raise ValueError(
                    f"expected {self.num_d_tensors} D tensors, got {len(Ds)}"
                )
        else:
            Ds = []

        t = self._lib.run(A2, B2, E, prob, Ds=Ds if Ds else None)
        return BatchedContractionResult(
            E=E, time_ms=t, kernel_name=self.kernel_name, Ds=Ds if Ds else None
        )

    def reference(self, A, B, prob: BatchedContractionProblem, Ds=None):
        """fp32 reference with the compiled-in D epilogue applied.

        num_d==0 / PassThrough : E[g,m,n] = sum_k A[g,m,k]*B[g,n,k]
        MultiDAdd              : E = C + D0 + D1 + ...   (elementwise, D shape==E)
        MultiDMultiply         : E = C * D0 * D1 * ...
        Matches ck_tile::element_wise::MultiDAdd / MultiDMultiply and the host
        reference in reference_batched_contraction.hpp."""
        A2 = np.asarray(A).astype(np.float32).reshape(prob.G, prob.M, prob.K)
        B2 = np.asarray(B).astype(np.float32).reshape(prob.G, prob.N, prob.K)
        C = np.einsum("gmk,gnk->gmn", A2, B2)
        if self.num_d_tensors == 0 or self.elementwise == "PassThrough":
            return C
        if Ds is None:
            raise ValueError("reference() with num_d>0 requires the same Ds passed to run()")
        Ds32 = [np.asarray(d).astype(np.float32).reshape(prob.G, prob.M, prob.N) for d in Ds]
        if self.elementwise == "MultiDAdd":
            for d in Ds32:
                C = C + d
        elif self.elementwise == "MultiDMultiply":
            for d in Ds32:
                C = C * d
        else:
            raise ValueError(f"unsupported elementwise op {self.elementwise}")
        return C

    @staticmethod
    def reference_contraction(A, B, prob: BatchedContractionProblem):
        """Plain fp32 contraction (no D epilogue): E[g,m,n] = sum_k A[g,m,k]*B[g,n,k]."""
        A2 = np.asarray(A).astype(np.float32).reshape(prob.G, prob.M, prob.K)
        B2 = np.asarray(B).astype(np.float32).reshape(prob.G, prob.N, prob.K)
        return np.einsum("gmk,gnk->gmn", A2, B2)


# =============================================================================
# Build pipeline
# =============================================================================


def _generate_kernel(cfg: BatchedContractionKernelConfig, headers_dir: Path) -> Optional[Path]:
    cmd = [
        sys.executable, str(_CODEGEN_SCRIPT), "--output-dir", str(headers_dir),
        "--config-json", json.dumps(cfg.to_codegen_config()),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        log.error("codegen failed for %s:\n%s", cfg.name, r.stderr[-1500:])
        return None
    hpp = headers_dir / f"{cfg.name}.hpp"
    return hpp if hpp.exists() else None


def _compile_kernel(hpp: Path, so: Path, arch: str) -> bool:
    ck_root = _CTYPES_LIB_SRC.parent.parent.parent.parent  # .../composablekernel
    obj = so.with_suffix(".o")
    compile_cmd = [
        _HIPCC, "-c", "-fPIC", "-O3", "-std=c++17",
        f"-I{ck_root}/include", f"-I{ck_root}",
        "-DCK_TILE_SINGLE_KERNEL_INCLUDE", f"-include{hpp}",
        "-D__HIP_PLATFORM_AMD__", f"--offload-arch={arch}", f'-DGFX_ARCH="{arch}"',
        # Match Tile Engine's AMDGPU codegen flags exactly so the bridge .so
        # produces the same machine code as Old-TE (inlining, register
        # allocation, occupancy).  Without these, persistent kernels size their
        # grid differently and perf gaps appear.
        *_tile_engine_codegen_flags(),
        "-Wno-undefined-func-template", "-Wno-float-equal",
        str(_CTYPES_LIB_SRC), "-o", str(obj),
    ]
    r = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        log.error("compile failed for %s:\n%s", so.name, r.stderr[-2500:])
        return False
    r = subprocess.run(
        [_HIPCC, "-shared", "-fPIC", f"--offload-arch={arch}", "--hip-link", str(obj), "-o", str(so)],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        log.error("link failed for %s:\n%s", so.name, r.stderr[-1500:])
        return False
    return True


def setup_multiple_batched_contraction_dispatchers(
    configs: List[BatchedContractionKernelConfig],
    output_dir: Optional[Path] = None,
    gfx_arch: Optional[str] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """codegen -> hipcc -> .so for each config. Returns paths aligned with `configs`
    (None on failure). Dedups by .name; cross-arch cache via _{arch}.so suffix."""
    if not configs:
        return []
    explicit = gfx_arch or configs[0].gfx_arch
    arch = _validate_arch(explicit) if explicit else _get_arch()
    base = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="bc_bridge_"))
    headers = base / "generated_kernels"
    libs = base / "libs"
    headers.mkdir(parents=True, exist_ok=True)
    libs.mkdir(parents=True, exist_ok=True)

    seen: Dict[str, int] = {}
    deduped: List[Tuple[int, BatchedContractionKernelConfig]] = []
    for i, c in enumerate(configs):
        if c.name not in seen:
            seen[c.name] = i
            deduped.append((i, c))
    results: List[Optional[Path]] = [None] * len(configs)

    def build_one(idx: int, cfg: BatchedContractionKernelConfig):
        hpp = _generate_kernel(cfg, headers)
        if hpp is None:
            return idx, None
        so = libs / f"lib{cfg.name}_{arch}.so"
        if so.exists():
            return idx, so
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

    for i, c in enumerate(configs):
        if results[i] is None and seen.get(c.name, i) != i:
            results[i] = results[seen[c.name]]
    built = sum(1 for r in results if r)
    log.info("built %d/%d batched-contraction kernels for %s", built, len(configs), arch)
    return results


# =============================================================================
# Sweep expansion (for the TE driver)
# =============================================================================


def _range_values(spec: dict) -> List[int]:
    if "values" in spec:
        return list(spec["values"])
    lo, hi, step = spec["min"], spec["max"], spec.get("step", 1)
    out, v = [], lo
    while v <= hi:
        out.append(v)
        v += step
    return out


def expand_sweep(config: dict, dtype: str = "fp16", layout: str = "rcr") -> List[BatchedContractionKernelConfig]:
    """Expand a tile_config x trait_config sweep into valid kernel configs (deduped)."""
    tc = config["tile_config"]
    trc = config.get("trait_config", {})
    axes = {k: _range_values(tc[k]) for k in
            ["tile_m", "tile_n", "tile_k", "warp_m", "warp_n", "warp_k",
             "warp_tile_m", "warp_tile_n", "warp_tile_k"]}
    traits = {k: (trc[k]["values"] if k in trc else [d]) for k, d in
              [("pipeline", "compv3"), ("scheduler", "intrawave"), ("epilogue", "cshuffle"),
               ("pad_m", False), ("pad_n", False), ("pad_k", False), ("persistent", False)]}

    out: List[BatchedContractionKernelConfig] = []
    seen = set()
    keys = list(axes.keys())
    tkeys = list(traits.keys())
    for tile_vals in itertools.product(*[axes[k] for k in keys]):
        tile = dict(zip(keys, tile_vals))
        for trait_vals in itertools.product(*[traits[k] for k in tkeys]):
            tr = dict(zip(tkeys, trait_vals))
            cfg = BatchedContractionKernelConfig(
                dtype=dtype, layout=layout, **tr, **tile,
                num_dim_g=config.get("num_dim_g", 1), num_dim_m=config.get("num_dim_m", 1),
                num_dim_n=config.get("num_dim_n", 1), num_dim_k=config.get("num_dim_k", 1),
                num_d_tensors=config.get("num_d_tensors", 0),
                # Project elementwise alongside the D count: with num_d>0 the default
                # PassThrough would fail is_valid(), so a MultiD* sweep needs its op.
                elementwise=config.get(
                    "elementwise",
                    "PassThrough" if config.get("num_d_tensors", 0) == 0 else "MultiDAdd",
                ),
                k_block_per_cu=config.get("k_block_per_cu", 1),
            )
            if not cfg.is_valid():
                continue
            if cfg.name in seen:
                continue
            seen.add(cfg.name)
            out.append(cfg)
    return out


def default_fp16_config(gfx_arch: Optional[str] = None) -> BatchedContractionKernelConfig:
    gfx_arch = _validate_arch(gfx_arch) if gfx_arch else _get_arch()
    return BatchedContractionKernelConfig(
        dtype="fp16", layout="rcr", pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=64, warp_m=2, warp_n=2, warp_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        num_dim_g=1, num_dim_m=1, num_dim_n=1, num_dim_k=1, gfx_arch=gfx_arch,
    )
