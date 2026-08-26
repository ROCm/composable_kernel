#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Batched Contraction Multiple ABD dispatcher utilities.

Three-layer Python bridge for the dispatcher's batched_contraction_multi_abd path:

  ContractionMultiABDKernelConfig  — describes one kernel; .name is byte-exact
                                     with the codegen's KERNEL_NAME
  ContractionMultiABDDispatcherLib — thin ctypes wrapper around a compiled .so
  ContractionMultiABDRunner        — high-level runner accepting numpy arrays

Build helper (self-contained, does not import from ctypes_utils.py):
  setup_multiple_contraction_multi_abd_dispatchers(configs, ...)
       codegen → hipcc → list of .so paths, all in parallel

Usage (end-to-end):
  import numpy as np
  configs = [ContractionMultiABDKernelConfig(dtype="fp16", layout="rcr", ...)]
  so_paths = setup_multiple_contraction_multi_abd_dispatchers(
      configs, output_dir=Path("/tmp/cma"))
  runner = ContractionMultiABDRunner(so_paths[0])
  result = runner.run(As, Bs, Ds, problem)
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

import numpy as np

log = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_contraction_multi_abd_codegen.py"
_CTYPES_LIB_SRC = (
    Path(__file__).parent.parent
    / "bindings" / "ctypes"
    / "batched_contraction_multi_abd_ctypes_lib.cpp"
)

# Import the shared name-construction helper from codegen so both sides
# produce byte-exact names without duplicating the logic.
_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from unified_contraction_multi_abd_codegen import (  # noqa: E402
    make_contraction_multi_abd_kernel_name,
    validate_contraction_multi_abd_params,
)

_DEFAULT_HIPCC = "hipcc"

# Archs this bridge is known to build for. Mirrors batched_contraction_utils;
# there is deliberately no default -- see _detect_gpu_arch().
_SUPPORTED_ARCHS = ("gfx90a", "gfx942", "gfx950")

_HIPCC_BASE_FLAGS = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-DCK_TILE_SINGLE_KERNEL_INCLUDE",
    "-w",
]


# =============================================================================
# ContractionMultiABDKernelConfig — byte-exact naming with codegen
# =============================================================================


@dataclass
class ContractionMultiABDKernelConfig:
    """
    Complete description of one batched_contraction_multi_abd kernel.

    The .name property produces the exact string that unified_contraction_multi_abd_codegen.py
    emits as KERNEL_NAME, ensuring the Python side and compiled .so always agree.
    """

    dtype: str           # "fp16", "bf16"
    layout: str          # 3-char: "rcr", "rrr", etc.
    pipeline: str        # "compv3", "compv4", "mem"
    epilogue: str        # "cshuffle", "default2d"
    scheduler: str       # "intrawave", "interwave"

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
    pad_k: bool = False
    persistent: bool = False

    num_a_tensor: int = 1
    num_b_tensor: int = 1
    num_d_tensor: int = 1
    num_dim_g: int = 1
    num_dim_m: int = 2
    num_dim_n: int = 2
    num_dim_k: int = 1

    a_elementwise: str = "PassThrough"
    b_elementwise: str = "PassThrough"
    cde_elementwise: str = "MultiDAdd"

    # Empty means "detect at build time"; there is deliberately no hard-coded
    # default arch, so a wrong-GPU build fails loudly instead of silently.
    gfx_arch: str = ""

    def __post_init__(self):
        # Same rules the codegen spec enforces, via the same function. Without
        # this, an unsupported combination is accepted here and only surfaces
        # much later as codegen subprocess stderr, by which point the caller has
        # lost the connection to the field that was wrong.
        validate_contraction_multi_abd_params(
            epilogue=self.epilogue,
            persistent=self.persistent,
            num_a_tensor=self.num_a_tensor,
            num_b_tensor=self.num_b_tensor,
        )

    @property
    def name(self) -> str:
        """Byte-exact match to codegen KERNEL_NAME."""
        return make_contraction_multi_abd_kernel_name(
            dtype=self.dtype,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            pad_m=self.pad_m,
            pad_n=self.pad_n,
            pad_k=self.pad_k,
            persistent=self.persistent,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m,
            warp_tile_n=self.warp_tile_n,
            warp_tile_k=self.warp_tile_k,
            num_a_tensor=self.num_a_tensor,
            num_b_tensor=self.num_b_tensor,
            num_d_tensor=self.num_d_tensor,
            num_dim_g=self.num_dim_g,
            num_dim_m=self.num_dim_m,
            num_dim_n=self.num_dim_n,
            num_dim_k=self.num_dim_k,
            a_elementwise=self.a_elementwise,
            b_elementwise=self.b_elementwise,
            cde_elementwise=self.cde_elementwise,
        )

    def to_codegen_config(self) -> dict:
        """Produce the config dict for unified_contraction_multi_abd_codegen.py."""
        return {
            "dtypes":     [self.dtype],
            "layouts":    [self.layout],
            "pipelines":  [self.pipeline],
            "epilogues":  [self.epilogue],
            "schedulers": [self.scheduler],
            # persistent is part of the kernel name, so it must be projected here
            # too -- otherwise codegen defaults it to False and emits a header whose
            # name does not match the one this spec reports.
            "pad_options": [{
                "pad_m": self.pad_m,
                "pad_n": self.pad_n,
                "pad_k": self.pad_k,
                "persistent": self.persistent,
            }],
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
            "num_a_tensors":   [self.num_a_tensor],
            "num_b_tensors":   [self.num_b_tensor],
            "num_d_tensors":   [self.num_d_tensor],
            "dim_combos": [{
                "num_dim_g": self.num_dim_g,
                "num_dim_m": self.num_dim_m,
                "num_dim_n": self.num_dim_n,
                "num_dim_k": self.num_dim_k,
            }],
            "a_elementwise":   self.a_elementwise,
            "b_elementwise":   self.b_elementwise,
            "cde_elementwise": self.cde_elementwise,
        }


# =============================================================================
# ContractionMultiABDProblem — runtime problem shape
# =============================================================================


@dataclass
class ContractionMultiABDProblem:
    """Runtime contraction problem shape.

    g_dims  : list of G (batch) dimension sizes
    m_dims  : list of M dimension sizes
    n_dims  : list of N dimension sizes
    k_dims  : list of K dimension sizes
    k_batch : split-K factor (default 1)
    """

    g_dims: List[int]
    m_dims: List[int]
    n_dims: List[int]
    k_dims: List[int]
    k_batch: int = 1

    @property
    def G_total(self) -> int:
        r = 1
        for d in self.g_dims: r *= d
        return r

    @property
    def M_total(self) -> int:
        r = 1
        for d in self.m_dims: r *= d
        return r

    @property
    def N_total(self) -> int:
        r = 1
        for d in self.n_dims: r *= d
        return r

    @property
    def K_total(self) -> int:
        r = 1
        for d in self.k_dims: r *= d
        return r


@dataclass
class ContractionMultiABDResult:
    E: object     # numpy array
    time_ms: float
    kernel_name: str


# =============================================================================
# ContractionMultiABDDispatcherLib — thin ctypes wrapper
# =============================================================================


class ContractionMultiABDDispatcherLib:
    """Loads a compiled batched_contraction_multi_abd .so and wraps its C API."""

    def __init__(self, so_path: Path):
        self.so_path = Path(so_path)
        if not self.so_path.exists():
            raise FileNotFoundError(f"contraction_multi_abd .so not found: {self.so_path}")
        self._lib = ctypes.CDLL(str(self.so_path))
        self._setup()
        rc = self._lib.dispatcher_initialize()
        if rc != 0:
            raise RuntimeError(f"dispatcher_initialize() returned {rc}")

    def _setup(self):
        lib = self._lib

        lib.dispatcher_initialize.restype  = ctypes.c_int
        lib.dispatcher_initialize.argtypes = []

        lib.dispatcher_get_kernel_name.restype  = ctypes.c_char_p
        lib.dispatcher_get_kernel_name.argtypes = []

        lib.dispatcher_get_kernel_count.restype  = ctypes.c_int
        lib.dispatcher_get_kernel_count.argtypes = []

        lib.dispatcher_get_num_a_tensors.restype  = ctypes.c_int
        lib.dispatcher_get_num_a_tensors.argtypes = []

        lib.dispatcher_get_num_b_tensors.restype  = ctypes.c_int
        lib.dispatcher_get_num_b_tensors.argtypes = []

        lib.dispatcher_get_num_d_tensors.restype  = ctypes.c_int
        lib.dispatcher_get_num_d_tensors.argtypes = []

        lib.dispatcher_get_num_dim_g.restype  = ctypes.c_int
        lib.dispatcher_get_num_dim_g.argtypes = []

        lib.dispatcher_get_num_dim_m.restype  = ctypes.c_int
        lib.dispatcher_get_num_dim_m.argtypes = []

        lib.dispatcher_get_num_dim_n.restype  = ctypes.c_int
        lib.dispatcher_get_num_dim_n.argtypes = []

        lib.dispatcher_get_num_dim_k.restype  = ctypes.c_int
        lib.dispatcher_get_num_dim_k.argtypes = []

        lib.dispatcher_cleanup.restype  = None
        lib.dispatcher_cleanup.argtypes = []

        # Main run function
        # as_ptrs, bs_ptrs, ds_ptrs: void** (array of void*)
        # e_ptr: void*
        # num_a, num_b, num_d: int
        # g_dims, m_dims, n_dims, k_dims: int64*
        # num_dim_g, ...: int
        # a_strides_flat, b_strides_flat, d_strides_flat, e_strides: int64*
        # elem_a, elem_b, elem_d, elem_e: int
        # k_batch: int64
        # time_ms: float*
        lib.dispatcher_run_batched_contraction_multi_abd.restype  = ctypes.c_int
        lib.dispatcher_run_batched_contraction_multi_abd.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),   # as_ptrs
            ctypes.POINTER(ctypes.c_void_p),   # bs_ptrs
            ctypes.POINTER(ctypes.c_void_p),   # ds_ptrs
            ctypes.c_void_p,                   # e_ptr
            ctypes.c_int,                      # num_a
            ctypes.c_int,                      # num_b
            ctypes.c_int,                      # num_d
            ctypes.POINTER(ctypes.c_int64),    # g_dims
            ctypes.POINTER(ctypes.c_int64),    # m_dims
            ctypes.POINTER(ctypes.c_int64),    # n_dims
            ctypes.POINTER(ctypes.c_int64),    # k_dims
            ctypes.c_int,                      # num_dim_g
            ctypes.c_int,                      # num_dim_m
            ctypes.c_int,                      # num_dim_n
            ctypes.c_int,                      # num_dim_k
            ctypes.POINTER(ctypes.c_int64),    # a_strides_flat
            ctypes.POINTER(ctypes.c_int64),    # b_strides_flat
            ctypes.POINTER(ctypes.c_int64),    # d_strides_flat (may be NULL if num_d==0)
            ctypes.POINTER(ctypes.c_int64),    # e_strides
            ctypes.c_int,                      # elem_a
            ctypes.c_int,                      # elem_b
            ctypes.c_int,                      # elem_d
            ctypes.c_int,                      # elem_e
            ctypes.c_int64,                    # k_batch
            ctypes.POINTER(ctypes.c_float),    # time_ms
        ]

    def run(
        self,
        As: List[np.ndarray],
        Bs: List[np.ndarray],
        Ds: List[np.ndarray],
        E: np.ndarray,
        problem: ContractionMultiABDProblem,
    ) -> Tuple[int, float]:
        """
        Call dispatcher_run_batched_contraction_multi_abd.

        All arrays must be C-contiguous numpy arrays.
        Returns (return_code, time_ms).
        """
        # Validate tensor counts and shapes before passing raw pointers to C.
        # The C shim uses G*M*K, G*N*K, G*M*N element counts -- a smaller array
        # would cause an out-of-bounds read inside hipMemcpy.
        expected_a_elems = problem.G_total * problem.M_total * problem.K_total
        expected_b_elems = problem.G_total * problem.N_total * problem.K_total
        expected_e_elems = problem.G_total * problem.M_total * problem.N_total

        for i, a in enumerate(As):
            if a.size != expected_a_elems:
                raise ValueError(
                    f"As[{i}] has {a.size} elements but problem requires "
                    f"G*M*K = {expected_a_elems}"
                )
            if not a.flags["C_CONTIGUOUS"]:
                raise ValueError(f"As[{i}] must be C-contiguous")
        for i, b in enumerate(Bs):
            if b.size != expected_b_elems:
                raise ValueError(
                    f"Bs[{i}] has {b.size} elements but problem requires "
                    f"G*N*K = {expected_b_elems}"
                )
            if not b.flags["C_CONTIGUOUS"]:
                raise ValueError(f"Bs[{i}] must be C-contiguous")
        for i, d in enumerate(Ds):
            if d.size != expected_e_elems:
                raise ValueError(
                    f"Ds[{i}] has {d.size} elements but problem requires "
                    f"G*M*N = {expected_e_elems}"
                )
            if not d.flags["C_CONTIGUOUS"]:
                raise ValueError(f"Ds[{i}] must be C-contiguous")
        # All D tensors must share the same element size; a mismatch causes the C shim
        # to allocate all D device buffers with Ds[0].itemsize and overflow on copy.
        if Ds:
            d0_itemsize = Ds[0].itemsize
            for i, d in enumerate(Ds[1:], start=1):
                if d.itemsize != d0_itemsize:
                    raise ValueError(
                        f"Ds[{i}].dtype ({d.dtype}, itemsize={d.itemsize}) does not match "
                        f"Ds[0].dtype ({Ds[0].dtype}, itemsize={d0_itemsize}); "
                        "all D tensors must have the same element size."
                    )
        if E.size != expected_e_elems:
            raise ValueError(
                f"E has {E.size} elements but problem requires G*M*N = {expected_e_elems}"
            )
        if not E.flags["C_CONTIGUOUS"]:
            raise ValueError("E must be C-contiguous")

        num_a, num_b, num_d = len(As), len(Bs), len(Ds)

        # Build void** arrays of pointers
        as_arr = (ctypes.c_void_p * num_a)(
            *[a.ctypes.data_as(ctypes.c_void_p) for a in As]
        )
        bs_arr = (ctypes.c_void_p * num_b)(
            *[b.ctypes.data_as(ctypes.c_void_p) for b in Bs]
        )
        if num_d > 0:
            ds_arr = (ctypes.c_void_p * num_d)(
                *[d.ctypes.data_as(ctypes.c_void_p) for d in Ds]
            )
        else:
            ds_arr = (ctypes.c_void_p * 0)()

        e_ptr = E.ctypes.data_as(ctypes.c_void_p)

        # Dimension arrays
        g_arr = np.array(problem.g_dims, dtype=np.int64)
        m_arr = np.array(problem.m_dims, dtype=np.int64)
        n_arr = np.array(problem.n_dims, dtype=np.int64)
        k_arr = np.array(problem.k_dims, dtype=np.int64)

        g_ptr = g_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        m_ptr = m_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        n_ptr = n_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        k_ptr = k_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

        num_dim_g = len(problem.g_dims)
        num_dim_m = len(problem.m_dims)
        num_dim_n = len(problem.n_dims)
        num_dim_k = len(problem.k_dims)

        a_dim_size = num_dim_g + num_dim_m + num_dim_k
        b_dim_size = num_dim_g + num_dim_n + num_dim_k
        e_dim_size = num_dim_g + num_dim_m + num_dim_n

        # The C shim needs one stride per logical dimension of each tensor, in the
        # order the kernel indexes it (A: G,M,K -- B: G,N,K -- D/E: G,M,N).
        #
        # Deriving these from arr.strides is not safe: a caller may hand us a flat
        # (G*M*K,) buffer, whose numpy strides have length 1 rather than a_dim_size,
        # which would leave the flat stride array short and make the C side read
        # past its end. Every array reaching this point is already validated as
        # C-contiguous with exactly product(dims) elements, so its layout is packed
        # row-major over the logical dims -- compute the strides from the dims.
        def _packed_strides(dims: List[int]) -> List[int]:
            strides = [1] * len(dims)
            for i in range(len(dims) - 2, -1, -1):
                strides[i] = strides[i + 1] * int(dims[i + 1])
            return strides

        a_dims = list(problem.g_dims) + list(problem.m_dims) + list(problem.k_dims)
        b_dims = list(problem.g_dims) + list(problem.n_dims) + list(problem.k_dims)
        e_dims = list(problem.g_dims) + list(problem.m_dims) + list(problem.n_dims)

        a_strides = np.array(_packed_strides(a_dims) * num_a, dtype=np.int64)
        b_strides = np.array(_packed_strides(b_dims) * num_b, dtype=np.int64)

        if num_d > 0:
            d_strides = np.array(_packed_strides(e_dims) * num_d, dtype=np.int64)
            d_strides_ptr = d_strides.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        else:
            d_strides = np.array([], dtype=np.int64)
            d_strides_ptr = ctypes.cast(None, ctypes.POINTER(ctypes.c_int64))

        e_strides = np.array(_packed_strides(e_dims), dtype=np.int64)

        assert a_strides.size == num_a * a_dim_size
        assert b_strides.size == num_b * b_dim_size
        assert e_strides.size == e_dim_size

        elem_a = As[0].itemsize  if As  else 2
        elem_b = Bs[0].itemsize  if Bs  else 2
        elem_d = Ds[0].itemsize  if Ds  else 2
        elem_e = E.itemsize

        time_ms = ctypes.c_float(0.0)

        rc = self._lib.dispatcher_run_batched_contraction_multi_abd(
            ctypes.cast(as_arr, ctypes.POINTER(ctypes.c_void_p)),
            ctypes.cast(bs_arr, ctypes.POINTER(ctypes.c_void_p)),
            ctypes.cast(ds_arr, ctypes.POINTER(ctypes.c_void_p)),
            e_ptr,
            ctypes.c_int(num_a),
            ctypes.c_int(num_b),
            ctypes.c_int(num_d),
            g_ptr, m_ptr, n_ptr, k_ptr,
            ctypes.c_int(num_dim_g),
            ctypes.c_int(num_dim_m),
            ctypes.c_int(num_dim_n),
            ctypes.c_int(num_dim_k),
            a_strides.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            b_strides.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            d_strides_ptr,
            e_strides.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int(elem_a),
            ctypes.c_int(elem_b),
            ctypes.c_int(elem_d),
            ctypes.c_int(elem_e),
            ctypes.c_int64(problem.k_batch),
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
# ContractionMultiABDRunner — high-level runner
# =============================================================================


class ContractionMultiABDRunner:
    """
    High-level runner that loads a contraction_multi_abd .so and runs on the GPU.

    Accepts numpy arrays for As, Bs, Ds; allocates E; returns result.
    """

    def __init__(self, so_path: Path):
        self._lib = ContractionMultiABDDispatcherLib(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(
        self,
        As: List[np.ndarray],
        Bs: List[np.ndarray],
        Ds: List[np.ndarray],
        problem: ContractionMultiABDProblem,
        e_dtype=None,
    ) -> ContractionMultiABDResult:
        """
        Run batched contraction multi-ABD.

        As : list of A tensors, each shape matching [G..., M..., K...]
        Bs : list of B tensors, each shape matching [G..., N..., K...]
        Ds : list of D tensors, each shape matching [G..., M..., N...]
        problem : ContractionMultiABDProblem with dimension lists
        e_dtype : numpy dtype for the output E buffer (default: same as As[0])

        Returns ContractionMultiABDResult with E shape [G..., M..., N...].
        """
        if not As:
            raise ValueError("As must be non-empty")
        if not Bs:
            raise ValueError("Bs must be non-empty")

        As = [np.ascontiguousarray(a) for a in As]
        Bs = [np.ascontiguousarray(b) for b in Bs]
        Ds = [np.ascontiguousarray(d) for d in Ds]

        if e_dtype is None:
            e_dtype = As[0].dtype
        else:
            e_dtype = np.dtype(e_dtype)
            # The compiled kernel has a fixed EDataType (same element width as A/B).
            # A mismatched e_dtype would silently produce garbage output.
            if e_dtype.itemsize != As[0].dtype.itemsize:
                raise ValueError(
                    f"e_dtype={e_dtype} (itemsize={e_dtype.itemsize}) does not match "
                    f"the kernel's compiled output element size "
                    f"(As[0].dtype={As[0].dtype}, itemsize={As[0].dtype.itemsize}). "
                    "Use e_dtype=None to let the runner pick the correct dtype automatically."
                )

        e_shape = tuple(problem.g_dims + problem.m_dims + problem.n_dims)
        E = np.zeros(e_shape, dtype=e_dtype)

        rc, time_ms = self._lib.run(As, Bs, Ds, E, problem)
        if rc != 0:
            raise RuntimeError(
                f"dispatcher_run_batched_contraction_multi_abd failed with code {rc} "
                f"for kernel {self.kernel_name}"
            )
        return ContractionMultiABDResult(E=E, time_ms=time_ms, kernel_name=self.kernel_name)


# =============================================================================
# Subprocess helpers (self-contained)
# =============================================================================


def _validate_arch(arch: str) -> str:
    """Validate an explicitly supplied arch against the supported set."""
    if arch not in _SUPPORTED_ARCHS:
        raise ValueError(
            f"Unsupported GPU architecture {arch!r}; supported: {list(_SUPPORTED_ARCHS)}"
        )
    return arch


def _detect_gpu_arch() -> str:
    """Detect the current GPU arch via rocm_agent_enumerator; raise on failure.

    Never defaults: a wrong arch compiles silently and then mis-tunes (or fails
    at load) on the actual device, which is far harder to diagnose than an
    up-front error. Callers that know the target must pass gfx_arch explicitly.
    """
    arch = ""
    try:
        result = subprocess.run(
            ["rocm_agent_enumerator"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("gfx") and line != "gfx000":
                arch = line
                break
    except Exception:
        arch = ""
    if not arch:
        raise RuntimeError(
            "Could not detect GPU architecture from rocm_agent_enumerator; refusing "
            "to default to a specific GPU. Pass gfx_arch explicitly."
        )
    return _validate_arch(arch)


def _get_ck_include_dir() -> Optional[Path]:
    """Attempt to locate the CK include directory relative to this file."""
    here = Path(__file__).resolve().parent
    for parent in [here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "include"
        if (candidate / "ck_tile").is_dir():
            return candidate
    return None


def _generate_kernel_header(
    config: ContractionMultiABDKernelConfig,
    output_dir: Path,
) -> Optional[Path]:
    """Run unified_contraction_multi_abd_codegen.py for one config; return .hpp path or None."""
    config_dict = config.to_codegen_config()
    config_json = json.dumps(config_dict)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(config_json)
        cfg_file = f.name

    try:
        cmd = [
            sys.executable,
            str(_CODEGEN_SCRIPT),
            "--output-dir", str(output_dir),
            "--config", cfg_file,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log.error("Codegen failed for %s:\n%s", config.name, result.stderr)
            return None
    except subprocess.TimeoutExpired:
        log.error("Codegen timed out for %s", config.name)
        return None
    finally:
        Path(cfg_file).unlink(missing_ok=True)

    hpp = output_dir / f"{config.name}.hpp"
    if not hpp.exists():
        log.error("Codegen succeeded but %s not found", hpp)
        return None
    return hpp


def _compile_kernel(
    hpp_path: Path,
    so_path: Path,
    gfx_arch: str,
    hipcc: str = _DEFAULT_HIPCC,
    extra_include_dirs: Optional[List[str]] = None,
) -> bool:
    """Compile a generated .hpp into a .so via hipcc (compile then link)."""
    ck_include = _get_ck_include_dir()
    obj_path   = so_path.with_suffix(".o")

    arch_defines = []
    if "gfx12" in gfx_arch or "gfx950" in gfx_arch:
        arch_defines += ["-DCK_USE_OCP_FP8", "-DCK_TILE_USE_OCP_FP8"]
    if "gfx950" in gfx_arch:
        arch_defines += ["-DCK_USE_NATIVE_MX_SUPPORT", "-DCK_GFX950_SUPPORT"]

    compile_cmd = [
        hipcc, "-c", "-fPIC", "-O3", "-std=c++17",
        "-DCK_TILE_SINGLE_KERNEL_INCLUDE", "-w",
        f"--offload-arch={gfx_arch}",
        f"-DGFX_ARCH=\"{gfx_arch}\"",
        *arch_defines,
        "-include", str(hpp_path),
        str(_CTYPES_LIB_SRC),
        "-o", str(obj_path),
    ]
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
        return False

    link_cmd = [
        hipcc, "-shared", "-fPIC",
        f"--offload-arch={gfx_arch}", "--hip-link",
        str(obj_path), "-o", str(so_path),
    ]
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
# setup_multiple_contraction_multi_abd_dispatchers — build pipeline
# =============================================================================


import concurrent.futures


def setup_multiple_contraction_multi_abd_dispatchers(
    configs: List[ContractionMultiABDKernelConfig],
    output_dir: Optional[Path] = None,
    hipcc: str = _DEFAULT_HIPCC,
    gfx_arch: Optional[str] = None,
    extra_include_dirs: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """
    For each ContractionMultiABDKernelConfig: codegen → hipcc compile → .so path.

    Returns a list parallel to `configs` — each entry is the Path to the
    compiled .so, or None if that config failed.

    No GPU is required to call this function.
    """
    if not configs:
        return []

    arch     = _validate_arch(gfx_arch) if gfx_arch else _detect_gpu_arch()
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix="contraction_multi_abd_"))
    base_dir.mkdir(parents=True, exist_ok=True)

    headers_dir = base_dir / "generated_kernels"
    so_dir      = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    log.info(
        "Building %d contraction_multi_abd kernel(s) for %s into %s",
        len(configs), arch, base_dir,
    )

    # Deduplicate by name
    seen: Dict[str, int] = {}
    deduped: List[Tuple[int, ContractionMultiABDKernelConfig]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx: int, cfg: ContractionMultiABDKernelConfig) -> Tuple[int, Optional[Path]]:
        hpp = _generate_kernel_header(cfg, headers_dir)
        if hpp is None:
            return idx, None

        so = so_dir / f"lib{cfg.name}_{arch}.so"
        if so.exists():
            log.info("  [cached] %s", so.name)
            return idx, so

        ok = _compile_kernel(
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
            futures = {
                ex.submit(_build_one, idx, cfg): (idx, cfg)
                for idx, cfg in deduped
            }
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

    # Fill duplicates
    for i, cfg in enumerate(configs):
        if results[i] is None:
            first_idx = seen.get(cfg.name)
            if first_idx is not None and first_idx != i:
                results[i] = results[first_idx]

    built = sum(1 for r in results if r is not None)
    log.info("Built %d / %d contraction_multi_abd kernels", built, len(configs))
    return results
