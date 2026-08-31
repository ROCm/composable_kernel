# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Batched GEMM Tile Engine <-> Dispatcher bridge.

Batched counterpart of ``gemm_utils.py``. Batched GEMM adds a leading batch
dimension with per-batch strides, giving it a DIVERGENT ABI from the
single-problem GEMM library: the ctypes entry point
``dispatcher_run_batched`` carries ``batch_count``, ``k_batch`` (split-K), the
three batch strides, and the benchmarking knobs (warmup/repeat/flush_cache/
rotating_count) in addition to M/N/K, and the .so launches the force-included
kernel directly via ``SelectedKernel::launch(BatchedGemmHostArgs{...}, ...)``
(registry bypass). The benchmarking knobs let the bridge build the SAME
``stream_config`` the Tile Engine batched_gemm profiler uses, so A/B timing is
fair; ``k_batch`` mirrors Old-TE's split-K support.

Public surface (mirrors gemm_utils):

    BatchedGemmKernelConfig  -- shared contract dataclass (variant="batched")
        .name                -- registry/runtime lookup key (byte-exact vs the
                                codegen kernel header stem)
        .to_codegen_json()   -- feeds unified_gemm_codegen.py
    BatchedGemmProblem       -- a single (batch, M, N, K) problem
    setup_multiple_batched_gemm_dispatchers -- codegen + hipcc -> .so (NO GPU)
    BatchedGemmDispatcherLib -- thin ctypes ABI wrapper
    GpuBatchedGemmRunner     -- GPU run + time (from a .so path)
    expand_sweep             -- TE JSON sweep config -> [BatchedGemmKernelConfig]

The build half reuses gemm_utils' proven codegen/compile flag machinery
(``_tile_engine_codegen_flags`` etc.) so both bridges compile kernels with
byte-identical AMDGPU backend flags -- a prerequisite for fair A/B parity.
"""

from __future__ import annotations

import ctypes
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import ctypes_utils as _cu
import gemm_utils as _gu


# ============================================================================
# Old-TE IsSupportedArgument parity: reject odd-per-wave-repeat / 32-wide warp
# ============================================================================
#
# The batched_gemm default_config sweeps tile_m/tile_n = 64/128/192/256 across
# BOTH the ``cshuffle`` and ``default`` epilogues. The odd (>1) per-wave repeat
# paired with a 32-wide warp tile in that dimension -- e.g. tile=192 / wave=2 /
# warp_tile=32 => repeat = 192/(2*32) = 3 -- mis-maps the C accumulator and
# returns garbage (issue #9684). Old Tile-Engine's batched kernel refuses these
# configs at launch ("Arguments not supported" from IsSupportedArgument), so it
# never runs them; the bridge, however, would silently codegen and ship them.
#
# gemm_utils.expand_sweep already drops this signature for the ``cshuffle``
# epilogue via ``_cshuffle_store_ok`` -- but it exempts the ``default`` epilogue.
# In the batched sweep the SAME garbage occurs with the ``default`` epilogue
# (~420 distinct geometry stems survive), so we re-apply the identical
# odd-repeat/wt32 predicate here, epilogue-agnostically, before any .so is
# generated. This mirrors Old-TE's reject set EXACTLY (even repeats such as 2/4
# and MRepeat=3 with a 16-wide warp tile remain valid, so 64/128/256 tiles and
# the non-32 warp tiles are NOT over-pruned).
def _repeat_ok(
    tile_m: int,
    tile_n: int,
    wave_m: int,
    wave_n: int,
    warp_tile_m: int,
    warp_tile_n: int,
) -> bool:
    """Return False when either the M or N dimension has an odd per-wave repeat
    (>1) with a 32-wide warp tile -- the batched garbage signature Old-TE rejects
    (independent of epilogue). Returns True (allowed) for every other geometry."""

    def _dim_bad(tile: int, wave: int, warp_tile: int) -> bool:
        div = wave * warp_tile
        if div <= 0 or tile % div != 0:
            # Uneven split is already dropped upstream; treat as "not this bug".
            return False
        repeat = tile // div
        return repeat > 1 and repeat % 2 == 1 and warp_tile == 32

    return not (
        _dim_bad(tile_m, wave_m, warp_tile_m) or _dim_bad(tile_n, wave_n, warp_tile_n)
    )


# ============================================================================
# GPU architecture resolution (never default to gfx942)
# ============================================================================

_SUPPORTED_ARCHES: Tuple[str, ...] = ("gfx90a", "gfx942", "gfx950", "gfx1250")

# Byte size of each C output dtype as the compiled kernel writes it
# (sizeof(CDataType)). The host numpy buffer is memcpy'd verbatim to/from the
# device, so its element size must match these exactly -- see the F6 check in
# GpuBatchedGemmRunner.run(). ck_tile bf16_t is a 2-byte type (mirrored on the
# host by np.uint16); fp16 is 2 bytes; the int8 path accumulates into int32.
_C_SIZEOF: Dict[str, int] = {"fp16": 2, "bf16": 2, "int32": 4}


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
    if arch not in _SUPPORTED_ARCHES:
        raise ValueError(
            f"Unsupported GPU architecture {arch!r}; supported: {list(_SUPPORTED_ARCHES)}"
        )
    return arch


def _resolve_arch(arch: Optional[str]) -> str:
    """Resolve an explicit arch (validated) or auto-detect via rocminfo."""
    if arch is None:
        return _get_arch()
    if arch not in _SUPPORTED_ARCHES:
        raise ValueError(
            f"Unsupported GPU architecture {arch!r}; supported: {list(_SUPPORTED_ARCHES)}"
        )
    return arch


# ============================================================================
# The shared contract: BatchedGemmKernelConfig
# ============================================================================


@dataclass
class BatchedGemmKernelConfig(_gu.GemmKernelConfig):
    """GEMM kernel config specialised to the batched variant.

    Inherits every field/property from GemmKernelConfig and only pins the
    variant to ``"batched"`` so ``.name`` gains the ``_batched`` suffix that the
    codegen (unified_gemm_codegen.py::KernelNaming.generate) also appends -- the
    single thread tying config -> codegen -> runtime name together.

    ``gfx_arch`` is overridden to default to ``None`` (rather than the parent's
    hardcoded ``"gfx942"``) so the batched bridge never silently builds for the
    wrong GPU: the build path resolves ``None`` via ``_get_arch()`` (rocminfo)
    and raises on an unsupported / undetectable arch.
    """

    variant: str = "batched"
    gfx_arch: Optional[str] = None  # type: ignore[assignment]


# Extend GemmKernelConfig.name handling for the batched variant. The parent
# already appends "_streamk"/"_preshuffle" for those variants; "batched" needs
# the same treatment, so we patch the property lookup by overriding name here.
def _batched_name(self: BatchedGemmKernelConfig) -> str:  # pragma: no cover - thin
    base = _gu.GemmKernelConfig.name.fget(self)  # type: ignore[attr-defined]
    if self.variant == "batched" and not base.endswith("_batched"):
        return base + "_batched"
    return base


BatchedGemmKernelConfig.name = property(_batched_name)  # type: ignore[assignment]


# ============================================================================
# Problem / result
# ============================================================================


@dataclass
class BatchedGemmProblem:
    """A batched GEMM problem: C[b,M,N] = A[b,M,K] @ B[b,K,N] for b in [0,batch)."""

    batch_count: int
    M: int
    N: int
    K: int

    @property
    def flops(self) -> float:
        return 2.0 * self.batch_count * self.M * self.N * self.K

    def to_dict(self) -> Dict[str, int]:
        return {
            "batch_count": self.batch_count,
            "M": self.M,
            "N": self.N,
            "K": self.K,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "BatchedGemmProblem":
        return cls(
            batch_count=int(d["batch_count"]),
            M=int(d["M"]),
            N=int(d["N"]),
            K=int(d["K"]),
        )


@dataclass
class BatchedGemmResult:
    output: np.ndarray
    time_ms: float
    status: int
    tflops: float
    kernel_name: str

    @property
    def success(self) -> bool:
        return self.status == 0


# ============================================================================
# ctypes ABI wrapper (divergent from single-problem GEMM)
# ============================================================================


class BatchedGemmDispatcherLib:
    """Thin ctypes wrapper around a compiled batched GEMM dispatcher .so.

    The batched .so exposes exactly one kernel (the force-included header), so
    ``kernel_names`` returns a single-element list.
    """

    def __init__(self, so_path: Path):
        self._path = Path(so_path)
        self._lib = ctypes.CDLL(str(self._path))
        self._setup_functions()

    def _setup_functions(self) -> None:
        lib = self._lib

        lib.dispatcher_initialize.argtypes = []
        lib.dispatcher_initialize.restype = ctypes.c_int

        lib.dispatcher_get_kernel_count.argtypes = []
        lib.dispatcher_get_kernel_count.restype = ctypes.c_int

        lib.dispatcher_get_kernel_name.argtypes = []
        lib.dispatcher_get_kernel_name.restype = ctypes.c_char_p

        lib.dispatcher_get_kernel_name_at.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        lib.dispatcher_get_kernel_name_at.restype = ctypes.c_int

        # Divergent ABI: batch_count + k_batch (split-K) + three batch strides
        # on top of M/N/K, plus the benchmarking knobs (warmup/repeat/
        # flush_cache/rotating_count) so the .so drives stream_config identically
        # to the Tile Engine batched_gemm profiler for fair A/B timing.
        lib.dispatcher_run_batched.argtypes = [
            ctypes.c_void_p,  # A (host)
            ctypes.c_void_p,  # B (host)
            ctypes.c_void_p,  # C (host)
            ctypes.c_int64,  # M
            ctypes.c_int64,  # N
            ctypes.c_int64,  # K
            ctypes.c_int64,  # batch_count
            ctypes.c_int64,  # k_batch (split-K; 0/1 -> no split)
            ctypes.c_int64,  # stride_A
            ctypes.c_int64,  # stride_B
            ctypes.c_int64,  # stride_C
            ctypes.c_int64,  # batch_stride_A
            ctypes.c_int64,  # batch_stride_B
            ctypes.c_int64,  # batch_stride_C
            ctypes.c_int64,  # warmup   (<=0 -> TE default 50)
            ctypes.c_int64,  # repeat   (<=0 -> TE default 100)
            ctypes.c_int64,  # flush_cache (0/1)
            ctypes.c_int64,  # rotating_count (<=0 -> 1)
            ctypes.POINTER(ctypes.c_float),  # time_ms
        ]
        lib.dispatcher_run_batched.restype = ctypes.c_int

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
        buf = ctypes.create_string_buffer(256)
        if self._lib.dispatcher_get_kernel_name_at(0, buf, 256) == 0:
            return [buf.value.decode("utf-8")]
        raw = self._lib.dispatcher_get_kernel_name()
        return [raw.decode("utf-8")] if raw else []

    def run(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        M: int,
        N: int,
        K: int,
        batch_count: int,
        k_batch: int = 1,
        stride_A: int = 0,
        stride_B: int = 0,
        stride_C: int = 0,
        batch_stride_A: int = 0,
        batch_stride_B: int = 0,
        batch_stride_C: int = 0,
        warmup: int = 50,
        repeat: int = 100,
        flush_cache: bool = True,
        rotating_count: int = 1000,
    ) -> Tuple[int, float]:
        time_ms = ctypes.c_float(0.0)
        status = self._lib.dispatcher_run_batched(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            M,
            N,
            K,
            batch_count,
            k_batch,
            stride_A,
            stride_B,
            stride_C,
            batch_stride_A,
            batch_stride_B,
            batch_stride_C,
            warmup,
            repeat,
            1 if flush_cache else 0,
            rotating_count,
            ctypes.byref(time_ms),
        )
        return status, time_ms.value

    def cleanup(self) -> None:
        self._lib.dispatcher_cleanup()


# ============================================================================
# GPU runner
# ============================================================================


class GpuBatchedGemmRunner:
    """High-level runner: construct from a .so path, call run(A, B, problem).

    A/B/C are batched tensors laid out per the compiled kernel's layout. The C
    ABI takes HOST pointers and manages GPU memory internally, so this runner
    hands numpy arrays (in the kernel's element dtype + memory order) straight
    to the .so. fp16/rcr is the only TE-supported batched signature today, but
    the dtype-encode helpers are reused from gemm_utils so extending to more
    dtypes only requires codegen support.
    """

    def __init__(self, lib_path: Path):
        self.lib = BatchedGemmDispatcherLib(lib_path)
        if not self.lib.initialize():
            raise RuntimeError(f"Failed to initialize batched dispatcher .so: {lib_path}")
        names = self.lib.kernel_names
        self._kernel_name = names[0] if names else "unknown"

    @property
    def kernel_name(self) -> str:
        return self._kernel_name

    def run(
        self,
        A: np.ndarray,
        B: np.ndarray,
        problem: BatchedGemmProblem,
        k_batch: int = 1,
        batch_pad_A: int = 0,
        batch_pad_B: int = 0,
        batch_pad_C: int = 0,
        warmup: int = 50,
        repeat: int = 100,
        flush_cache: bool = True,
        rotating_count: int = 1000,
    ) -> BatchedGemmResult:
        """Run a batched GEMM.

        ``k_batch`` selects split-K (default 1 == no split, matching Old-TE).

        ``batch_pad_{A,B,C}`` add padding elements between consecutive per-batch
        slabs so the batch stride is NON-packed
        (batch_stride = packed_slab + pad). When any pad is > 0 the host buffer
        is over-allocated to slab*batch + pad*(batch) and each logical slab is
        scattered into its padded slot, exercising the batch_stride path in the
        .so (B2). With all pads 0 this is the packed case (batch stride 0).

        ``warmup/repeat/flush_cache/rotating_count`` are threaded into the .so's
        stream_config so timing matches the Tile Engine batched_gemm profiler
        (B1).
        """
        batch, M, N, K = problem.batch_count, problem.M, problem.N, problem.K

        # E: validate the logical tensor shapes BEFORE any layout transform. The
        # C bridge copies batch*M*K (A) and batch*K*N (B) elements verbatim from
        # the host pointers regardless of the actual NumPy allocation, so a
        # smaller same-rank input would trigger an out-of-bounds host read. A/B
        # are always passed in logical (batch, M, K) / (batch, K, N) order (the
        # column-major transpose below happens after this check), so validate
        # against those shapes up front.
        expected_A = (batch, M, K)
        expected_B = (batch, K, N)
        if A.shape != expected_A:
            raise ValueError(
                f"A has shape {A.shape}, expected logical (batch, M, K) "
                f"{expected_A} for problem batch={batch} M={M} K={K}"
            )
        if B.shape != expected_B:
            raise ValueError(
                f"B has shape {B.shape}, expected logical (batch, K, N) "
                f"{expected_B} for problem batch={batch} K={K} N={N}"
            )

        dtype = _gu._dtype_from_kernel_name(self._kernel_name)
        la, lb, lc = _gu._layout_from_kernel_name(self._kernel_name)

        # Per-batch layout transform: a 'c' (column-major) operand is stored as
        # the transpose of its logical (rows, cols) so the contiguous per-batch
        # slab matches column-major memory order (leading batch axis untouched).
        A_lay = A if la == "r" else np.transpose(A, (0, 2, 1))
        B_lay = B if lb == "r" else np.transpose(B, (0, 2, 1))
        C_shape = (batch, M, N) if lc == "r" else (batch, N, M)

        if dtype == "bf16":
            A_h = _gu._fp32_to_bf16_u16(A_lay)
            B_h = _gu._fp32_to_bf16_u16(B_lay)
        elif dtype == "fp8":
            A_h = _gu._fp32_to_fp8_u8(A_lay)
            B_h = _gu._fp32_to_fp8_u8(B_lay)
        elif dtype == "bf8":
            A_h = _gu._fp32_to_bf8_u8(A_lay)
            B_h = _gu._fp32_to_bf8_u8(B_lay)
        elif dtype == "int8":
            A_h = np.ascontiguousarray(A_lay, dtype=np.int8)
            B_h = np.ascontiguousarray(B_lay, dtype=np.int8)
        else:  # fp16 (default / only TE batched dtype)
            A_h = np.ascontiguousarray(A_lay, dtype=np.float16)
            B_h = np.ascontiguousarray(B_lay, dtype=np.float16)

        out_dtype = _gu._output_dtype(dtype)
        _C_NP = {"fp16": np.float16, "bf16": np.uint16, "int32": np.int32}
        if out_dtype not in _C_NP:
            raise ValueError(
                f"unsupported C dtype {out_dtype!r} (from input dtype {dtype!r}); "
                "add it to _C_NP so the host buffer matches sizeof(CDataType)"
            )
        # F6: the host C buffer is memcpy'd byte-for-byte to/from the device
        # buffer the kernel writes as CDataType, so the numpy element size MUST
        # equal the C++ sizeof(CDataType). Assert it here so a future _C_NP edit
        # (e.g. mapping fp16 -> np.float32) fails loudly instead of silently
        # copying the wrong byte count.
        if np.dtype(_C_NP[out_dtype]).itemsize != _C_SIZEOF[out_dtype]:
            raise ValueError(
                f"host C dtype size mismatch for {out_dtype!r}: numpy "
                f"{np.dtype(_C_NP[out_dtype]).itemsize} bytes != kernel "
                f"sizeof(CDataType) {_C_SIZEOF[out_dtype]} bytes"
            )

        # Packed per-batch slab element counts (row-major flattened).
        slab_A = M * K
        slab_B = K * N
        slab_C = M * N

        if batch_pad_A or batch_pad_B or batch_pad_C:
            # NON-PACKED (B2): interleave a padding gap between per-batch slabs.
            # Batch stride = packed slab + pad; host buffer is over-allocated to
            # batch_stride * batch and each slab is copied into its slot. The .so
            # sizes its device allocation from the same batch stride.
            bsa = slab_A + int(batch_pad_A)
            bsb = slab_B + int(batch_pad_B)
            bsc = slab_C + int(batch_pad_C)

            A_flat = A_h.reshape(batch, slab_A)
            B_flat = B_h.reshape(batch, slab_B)
            A_buf = np.zeros(bsa * batch, dtype=A_h.dtype)
            B_buf = np.zeros(bsb * batch, dtype=B_h.dtype)
            for b in range(batch):
                A_buf[b * bsa : b * bsa + slab_A] = A_flat[b]
                B_buf[b * bsb : b * bsb + slab_B] = B_flat[b]
            C_buf = np.zeros(bsc * batch, dtype=_C_NP[out_dtype])

            status, time_ms = self.lib.run(
                A_buf,
                B_buf,
                C_buf,
                M,
                N,
                K,
                batch,
                k_batch,
                0,
                0,
                0,
                bsa,
                bsb,
                bsc,
                warmup,
                repeat,
                flush_cache,
                rotating_count,
            )
            # Gather the C slabs back out of their padded slots.
            C_h = np.empty((batch, slab_C), dtype=_C_NP[out_dtype])
            for b in range(batch):
                C_h[b] = C_buf[b * bsc : b * bsc + slab_C]
            C_h = C_h.reshape(C_shape)
        else:
            # PACKED: batch strides default (0) -> kernel/.so derive them.
            C_h = np.zeros(C_shape, dtype=_C_NP[out_dtype])
            status, time_ms = self.lib.run(
                A_h,
                B_h,
                C_h,
                M,
                N,
                K,
                batch,
                k_batch,
                warmup=warmup,
                repeat=repeat,
                flush_cache=flush_cache,
                rotating_count=rotating_count,
            )

        if out_dtype == "bf16":
            C_dec = _gu._bf16_u16_to_fp32(C_h)
        else:
            C_dec = C_h
        C_out = C_dec if lc == "r" else np.transpose(C_dec, (0, 2, 1))

        tflops = (problem.flops / (time_ms * 1e-3)) / 1e12 if time_ms > 0 else 0.0
        return BatchedGemmResult(
            output=C_out,
            time_ms=time_ms,
            status=status,
            tflops=tflops,
            kernel_name=self._kernel_name,
        )


# ============================================================================
# Build API: codegen + hipcc -> .so paths (no GPU)
# ============================================================================


def _build_batched_compile_jobs(
    config: BatchedGemmKernelConfig, header: Path
) -> Tuple[Dict[str, Any], Path]:
    """Compile+link commands for the batched ctypes lib. Identical to the
    single-problem job (same flags, same includes) except it force-links the
    batched ctypes source (divergent ABI) instead of gemm_ctypes_lib.cpp."""
    root = _cu.get_dispatcher_root()
    ck_root = root.parent
    build_dir = _cu.get_build_dir()
    output_dir = _cu.get_generated_kernels_dir()
    ctypes_source = root / "bindings" / "ctypes" / "batched_gemm_ctypes_lib.cpp"

    lib_path = build_dir / "examples" / f"lib{config.name}.so"
    obj_file = lib_path.with_suffix(".o")

    # Never default to gfx942: resolve None via rocminfo (_get_arch) and validate
    # an explicit arch. GFX_ARCH is threaded to the .cpp as a hard requirement
    # (the source #errors if it is missing).
    gfx_arch = _resolve_arch(config.gfx_arch)

    compile_cmd = [
        _gu._resolve_hipcc(),
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
        f"--offload-arch={gfx_arch}",
        f'-DGFX_ARCH="{gfx_arch}"',
        # Byte-identical AMDGPU backend flags to the single-problem bridge and
        # Old-TE (see gemm_utils._tile_engine_codegen_flags) -- required for a
        # fair A/B parity comparison.
        *_gu._tile_engine_codegen_flags(),
        "-Wno-undefined-func-template",
        "-Wno-float-equal",
        str(ctypes_source),
        "-o",
        str(obj_file),
    ]
    # The batched ctypes lib bypasses the registry (it launches the kernel
    # directly), so unlike the single-problem lib it does NOT need to link the
    # dispatcher static archive -- only the kernel + ck_tile headers.
    link_cmd = [
        _gu._resolve_hipcc(),
        "-shared",
        "-fPIC",
        f"--offload-arch={gfx_arch}",
        "--hip-link",
        str(obj_file),
        "-o",
        str(lib_path),
    ]
    job = {"compile_cmd": compile_cmd, "link_cmd": link_cmd, "lib_path": str(lib_path)}
    return job, lib_path


def setup_multiple_batched_gemm_dispatchers(
    configs: List[BatchedGemmKernelConfig],
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """Codegen + compile each batched config into its own .so. Returns .so paths
    aligned to input order (None for configs that failed). Pure CPU (no GPU)."""
    import sys

    n = len(configs)
    results: List[Optional[Path]] = [None] * n
    if n == 0:
        return results

    max_workers = max_workers or min(multiprocessing.cpu_count(), 8)

    first_index: Dict[str, int] = {}
    unique: List[int] = []
    for i, c in enumerate(configs):
        key = c.name
        if key not in first_index:
            first_index[key] = i
            unique.append(i)

    codegen_script = _cu.get_codegen_path()
    output_dir = _cu.get_generated_kernels_dir()
    ctypes_source = (
        _cu.get_dispatcher_root()
        / "bindings"
        / "ctypes"
        / "batched_gemm_ctypes_lib.cpp"
    )
    if not ctypes_source.exists():
        raise FileNotFoundError(
            f"Missing batched ctypes source required for compilation:\n  {ctypes_source}"
        )

    # -- Step 1: parallel codegen (one header per unique config) --------------
    codegen_args = []
    for i in unique:
        c = configs[i]
        # Resolve arch here too so codegen and compile target the same detected
        # (or explicitly-validated) GPU -- never a silent gfx942 default.
        gpu_target = _resolve_arch(c.gfx_arch)
        codegen_args.append(
            {
                "index": i,
                "python": sys.executable,
                "codegen_script": str(codegen_script),
                "output_dir": str(output_dir),
                "dtype": c.dtype_a,
                "layout": c.layout,
                "gpu_target": gpu_target,
                "tile_config_json": c.to_codegen_json(),
                "hpp_glob_pattern": f"{c.name}.hpp",
                # Thread the batched variant so the emitted header is the batched
                # kernel and its name carries the _batched suffix (matching the
                # glob pattern above).
                "variant": c.variant,
            }
        )

    if verbose:
        print(
            f"[batched-gemm-bridge] codegen: {len(codegen_args)} headers "
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

    # -- Step 2: parallel compile + link --------------------------------------
    compile_jobs = []
    job_index: List[int] = []
    for i in unique:
        hdr = headers.get(i)
        if hdr is None:
            continue
        job, _ = _build_batched_compile_jobs(configs[i], hdr)
        compile_jobs.append(job)
        job_index.append(i)

    if verbose and compile_jobs:
        print(
            f"[batched-gemm-bridge] compile: {len(compile_jobs)} .so "
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

    for i, c in enumerate(configs):
        if results[i] is None:
            results[i] = results[first_index[c.name]]

    if verbose:
        ok_count = sum(1 for r in results if r is not None)
        print(f"[batched-gemm-bridge] setup complete: {ok_count}/{n} configs -> .so")

    return results


# ============================================================================
# TE sweep config expansion
# ============================================================================


def expand_sweep(
    config_path: str,
    arch: Optional[str] = None,
    dtype: str = "fp16",
    layout: str = "rcr",
) -> List[BatchedGemmKernelConfig]:
    """Expand a Tile Engine batched GEMM JSON sweep config into
    [BatchedGemmKernelConfig]. Reuses gemm_utils.expand_sweep (same tile/trait
    sweep + validity gates) and re-stamps each result as the batched variant.

    ``arch`` is resolved via ``_get_arch()`` (rocminfo) when omitted and
    validated against the supported set otherwise -- never a silent gfx942
    default."""
    arch = _resolve_arch(arch)
    # Match Old-TE's validated set EXACTLY: the batched_gemm instance builder
    # (tile_engine/ops/gemm/batched_gemm/batched_gemm_instance_builder.py)
    # declares --datatype choices=["fp16"] and --layout choices=["rcr"], so
    # anything else would codegen/compile/launch a kernel Old-TE never validated
    # (claimed parity != exercised parity). Reject it up front with a clear error
    # rather than silently building an untested signature.
    if dtype != "fp16":
        raise ValueError(
            f"batched_gemm bridge supports only dtype 'fp16' (Old-TE "
            f"batched_gemm_instance_builder declares --datatype choices=['fp16']); "
            f"got {dtype!r}"
        )
    if layout != "rcr":
        raise ValueError(
            f"batched_gemm bridge supports only layout 'rcr' (Old-TE "
            f"batched_gemm_instance_builder declares --layout choices=['rcr']); "
            f"got {layout!r}"
        )
    base_configs = _gu.expand_sweep(config_path, arch, dtype=dtype, layout=layout)
    out: List[BatchedGemmKernelConfig] = []
    seen: set = set()
    for b in base_configs:
        # Old-TE IsSupportedArgument parity gate (issue #9684): drop the
        # odd-per-wave-repeat / 32-wide-warp-tile signature (e.g. tile=192 /
        # wave=2 / warp_tile=32 => repeat=3) that returns garbage. gemm_utils
        # only gates this for the cshuffle epilogue; the batched sweep also
        # emits it with the default epilogue, so re-apply it here regardless of
        # epilogue -- before any .so is generated. See _repeat_ok for scope.
        if not _repeat_ok(
            b.tile_m,
            b.tile_n,
            b.wave_m,
            b.wave_n,
            b.warp_tile_m,
            b.warp_tile_n,
        ):
            continue
        c = BatchedGemmKernelConfig(
            dtype_a=b.dtype_a,
            dtype_b=b.dtype_b,
            dtype_c=b.dtype_c,
            dtype_acc=b.dtype_acc,
            layout_a=b.layout_a,
            layout_b=b.layout_b,
            layout_c=b.layout_c,
            tile_m=b.tile_m,
            tile_n=b.tile_n,
            tile_k=b.tile_k,
            wave_m=b.wave_m,
            wave_n=b.wave_n,
            wave_k=b.wave_k,
            warp_tile_m=b.warp_tile_m,
            warp_tile_n=b.warp_tile_n,
            warp_tile_k=b.warp_tile_k,
            pipeline=b.pipeline,
            scheduler=b.scheduler,
            epilogue=b.epilogue,
            pad_m=b.pad_m,
            pad_n=b.pad_n,
            pad_k=b.pad_k,
            persistent=b.persistent,
            gfx_arch=b.gfx_arch,
            variant="batched",
        )
        if c.name in seen:
            continue
        seen.add(c.name)
        out.append(c)
    return out
