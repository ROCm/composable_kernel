#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CK Tile Convolution Dispatcher Utilities

Common utilities for convolution kernel specification using the
Signature/Algorithm/Arch pattern from experimental/builder/reflect.

Structure:
  - Signature: WHAT operation (types, layouts, direction, element ops)
  - Algorithm: HOW it's computed (tiles, warps, pipeline, scheduler, padding)
  - Arch:      WHERE it runs (target GPU architecture)

Usage:
    from conv_utils import (
        ConvSignature, ConvAlgorithm, ArchInfo,
        ConvKernelConfig, ConvKernelSet, ConvProblem
    )

    # Define signature (WHAT)
    sig = ConvSignature()
    sig.dtype("fp16")
    sig.layout = "nhwc"
    sig.direction = "forward"

    # Define algorithm (HOW)
    algo = ConvAlgorithm()
    algo.tile(1, 128, 128)
    algo.wave(2, 2, 1)
    algo.warp(32, 32, 16)
    algo.pipeline = "compv4"

    # Define arch (WHERE)
    arch = ArchInfo(name="gfx942")

    # Combine into config
    config = ConvKernelConfig(signature=sig, algorithm=algo, arch=arch)
"""

import ctypes
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# =============================================================================
# PATH CONFIGURATION
# =============================================================================


def get_dispatcher_root() -> Path:
    """Get the dispatcher root directory"""
    # This file is in dispatcher/python/
    return Path(__file__).parent.parent


def get_ck_root() -> Path:
    """Get the CK root directory"""
    return get_dispatcher_root().parent


def get_build_dir() -> Path:
    """Get the build directory"""
    return get_dispatcher_root() / "build"


def get_generated_kernels_dir() -> Path:
    """Get the generated kernels directory"""
    return get_build_dir() / "generated_kernels"


def get_codegen_dir() -> Path:
    """Get the codegen scripts directory"""
    return get_dispatcher_root() / "codegen"


# =============================================================================
# ARCH FILTER AND VALIDATION
# =============================================================================


def get_arch_filter_data() -> Dict[str, Any]:
    """Load arch filter data from arch_specs_generated if available."""
    codegen_dir = get_dispatcher_root() / "codegen"
    import sys

    sys.path.insert(0, str(codegen_dir))

    try:
        from arch_specs_generated import (
            TRAIT_UNSUPPORTED_COMBINATIONS,
            WARP_SUPPORTED_COMBINATIONS,
            WARP_TILE_SUPPORTED_COMBINATIONS,
            get_supported_archs,
        )

        return {
            "trait_unsupported": TRAIT_UNSUPPORTED_COMBINATIONS,
            "warp_combos": WARP_SUPPORTED_COMBINATIONS,
            "warp_tile_combos": WARP_TILE_SUPPORTED_COMBINATIONS,
            "supported_archs": get_supported_archs(),
        }
    except ImportError:
        # Fallback defaults
        return {
            "trait_unsupported": {
                ("compv3", "cshuffle", "interwave"),
                ("compv3", "default", "interwave"),
                ("compv4", "cshuffle", "interwave"),
                ("compv4", "default", "interwave"),
            },
            "warp_combos": {
                "gfx942": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
            },
            "warp_tile_combos": {
                "gfx942": {"fp16_fp16_fp16": [[16, 16, 16], [32, 32, 16]]},
            },
            "supported_archs": ["gfx90a", "gfx942", "gfx950"],
        }


@dataclass
class ConvValidationResult:
    """Result of conv kernel config validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggested_fixes: Dict[str, Any] = field(default_factory=dict)

    def print_result(self, indent: str = "  "):
        """Print validation result."""
        if self.is_valid:
            print(f"{indent}✓ Conv configuration valid")
        else:
            print(f"{indent}⚠ Conv configuration has issues:")
            for err in self.errors:
                print(f"{indent}  - {err}")

        if self.warnings:
            for warn in self.warnings:
                print(f"{indent}  Warning: {warn}")

        if self.suggested_fixes:
            print(f"{indent}  Suggested fixes:")
            for key, val in self.suggested_fixes.items():
                print(f"{indent}    {key}: {val}")


def validate_conv_config(
    pipeline: str = "compv3",
    scheduler: str = "intrawave",
    epilogue: str = "cshuffle",
    wave_m: int = 2,
    wave_n: int = 2,
    wave_k: int = 1,
    warp_m: int = 32,
    warp_n: int = 32,
    warp_k: int = 16,
    dtype: str = "fp16",
    arch: str = "gfx942",
    direction: str = "forward",
) -> ConvValidationResult:
    """
    Validate a conv kernel configuration against arch filter rules.

    Args:
        pipeline: Pipeline type (compv3, compv4, etc.)
        scheduler: Scheduler type (intrawave, interwave)
        epilogue: Epilogue type (cshuffle, default)
        wave_m, wave_n, wave_k: Wave/warp configuration
        warp_m, warp_n, warp_k: Warp tile dimensions
        dtype: Data type (fp16, bf16, etc.)
        arch: Target architecture (gfx942, gfx90a, etc.)
        direction: Convolution direction (forward, bwd_data, bwd_weight)
                  Affects operator-specific validation constraints.

    Returns ConvValidationResult with is_valid, errors, and suggested fixes.
    """
    arch_data = get_arch_filter_data()

    errors = []
    warnings = []
    suggested_fixes = {}

    # Check trait combination (pipeline, epilogue, scheduler)
    combo = (pipeline, epilogue, scheduler)
    if combo in arch_data["trait_unsupported"]:
        errors.append(
            f"Unsupported trait combination: pipeline={pipeline}, epilogue={epilogue}, scheduler={scheduler}"
        )
        suggested_fixes["scheduler"] = "intrawave"

    # Check wave configuration for this arch
    warp_combos = arch_data["warp_combos"].get(arch, [[2, 2, 1]])
    wave_cfg = [wave_m, wave_n, wave_k]
    if wave_cfg not in warp_combos:
        valid_str = ", ".join(f"[{c[0]},{c[1]},{c[2]}]" for c in warp_combos)
        errors.append(
            f"Unsupported wave configuration [{wave_m},{wave_n},{wave_k}] for {arch}. Valid: {valid_str}"
        )
        if warp_combos:
            suggested_fixes["wave_m"] = warp_combos[0][0]
            suggested_fixes["wave_n"] = warp_combos[0][1]
            suggested_fixes["wave_k"] = warp_combos[0][2]

    # Check warp tile configuration for this arch and dtype
    dtype_key = f"{dtype}_{dtype}_{dtype}"
    warp_tile_combos = (
        arch_data["warp_tile_combos"]
        .get(arch, {})
        .get(dtype_key, [[32, 32, 16], [16, 16, 16]])
    )
    warp_cfg = [warp_m, warp_n, warp_k]
    if warp_cfg not in warp_tile_combos:
        valid_str = ", ".join(f"[{c[0]},{c[1]},{c[2]}]" for c in warp_tile_combos[:5])
        errors.append(
            f"Unsupported warp tile [{warp_m},{warp_n},{warp_k}] for {arch}/{dtype}. Valid: {valid_str}"
        )
        if warp_tile_combos:
            suggested_fixes["warp_m"] = warp_tile_combos[0][0]
            suggested_fixes["warp_n"] = warp_tile_combos[0][1]
            suggested_fixes["warp_k"] = warp_tile_combos[0][2]

    # Check arch is supported
    if arch not in arch_data["supported_archs"]:
        errors.append(
            f"Unsupported architecture: {arch}. Supported: {', '.join(arch_data['supported_archs'])}"
        )

    return ConvValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        suggested_fixes=suggested_fixes,
    )


def find_matching_conv_kernel_header(
    dtype: str = "fp16",
    conv_type: str = "forward",
    ndim: int = 2,
    pipeline: str = "compv3",
    scheduler: str = "intrawave",
    tile_k: int = 128,
    tile_c: int = 128,
    wave_m: int = 2,
    wave_n: int = 2,
    wave_k: int = 1,
) -> Optional[Path]:
    """
    Find a conv kernel header that matches the config.

    Kernel filename format:
    conv_{fwd|bwdd|bwdw}_{dtype}_{layout}_{ndim}d_{pipeline}_{epilogue}_{scheduler}_{tile_m}x{tile_n}x{tile_k}_{wave}_{warp}[_dsb].hpp

    Searches in:
    - build/generated_kernels/ (JIT-generated kernels)
    - build/examples/conv_python_fallback/ (pre-compiled fallback kernels)
    """
    build_dir = get_build_dir()

    # Kernel directories to search
    kernel_dirs = []
    gen_dir = build_dir / "generated_kernels"
    if gen_dir.exists():
        kernel_dirs.append(gen_dir)
    fallback_dir = build_dir / "examples" / "conv_python_fallback"
    if fallback_dir.exists():
        kernel_dirs.append(fallback_dir)

    # Map conv_type to prefix
    if conv_type == "forward":
        type_prefix = "fwd"
    elif conv_type == "bwd_data":
        type_prefix = "bwdd"
    elif conv_type == "bwd_weight":
        type_prefix = "bwdw"
    else:
        type_prefix = conv_type

    wave_str = f"{wave_m}x{wave_n}x{wave_k}"

    # Search patterns in priority order
    patterns = [
        # Strategy 1: Match with pipeline, scheduler, wave config
        f"conv_{type_prefix}_{dtype}_*_{ndim}d_{pipeline}_*_{scheduler}_{tile_k}x{tile_c}*_{wave_str}_*.hpp",
        # Strategy 2: Match with pipeline and scheduler
        f"conv_{type_prefix}_{dtype}_*_{ndim}d_{pipeline}_*_{scheduler}_*.hpp",
        # Strategy 3: Match with just pipeline
        f"conv_{type_prefix}_{dtype}_*_{ndim}d_{pipeline}_*.hpp",
        # Strategy 4: Any kernel with matching type/dtype/ndim
        f"conv_{type_prefix}_{dtype}_*_{ndim}d_*.hpp",
    ]

    for pattern in patterns:
        for kernel_dir in kernel_dirs:
            matches = list(kernel_dir.glob(pattern))
            if matches:
                return matches[0]

    return None


# =============================================================================
# ENUMS (matching conv_config.hpp)
# =============================================================================


class DataType(Enum):
    """
    Data types for convolution - matches CK Tile numeric types.

    Floating Point Types:
      - FP32: 32-bit float (float)
      - FP16: 16-bit float (half_t)
      - BF16: 16-bit bfloat (bf16_t/bfloat16_t)

    8-bit Float Types (FP8):
      - FP8_E4M3: 8-bit E4M3 format (FP8, OCP or FNUZ)
      - FP8_E5M2: 8-bit E5M2 format (BF8, OCP or FNUZ)
      - FP8: Alias for FP8_E4M3

    Integer Types:
      - INT8/I8: 8-bit signed integer
      - UINT8/U8: 8-bit unsigned integer
      - INT32: 32-bit signed integer (for accumulator)

    4-bit Types (gfx950+ only):
      - FP4: 4-bit float (MXFP4)
      - INT4: 4-bit integer
    """

    # Standard floating point
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    # 8-bit float variants (FP8/BF8)
    FP8_E4M3 = "fp8_e4m3"  # E4M3 format (more precision)
    FP8_E5M2 = "fp8_e5m2"  # E5M2 format (more range, BF8)
    FP8 = "fp8"  # Alias for fp8_e4m3
    BF8 = "bf8"  # Alias for fp8_e5m2

    # OCP vs FNUZ variants
    FP8_E4M3_OCP = "fp8_e4m3_ocp"
    FP8_E5M2_OCP = "fp8_e5m2_ocp"
    FP8_E4M3_FNUZ = "fp8_e4m3_fnuz"
    FP8_E5M2_FNUZ = "fp8_e5m2_fnuz"

    # Integer types
    INT8 = "int8"
    I8 = "i8"  # Alias for int8
    UINT8 = "uint8"
    U8 = "u8"  # Alias for uint8
    INT32 = "int32"  # For accumulator

    # 4-bit types (gfx950+ only)
    FP4 = "fp4"  # MXFP4
    INT4 = "int4"


class ConvDirection(Enum):
    """Convolution operation direction"""

    FORWARD = "forward"
    BACKWARD_DATA = "bwd_data"
    BACKWARD_WEIGHT = "bwd_weight"


class ConvLayout(Enum):
    """Memory layout for convolution tensors"""

    NHWC = "nhwc"
    NHWGC = "nhwgc"  # Grouped
    NCHW = "nchw"
    NGCHW = "ngchw"  # Grouped


class PipelineVersion(Enum):
    """Pipeline versions - matches CK Tile GemmPipeline enum"""

    COMPUTE_V3 = "compv3"
    COMPUTE_V4 = "compv4"
    COMPUTE_V5 = "compv5"
    COMPUTE_V6 = "compv6"
    COMPUTE_ASYNC = "compute_async"
    MEMORY = "mem"
    BASIC_V1 = "basic_v1"
    BASIC_V2 = "basic_v2"
    PRESHUFFLE_V2 = "preshuffle_v2"

    # Aliases for convenience
    V3 = "compv3"
    V4 = "compv4"
    V5 = "compv5"
    V6 = "compv6"


class PipelineScheduler(Enum):
    """Pipeline schedulers"""

    DEFAULT = "default"
    INTRAWAVE = "intrawave"
    INTERWAVE = "interwave"


class ElementwiseOp(Enum):
    """Elementwise operations"""

    PASS_THROUGH = "passthrough"
    BIAS = "bias"
    BIAS_CLAMP = "bias_clamp"
    SCALE = "scale"
    BILINEAR = "bilinear"


class ConvSpecialization(Enum):
    """Convolution specializations"""

    DEFAULT = "default"
    FILTER_1X1_PAD0 = "filter_1x1_pad0"
    FILTER_1X1_STRIDE1_PAD0 = "filter_1x1_stride1_pad0"
    FILTER_3X3 = "filter_3x3"


class GemmPadding(Enum):
    """GEMM padding modes"""

    DEFAULT = "default"
    M_PADDING = "m_padding"
    N_PADDING = "n_padding"
    K_PADDING = "k_padding"
    MN_PADDING = "mn_padding"
    MK_PADDING = "mk_padding"
    NK_PADDING = "nk_padding"
    MNK_PADDING = "mnk_padding"


class MemoryOperation(Enum):
    """Memory operation modes - for split-k accumulation"""

    SET = "set"  # Normal write
    ATOMIC_ADD = "atomic_add"  # Atomic add for split-k
    ATOMIC_MAX = "atomic_max"  # Atomic max
    ADD = "add"  # Non-atomic add


class EpilogueType(Enum):
    """Epilogue types"""

    CSHUFFLE = "cshuffle"
    DEFAULT_2D = "default_2d"
    DEFAULT_GEMM_2D = "default_gemm_2d"


# =============================================================================
# SIGNATURE: WHAT operation (types, layouts, direction)
# =============================================================================


@dataclass
class ConvSignature:
    """
    Convolution Signature - describes WHAT operation to perform.

    This groups all the "what" parameters:
      - Data types (input, weight, output, accumulator)
      - Memory layout (nhwc, nchw)
      - Operation direction (forward, backward data, backward weight)
      - Spatial dimensions (1D, 2D, 3D)
      - Grouping
      - Elementwise operations

    Attributes:
        dtype_in:       Input data type (fp16, fp32, bf16, etc.)
        dtype_wei:      Weight data type
        dtype_out:      Output data type
        dtype_acc:      Accumulator data type
        layout:         Memory layout (nhwc, nchw, nhwgc)
        direction:      Convolution direction (forward, bwd_data, bwd_weight)
        num_dims:       Spatial dimensions (1, 2, or 3)
        groups:         Number of groups for grouped convolution
        in_element_op:  Input elementwise operation
        wei_element_op: Weight elementwise operation
        out_element_op: Output elementwise operation
        specialization: Convolution specialization (default, 1x1, 3x3)
    """

    dtype_in: str = "fp16"
    dtype_wei: str = "fp16"
    dtype_out: str = "fp16"
    dtype_acc: str = "fp32"
    dtype_workspace: str = "fp32"  # Workspace type for two-stage algorithms
    dtype_bias: str = "fp16"  # Bias data type (when using bias epilogue)
    layout: str = "nhwc"
    direction: str = "forward"
    num_dims: int = 2
    groups: int = 1
    in_element_op: str = "passthrough"
    wei_element_op: str = "passthrough"
    out_element_op: str = "passthrough"
    specialization: str = "default"

    def dtype(
        self,
        in_type: str,
        wei_type: str = None,
        out_type: str = None,
        acc_type: str = "fp32",
        workspace_type: str = None,
        bias_type: str = None,
    ):
        """Set all data types at once"""
        self.dtype_in = in_type
        self.dtype_wei = wei_type or in_type
        self.dtype_out = out_type or in_type
        self.dtype_acc = acc_type
        self.dtype_workspace = workspace_type or acc_type
        self.dtype_bias = bias_type or out_type or in_type
        return self

    def copy(self):
        """Create a deep copy"""
        return ConvSignature(
            dtype_in=self.dtype_in,
            dtype_wei=self.dtype_wei,
            dtype_out=self.dtype_out,
            dtype_acc=self.dtype_acc,
            dtype_workspace=self.dtype_workspace,
            dtype_bias=self.dtype_bias,
            layout=self.layout,
            direction=self.direction,
            num_dims=self.num_dims,
            groups=self.groups,
            in_element_op=self.in_element_op,
            wei_element_op=self.wei_element_op,
            out_element_op=self.out_element_op,
            specialization=self.specialization,
        )

    def direction_short(self) -> str:
        """Get short direction string"""
        if self.direction == "forward":
            return "fwd"
        elif self.direction == "bwd_data":
            return "bwdd"
        elif self.direction == "bwd_weight":
            return "bwdw"
        return self.direction

    def __repr__(self):
        return (
            f"Signature(dtype={self.dtype_in}, layout={self.layout}, "
            f"dir={self.direction}, dims={self.num_dims}D)"
        )


# =============================================================================
# ALGORITHM: HOW it's computed (tiles, warps, pipeline, scheduler)
# =============================================================================


@dataclass
class ConvAlgorithm:
    """
    Convolution Algorithm - describes HOW the operation is computed.

    This groups all the "how" parameters matching CK Tile conv_configs.hpp:
      - Block tile dimensions
      - Warp distribution (M_Warp, N_Warp, K_Warp)
      - Warp tile sizes (M_Warp_Tile, N_Warp_Tile, K_Warp_Tile)
      - Vector sizes for memory access (VectorSizeA/B/C)
      - Pipeline version and scheduler
      - Epilogue configuration
      - Occupancy and parallelism hints

    For convolution, tile dimensions map to:
      - tile_n: Batch tile (usually 1)
      - tile_k: Output channel tile (K dimension)
      - tile_c: Input channel tile (C dimension, reduction)

    In CK Tile terminology:
      - M_Tile = output spatial (N * Ho * Wo)
      - N_Tile = output channels (K)
      - K_Tile = input channels * filter (C * Y * X)

    Attributes:
        tile_n:         Batch tile dimension (usually 1)
        tile_k:         Output channel tile (K)
        tile_c:         Input channel tile (C * filter)
        tile_ho:        Output tile height
        tile_wo:        Output tile width
        wave_m:         Number of warps along M dimension
        wave_n:         Number of warps along N dimension
        wave_k:         Number of warps along K dimension
        warp_m:         Warp tile M size (M_Warp_Tile)
        warp_n:         Warp tile N size (N_Warp_Tile)
        warp_k:         Warp tile K size (K_Warp_Tile)
        vector_size_a:  Vector size for input tensor A (default: 4)
        vector_size_b:  Vector size for weight tensor B (default: 8)
        vector_size_c:  Vector size for output tensor C (default: 8)
        pipeline:       Pipeline version (compv3, compv4, compv5, compv6, mem, etc.)
        scheduler:      Scheduler type (default, intrawave, interwave)
        epilogue:       Epilogue type (cshuffle, default_2d)
        padding:        GEMM padding mode
        double_buffer:  Use double buffering for LDS (DoubleSmemBuffer)
        block_per_cu:   Blocks per CU hint for occupancy (kBlockPerCu)
        num_wave_groups: Number of wave groups (NumWaveGroups, for V5 pipeline)
        num_groups_to_merge: Groups to merge optimization (NumGroupsToMerge)
        memory_op:      Memory operation for output (set, atomic_add for split-k)
    """

    # Block tile dimensions (backward compatible naming)
    tile_n: int = 1  # Batch tile (usually 1)
    tile_k: int = 128  # Output channel tile (K)
    tile_c: int = 128  # Input channel tile (C * filter)
    tile_ho: int = 1  # Output spatial tile height
    tile_wo: int = 16  # Output spatial tile width

    # Wave/warp distribution (maps to M_Warp, N_Warp, K_Warp in CK)
    wave_m: int = 2
    wave_n: int = 2
    wave_k: int = 1

    # Warp tile sizes (maps to M_Warp_Tile, N_Warp_Tile, K_Warp_Tile in CK)
    warp_m: int = 32
    warp_n: int = 32
    warp_k: int = 16

    # Vector sizes for memory access optimization (NEW)
    vector_size_a: int = 4  # VectorSizeA - input tensor
    vector_size_b: int = 8  # VectorSizeB - weight tensor
    vector_size_c: int = 8  # VectorSizeC - output tensor

    # Pipeline and scheduler
    pipeline: str = "compv4"  # GemmPipeline enum
    scheduler: str = "intrawave"  # GemmPipelineScheduler enum
    epilogue: str = "cshuffle"

    # Padding and buffering
    padding: str = "mnk_padding"
    double_buffer: bool = False  # DoubleSmemBuffer
    block_size: int = 256  # Thread block size

    # Occupancy and parallelism (NEW)
    block_per_cu: int = 1  # kBlockPerCu
    num_wave_groups: int = 1  # NumWaveGroups (for V5 pipeline)
    num_groups_to_merge: int = 1  # NumGroupsToMerge

    # Memory operation (NEW - for split-k)
    memory_op: str = "set"  # set, atomic_add, atomic_max

    # Split-K parallelism (NEW)
    split_k: int = 1  # k_batch - number of split-K batches

    # Large tensor support (NEW)
    enable_split_image: bool = False  # EnableSplitImage for large tensors

    # GEMM traits (NEW - from FixedGemmParams)
    transpose_c: bool = False  # TransposeC
    use_structured_sparsity: bool = False  # UseStructuredSparsity
    persistent: bool = False  # Persistent kernel launch
    fixed_vector_size: bool = True  # FixedVectorSize

    # Tile partitioner params (NEW)
    tile_partitioner_group_num: int = 8  # TilePartitionerGroupNum
    tile_partitioner_m01: int = 4  # TilePartitionerM01

    # Explicit padding flags (NEW)
    pad_m: bool = True  # kPadM
    pad_n: bool = True  # kPadN
    pad_k: bool = True  # kPadK

    # Activation/Clamp parameters (NEW - for bias_clamp epilogue)
    clamp_min: float = -float("inf")  # Floor for clamp activation
    clamp_max: float = float("inf")  # Ceil for clamp activation

    def tile(self, n: int, k: int, c: int):
        """Set block tile dimensions (N, K, C)"""
        self.tile_n = n
        self.tile_k = k
        self.tile_c = c
        return self

    def tile_output(self, ho: int, wo: int):
        """Set output spatial tile dimensions"""
        self.tile_ho = ho
        self.tile_wo = wo
        return self

    def wave(self, m: int, n: int, k: int = 1):
        """Set warp distribution across M, N, K"""
        self.wave_m = m
        self.wave_n = n
        self.wave_k = k
        return self

    def warp(self, m: int, n: int, k: int = 16):
        """Set warp tile sizes"""
        self.warp_m = m
        self.warp_n = n
        self.warp_k = k
        return self

    def vector_sizes(self, a: int = 4, b: int = 8, c: int = 8):
        """Set vector sizes for A, B, C tensors"""
        self.vector_size_a = a
        self.vector_size_b = b
        self.vector_size_c = c
        return self

    def occupancy(self, block_per_cu: int = 1, num_wave_groups: int = 1):
        """Set occupancy hints"""
        self.block_per_cu = block_per_cu
        self.num_wave_groups = num_wave_groups
        return self

    # MNK convention properties (for unified codegen interface)
    # Conv uses tile_n/tile_k/tile_c, but codegen uses tile_m/tile_n/tile_k
    @property
    def tile_m(self) -> int:
        """Tile M dimension (maps to tile_n in conv - batch tile)"""
        return self.tile_n

    @tile_m.setter
    def tile_m(self, value: int):
        self.tile_n = value

    # Note: tile_n and tile_k already exist, but for complete MNK coverage:
    # - tile_n (conv) = tile_k (MNK) = output channels
    # - tile_c (conv) = tile_k (MNK) = reduction dimension

    def copy(self):
        """Create a deep copy"""
        return ConvAlgorithm(
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            tile_c=self.tile_c,
            tile_ho=self.tile_ho,
            tile_wo=self.tile_wo,
            wave_m=self.wave_m,
            wave_n=self.wave_n,
            wave_k=self.wave_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_k=self.warp_k,
            vector_size_a=self.vector_size_a,
            vector_size_b=self.vector_size_b,
            vector_size_c=self.vector_size_c,
            pipeline=self.pipeline,
            scheduler=self.scheduler,
            epilogue=self.epilogue,
            padding=self.padding,
            double_buffer=self.double_buffer,
            block_size=self.block_size,
            block_per_cu=self.block_per_cu,
            num_wave_groups=self.num_wave_groups,
            num_groups_to_merge=self.num_groups_to_merge,
            memory_op=self.memory_op,
            split_k=self.split_k,
            enable_split_image=self.enable_split_image,
            transpose_c=self.transpose_c,
            use_structured_sparsity=self.use_structured_sparsity,
            persistent=self.persistent,
            fixed_vector_size=self.fixed_vector_size,
            tile_partitioner_group_num=self.tile_partitioner_group_num,
            tile_partitioner_m01=self.tile_partitioner_m01,
            pad_m=self.pad_m,
            pad_n=self.pad_n,
            pad_k=self.pad_k,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
        )

    def __repr__(self):
        return (
            f"Algorithm(tile={self.tile_k}x{self.tile_c}, "
            f"wave={self.wave_m}x{self.wave_n}, pipeline={self.pipeline})"
        )


# =============================================================================
# ARCH: WHERE it runs (target GPU)
# =============================================================================


@dataclass
class ArchInfo:
    """
    Architecture Info - describes WHERE the kernel runs.

    Attributes:
        name:             GPU architecture name (gfx942, gfx1100, etc.)
        max_waves_per_cu: Maximum waves per compute unit
        lds_size_kb:      LDS size in KB
        sgpr_count:       Number of SGPRs
        vgpr_count:       Number of VGPRs
    """

    name: str = "gfx942"
    max_waves_per_cu: int = 8
    lds_size_kb: int = 64
    sgpr_count: int = 108
    vgpr_count: int = 512

    def supports_mfma_fp16(self) -> bool:
        """Check if architecture supports FP16 MFMA"""
        return "gfx9" in self.name

    def supports_wmma(self) -> bool:
        """Check if architecture supports WMMA"""
        return "gfx11" in self.name

    def is_mi300(self) -> bool:
        """Check if MI300 series"""
        return self.name in ("gfx940", "gfx941", "gfx942")

    def is_mi200(self) -> bool:
        """Check if MI200 series"""
        return self.name in ("gfx90a",)

    def __repr__(self):
        return f"Arch({self.name})"


# =============================================================================
# COMPLETE KERNEL CONFIG (Signature + Algorithm + Arch)
# =============================================================================


@dataclass
class ConvKernelConfig:
    """
    Complete convolution kernel configuration.
    Combines Signature + Algorithm + Arch into a single config.
    """

    signature: ConvSignature = field(default_factory=ConvSignature)
    algorithm: ConvAlgorithm = field(default_factory=ConvAlgorithm)
    arch: ArchInfo = field(default_factory=ArchInfo)

    def name(self) -> str:
        """Generate unique kernel name"""
        sig = self.signature
        algo = self.algorithm
        return (
            f"conv_{sig.direction_short()}_{sig.dtype_in}_"
            f"{sig.num_dims}d_{algo.pipeline}_{algo.tile_k}x{algo.tile_c}"
        )

    def brief(self) -> str:
        """One-line summary"""
        sig = self.signature
        return f"{sig.num_dims}D {sig.direction} convolution ({sig.dtype_in})"

    def detailed(self) -> str:
        """Detailed hierarchical description"""
        sig = self.signature
        algo = self.algorithm
        arch = self.arch

        lines = [
            f"{sig.num_dims}D {sig.direction} Convolution Kernel",
            "",
            "  Signature (WHAT):",
            f"    Data Type:     {sig.dtype_in} -> {sig.dtype_out} (acc: {sig.dtype_acc})",
            f"    Layout:        {sig.layout}",
            f"    Direction:     {sig.direction}",
            f"    Spatial Dims:  {sig.num_dims}D",
            f"    Groups:        {sig.groups}",
            f"    Specialization: {sig.specialization}",
            "",
            "  Algorithm (HOW):",
            f"    Block Tile:    N={algo.tile_n}, K={algo.tile_k}, C={algo.tile_c}",
            f"    Output Tile:   Ho={algo.tile_ho}, Wo={algo.tile_wo}",
            f"    Wave Config:   {algo.wave_m}x{algo.wave_n}x{algo.wave_k}",
            f"    Warp Tile:     {algo.warp_m}x{algo.warp_n}x{algo.warp_k}",
            f"    Pipeline:      {algo.pipeline}",
            f"    Scheduler:     {algo.scheduler}",
            f"    Epilogue:      {algo.epilogue}",
            f"    Padding:       {algo.padding}",
            f"    Block Size:    {algo.block_size}",
            "",
            "  Arch (WHERE):",
            f"    Target:        {arch.name}",
            f"    MFMA FP16:     {arch.supports_mfma_fp16()}",
            f"    WMMA:          {arch.supports_wmma()}",
        ]
        return "\n".join(lines)

    def copy(self):
        """Create a deep copy"""
        return ConvKernelConfig(
            signature=self.signature.copy(),
            algorithm=self.algorithm.copy(),
            arch=ArchInfo(
                name=self.arch.name,
                max_waves_per_cu=self.arch.max_waves_per_cu,
                lds_size_kb=self.arch.lds_size_kb,
            ),
        )


# =============================================================================
# KERNEL SET (Collection of configs)
# =============================================================================


class ConvKernelSet:
    """
    Collection of convolution kernel configurations.

    Provides both simple and full APIs for adding kernels.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.configs: List[ConvKernelConfig] = []

    def add_simple(
        self,
        dtype: str,
        layout: str,
        direction: str,
        tile_k: int,
        tile_c: int,
        arch: str = "gfx942",
    ):
        """
        Simple add with basic parameters.

        Args:
            dtype:     Data type (fp16, fp32, bf16)
            layout:    Memory layout (nhwc, nchw)
            direction: Operation direction (forward, bwd_data, bwd_weight)
            tile_k:    K tile size
            tile_c:    C tile size
            arch:      Target architecture
        """
        sig = ConvSignature()
        sig.dtype(dtype)
        sig.layout = layout
        sig.direction = direction

        algo = ConvAlgorithm()
        algo.tile_k = tile_k
        algo.tile_c = tile_c

        self.configs.append(
            ConvKernelConfig(signature=sig, algorithm=algo, arch=ArchInfo(name=arch))
        )
        return self

    def add(
        self, signature: ConvSignature, algorithm: ConvAlgorithm, arch: ArchInfo = None
    ):
        """
        Add with full Signature + Algorithm + Arch.

        Args:
            signature: ConvSignature instance
            algorithm: ConvAlgorithm instance
            arch:      ArchInfo instance (defaults to gfx942)
        """
        self.configs.append(
            ConvKernelConfig(
                signature=signature.copy(),
                algorithm=algorithm.copy(),
                arch=arch or ArchInfo(),
            )
        )
        return self

    def merge(self, other: "ConvKernelSet"):
        """Merge another kernel set into this one"""
        self.configs.extend(other.configs)
        return self

    def __len__(self):
        return len(self.configs)

    def __iter__(self):
        return iter(self.configs)

    def print(self, detailed: bool = False):
        """Print all configurations"""
        print(f"ConvKernelSet '{self.name}' ({len(self.configs)} configs):")
        for cfg in self.configs:
            if detailed:
                print(cfg.detailed())
                print()
            else:
                print(f"  - {cfg.name()}")


# =============================================================================
# CONV PROBLEM (Runtime problem specification)
# =============================================================================


@dataclass
class ConvProblem:
    """
    Convolution problem specification for runtime.

    Describes the actual sizes of a convolution to be computed.
    """

    # Batch and channels
    N: int = 1  # Batch size
    C: int = 64  # Input channels
    K: int = 128  # Output channels
    G: int = 1  # Groups

    # Spatial dimensions (2D default)
    Hi: int = 28  # Input height
    Wi: int = 28  # Input width
    Di: int = 1  # Input depth (for 3D)

    # Filter dimensions
    Y: int = 3  # Filter height
    X: int = 3  # Filter width
    Z: int = 1  # Filter depth (for 3D)

    # Stride
    stride_h: int = 1
    stride_w: int = 1
    stride_d: int = 1

    # Padding
    pad_h: int = 0
    pad_w: int = 0
    pad_d: int = 0

    # Dilation
    dilation_h: int = 1
    dilation_w: int = 1
    dilation_d: int = 1

    # Operation
    direction: str = "forward"

    @property
    def Ho(self) -> int:
        """Output height"""
        eff_y = (self.Y - 1) * self.dilation_h + 1
        return (self.Hi + 2 * self.pad_h - eff_y) // self.stride_h + 1

    @property
    def Wo(self) -> int:
        """Output width"""
        eff_x = (self.X - 1) * self.dilation_w + 1
        return (self.Wi + 2 * self.pad_w - eff_x) // self.stride_w + 1

    @property
    def Do(self) -> int:
        """Output depth (for 3D)"""
        eff_z = (self.Z - 1) * self.dilation_d + 1
        return (self.Di + 2 * self.pad_d - eff_z) // self.stride_d + 1

    @property
    def flops(self) -> float:
        """Total FLOPs for forward convolution"""
        c_per_group = self.C // self.G
        return 2.0 * self.N * self.K * self.Ho * self.Wo * c_per_group * self.Y * self.X

    @property
    def flops_3d(self) -> float:
        """Total FLOPs for 3D forward convolution"""
        c_per_group = self.C // self.G
        return (
            2.0
            * self.N
            * self.K
            * self.Do
            * self.Ho
            * self.Wo
            * c_per_group
            * self.Z
            * self.Y
            * self.X
        )

    def is_pointwise(self) -> bool:
        """Check if 1x1 convolution"""
        return self.Y == 1 and self.X == 1 and self.Z == 1

    def is_depthwise(self) -> bool:
        """Check if depthwise convolution"""
        return self.G == self.C == self.K

    def compute_flops(self) -> float:
        """
        Compute FLOPs for this convolution problem.

        Automatically handles 2D vs 3D based on problem dimensions.
        Note: FLOPs are the same for forward, backward data, and backward weight
        operations since they all involve the same number of multiply-accumulate ops.
        """
        return self.flops_3d if self.is_3d() else self.flops

    def is_3d(self) -> bool:
        """Check if 3D convolution"""
        return self.Di > 1 or self.Z > 1

    def input_size(self) -> Tuple[int, ...]:
        """Get input tensor size (N, C, D, H, W) or (N, C, H, W)"""
        if self.is_3d():
            return (self.N, self.C, self.Di, self.Hi, self.Wi)
        return (self.N, self.C, self.Hi, self.Wi)

    def output_size(self) -> Tuple[int, ...]:
        """Get output tensor size"""
        if self.is_3d():
            return (self.N, self.K, self.Do, self.Ho, self.Wo)
        return (self.N, self.K, self.Ho, self.Wo)

    def filter_size(self) -> Tuple[int, ...]:
        """Get filter tensor size"""
        c_per_group = self.C // self.G
        if self.is_3d():
            return (self.K, c_per_group, self.Z, self.Y, self.X)
        return (self.K, c_per_group, self.Y, self.X)

    def __repr__(self):
        if self.is_3d():
            return (
                f"ConvProblem(N={self.N}, C={self.C}, K={self.K}, "
                f"Di={self.Di}, Hi={self.Hi}, Wi={self.Wi}, "
                f"Z={self.Z}, Y={self.Y}, X={self.X})"
            )
        return (
            f"ConvProblem(N={self.N}, C={self.C}, K={self.K}, "
            f"Hi={self.Hi}, Wi={self.Wi}, Y={self.Y}, X={self.X})"
        )


# =============================================================================
# CODEGEN RUNNER
# =============================================================================


class ConvCodegenRunner:
    """
    Runner for convolution kernel code generation.

    Generates kernels using unified_conv_codegen.py.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.codegen_script = get_codegen_dir() / "unified_conv_codegen.py"
        self.output_dir = get_generated_kernels_dir()

    def generate(self, config: ConvKernelConfig) -> Optional[Path]:
        """Generate a single kernel from config"""
        sig = config.signature
        algo = config.algorithm
        arch = config.arch

        cmd = [
            "python3",
            str(self.codegen_script),
            "--dtype",
            sig.dtype_in,
            "--layout",
            sig.layout,
            "--conv-type",
            sig.direction,
            "--spatial-dims",
            str(sig.num_dims),
            "--tile-k",
            str(algo.tile_k),
            "--tile-c",
            str(algo.tile_c),
            "--wave-m",
            str(algo.wave_m),
            "--wave-n",
            str(algo.wave_n),
            "--pipeline",
            algo.pipeline,
            "--scheduler",
            algo.scheduler,
            "--arch",
            arch.name,
            "--output-dir",
            str(self.output_dir),
        ]

        if self.verbose:
            print(f"  Generating: {config.name()}")

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Find generated file
            pattern = f"conv_{sig.direction_short()}_{sig.dtype_in}_*.hpp"
            files = list(self.output_dir.glob(pattern))
            return files[0] if files else None

        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"  Error: {e.stderr}")
            return None

    def generate_set(
        self, kernel_set: ConvKernelSet, parallel: bool = True
    ) -> List[Path]:
        """Generate all kernels in a set"""
        generated = []

        if parallel and len(kernel_set) > 1:
            max_workers = min(len(kernel_set), multiprocessing.cpu_count())
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.generate, cfg): cfg for cfg in kernel_set
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        generated.append(result)
        else:
            for cfg in kernel_set:
                result = self.generate(cfg)
                if result:
                    generated.append(result)

        return generated


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


class ConvValidator:
    """Validation utilities for convolution results"""

    def __init__(self, rtol: float = 1e-3, atol: float = 1e-3):
        self.rtol = rtol
        self.atol = atol

    def check(self, result: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
        """Compare result against reference"""
        if result.shape != reference.shape:
            return {
                "passed": False,
                "error": f"Shape mismatch: {result.shape} vs {reference.shape}",
            }

        abs_diff = np.abs(result - reference)
        max_abs_diff = np.max(abs_diff)

        ref_norm = np.linalg.norm(reference.flatten())
        rel_diff = max_abs_diff / (ref_norm + 1e-10)

        passed = np.allclose(result, reference, rtol=self.rtol, atol=self.atol)

        return {
            "passed": passed,
            "max_abs_diff": float(max_abs_diff),
            "rel_diff": float(rel_diff),
            "rtol": self.rtol,
            "atol": self.atol,
        }

    def reference_conv2d_forward(
        self,
        input: np.ndarray,
        weight: np.ndarray,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """CPU reference for 2D forward convolution (NHWC layout)"""
        N, Hi, Wi, C = input.shape
        K, Y, X, _ = weight.shape

        pad_h, pad_w = padding
        stride_h, stride_w = stride

        # Pad input
        if pad_h > 0 or pad_w > 0:
            input = np.pad(input, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))

        Ho = (Hi + 2 * pad_h - Y) // stride_h + 1
        Wo = (Wi + 2 * pad_w - X) // stride_w + 1

        output = np.zeros((N, Ho, Wo, K), dtype=input.dtype)

        for n in range(N):
            for ho in range(Ho):
                for wo in range(Wo):
                    for k in range(K):
                        for y in range(Y):
                            for x in range(X):
                                for c in range(C):
                                    hi = ho * stride_h + y
                                    wi = wo * stride_w + x
                                    output[n, ho, wo, k] += (
                                        input[n, hi, wi, c] * weight[k, y, x, c]
                                    )

        return output


# =============================================================================
# C STRUCTURE FOR CTYPES
# =============================================================================


class ConvProblemC(ctypes.Structure):
    """C structure matching ConvProblemC in conv_ctypes_lib.cpp"""

    _fields_ = [
        ("N", ctypes.c_int),
        ("G", ctypes.c_int),
        ("C", ctypes.c_int),
        ("K", ctypes.c_int),
        ("input_d", ctypes.c_int),
        ("input_h", ctypes.c_int),
        ("input_w", ctypes.c_int),
        ("filter_z", ctypes.c_int),
        ("filter_y", ctypes.c_int),
        ("filter_x", ctypes.c_int),
        ("stride_d", ctypes.c_int),
        ("stride_h", ctypes.c_int),
        ("stride_w", ctypes.c_int),
        ("pad_d", ctypes.c_int),
        ("pad_h", ctypes.c_int),
        ("pad_w", ctypes.c_int),
        ("dilation_d", ctypes.c_int),
        ("dilation_h", ctypes.c_int),
        ("dilation_w", ctypes.c_int),
        ("direction", ctypes.c_int),  # 0=forward, 1=bwd_data, 2=bwd_weight
    ]

    @classmethod
    def from_problem(cls, p: "ConvProblem") -> "ConvProblemC":
        """Create C struct from Python ConvProblem"""
        c = cls()
        c.N = p.N
        c.G = p.G
        c.C = p.C
        c.K = p.K
        c.input_d = p.Di
        c.input_h = p.Hi
        c.input_w = p.Wi
        c.filter_z = p.Z
        c.filter_y = p.Y
        c.filter_x = p.X
        c.stride_d = p.stride_d
        c.stride_h = p.stride_h
        c.stride_w = p.stride_w
        c.pad_d = p.pad_d
        c.pad_h = p.pad_h
        c.pad_w = p.pad_w
        c.dilation_d = p.dilation_d
        c.dilation_h = p.dilation_h
        c.dilation_w = p.dilation_w
        direction_map = {"forward": 0, "bwd_data": 1, "bwd_weight": 2}
        c.direction = direction_map.get(p.direction, 0)
        return c


# =============================================================================
# LIBRARY LOADING (for compiled kernels)
# =============================================================================


class ConvDispatcherLib:
    """
    Wrapper for the convolution dispatcher dynamic library.

    Provides Python interface to the C API in conv_ctypes_lib.cpp.

    Usage:
        lib = ConvDispatcherLib.find()
        lib.initialize()

        # Run convolution
        result = lib.run_conv(input, weight, output, problem)
    """

    SEARCH_PATHS = [
        "build/bindings/libdispatcher_conv_lib.so",
        "build/examples/libdispatcher_conv_lib.so",
        "build/lib/libdispatcher_conv.so",
        "bindings/ctypes/libdispatcher_conv_lib.so",
    ]

    def __init__(self, lib: ctypes.CDLL, path: Path):
        self._lib = lib
        self._path = path
        self._setup_functions()

    def _setup_functions(self):
        """Setup ctypes function signatures"""
        # Initialize
        self._lib.conv_dispatcher_init.argtypes = []
        self._lib.conv_dispatcher_init.restype = ctypes.c_int

        # Cleanup
        self._lib.conv_dispatcher_cleanup.argtypes = []
        self._lib.conv_dispatcher_cleanup.restype = ctypes.c_int

        # Get kernel count
        self._lib.conv_dispatcher_get_kernel_count.argtypes = []
        self._lib.conv_dispatcher_get_kernel_count.restype = ctypes.c_int

        # Version
        self._lib.conv_dispatcher_version.argtypes = []
        self._lib.conv_dispatcher_version.restype = ctypes.c_char_p

        # Has kernels
        self._lib.conv_dispatcher_has_kernels.argtypes = []
        self._lib.conv_dispatcher_has_kernels.restype = ctypes.c_int

        # Run convolution (actual GPU execution)
        self._lib.conv_dispatcher_run.argtypes = [
            ctypes.c_void_p,  # input_ptr
            ctypes.c_void_p,  # weight_ptr
            ctypes.c_void_p,  # output_ptr
            ctypes.POINTER(ConvProblemC),  # problem
            ctypes.c_void_p,  # stream
        ]
        self._lib.conv_dispatcher_run.restype = ctypes.c_float

        # Get kernel name by index
        self._lib.conv_dispatcher_get_kernel_name.argtypes = [
            ctypes.c_int,  # index
            ctypes.c_char_p,  # buffer
            ctypes.c_int,  # buffer_size
        ]
        self._lib.conv_dispatcher_get_kernel_name.restype = ctypes.c_int

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> bool:
        """Initialize the dispatcher"""
        return self._lib.conv_dispatcher_init() == 0

    def cleanup(self):
        """Cleanup dispatcher resources"""
        self._lib.conv_dispatcher_cleanup()

    def get_kernel_count(self) -> int:
        """Get number of registered kernels"""
        return self._lib.conv_dispatcher_get_kernel_count()

    def get_version(self) -> str:
        """Get library version"""
        version = self._lib.conv_dispatcher_version()
        return version.decode("utf-8") if version else "unknown"

    def has_kernels(self) -> bool:
        """Check if library was compiled with kernels"""
        return self._lib.conv_dispatcher_has_kernels() == 1

    def get_kernel_name(self, index: int = 0) -> Optional[str]:
        """Get kernel name by index (default: first kernel)"""
        buffer = ctypes.create_string_buffer(512)
        result = self._lib.conv_dispatcher_get_kernel_name(index, buffer, 512)
        if result == 0:
            return buffer.value.decode("utf-8")
        return None

    def get_all_kernel_names(self) -> List[str]:
        """Get names of all registered kernels"""
        names = []
        count = self.get_kernel_count()
        for i in range(count):
            name = self.get_kernel_name(i)
            if name:
                names.append(name)
        return names

    def run(
        self,
        input_ptr: int,
        weight_ptr: int,
        output_ptr: int,
        problem: "ConvProblem",
        stream: int = 0,
    ) -> float:
        """
        Run convolution on GPU.

        Args:
            input_ptr:  Device pointer to input data
            weight_ptr: Device pointer to weight data
            output_ptr: Device pointer to output data
            problem:    ConvProblem describing the convolution
            stream:     HIP stream (0 for default)

        Returns:
            Elapsed time in milliseconds, or -1.0 on error
        """
        prob_c = ConvProblemC.from_problem(problem)
        return self._lib.conv_dispatcher_run(
            ctypes.c_void_p(input_ptr),
            ctypes.c_void_p(weight_ptr),
            ctypes.c_void_p(output_ptr),
            ctypes.byref(prob_c),
            ctypes.c_void_p(stream),
        )

    @classmethod
    def load(cls, path: str) -> "ConvDispatcherLib":
        """Load library from explicit path"""
        lib = ctypes.CDLL(path)
        return cls(lib, Path(path))

    @classmethod
    def find(cls) -> Optional["ConvDispatcherLib"]:
        """Find and load the library from common locations"""
        dispatcher_root = get_dispatcher_root()

        for rel_path in cls.SEARCH_PATHS:
            full_path = dispatcher_root / rel_path
            if full_path.exists():
                try:
                    return cls.load(str(full_path))
                except OSError:
                    continue

        return None

    @classmethod
    def auto(cls, recompile: bool = False) -> Optional["ConvDispatcherLib"]:
        """Auto-find the library and initialize it"""
        lib = cls.find()
        if lib is not None:
            lib.initialize()
            return lib
        return None


# =============================================================================
# REGISTRY AND DISPATCHER (Explicit API)
# =============================================================================


class ConvRegistry:
    """
    Convolution kernel registry - stores and manages kernel instances.

    This provides an explicit registry API that mirrors the C++ ConvRegistry class.

    Usage:
        registry = ConvRegistry()
        registry.register_kernel(kernel_config)
        dispatcher = ConvDispatcher(registry)
    """

    def __init__(self, lib: Optional[ConvDispatcherLib] = None, name: str = "default"):
        self._lib = lib
        self._name = name
        self._kernels: List[ConvKernelConfig] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def kernel_count(self) -> int:
        if self._lib:
            return self._lib.get_kernel_count()
        return len(self._kernels)

    def register_kernel(self, config: ConvKernelConfig) -> bool:
        """Register a kernel configuration."""
        self._kernels.append(config)
        return True

    def get_kernels(self) -> List[ConvKernelConfig]:
        """Get all registered kernel configs."""
        return self._kernels.copy()

    def clear(self):
        """Clear all kernels."""
        self._kernels.clear()

    def bind_library(self, lib: ConvDispatcherLib):
        """Bind to a loaded dispatcher library."""
        self._lib = lib

    def __repr__(self) -> str:
        return f"ConvRegistry(name='{self._name}', kernels={self.kernel_count})"


class ConvDispatcher:
    """
    Convolution kernel dispatcher - selects and runs kernels for problems.

    This provides an explicit dispatcher API that mirrors the C++ ConvDispatcher class.

    Usage:
        registry = ConvRegistry()
        registry.register_kernel(config)

        dispatcher = ConvDispatcher(registry)
        result = dispatcher.run(input, weight, problem)
    """

    def __init__(self, registry: ConvRegistry, lib: Optional[ConvDispatcherLib] = None):
        self._registry = registry
        self._lib = lib or registry._lib

    @property
    def registry(self) -> ConvRegistry:
        return self._registry

    def select_kernel(self, problem: ConvProblem) -> Optional[str]:
        """Select best kernel for problem."""
        # Fallback: return first matching kernel
        for config in self._registry.get_kernels():
            return config.name()
        return None

    def is_supported(self, problem: ConvProblem) -> bool:
        """Check if problem size is supported."""
        return len(self._registry.get_kernels()) > 0

    def __repr__(self) -> str:
        return f"ConvDispatcher(registry={self._registry.name}, kernels={self._registry.kernel_count})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


# =============================================================================
# GPU EXECUTION HELPER
# =============================================================================


class GpuConvRunner:
    """
    Simple helper for running convolution on GPU.

    Handles library loading, HIP memory management, and kernel execution.

    Benchmark Parameters (matching CK Tile stream_config):
        warmup (int): Number of warmup iterations (default: 5)
        repeat (int): Number of benchmark iterations (default: 20)
        flush_cache (bool): Flush GPU L2 cache between iterations (default: False)
        rotating_count (int): Rotating buffer count for cache simulation (default: 1)
        timer (str): Timer type - "gpu" or "cpu" (default: "gpu")

    Usage:
        # Basic usage
        runner = GpuConvRunner()
        if runner.is_available():
            result = runner.run(input_np, weight_np, problem)
            print(f"Time: {result['time_ms']:.4f} ms")

        # With custom benchmark settings
        runner = GpuConvRunner(
            warmup=10,
            repeat=100,
            flush_cache=True,
            timer="gpu"
        )
        result = runner.run(input_np, weight_np, problem)
    """

    def __init__(
        self,
        lib_path: Optional[str] = None,
        warmup: int = 5,
        repeat: int = 20,
        flush_cache: bool = False,
        rotating_count: int = 1,
        timer: str = "gpu",
    ):
        """
        Initialize GPU Conv runner.

        Args:
            lib_path: Optional path to the dispatcher library
            warmup: Number of warmup iterations (default: 5)
            repeat: Number of benchmark iterations (default: 20)
            flush_cache: Flush GPU cache between iterations (default: False)
            rotating_count: Rotating buffer count (default: 1)
            timer: Timer type - "gpu" or "cpu" (default: "gpu")
        """
        self._lib = None
        self._hip = None
        self._initialized = False
        self._lib_path = lib_path

        # Benchmark settings (matching CK Tile stream_config)
        self.warmup = warmup
        self.repeat = repeat
        self.flush_cache = flush_cache
        self.rotating_count = rotating_count
        self.timer = timer
        self.is_gpu_timer = timer == "gpu"

        self._init()

    def _init(self):
        """Initialize library and HIP"""
        try:
            if self._lib_path:
                self._lib = ConvDispatcherLib(Path(self._lib_path))
            else:
                self._lib = ConvDispatcherLib.find()
            if self._lib is None:
                return

            self._hip = ctypes.CDLL("libamdhip64.so")
            self._hip.hipMalloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
            ]
            self._hip.hipMalloc.restype = ctypes.c_int
            self._hip.hipFree.argtypes = [ctypes.c_void_p]
            self._hip.hipFree.restype = ctypes.c_int
            self._hip.hipMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            self._hip.hipMemcpy.restype = ctypes.c_int
            self._hip.hipDeviceSynchronize.argtypes = []
            self._hip.hipDeviceSynchronize.restype = ctypes.c_int

            self._lib.initialize()
            self._initialized = True
        except Exception:
            self._initialized = False

    def is_available(self) -> bool:
        """Check if GPU execution is available"""
        return self._initialized and self._lib is not None

    @property
    def library_path(self) -> Optional[str]:
        """Get library path"""
        return str(self._lib.path) if self._lib else None

    def run(
        self,
        input_np: np.ndarray,
        weight_np: np.ndarray,
        problem: ConvProblem,
        output_np: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run convolution on GPU.

        Args:
            input_np: Input tensor (NHWGC layout)
            weight_np: Weight tensor (GKYXC layout)
            problem: ConvProblem specification
            output_np: Optional output buffer (for copy-back)

        Returns:
            Dict with 'time_ms', 'tflops', 'success', and optionally 'output'
        """
        if not self.is_available():
            return {"success": False, "error": "GPU not available"}

        try:
            # Calculate sizes
            input_size = input_np.nbytes
            weight_size = weight_np.nbytes

            # Output size depends on direction
            # Forward: output is (N, Ho, Wo, G, K)
            # Bwd_data: output is grad_input (N, Hi, Wi, G, C)
            # Bwd_weight: output is grad_weight (G, K, Y, X, C)
            direction = getattr(problem, "direction", "forward")

            if direction == "bwd_data":
                # Output is grad_input: (N, Hi, Wi, G, C)
                if hasattr(problem, "Di") and problem.Di > 0:
                    output_elements = (
                        problem.N
                        * problem.Di
                        * problem.Hi
                        * problem.Wi
                        * problem.G
                        * problem.C
                    )
                else:
                    output_elements = (
                        problem.N * problem.Hi * problem.Wi * problem.G * problem.C
                    )
            elif direction == "bwd_weight":
                # Output is grad_weight: (G, K, Y, X, C)
                if hasattr(problem, "Z") and problem.Z > 0:
                    output_elements = (
                        problem.G
                        * problem.K
                        * problem.Z
                        * problem.Y
                        * problem.X
                        * problem.C
                    )
                else:
                    output_elements = (
                        problem.G * problem.K * problem.Y * problem.X * problem.C
                    )
            else:
                # Forward: output is (N, Ho, Wo, G, K)
                if hasattr(problem, "Do") and problem.Do > 0:
                    output_elements = (
                        problem.N
                        * problem.Do
                        * problem.Ho
                        * problem.Wo
                        * problem.G
                        * problem.K
                    )
                else:
                    output_elements = (
                        problem.N * problem.Ho * problem.Wo * problem.G * problem.K
                    )

            output_size = output_elements * input_np.dtype.itemsize

            # Create output buffer if not provided
            if output_np is None:
                direction = getattr(problem, "direction", "forward")
                if direction == "bwd_data":
                    # grad_input: (N, Hi, Wi, G, C)
                    output_np = np.zeros(
                        (problem.N, problem.Hi, problem.Wi, problem.G, problem.C),
                        dtype=input_np.dtype,
                    )
                elif direction == "bwd_weight":
                    # grad_weight: (G, K, Y, X, C)
                    output_np = np.zeros(
                        (problem.G, problem.K, problem.Y, problem.X, problem.C),
                        dtype=input_np.dtype,
                    )
                else:
                    # Forward output: (N, Ho, Wo, G, K)
                    output_np = np.zeros(
                        (problem.N, problem.Ho, problem.Wo, problem.G, problem.K),
                        dtype=input_np.dtype,
                    )

            # Allocate GPU memory
            input_dev = ctypes.c_void_p()
            weight_dev = ctypes.c_void_p()
            output_dev = ctypes.c_void_p()

            self._hip.hipMalloc(ctypes.byref(input_dev), input_size)
            self._hip.hipMalloc(ctypes.byref(weight_dev), weight_size)
            self._hip.hipMalloc(ctypes.byref(output_dev), output_size)

            # Copy to device
            self._hip.hipMemcpy(input_dev, input_np.ctypes.data, input_size, 1)  # H2D
            self._hip.hipMemcpy(weight_dev, weight_np.ctypes.data, weight_size, 1)

            # Run kernel
            time_ms = self._lib.run(
                input_dev.value, weight_dev.value, output_dev.value, problem
            )
            self._hip.hipDeviceSynchronize()

            # Copy back results
            result = {
                "success": time_ms > 0,
                "time_ms": time_ms if time_ms > 0 else 0,
                "tflops": problem.compute_flops() / (time_ms * 1e9)
                if time_ms > 0
                else 0,
            }

            if time_ms > 0:
                self._hip.hipMemcpy(
                    output_np.ctypes.data, output_dev, output_np.nbytes, 2
                )  # D2H
                result["output"] = output_np

            # Free GPU memory
            self._hip.hipFree(input_dev)
            self._hip.hipFree(weight_dev)
            self._hip.hipFree(output_dev)

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def cleanup(self):
        """Cleanup resources"""
        if self._lib:
            try:
                self._lib.cleanup()
            except Exception:
                pass


# =============================================================================
# BACKWARD WEIGHT LIBRARY (separate to avoid template conflicts)
# =============================================================================


class ConvBwdwProblemC(ctypes.Structure):
    """C structure for backward weight problem"""

    _fields_ = [
        ("N", ctypes.c_int),
        ("G", ctypes.c_int),
        ("C", ctypes.c_int),
        ("K", ctypes.c_int),
        ("input_d", ctypes.c_int),
        ("input_h", ctypes.c_int),
        ("input_w", ctypes.c_int),
        ("filter_z", ctypes.c_int),
        ("filter_y", ctypes.c_int),
        ("filter_x", ctypes.c_int),
        ("stride_d", ctypes.c_int),
        ("stride_h", ctypes.c_int),
        ("stride_w", ctypes.c_int),
        ("pad_d", ctypes.c_int),
        ("pad_h", ctypes.c_int),
        ("pad_w", ctypes.c_int),
        ("dilation_d", ctypes.c_int),
        ("dilation_h", ctypes.c_int),
        ("dilation_w", ctypes.c_int),
    ]

    @classmethod
    def from_problem(cls, p: "ConvProblem") -> "ConvBwdwProblemC":
        """Create C struct from Python ConvProblem"""
        c = cls()
        c.N = p.N
        c.G = p.G
        c.C = p.C
        c.K = p.K
        c.input_d = p.Di
        c.input_h = p.Hi
        c.input_w = p.Wi
        c.filter_z = p.Z
        c.filter_y = p.Y
        c.filter_x = p.X
        c.stride_d = p.stride_d
        c.stride_h = p.stride_h
        c.stride_w = p.stride_w
        c.pad_d = p.pad_d
        c.pad_h = p.pad_h
        c.pad_w = p.pad_w
        c.dilation_d = p.dilation_d
        c.dilation_h = p.dilation_h
        c.dilation_w = p.dilation_w
        return c


class ConvBwdWeightLib:
    """
    Wrapper for the backward weight convolution library.

    This is a SEPARATE library from the main conv library to avoid
    CK Tile template conflicts.

    Usage:
        lib = ConvBwdWeightLib.find()
        lib.initialize()
        time_ms = lib.run(input_ptr, grad_output_ptr, grad_weight_ptr, problem)
    """

    SEARCH_PATHS = [
        "build/examples/libdispatcher_conv_bwdw_lib.so",
        "build/bindings/libdispatcher_conv_bwdw_lib.so",
        "examples/build/libdispatcher_conv_bwdw_lib.so",
    ]

    def __init__(self, lib: ctypes.CDLL, path: Path):
        self._lib = lib
        self._path = path
        self._setup_functions()

    def _setup_functions(self):
        """Setup ctypes function signatures"""
        self._lib.conv_bwdw_init.argtypes = []
        self._lib.conv_bwdw_init.restype = ctypes.c_int

        self._lib.conv_bwdw_cleanup.argtypes = []
        self._lib.conv_bwdw_cleanup.restype = None

        self._lib.conv_bwdw_version.argtypes = []
        self._lib.conv_bwdw_version.restype = ctypes.c_char_p

        self._lib.conv_bwdw_has_kernels.argtypes = []
        self._lib.conv_bwdw_has_kernels.restype = ctypes.c_int

        self._lib.conv_bwdw_get_kernel_count.argtypes = []
        self._lib.conv_bwdw_get_kernel_count.restype = ctypes.c_int

        self._lib.conv_bwdw_run.argtypes = [
            ctypes.c_void_p,  # input_ptr
            ctypes.c_void_p,  # grad_output_ptr
            ctypes.c_void_p,  # grad_weight_ptr
            ctypes.POINTER(ConvBwdwProblemC),  # problem
            ctypes.c_void_p,  # stream
        ]
        self._lib.conv_bwdw_run.restype = ctypes.c_float

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> bool:
        """Initialize the backward weight dispatcher"""
        return self._lib.conv_bwdw_init() == 1

    def cleanup(self):
        """Cleanup resources"""
        self._lib.conv_bwdw_cleanup()

    def has_kernels(self) -> bool:
        """Check if backward weight kernels are available"""
        return self._lib.conv_bwdw_has_kernels() == 1

    def get_kernel_count(self) -> int:
        """Get number of registered kernels"""
        return self._lib.conv_bwdw_get_kernel_count()

    def run(
        self,
        input_ptr: int,
        grad_output_ptr: int,
        grad_weight_ptr: int,
        problem: "ConvProblem",
        stream: int = 0,
    ) -> float:
        """
        Run backward weight convolution on GPU.

        Args:
            input_ptr:       Device pointer to input data
            grad_output_ptr: Device pointer to gradient output (dY)
            grad_weight_ptr: Device pointer to gradient weight (dW) - OUTPUT
            problem:         ConvProblem describing the convolution
            stream:          HIP stream (0 for default)

        Returns:
            Elapsed time in milliseconds, or -1.0 on error
        """
        prob_c = ConvBwdwProblemC.from_problem(problem)
        return self._lib.conv_bwdw_run(
            ctypes.c_void_p(input_ptr),
            ctypes.c_void_p(grad_output_ptr),
            ctypes.c_void_p(grad_weight_ptr),
            ctypes.byref(prob_c),
            ctypes.c_void_p(stream),
        )

    @classmethod
    def find(cls) -> Optional["ConvBwdWeightLib"]:
        """Find and load the backward weight library"""
        # This file is in dispatcher/python/
        dispatcher_dir = get_dispatcher_root()

        search_paths = [dispatcher_dir / p for p in cls.SEARCH_PATHS]

        for path in search_paths:
            if path.exists():
                try:
                    lib = ctypes.CDLL(str(path))
                    return cls(lib, path)
                except OSError:
                    continue

        return None


class GpuConvBwdWeightRunner:
    """
    Runs backward weight convolution on GPU.

    Handles HIP memory allocation and the separate backward weight library.

    Usage:
        runner = GpuConvBwdWeightRunner()
        if runner.is_available():
            result = runner.run(input_np, grad_output_np, problem, grad_weight_np)
            print(f"Time: {result['time_ms']:.4f} ms")
    """

    def __init__(self):
        self._lib = None
        self._hip = None
        self._initialized = False
        self._init()

    def _init(self):
        """Initialize library and HIP"""
        try:
            self._lib = ConvBwdWeightLib.find()
            if self._lib is None:
                return

            self._lib.initialize()

            # Load HIP runtime
            try:
                self._hip = ctypes.CDLL("libamdhip64.so")
                self._hip.hipMalloc.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_size_t,
                ]
                self._hip.hipMalloc.restype = ctypes.c_int
                self._hip.hipFree.argtypes = [ctypes.c_void_p]
                self._hip.hipFree.restype = ctypes.c_int
                self._hip.hipMemcpy.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_int,
                ]
                self._hip.hipMemcpy.restype = ctypes.c_int
                self._hip.hipDeviceSynchronize.argtypes = []
                self._hip.hipDeviceSynchronize.restype = ctypes.c_int
            except OSError:
                self._hip = None
                return

            self._initialized = True
        except Exception:
            pass

    def is_available(self) -> bool:
        """Check if GPU backward weight is available"""
        return self._initialized and self._lib is not None and self._hip is not None

    @property
    def library_path(self) -> Optional[str]:
        """Get library path"""
        return str(self._lib.path) if self._lib else None

    def run(
        self,
        input_np: np.ndarray,
        grad_output_np: np.ndarray,
        problem: ConvProblem,
        grad_weight_np: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run backward weight convolution on GPU.

        Args:
            input_np:       Input tensor (NHWGC layout)
            grad_output_np: Gradient output tensor (NHWGK layout)
            problem:        ConvProblem specification (with direction='bwd_weight')
            grad_weight_np: Optional output buffer for gradient weight (GKYXC layout)

        Returns:
            Dict with 'time_ms', 'tflops', 'success', and optionally 'output'
        """
        if not self.is_available():
            return {"success": False, "error": "GPU backward weight not available"}

        try:
            # Calculate sizes
            input_size = input_np.nbytes
            grad_output_size = grad_output_np.nbytes

            # Grad weight output: (G, K, Y, X, C)
            grad_weight_elements = (
                problem.G * problem.K * problem.Y * problem.X * problem.C
            )
            grad_weight_size = grad_weight_elements * input_np.dtype.itemsize

            # Create output buffer if not provided
            if grad_weight_np is None:
                grad_weight_np = np.zeros(
                    (problem.G, problem.K, problem.Y, problem.X, problem.C),
                    dtype=input_np.dtype,
                )

            # Allocate GPU memory
            input_dev = ctypes.c_void_p()
            grad_output_dev = ctypes.c_void_p()
            grad_weight_dev = ctypes.c_void_p()

            self._hip.hipMalloc(ctypes.byref(input_dev), input_size)
            self._hip.hipMalloc(ctypes.byref(grad_output_dev), grad_output_size)
            self._hip.hipMalloc(ctypes.byref(grad_weight_dev), grad_weight_size)

            # Copy input data to device
            self._hip.hipMemcpy(input_dev, input_np.ctypes.data, input_size, 1)  # H2D
            self._hip.hipMemcpy(
                grad_output_dev, grad_output_np.ctypes.data, grad_output_size, 1
            )

            # Run kernel
            time_ms = self._lib.run(
                input_dev.value, grad_output_dev.value, grad_weight_dev.value, problem
            )
            self._hip.hipDeviceSynchronize()

            result = {
                "success": time_ms > 0,
                "time_ms": time_ms if time_ms > 0 else 0,
                "tflops": problem.compute_flops() / (time_ms * 1e9)
                if time_ms > 0
                else 0,
            }

            # Copy back results
            if time_ms > 0:
                self._hip.hipMemcpy(
                    grad_weight_np.ctypes.data,
                    grad_weight_dev,
                    grad_weight_np.nbytes,
                    2,
                )  # D2H
                result["output"] = grad_weight_np

            # Free GPU memory
            self._hip.hipFree(input_dev)
            self._hip.hipFree(grad_output_dev)
            self._hip.hipFree(grad_weight_dev)

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def cleanup(self):
        """Cleanup resources"""
        if self._lib:
            try:
                self._lib.cleanup()
            except Exception:
                pass


# =============================================================================
# HIGH-LEVEL HELPER FUNCTIONS
# =============================================================================


def cleanup_conv():
    """
    Cleanup function to call after running Conv examples.
    """
    import gc

    gc.collect()


def cleanup_generated_conv_kernels(
    keep_default: bool = True,
    verbose: bool = False,
) -> int:
    """
    Clean up generated conv kernel files.

    Call this at the start of examples to ensure fresh state.

    Args:
        keep_default: Keep the default fp16 forward kernel (True) or delete all (False)
        verbose: Print what's being deleted

    Returns:
        Number of files deleted
    """
    kernel_dir = get_generated_kernels_dir()
    if not kernel_dir.exists():
        return 0

    deleted = 0

    # Default kernel pattern to keep
    default_pattern = "conv_fwd_fp16_2d_compv*_128x128_2x2x1.hpp"

    for f in kernel_dir.glob("conv_*.hpp"):
        # Skip directories
        if f.is_dir():
            continue

        # Optionally keep default kernel
        if keep_default and f.match(default_pattern):
            continue

        if verbose:
            print(f"  Deleting: {f.name}")
        f.unlink()
        deleted += 1

    # Also clean up any temp libs
    build_dir = get_build_dir()
    examples_dir = build_dir / "examples"
    if examples_dir.exists():
        for f in examples_dir.glob("libdispatcher_conv_*_lib.so"):
            if f.name not in (
                "libdispatcher_conv_lib.so",
                "libdispatcher_conv_bwdw_lib.so",
            ):
                if verbose:
                    print(f"  Deleting: {f.name}")
                f.unlink()
                deleted += 1

    return deleted


def reset_for_conv_example(verbose: bool = False):
    """
    Reset state for a fresh Conv example run.

    Cleans up generated kernels (except default) and resets globals.
    """
    # Cleanup any previously generated kernels
    deleted = cleanup_generated_conv_kernels(keep_default=True, verbose=verbose)
    if verbose and deleted > 0:
        print(f"  Cleaned up {deleted} generated files")

    # Clear any cached state
    cleanup_conv()


def auto_correct_conv_config(
    pipeline: str = "compv3",
    scheduler: str = "intrawave",
    epilogue: str = "cshuffle",
    wave_m: int = 2,
    wave_n: int = 2,
    wave_k: int = 1,
    warp_m: int = 32,
    warp_n: int = 32,
    warp_k: int = 16,
    dtype: str = "fp16",
    arch: str = "gfx942",
    direction: str = "forward",
    verbose: bool = False,
) -> Tuple[Dict[str, Any], bool, List[str]]:
    """
    Validate and auto-correct a conv kernel configuration.

    Args:
        pipeline, scheduler, epilogue: Trait configuration
        wave_m, wave_n, wave_k: Wave/warp configuration
        warp_m, warp_n, warp_k: Warp tile dimensions
        dtype: Data type
        arch: Target architecture
        direction: Convolution direction (forward, bwd_data, bwd_weight)
        verbose: Print verbose output

    Returns (corrected_config_dict, was_modified, corrections_list).
    If the config was valid, returns (original_config, False, []).
    If corrections were made, returns (new_config, True, [list of correction descriptions]).
    """
    validation = validate_conv_config(
        pipeline=pipeline,
        scheduler=scheduler,
        epilogue=epilogue,
        wave_m=wave_m,
        wave_n=wave_n,
        wave_k=wave_k,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_k=warp_k,
        dtype=dtype,
        arch=arch,
        direction=direction,
    )

    original = {
        "pipeline": pipeline,
        "scheduler": scheduler,
        "epilogue": epilogue,
        "wave_m": wave_m,
        "wave_n": wave_n,
        "wave_k": wave_k,
        "warp_m": warp_m,
        "warp_n": warp_n,
        "warp_k": warp_k,
        "dtype": dtype,
        "arch": arch,
    }

    if validation.is_valid:
        return original, False, []

    # Apply suggested fixes and track what changed
    fixes = validation.suggested_fixes
    corrections = []

    # Check each fix and describe what changed
    if "scheduler" in fixes and fixes["scheduler"] != scheduler:
        corrections.append(
            f"Scheduler: {scheduler} → {fixes['scheduler']} "
            f"('{scheduler}' not supported with pipeline={pipeline}, epilogue={epilogue})"
        )

    if "wave_m" in fixes or "wave_n" in fixes or "wave_k" in fixes:
        old_wave = f"[{wave_m}, {wave_n}, {wave_k}]"
        new_wave = f"[{fixes.get('wave_m', wave_m)}, {fixes.get('wave_n', wave_n)}, {fixes.get('wave_k', wave_k)}]"
        if old_wave != new_wave:
            corrections.append(
                f"Wave config: {old_wave} → {new_wave} "
                f"(original not supported on {arch})"
            )

    if "warp_m" in fixes or "warp_n" in fixes or "warp_k" in fixes:
        old_warp = f"[{warp_m}, {warp_n}, {warp_k}]"
        new_warp = f"[{fixes.get('warp_m', warp_m)}, {fixes.get('warp_n', warp_n)}, {fixes.get('warp_k', warp_k)}]"
        if old_warp != new_warp:
            corrections.append(
                f"Warp tile: {old_warp} → {new_warp} "
                f"(original not supported for {dtype} on {arch})"
            )

    corrected = {
        "pipeline": fixes.get("pipeline", pipeline),
        "scheduler": fixes.get("scheduler", scheduler),
        "epilogue": fixes.get("epilogue", epilogue),
        "wave_m": fixes.get("wave_m", wave_m),
        "wave_n": fixes.get("wave_n", wave_n),
        "wave_k": fixes.get("wave_k", wave_k),
        "warp_m": fixes.get("warp_m", warp_m),
        "warp_n": fixes.get("warp_n", warp_n),
        "warp_k": fixes.get("warp_k", warp_k),
        "dtype": dtype,
        "arch": arch,
    }

    if verbose and corrections:
        print("  ⚠ Auto-correcting configuration:")
        for correction in corrections:
            print(f"    • {correction}")

    return corrected, True, corrections


def print_conv_kernel_config(sig, algo, arch, title: str = "KERNEL CONFIGURATION"):
    """
    Print a formatted kernel configuration for Conv.

    Args:
        sig: ConvSignature object
        algo: ConvAlgorithm object
        arch: ArchInfo object
        title: Title to display (e.g., "REQUESTED KERNEL CONFIGURATION")
    """
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(
        f"  Data Type:     {sig.dtype_in} (input) / {sig.dtype_wei} (weight) / {sig.dtype_out} (output)"
    )
    print(f"  Accumulator:   {sig.dtype_acc}")
    print(f"  Direction:     {sig.direction}")
    print(f"  Spatial Dims:  {sig.num_dims}D")
    print(f"  Layout:        {sig.layout}")
    print(f"  Groups:        {sig.groups}")
    print()
    print(f"  Tile N x K x C: {algo.tile_n} x {algo.tile_k} x {algo.tile_c}")
    print(f"  Wave Config:    {algo.wave_m} x {algo.wave_n} x {algo.wave_k}")
    print(f"  Warp Tile:      {algo.warp_m} x {algo.warp_n} x {algo.warp_k}")
    print(f"  Pipeline:       {algo.pipeline}")
    print(f"  Scheduler:      {algo.scheduler}")
    print(f"  Epilogue:       {algo.epilogue}")
    print()
    print(f"  Target Arch:    {arch.name}")
    print("=" * 70)
    print()


def print_conv_auto_correction(corrections: List[str], indent: str = "  "):
    """
    Print what was auto-corrected and why.

    Args:
        corrections: List of correction descriptions
        indent: Indentation for output
    """
    if not corrections:
        print(f"{indent}✓ Configuration valid - no corrections needed")
        return

    print(f"\n{indent}⚠ AUTO-CORRECTION APPLIED:")
    print(f"{indent}" + "-" * 50)
    for correction in corrections:
        print(f"{indent}  • {correction}")
    print(f"{indent}" + "-" * 50)
    print()


# =============================================================================
# ENHANCED CONV CODEGEN RUNNER
# =============================================================================


@dataclass
class ConvCodegenResult:
    """Result of conv kernel code generation"""

    success: bool
    output_dir: Optional[Path] = None
    kernel_path: Optional[Path] = None
    kernel_count: int = 0
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0


class EnhancedConvCodegenRunner:
    """
    Enhanced runner for convolution kernel code generation.

    Features:
    - generate_from_config: Generate specific kernel from ConvKernelConfig
    - rebuild_library: Rebuild the conv library after generation
    - Matches GEMM CodegenRunner feature parity
    """

    def __init__(
        self,
        datatype: str = "fp16",
        direction: str = "forward",
        ndim: int = 2,
        gpu_target: str = "gfx942",
    ):
        self.datatype = datatype
        self.direction = direction
        self.ndim = ndim
        self.gpu_target = gpu_target
        self.codegen_path = get_codegen_dir() / "unified_conv_codegen.py"
        self.output_dir = get_generated_kernels_dir()

    def generate_from_config(
        self,
        config: ConvKernelConfig,
        output_dir: Optional[Path] = None,
        force: bool = False,
        show_instances: bool = False,
    ) -> ConvCodegenResult:
        """
        Generate kernel from a specific ConvKernelConfig.

        Args:
            config: ConvKernelConfig with all kernel parameters
            output_dir: Override output directory
            force: Force regeneration even if kernel exists
            show_instances: Print instance names when generating

        Returns:
            ConvCodegenResult with success status and paths
        """
        import time

        out_dir = output_dir or self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        sig = config.signature
        algo = config.algorithm
        arch = config.arch

        # Build expected kernel name pattern
        direction_short = sig.direction_short()
        tile_str = f"{algo.tile_k}x{algo.tile_c}"
        wave_str = f"{algo.wave_m}x{algo.wave_n}x{algo.wave_k}"

        # Check if kernel already exists - use broader pattern for initial check
        pattern = f"conv_{direction_short}_{sig.dtype_in}_{sig.num_dims}d_*.hpp"
        existing = list(out_dir.glob(pattern))

        if existing and not force:
            # Filter to find best match
            matching = [k for k in existing if tile_str in k.name or wave_str in k.name]
            if not matching:
                matching = existing  # Fall back to any kernel of right type

            instance_names = sorted([k.stem for k in matching])
            if show_instances:
                for name in instance_names[:3]:  # Show first 3
                    print(f"  Kernel exists: {name}")
                if len(instance_names) > 3:
                    print(f"  ... and {len(instance_names) - 3} more")

            return ConvCodegenResult(
                success=True,
                output_dir=out_dir,
                kernel_path=matching[0] if matching else existing[0],
                kernel_count=len(matching) if matching else len(existing),
                stdout="Using existing kernel(s)",
            )

        if not self.codegen_path.exists():
            return ConvCodegenResult(
                success=False,
                output_dir=out_dir,
                stderr=f"Codegen not found at {self.codegen_path}",
            )

        start = time.time()

        try:
            # Build command with all algorithm parameters
            cmd = [
                "python3",
                str(self.codegen_path),
                "--datatype",
                sig.dtype_in,
                "--variant",
                sig.direction,
                "--ndim",
                str(sig.num_dims),
                "--arch",
                arch.name,
                "--output",
                str(out_dir),
                # Tile dimensions
                "--tile-m",
                str(algo.tile_m),
                "--tile-n",
                str(algo.tile_n),
                "--tile-k",
                str(algo.tile_k),
                # Wave distribution
                "--warp-m",
                str(algo.wave_m),
                "--warp-n",
                str(algo.wave_n),
                "--warp-k",
                str(algo.wave_k),
                # Warp tile sizes
                "--warp-tile-m",
                str(algo.warp_m),
                "--warp-tile-n",
                str(algo.warp_n),
                "--warp-tile-k",
                str(algo.warp_k),
                # Pipeline and scheduler
                "--pipeline",
                algo.pipeline,
                "--scheduler",
                algo.scheduler,
                "--epilogue",
                algo.epilogue,
                # Vector sizes
                "--vector-a",
                str(algo.vector_size_a),
                "--vector-b",
                str(algo.vector_size_b),
                "--vector-c",
                str(algo.vector_size_c),
                # Occupancy
                "--block-per-cu",
                str(algo.block_per_cu),
                "--num-wave-groups",
                str(algo.num_wave_groups),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Find generated kernels
            matching = list(out_dir.glob(pattern))
            kernel_count = len(matching)
            elapsed = time.time() - start

            instance_names = sorted([k.stem for k in matching])
            if show_instances and instance_names:
                for name in instance_names[:5]:  # Show first 5
                    print(f"  Generated: {name}")
                if len(instance_names) > 5:
                    print(f"  ... and {len(instance_names) - 5} more")

            return ConvCodegenResult(
                success=result.returncode == 0 or kernel_count > 0,
                output_dir=out_dir,
                kernel_path=matching[0] if matching else None,
                stdout=result.stdout,
                stderr=result.stderr,
                kernel_count=kernel_count,
                elapsed_seconds=elapsed,
            )
        except subprocess.TimeoutExpired:
            return ConvCodegenResult(
                success=False,
                output_dir=out_dir,
                stderr="Codegen timed out after 120 seconds",
            )
        except Exception as e:
            return ConvCodegenResult(
                success=False,
                output_dir=out_dir,
                stderr=str(e),
            )

    def _rebuild_library_for_config(
        self,
        config: ConvKernelConfig,
        kernel_header: Path,
    ) -> Optional[Path]:
        """
        Rebuild the conv library with a specific kernel.

        Args:
            config: ConvKernelConfig
            kernel_header: Path to the kernel header file

        Returns:
            Path to the rebuilt library, or None on failure
        """
        build_dir = get_build_dir()

        if not build_dir.exists():
            print(f"  Build directory not found: {build_dir}")
            return None

        sig = config.signature

        # Determine which library to build
        if sig.direction == "bwd_weight":
            lib_target = "dispatcher_conv_bwdw_lib"
            lib_name = "libdispatcher_conv_bwdw_lib.so"
        else:
            lib_target = "dispatcher_conv_lib"
            lib_name = "libdispatcher_conv_lib.so"

        # Build unique library name to avoid overwriting loaded lib
        unique_name = (
            f"libdispatcher_conv_{sig.dtype_in}_{sig.direction_short()}_lib.so"
        )

        try:
            # Run cmake to pick up new kernel headers
            cmake_cmd = ["cmake", ".."]
            subprocess.run(
                cmake_cmd,
                cwd=str(build_dir),
                capture_output=True,
                timeout=30,
            )

            # Build the library
            make_cmd = ["make", lib_target, "-j4"]
            result = subprocess.run(
                make_cmd,
                cwd=str(build_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                print(f"  Build failed: {result.stderr[:200]}")
                return None

            # Copy to unique name
            lib_path = build_dir / "examples" / lib_name
            unique_path = build_dir / "examples" / unique_name

            if lib_path.exists():
                import shutil

                shutil.copy2(lib_path, unique_path)
                return unique_path

            return lib_path if lib_path.exists() else None

        except subprocess.TimeoutExpired:
            print("  Build timed out")
            return None
        except Exception as e:
            print(f"  Build error: {e}")
            return None
