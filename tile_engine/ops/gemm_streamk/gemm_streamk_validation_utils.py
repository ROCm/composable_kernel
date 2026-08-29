#!/usr/bin/env python
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Validation utilities for GEMM kernel generation.
Extracted from tile_engine_develop for consistency.
"""

import logging
from typing import Tuple, List

# Element size mapping for different data types
ELEMENT_SIZE_MAP = {
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "fp8": 1,
    "bf8": 1,
    "int4": 0.5,
    "int32": 4,
    "fp32": 4,
    "fp64": 8,
}

def get_warp_size_for_gpu(gpu_target: str) -> int:
    """Get the warp size for a given GPU target.

    CDNA architectures (gfx9xx) use WAVE64 (64 threads per wavefront).
    RDNA architectures (gfx10xx, gfx11xx, gfx12xx) use WAVE32 (32 threads per wavefront).
    """
    if gpu_target.startswith("gfx9"):
        return 64  # CDNA - WAVE64
    return 32  # RDNA and others - WAVE32

# Supported warp combinations for different GPU architectures and data types
WARP_SUPPORTED_COMBINATIONS = {
    "gfx90a": [
        [1, 4, 1],
        [2, 2, 1],
        [4, 1, 1],
    ],
    "gfx942": [
        [1, 4, 1],
        [2, 2, 1],
        [4, 1, 1],
    ],
    "gfx950": [
        [1, 4, 1],
        [2, 2, 1],
        [4, 1, 1],
    ],
}

# Supported warp tile combinations for different GPU architectures and data types
WARP_TILE_SUPPORTED_COMBINATIONS = {
    "gfx90a": {
        "fp16_fp16_fp16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [64, 4, 16],
        ],
        "bf16_bf16_bf16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [64, 4, 16],
        ],
        "fp8_fp8_fp16": [[32, 32, 16], [32, 32, 32]],
        "bf8_bf8_fp16": [[32, 32, 16], [32, 32, 32]],
    },
    "gfx942": {
        "fp16_fp16_fp16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [64, 4, 16],
        ],
        "bf16_bf16_bf16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [64, 4, 16],
        ],
        "fp8_fp8_fp16": [[32, 32, 16], [32, 32, 32], [16, 16, 32], [16, 16, 64]],
        "bf8_bf8_fp16": [[32, 32, 16], [32, 32, 32], [16, 16, 64], [16, 16, 32]],
        "int8_int8_int32": [[16, 16, 32], [32, 32, 16]],
    },
    "gfx950": {
        "fp16_fp16_fp16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [64, 4, 16],
        ],
        "bf16_bf16_bf16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [64, 4, 16],
        ],
        "fp8_fp8_fp16": [
            [32, 32, 16],
            [32, 32, 32],
            [16, 16, 32],
            [16, 16, 64],
            [16, 16, 128],
            [32, 32, 64],
        ],
        "bf8_bf8_fp16": [
            [32, 32, 16],
            [32, 32, 32],
            [16, 16, 64],
            [16, 16, 32],
            [16, 16, 128],
            [32, 32, 64],
        ],
    },
}

# Unsupported trait combinations
TRAIT_UNSUPPORTED_COMBINATIONS = {
    ("compv3", "cshuffle", "interwave", "reduction"),
    ("compv3", "default", "interwave", "reduction"),
    ("compv3", "cshuffle", "interwave", "atomic"),
    ("compv3", "default", "interwave", "atomic"),
}


def element_size(data_type: str) -> float:
    """Calculate the size (in bytes) of a single element for given data type."""
    data_type = data_type.lower()
    if data_type not in ELEMENT_SIZE_MAP:
        raise ValueError(f"Unsupported data type: {data_type}")
    return ELEMENT_SIZE_MAP[data_type]

def is_trait_combination_valid(
    pipeline: str, epilogue: str, scheduler: str, reduction_strategy: str
) -> bool:
    """Check if a trait combination is valid."""
    return (
        pipeline,
        epilogue,
        scheduler,
        reduction_strategy,
    ) not in TRAIT_UNSUPPORTED_COMBINATIONS


def validate_warp_configuration(
    warp_m: int,
    warp_n: int,
    warp_k: int,
    gpu_name: str,
) -> bool:
    """Validate warp configuration."""
    base_gpu_name = gpu_name.split(":")[0] if gpu_name else gpu_name
    current_combination = [warp_m, warp_n, warp_k]

    allowed_combinations = WARP_SUPPORTED_COMBINATIONS.get(base_gpu_name)
    if not allowed_combinations:
        logging.debug(f"No warp_[m/n/k] combinations found for GPU: {base_gpu_name}; using defaults")
        allowed_combinations = [[1, 4, 1], [2, 2, 1], [4, 1, 1]]

    # Check if current combination is in the allowed list
    if current_combination not in allowed_combinations:
        return False

    return True

def validate_dimension_alignment(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
) -> Tuple[bool, List[str]]:
    """Check if tile dimensions are properly aligned with warp dimensions."""
    alignment_issues = []

    if tile_m % (warp_m * warp_tile_m) != 0:
        alignment_issues.append(
            f"tile_m({tile_m}) % [{warp_m}x{warp_tile_m}] = {tile_m % (warp_m * warp_tile_m)}"
        )
    if tile_n % (warp_n * warp_tile_n) != 0:
        alignment_issues.append(
            f"tile_n({tile_n}) % [{warp_n}x{warp_tile_n}] = {tile_n % (warp_n * warp_tile_n)}"
        )
    if tile_k % (warp_k * warp_tile_k) != 0:
        alignment_issues.append(
            f"tile_k({tile_k}) % [{warp_k}x{warp_tile_k}] = {tile_k % (warp_k * warp_tile_k)}"
        )

    return len(alignment_issues) == 0, alignment_issues

LDS_SIZE_MAP = {
    "gfx90a": 2**16,   # 64KB
    "gfx942": 2**16,   # 64KB
    "gfx950": 160 * 1024,  # 160KB
}

DEFAULT_LDS_SIZE = 2**16  # 64KB

def validate_lds_capacity(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_datatype: str,
    b_datatype: str,
    pipeline: str,
    gpu_target: str = "",
) -> Tuple[bool, str]:
    """Validate LDS capacity requirements."""
    matrix_a_size = (tile_m * tile_k) * element_size(a_datatype)
    matrix_b_size = (tile_n * tile_k) * element_size(b_datatype)
    total_tile_in_lds = matrix_a_size + matrix_b_size

    base_gpu_target = gpu_target.split(":")[0] if gpu_target else gpu_target
    hw_lds_size = LDS_SIZE_MAP.get(base_gpu_target, DEFAULT_LDS_SIZE)
    double_buffer = pipeline in ["compv4"]
    max_tile_size = hw_lds_size // 2 if double_buffer else hw_lds_size

    if total_tile_in_lds > max_tile_size:
        error_msg = (
            f"LDS capacity exceeded: Total required {total_tile_in_lds:,}B ({total_tile_in_lds / 1024:.1f}KB) > "
            f"maximum allowed {max_tile_size:,}B ({max_tile_size / 1024}KB) "
            f"[{base_gpu_target}, {'double' if double_buffer else 'single'} buffer]. Breakdown:\n"
            f"- Matrix A ({a_datatype}): {tile_m}x{tile_k} = {matrix_a_size:,}B\n"
            f"- Matrix B ({b_datatype}): {tile_n}x{tile_k} = {matrix_b_size:,}B"
        )
        return False, error_msg

    return True, ""

def validate_warp_tile_combination(
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
    a_datatype: str,
    b_datatype: str,
    c_datatype: str,
    gpu_name: str,
) -> Tuple[bool, str]:
    """Validate warp tile combination against GPU-specific supported combinations."""
    base_gpu_name = gpu_name.split(":")[0] if gpu_name else gpu_name

    # Construct the key for looking up supported combinations
    warp_tile_key = f"{a_datatype}_{b_datatype}_{c_datatype}"
    current_combination = [warp_tile_m, warp_tile_n, warp_tile_k]

    # Check if we have GPU-specific combinations
    gpu_warp_tile_combinations = WARP_TILE_SUPPORTED_COMBINATIONS.get(base_gpu_name, {})
    if not gpu_warp_tile_combinations:
        # If GPU not recognized, try to be permissive but log warning
        logging.warning(f"No warp tile combinations found for GPU: {base_gpu_name}")
        return True, ""

    # Check if we have combinations for this data type combination
    allowed_combinations = gpu_warp_tile_combinations.get(warp_tile_key, [])
    if not allowed_combinations:
        # For data type combinations not in the list, be permissive
        logging.debug(
            f"No warp tile combinations found for data types: {warp_tile_key}"
        )
        return True, ""

    # Check if current combination is in the allowed list
    if current_combination not in allowed_combinations:
        error_msg = (
            f"Invalid warp tile combination: {current_combination} not in allowed list. "
            f"Valid combinations for '{warp_tile_key}' on {base_gpu_name}: {allowed_combinations}"
        )
        return False, error_msg

    return True, ""


def is_tile_config_valid(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
    a_datatype: str,
    b_datatype: str,
    c_datatype: str,
    pipeline: str,
    layout: str,
    gpu_target: str,
) -> bool:
    """
    Comprehensive tile configuration validation.
    Returns True if configuration is valid, False otherwise.
    """
    # Basic sanity checks
    if tile_m <= 0 or tile_n <= 0 or tile_k <= 0:
        return False
    if warp_m <= 0 or warp_n <= 0 or warp_k <= 0:
        return False
    if warp_tile_m <= 0 or warp_tile_n <= 0 or warp_tile_k <= 0:
        return False

    # Check that warp tiles fit within block tiles
    if warp_m * warp_tile_m > tile_m:
        return False
    if warp_n * warp_tile_n > tile_n:
        return False
    if warp_k * warp_tile_k > tile_k:
        return False

    # Validate warp configuration
    if not validate_warp_configuration(warp_m, warp_n, warp_k, gpu_target):
        logging.debug(
            f"Invalid warp configuration: warp_m({warp_m}), warp_n({warp_n}), warp_k({warp_k})"
        )
        return False

    # Validate dimension alignment
    is_aligned, alignment_issues = validate_dimension_alignment(
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        warp_k,
        warp_tile_m,
        warp_tile_n,
        warp_tile_k,
    )
    if not is_aligned:
        logging.debug(
            f"Dimension alignment failed: {', '.join(alignment_issues)}. "
            f"Tile dimensions {tile_m}x{tile_n}x{tile_k} must be divisible by "
            f"[warp]: {warp_m}x{warp_n}x{warp_k} x [warp_tile]: {warp_tile_m}x{warp_tile_n}x{warp_tile_k}"
        )
        return False

    # Validate LDS capacity
    lds_valid, lds_error = validate_lds_capacity(
        tile_m, tile_n, tile_k, a_datatype, b_datatype, pipeline, gpu_target
    )
    if not lds_valid:
        logging.debug(f"LDS validation failed: {lds_error}")
        return False

    gemm_valid, gemm_error = validate_gemm(
        tile_m, tile_n, tile_k,
        warp_m, warp_n, warp_k,
        warp_tile_m, warp_tile_n, warp_tile_k,
        a_datatype, b_datatype, c_datatype,
        pipeline, layout,
        gpu_target,
    )
    if not gemm_valid:
        logging.debug(f"GEMM validation failed: {gemm_error}")
        return False

    # Validate warp tile combination
    warp_tile_valid, warp_tile_error = validate_warp_tile_combination(
        warp_tile_m, warp_tile_n, warp_tile_k, a_datatype, b_datatype, c_datatype,
        gpu_target,
    )
    if not warp_tile_valid:
        logging.debug(f"Warp tile validation failed: {warp_tile_error}")
        return False

    return True

def validate_whole_wg_cover_configuration(
    tile_m,
    tile_n,
    tile_k,
    warp_m,
    warp_n,
    warp_k,
    layout,
    a_datatype,
    b_datatype,
    gpu_target: str = "gfx90a",
) -> Tuple[bool, str]:
    # Validate whole workgroup cover configuration

    warp_size = get_warp_size_for_gpu(gpu_target)
    NumWarps = warp_m * warp_n * warp_k
    BlockSize = NumWarps * warp_size

    XPerTile = 0
    YPerTile = 0
    vector_load_size = 0

    # A matrix validation
    if layout[0] == "r":
        vector_load_size = get_global_vector_load_size(
            BlockSize, tile_k, a_datatype, tile_m, tile_k
        )

        XPerTile = tile_k
        YPerTile = tile_m

    elif layout[0] == "c":
        vector_load_size = get_global_vector_load_size(
            BlockSize, tile_k, a_datatype, tile_m, tile_m
        )

        # Validate distribution
        XPerTile = tile_k
        YPerTile = tile_m

        wg_cover_core_valid, wg_cover_core_error = wg_cover_core_validation(
            XPerTile, YPerTile, BlockSize, vector_load_size, warp_size
        )

        if not wg_cover_core_valid:
            logging.debug(
                f"whole workgroup cover failed for Matrix A distribution: {wg_cover_core_error}"
            )
            return False, wg_cover_core_error

        XPerTile = tile_m
        YPerTile = tile_k

    wg_cover_core_valid, wg_cover_core_error = wg_cover_core_validation(
        XPerTile, YPerTile, BlockSize, vector_load_size, warp_size
    )

    if not wg_cover_core_valid:
        logging.debug(
            f"whole workgroup cover failed for Matrix A: {wg_cover_core_error}"
        )
        return False, wg_cover_core_error

    # B matrix validation
    if layout[1] == "r":
        vector_load_size = get_global_vector_load_size(
            BlockSize, tile_k, b_datatype, tile_n, tile_n
        )

        # Validate distribution
        XPerTile = tile_k
        YPerTile = tile_n

        wg_cover_core_valid, wg_cover_core_error = wg_cover_core_validation(
            XPerTile, YPerTile, BlockSize, vector_load_size, warp_size
        )

        if not wg_cover_core_valid:
            logging.debug(
                f"whole workgroup cover failed for Matrix B distribution: {wg_cover_core_error}"
            )
            return False, wg_cover_core_error

        XPerTile = tile_n
        YPerTile = tile_k

    elif layout[1] == "c":
        XPerTile = tile_k
        YPerTile = tile_n

        vector_load_size = get_global_vector_load_size(
            BlockSize, tile_k, b_datatype, tile_n, tile_k
        )

    wg_cover_core_valid, wg_cover_core_error = wg_cover_core_validation(
        XPerTile, YPerTile, BlockSize, vector_load_size, warp_size
    )
    if not wg_cover_core_valid:
        logging.debug(
            f"whole workgroup cover failed for Matrix B: {wg_cover_core_error}"
        )
        return False, wg_cover_core_error

    return True, ""

def wg_cover_core_validation(
    XPerTile: int,
    YPerTile: int,
    BlockSize: int,
    vector_load_size: int,
    warp_size: int,
) -> Tuple[bool, str]:
    if XPerTile % vector_load_size != 0:
        return False, "XPerTile is not divisible by vector_load_size"

    num_warps = BlockSize / warp_size
    LargestVec = (XPerTile * YPerTile) / (num_warps * warp_size)

    X1 = LargestVec if vector_load_size > LargestVec else vector_load_size
    X0 = XPerTile / X1
    Y1 = warp_size // X0

    if X0 * Y1 != warp_size:
        return False, "X0 * Y1 != warp_size"

    return True, ""


def validate_cshuffle_epilogue_distribution(
    tile_m: int,
    tile_n: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    warp_tile_m: int,
    warp_tile_n: int,
    warp_size: int,
    c_datatype: str,
) -> Tuple[bool, str]:
    """
    Validate that the CShuffleEpilogue tile distribution pattern is valid.

    This mirrors the static_assert in static_encoding_pattern.hpp:
        static_assert(X0 * Y1 == warp_size, "X0 * Y1 must cover whole wavefront!");

    The CShuffleEpilogue creates a tile_distribution_encoding_pattern_2d<BlockSize, YPerTile, XPerTile, VecSize, thread_raked>
    where:
    - BlockSize = warp_m * warp_n * warp_k * warp_size
    - YPerTile = MPerIterationShuffle (derived from tile_m / (warp_m * warp_tile_m / some_factor))
    - XPerTile = NPerIterationShuffle (derived from tile_n)
    - VecSize = vector size based on element size (typically 8 for fp16)

    The key constraint is that X0 must evenly divide warp_size, where:
    - X0 = min(warp_size, XPerTile / X1)
    - X1 = min(VecSize, LargestVec)
    - LargestVec = (XPerTile * YPerTile) / (num_warps * warp_size)
    """
    NumWarps = warp_m * warp_n * warp_k
    BlockSize = NumWarps * warp_size

    elem_size = ELEMENT_SIZE_MAP.get(c_datatype, 2)
    VecSize = 16 // elem_size

    XPerTile = tile_n
    YPerTile = tile_m // warp_m

    if XPerTile <= 0 or YPerTile <= 0:
        return False, f"Invalid tile dimensions: XPerTile={XPerTile}, YPerTile={YPerTile}"

    num_warps = BlockSize // warp_size
    if num_warps * warp_size == 0:
        return False, "Invalid BlockSize or warp_size"

    LargestVec = (XPerTile * YPerTile) // (num_warps * warp_size)
    if LargestVec <= 0:
        LargestVec = 1

    X1 = min(VecSize, LargestVec) if LargestVec > 0 else VecSize
    if X1 <= 0:
        X1 = 1

    X0 = min(warp_size, XPerTile // X1) if X1 > 0 else warp_size

    Y1 = warp_size // X0 if X0 > 0 else 0

    if X0 * Y1 != warp_size:
        return (
            False,
            f"CShuffleEpilogue distribution invalid: X0({X0}) * Y1({Y1}) = {X0 * Y1} != warp_size({warp_size}). "
            f"XPerTile={XPerTile}, YPerTile={YPerTile}, VecSize={VecSize}, BlockSize={BlockSize}"
        )

    return True, ""


def get_global_vector_load_size(
    BlockSize: int,
    KPerBlock: int,
    DataType: str,
    MNPerBlock: int,
    XPerTile: int,
) -> int:
    elements_per_thread = MNPerBlock * KPerBlock / BlockSize
    PackedSize = 1

    if (
        PackedSize == 2
        and XPerTile % (PackedSize * 32 / element_size(DataType)) == 0
        and elements_per_thread % (PackedSize * 32 / element_size(DataType)) == 0
    ):
        return PackedSize * 32 / element_size(DataType)
    elif (
        XPerTile % (PackedSize * 16 / element_size(DataType)) == 0
        and elements_per_thread % (PackedSize * 16 / element_size(DataType)) == 0
    ):
        return int(PackedSize * 16 / element_size(DataType))

    elif (
        XPerTile % (PackedSize * 8 / element_size(DataType)) == 0
        and elements_per_thread % (PackedSize * 8 / element_size(DataType)) == 0
    ):
        return int(PackedSize * 8 / element_size(DataType))
    elif (
        element_size(DataType) >= PackedSize * 4
        and XPerTile % (PackedSize * 4 / element_size(DataType)) == 0
        and elements_per_thread % (PackedSize * 4 / element_size(DataType)) == 0
    ):
        return int(PackedSize * 4 / element_size(DataType))
    elif (
        element_size(DataType) >= PackedSize * 2
        and XPerTile % (PackedSize * 2 / element_size(DataType)) == 0
        and elements_per_thread % (PackedSize * 2 / element_size(DataType)) == 0
    ):
        return int(PackedSize * 2 / element_size(DataType))
    else:
        return PackedSize

def validate_gemm(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
    a_datatype: str,
    b_datatype: str,
    c_datatype: str,
    pipeline: str,
    layout: str,
    gpu_target: str,
    trait_name: str = None,
) -> Tuple[bool, str]:
    # GEMM Validation
    warp_size = get_warp_size_for_gpu(gpu_target)

    # Validate whole workgroup cover configuration
    whole_workgroup_cover_valid, whole_workgroup_cover_error = (
        validate_whole_wg_cover_configuration(
            tile_m,
            tile_n,
            tile_k,
            warp_m,
            warp_n,
            warp_k,
            layout,
            a_datatype,
            b_datatype,
            gpu_target,
        )
    )
    if not whole_workgroup_cover_valid:
        logging.debug(
            f"Whole workgroup cover configuration validation failed: {whole_workgroup_cover_error}"
        )
        return False, whole_workgroup_cover_error

    # Validate CShuffleEpilogue distribution pattern (for cshuffle epilogue)
    # This validation ensures the tile distribution pattern is valid for the output tile
    cshuffle_valid, cshuffle_error = validate_cshuffle_epilogue_distribution(
        tile_m,
        tile_n,
        warp_m,
        warp_n,
        warp_k,
        warp_tile_m,
        warp_tile_n,
        warp_size,
        c_datatype,
    )
    if not cshuffle_valid:
        logging.debug(f"CShuffleEpilogue validation failed: {cshuffle_error}")
        return False, cshuffle_error

    return True, ""

