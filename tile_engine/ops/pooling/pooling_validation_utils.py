#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Validation utilities for pooling tile_engine configurations.

Validates tile configurations, trait combinations, and datatype support for
pooling kernels.  Modelled after gemm_validation_utils.py — each constraint
from the CK PoolShape / PoolKernel static_asserts is mirrored here so that
invalid configs are rejected at code-generation time rather than at compile
or runtime.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------

# Default warp size (wave64 for CDNA architectures)
WARP_SIZE = 64
MAX_BLOCK_SIZE = 1024  # Maximum threads per workgroup on AMD GPUs
MAX_LDS_BYTES = 65536  # 64 KB LDS per workgroup

def get_warp_size_for_gpu(gpu_target: str) -> int:
    """Get the warp size for a given GPU target.

    CDNA architectures (gfx9xx) use WAVE64 (64 threads per wavefront).
    RDNA architectures (gfx10xx, gfx11xx, gfx12xx) use WAVE32 (32 threads per wavefront).
    """
    if gpu_target.startswith("gfx9"):
        return 64  # CDNA - WAVE64
    return 32  # RDNA and others - WAVE32

# ---------------------------------------------------------------------------
# Datatype helpers
# ---------------------------------------------------------------------------

ELEMENT_SIZE_MAP = {
    "fp8": 1,
    "bf8": 1,
    "int8": 1,
    "fp16": 2,
    "bf16": 2,
    "int4": 0.5,
    "int32": 4,
    "fp32": 4,
    "fp64": 8,
}

DTYPE_STRING_MAP = {
    "fp8": "ck_tile::fp8_t",
    "bf8": "ck_tile::bf8_t",
    "fp16": "ck_tile::fp16_t",
    "bf16": "ck_tile::bf16_t",
    "fp32": "float",
    "fp64": "double",
}

SUPPORTED_DATATYPES = list(DTYPE_STRING_MAP.keys())

# ---------------------------------------------------------------------------
# Reduce-op helpers
# ---------------------------------------------------------------------------

REDUCE_OP_STRING_MAP = {
    "max": "ck_tile::ReduceOp::Max",
    "min": "ck_tile::ReduceOp::Min",
    "avg": "ck_tile::ReduceOp::Add",
}

SUPPORTED_REDUCE_OPS = list(REDUCE_OP_STRING_MAP.keys())

SUPPORTED_POOLING_DIMS = ("2d", "3d")

# ---------------------------------------------------------------------------
# Public helper functions (used by the instance builder)
# ---------------------------------------------------------------------------


def element_size(datatype: str) -> float:
    """Return the byte-width of a single element for *datatype*."""
    datatype = datatype.lower()
    if datatype not in ELEMENT_SIZE_MAP:
        raise ValueError(
            f"Unsupported data type: '{datatype}'. "
            f"Supported: {list(ELEMENT_SIZE_MAP.keys())}"
        )
    return ELEMENT_SIZE_MAP[datatype]


def get_dtype_string(datatype: str) -> str:
    """Return the C++ type string (e.g. ``ck_tile::fp16_t``) for *datatype*."""
    return DTYPE_STRING_MAP.get(datatype, "float")


def get_reduce_op_string(reduce_op: str) -> str:
    """Return the C++ ReduceOp enumerator string for *reduce_op*."""
    return REDUCE_OP_STRING_MAP.get(reduce_op, "ck_tile::ReduceOp::Max")


# ---------------------------------------------------------------------------
# Individual tile-config validators
# ---------------------------------------------------------------------------


def validate_positivity(
    block_m: int,
    block_n: int,
    warp_m: int,
    warp_n: int,
    warp_tile_m: int,
    warp_tile_n: int,
    thread_tile_m: int,
    thread_tile_n: int,
) -> Tuple[bool, str]:
    """All tile parameters must be positive integers."""
    params = {
        "block_m": block_m,
        "block_n": block_n,
        "warp_m": warp_m,
        "warp_n": warp_n,
        "warp_tile_m": warp_tile_m,
        "warp_tile_n": warp_tile_n,
        "thread_tile_m": thread_tile_m,
        "thread_tile_n": thread_tile_n,
    }
    for name, val in params.items():
        if val <= 0:
            return False, f"{name} ({val}) must be > 0"
    return True, ""


def validate_power_of_two(
    block_m: int,
    block_n: int,
    warp_m: int,
    warp_n: int,
    warp_tile_m: int,
    warp_tile_n: int,
    thread_tile_m: int,
    thread_tile_n: int,
) -> Tuple[bool, str]:
    """All tile parameters should be powers of two for correct GPU addressing."""
    params = {
        "block_m": block_m,
        "block_n": block_n,
        "warp_m": warp_m,
        "warp_n": warp_n,
        "warp_tile_m": warp_tile_m,
        "warp_tile_n": warp_tile_n,
        "thread_tile_m": thread_tile_m,
        "thread_tile_n": thread_tile_n,
    }
    for name, val in params.items():
        if val > 0 and (val & (val - 1)) != 0:
            return False, f"{name} ({val}) is not a power of two"
    return True, ""


def validate_thread_tile_alignment(
    warp_tile_m: int,
    warp_tile_n: int,
    thread_tile_m: int,
    thread_tile_n: int,
) -> Tuple[bool, str]:
    """
    Mirrors pool_shape.hpp:
      static_assert(Warp_M % ThreadTile_M == 0);
      static_assert(Warp_N % ThreadTile_N == 0);
    """
    if warp_tile_m % thread_tile_m != 0:
        return (
            False,
            f"warp_tile_m ({warp_tile_m}) must be divisible by "
            f"thread_tile_m ({thread_tile_m})",
        )
    if warp_tile_n % thread_tile_n != 0:
        return (
            False,
            f"warp_tile_n ({warp_tile_n}) must be divisible by "
            f"thread_tile_n ({thread_tile_n})",
        )
    return True, ""


def validate_warp_thread_distribution(
    warp_tile_m: int,
    warp_tile_n: int,
    thread_tile_m: int,
    thread_tile_n: int,
    warp_size: int = WARP_SIZE,
) -> Tuple[bool, str]:
    """
    Mirrors pool_shape.hpp:
      static_assert((Warp_M * Warp_N / ThreadTile_M / ThreadTile_N)
                    % get_warp_size() == 0);
    """
    threads_per_warp = (warp_tile_m * warp_tile_n) // (thread_tile_m * thread_tile_n)
    if threads_per_warp % warp_size != 0:
        return (
            False,
            f"(warp_tile_m * warp_tile_n) / (thread_tile_m * thread_tile_n) = "
            f"{threads_per_warp} is not a multiple of warp_size ({warp_size})",
        )
    return True, ""


def _compute_warp_size_scale_factors(
    warp_tile_m: int,
    warp_tile_n: int,
    thread_tile_m: int,
    thread_tile_n: int,
    warp_size: int = WARP_SIZE,
) -> Tuple[int, int]:
    """
    Reproduce the WarpSizeScaleFactor_M / _N logic from pool_shape.hpp.
    """
    threads_per_warp = (warp_tile_m * warp_tile_n) // (thread_tile_m * thread_tile_n)
    scale = threads_per_warp // warp_size

    if warp_tile_m // thread_tile_m > warp_tile_n // thread_tile_n:
        return scale, 1
    return 1, scale


def validate_block_tile_coverage(
    block_m: int,
    block_n: int,
    warp_m: int,
    warp_n: int,
    warp_tile_m: int,
    warp_tile_n: int,
    thread_tile_m: int,
    thread_tile_n: int,
    warp_size: int = WARP_SIZE,
) -> Tuple[bool, str]:
    """
    Mirrors pool_shape.hpp:
      static_assert((Block_M * WarpSizeScaleFactor_M) %
                    (WarpPerBlock_M * Warp_M) == 0);
      static_assert((Block_N * WarpSizeScaleFactor_N) %
                    (WarpPerBlock_N * Warp_N) == 0);
    """
    sf_m, sf_n = _compute_warp_size_scale_factors(
        warp_tile_m, warp_tile_n, thread_tile_m, thread_tile_n, warp_size
    )

    if (block_m * sf_m) % (warp_m * warp_tile_m) != 0:
        return (
            False,
            f"block_m*ScaleFactor_M ({block_m}*{sf_m}={block_m * sf_m}) must be "
            f"divisible by warp_m*warp_tile_m ({warp_m}*{warp_tile_m}"
            f"={warp_m * warp_tile_m})",
        )
    if (block_n * sf_n) % (warp_n * warp_tile_n) != 0:
        return (
            False,
            f"block_n*ScaleFactor_N ({block_n}*{sf_n}={block_n * sf_n}) must be "
            f"divisible by warp_n*warp_tile_n ({warp_n}*{warp_tile_n}"
            f"={warp_n * warp_tile_n})",
        )
    return True, ""


def validate_block_size(
    warp_m: int,
    warp_n: int,
    warp_size: int = WARP_SIZE,
) -> Tuple[bool, str]:
    """BlockSize = warp_size * warp_m * warp_n must be <= MAX_BLOCK_SIZE."""
    block_size = warp_size * warp_m * warp_n
    if block_size > MAX_BLOCK_SIZE:
        return (
            False,
            f"BlockSize ({block_size} = {warp_size}*{warp_m}*{warp_n}) "
            f"exceeds maximum ({MAX_BLOCK_SIZE})",
        )
    return True, ""


def validate_vector_load_alignment(
    block_m: int,
    thread_tile_m: int,
    in_datatype: str,
) -> Tuple[bool, str]:
    """
    The M-dimension thread-tile determines the contiguous vector load width.
    It must produce a load whose byte-width divides 16 bytes (max global
    vector load width on AMD GPUs) and is at least 1 element wide.
    """
    elem_bytes = element_size(in_datatype)
    load_bytes = thread_tile_m * elem_bytes
    if load_bytes > 16:
        return (
            False,
            f"thread_tile_m ({thread_tile_m}) * element_size({in_datatype}, "
            f"{elem_bytes}B) = {load_bytes}B exceeds 16B max vector load",
        )
    if 16 % load_bytes != 0 and load_bytes % 16 != 0:
        return (
            False,
            f"Vector load width ({load_bytes}B) is not a divisor of 16B",
        )
    return True, ""


def validate_repeat_factors(
    block_m: int,
    block_n: int,
    warp_m: int,
    warp_n: int,
    warp_tile_m: int,
    warp_tile_n: int,
    thread_tile_m: int,
    thread_tile_n: int,
) -> Tuple[bool, str]:
    """
    Repeat_M and Repeat_N from pool_shape.hpp must be >= 1.  They are the
    number of tile iterations each warp performs within the block.
    """
    sf_m, sf_n = _compute_warp_size_scale_factors(
        warp_tile_m, warp_tile_n, thread_tile_m, thread_tile_n
    )
    repeat_m = (block_m * sf_m) // (warp_m * warp_tile_m)
    repeat_n = (block_n * sf_n) // (warp_n * warp_tile_n)
    if repeat_m < 1:
        return False, f"Repeat_M ({repeat_m}) must be >= 1"
    if repeat_n < 1:
        return False, f"Repeat_N ({repeat_n}) must be >= 1"
    return True, ""


# ---------------------------------------------------------------------------
# Comprehensive tile-config validation (entry point)
# ---------------------------------------------------------------------------


def is_tile_config_valid(
    block_m: int,
    block_n: int,
    warp_m: int,
    warp_n: int,
    warp_tile_m: int,
    warp_tile_n: int,
    thread_tile_m: int,
    thread_tile_n: int,
    in_datatype: str,
    out_datatype: str,
    fast_mode: bool = False,
    gpu_target: str = "gfx90a",
) -> bool:
    """
    Comprehensive pooling tile configuration validation.

    When *fast_mode* is True only cheap sanity checks are performed (useful
    for the ``--list_kernels`` path).  Full mode mirrors every
    ``static_assert`` in ``pool_shape.hpp``.

    Parameters
    ----------
    block_m, block_n : Block tile dimensions (M = output elems, N = window).
    warp_m, warp_n   : Warps per block along each dimension.
    warp_tile_m, warp_tile_n : Tile processed per warp.
    thread_tile_m, thread_tile_n : Contiguous elements per thread.
    in_datatype  : Input element type (e.g. ``"fp16"``).
    out_datatype : Output element type.
    fast_mode    : Skip expensive checks when True.
    """
    all_params = (
        block_m, block_n, warp_m, warp_n,
        warp_tile_m, warp_tile_n, thread_tile_m, thread_tile_n,
    )

    # --- Positivity (always) ---
    ok, err = validate_positivity(*all_params)
    if not ok:
        logger.debug(f"Positivity check failed: {err}")
        return False

    # --- Thread-tile alignment (always) ---
    ok, err = validate_thread_tile_alignment(
        warp_tile_m, warp_tile_n, thread_tile_m, thread_tile_n
    )
    if not ok:
        logger.debug(f"Thread tile alignment failed: {err}")
        return False

    if fast_mode:
        return True

    # Get the warp size for this GPU target
    warp_size = get_warp_size_for_gpu(gpu_target)

    # --- Power-of-two ---
    ok, err = validate_power_of_two(*all_params)
    if not ok:
        logger.debug(f"Power-of-two check failed: {err}")
        return False

    # --- Warp-thread distribution ---
    ok, err = validate_warp_thread_distribution(
        warp_tile_m, warp_tile_n, thread_tile_m, thread_tile_n, warp_size
    )
    if not ok:
        logger.debug(f"Warp thread distribution failed: {err}")
        return False

    # --- Block-tile coverage ---
    ok, err = validate_block_tile_coverage(*all_params, warp_size=warp_size)
    if not ok:
        logger.debug(f"Block tile coverage failed: {err}")
        return False

    # --- Block size ---
    ok, err = validate_block_size(warp_m, warp_n, warp_size)
    if not ok:
        logger.debug(f"Block size check failed: {err}")
        return False

    # --- Repeat factors ---
    ok, err = validate_repeat_factors(*all_params)
    if not ok:
        logger.debug(f"Repeat factor check failed: {err}")
        return False

    # --- Vector load alignment ---
    ok, err = validate_vector_load_alignment(block_m, thread_tile_m, in_datatype)
    if not ok:
        logger.debug(f"Vector load alignment failed: {err}")
        return False

    return True


# ---------------------------------------------------------------------------
# Trait-combination validation
# ---------------------------------------------------------------------------


def is_trait_combination_valid(
    reduce_op: str,
    output_index: bool,
    propagate_nan: bool,
    pooling_dim: str,
) -> bool:
    """
    Validate a pooling trait combination.

    Parameters
    ----------
    reduce_op    : ``"max"``, ``"min"``, or ``"avg"``.
    output_index : Whether to output indices of the selected elements.
    propagate_nan: Whether to propagate NaN values through the reduction.
    pooling_dim  : ``"2d"`` or ``"3d"``.
    """
    if reduce_op not in SUPPORTED_REDUCE_OPS:
        logger.debug(f"Unsupported reduce_op: '{reduce_op}'")
        return False

    if pooling_dim not in SUPPORTED_POOLING_DIMS:
        logger.debug(f"Invalid pooling dimension: '{pooling_dim}'")
        return False

    # output_index only makes sense for max pooling (CK constraint)
    if output_index and reduce_op != "max":
        logger.debug(
            f"output_index=True is only supported for 'max' pooling, "
            f"not '{reduce_op}'"
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Datatype validation
# ---------------------------------------------------------------------------


def is_datatype_supported(datatype: str) -> bool:
    """Return True if *datatype* is a known pooling datatype."""
    return datatype.lower() in ELEMENT_SIZE_MAP
