#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Validation utilities for pooling tile_engine configurations.
Validates tile configurations and trait combinations for pooling kernels.
"""

import logging

logging.basicConfig(level=logging.INFO)

WARP_SIZE = 64  # AMD wavefront size


def is_tile_config_valid(
    block_m,
    block_n,
    warp_m,
    warp_n,
    warp_tile_m,
    warp_tile_n,
    thread_tile_m,
    thread_tile_n,
    in_datatype,
    out_datatype,
):
    """
    Validate a pooling tile configuration.

    For pooling, the 2D tile is:
      M = output elements (N*Ho*Wo*C for 2D, N*Do*Ho*Wo*C for 3D)
      N = reduction dimension (window elements: Y*X for 2D, Z*Y*X for 3D)

    BlockShape params:
      BlockWarps = (warp_m, warp_n)
      BlockTile  = (block_m, block_n)
      WarpTile   = (warp_tile_m, warp_tile_n)
      ThreadTile = (thread_tile_m, thread_tile_n)
    """

    # Basic positivity checks
    if any(
        v <= 0
        for v in [
            block_m,
            block_n,
            warp_m,
            warp_n,
            warp_tile_m,
            warp_tile_n,
            thread_tile_m,
            thread_tile_n,
        ]
    ):
        logging.debug("All tile parameters must be positive")
        return False

    # WarpTile must be divisible by ThreadTile
    if warp_tile_m % thread_tile_m != 0:
        logging.debug(
            f"warp_tile_m ({warp_tile_m}) must be divisible by thread_tile_m ({thread_tile_m})"
        )
        return False
    if warp_tile_n % thread_tile_n != 0:
        logging.debug(
            f"warp_tile_n ({warp_tile_n}) must be divisible by thread_tile_n ({thread_tile_n})"
        )
        return False

    # WarpTile / ThreadTile product must be multiple of warp size
    threads_per_warp = (warp_tile_m * warp_tile_n) // (thread_tile_m * thread_tile_n)
    if threads_per_warp % WARP_SIZE != 0:
        logging.debug(
            f"warp_tile product / thread_tile product ({threads_per_warp}) "
            f"must be multiple of WARP_SIZE ({WARP_SIZE})"
        )
        return False

    # Calculate WarpSizeScaleFactor
    warp_size_scale_factor = threads_per_warp // WARP_SIZE

    if warp_tile_m // thread_tile_m > warp_tile_n // thread_tile_n:
        warp_size_scale_factor_m = warp_size_scale_factor
        warp_size_scale_factor_n = 1
    else:
        warp_size_scale_factor_m = 1
        warp_size_scale_factor_n = warp_size_scale_factor

    # Block dimensions must be properly divisible
    if (block_m * warp_size_scale_factor_m) % (warp_m * warp_tile_m) != 0:
        logging.debug(
            f"block_m*scale ({block_m * warp_size_scale_factor_m}) "
            f"must be divisible by warp_m*warp_tile_m ({warp_m * warp_tile_m})"
        )
        return False
    if (block_n * warp_size_scale_factor_n) % (warp_n * warp_tile_n) != 0:
        logging.debug(
            f"block_n*scale ({block_n * warp_size_scale_factor_n}) "
            f"must be divisible by warp_n*warp_tile_n ({warp_n * warp_tile_n})"
        )
        return False

    # BlockSize = WARP_SIZE * warp_m * warp_n; should be reasonable
    block_size = WARP_SIZE * warp_m * warp_n
    if block_size > 1024:
        logging.debug(f"BlockSize ({block_size}) exceeds maximum of 1024")
        return False

    return True


def is_trait_combination_valid(reduce_op, output_index, propagate_nan, pooling_dim):
    """
    Validate a pooling trait combination.

    Parameters:
        reduce_op: "max" or "avg"
        output_index: bool - whether to output indices
        propagate_nan: bool - whether to propagate NaN
        pooling_dim: "2d" or "3d"
    """
    # output_index only makes sense for max pooling
    if output_index and reduce_op != "max":
        logging.debug("output_index is only supported for max pooling")
        return False

    # Pooling dimension must be valid
    if pooling_dim not in ("2d", "3d"):
        logging.debug(f"Invalid pooling dimension: {pooling_dim}")
        return False

    return True
