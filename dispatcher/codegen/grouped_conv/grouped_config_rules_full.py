#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Full rule set for Grouped Convolution Tile Configurations.
"""

import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# ---------------------------------------------------------------------------
# Path setup — allow importing arch_specs_generated from the codegen directory
# ---------------------------------------------------------------------------
_CODEGEN_DIR = Path(__file__).resolve().parent.parent
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

from arch_specs_generated import WARP_TILE_SUPPORTED_COMBINATIONS

from .tile_math import (
    get_valid_vec_sizes as _tm_get_valid_vec_sizes,
    get_valid_wave_warp_pairs as _tm_get_valid_wave_warp_pairs,
)

# Dtype string to dtype_key mapping (for tile_math filter functions).
DTYPE_TO_DTYPE_KEY: Dict[str, str] = {
    "fp16": "fp16_fp16_fp32",
    "bf16": "bf16_bf16_fp32",
    "fp32": "fp32_fp32_fp32",
}

# =============================================================================
# Tile Lists
# =============================================================================
#
# ndim + concrete-dtype keyed tile tables
# 
# Tiles diverge by both ndim (3D is a subset of 2D) and concrete dtype. 
# Keyed by (ndim, dtype) where dtype is the concrete "fp16"/"bf16"/"fp32" string. 

_FWD_TILES: Dict[Tuple[int, str], List[Tuple[int, int, int]]] = {
    (2, 'bf16'): [(16, 16, 64), (16, 16, 128), (16, 32, 64), (16, 64, 64), (16, 128, 64), (16, 256, 64), (32, 16, 64), (32, 64, 32), (32, 64, 64), (32, 128, 32), (32, 128, 64), (32, 256, 64), (64, 16, 16), (64, 16, 64), (64, 32, 32), (64, 32, 64), (64, 64, 8), (64, 64, 32), (64, 64, 64), (64, 128, 32), (64, 128, 64), (128, 16, 64), (128, 32, 32), (128, 32, 64), (128, 64, 8), (128, 64, 16), (128, 64, 32), (128, 64, 64), (128, 128, 32), (128, 128, 64), (128, 256, 32), (224, 256, 64), (256, 16, 64), (256, 32, 64), (256, 64, 8), (256, 128, 32), (256, 224, 64), (256, 256, 32)],
    (2, 'fp16'): [(16, 16, 64), (16, 16, 128), (16, 32, 64), (16, 64, 64), (16, 128, 64), (16, 256, 64), (32, 16, 64), (32, 64, 32), (32, 64, 64), (32, 128, 32), (32, 128, 64), (32, 256, 64), (64, 16, 16), (64, 16, 64), (64, 32, 32), (64, 32, 64), (64, 64, 8), (64, 64, 32), (64, 64, 64), (64, 128, 32), (64, 128, 64), (128, 16, 64), (128, 32, 32), (128, 32, 64), (128, 64, 8), (128, 64, 16), (128, 64, 32), (128, 64, 64), (128, 128, 32), (128, 128, 64), (128, 256, 32), (224, 256, 64), (256, 16, 64), (256, 32, 64), (256, 64, 8), (256, 128, 32), (256, 224, 64), (256, 256, 32)],
    (2, 'fp32'): [(16, 16, 64), (16, 16, 128), (16, 32, 64), (16, 64, 64), (16, 128, 64), (32, 16, 64), (32, 64, 16), (32, 64, 64), (32, 128, 16), (32, 128, 64), (64, 16, 16), (64, 16, 64), (64, 32, 16), (64, 32, 64), (64, 64, 16), (64, 64, 32), (64, 128, 16), (128, 16, 64), (128, 32, 16), (128, 32, 64), (128, 64, 16), (128, 128, 16), (128, 128, 32), (128, 128, 64), (128, 192, 16), (128, 256, 16), (256, 128, 16)],
    (3, 'bf16'): [(16, 16, 64), (16, 16, 128), (16, 32, 64), (16, 64, 64), (16, 128, 64), (16, 256, 64), (32, 16, 64), (32, 64, 32), (32, 64, 64), (32, 128, 32), (32, 128, 64), (32, 256, 64), (64, 16, 16), (64, 16, 64), (64, 32, 32), (64, 32, 64), (64, 64, 32), (64, 64, 64), (64, 128, 32), (64, 128, 64), (128, 16, 64), (128, 32, 32), (128, 32, 64), (128, 64, 32), (128, 64, 64), (128, 128, 32), (128, 128, 64), (128, 256, 32), (224, 256, 64), (256, 16, 64), (256, 32, 64), (256, 128, 32), (256, 224, 64), (256, 256, 32)],
    (3, 'fp16'): [(16, 16, 64), (16, 16, 128), (16, 32, 64), (16, 64, 64), (16, 128, 64), (16, 256, 64), (32, 16, 64), (32, 64, 32), (32, 64, 64), (32, 128, 32), (32, 128, 64), (32, 256, 64), (64, 16, 16), (64, 16, 64), (64, 32, 32), (64, 32, 64), (64, 64, 32), (64, 64, 64), (64, 128, 32), (64, 128, 64), (128, 16, 64), (128, 32, 32), (128, 32, 64), (128, 64, 32), (128, 64, 64), (128, 128, 32), (128, 128, 64), (128, 256, 32), (224, 256, 64), (256, 16, 64), (256, 32, 64), (256, 128, 32), (256, 224, 64), (256, 256, 32)],
    (3, 'fp32'): [(16, 16, 64), (16, 16, 128), (16, 32, 64), (16, 64, 64), (16, 128, 64), (32, 16, 64), (32, 64, 16), (32, 64, 64), (32, 128, 16), (32, 128, 64), (64, 16, 16), (64, 16, 64), (64, 32, 16), (64, 32, 64), (64, 64, 16), (64, 64, 32), (64, 128, 16), (128, 16, 64), (128, 32, 16), (128, 32, 64), (128, 64, 16), (128, 128, 16), (128, 128, 32), (128, 128, 64), (128, 192, 16), (128, 256, 16), (256, 128, 16)],
}

_BWD_DATA_TILES: Dict[Tuple[int, str], List[Tuple[int, int, int]]] = {
    (2, 'bf16'): [(16, 64, 32), (32, 64, 32), (32, 128, 32), (64, 16, 16), (64, 16, 32), (64, 16, 64), (64, 32, 32), (64, 64, 32), (64, 64, 64), (64, 128, 32), (128, 32, 16), (128, 32, 32), (128, 32, 64), (128, 64, 32), (128, 64, 64), (128, 128, 32), (128, 256, 32), (256, 32, 64), (256, 128, 32), (256, 128, 64)],
    (2, 'fp16'): [(16, 64, 32), (32, 64, 32), (32, 128, 32), (64, 16, 16), (64, 16, 32), (64, 16, 64), (64, 32, 32), (64, 64, 32), (64, 64, 64), (64, 128, 32), (128, 32, 16), (128, 32, 32), (128, 32, 64), (128, 64, 32), (128, 64, 64), (128, 128, 32), (128, 256, 32), (256, 32, 64), (256, 128, 32), (256, 128, 64)],
    (2, 'fp32'): [(16, 64, 32), (32, 64, 32), (32, 128, 32), (64, 16, 16), (64, 16, 32), (64, 32, 32), (64, 64, 32), (64, 128, 32), (128, 32, 16), (128, 32, 32), (128, 64, 32), (128, 128, 32), (128, 256, 32), (256, 128, 32)],
    (3, 'bf16'): [(16, 64, 32), (32, 64, 32), (32, 128, 32), (64, 16, 16), (64, 16, 32), (64, 16, 64), (64, 32, 32), (64, 64, 32), (64, 128, 32), (128, 32, 16), (128, 32, 32), (128, 32, 64), (128, 64, 32), (128, 128, 32), (128, 256, 32), (256, 128, 32)],
    (3, 'fp16'): [(16, 64, 32), (32, 64, 32), (32, 128, 32), (64, 16, 16), (64, 16, 32), (64, 16, 64), (64, 32, 32), (64, 64, 32), (64, 128, 32), (128, 32, 16), (128, 32, 32), (128, 32, 64), (128, 64, 32), (128, 128, 32), (128, 256, 32), (256, 128, 32)],
    (3, 'fp32'): [(16, 64, 32), (32, 64, 32), (32, 128, 32), (64, 16, 16), (64, 16, 32), (64, 32, 32), (64, 64, 32), (64, 128, 32), (128, 32, 16), (128, 32, 32), (128, 64, 32), (128, 128, 32), (128, 256, 32), (256, 128, 32)],
}

_BWD_WEIGHT_TILES: Dict[Tuple[int, str], List[Tuple[int, int, int]]] = {
    (2, 'bf16'): [(16, 16, 32), (16, 16, 64), (16, 32, 64), (16, 64, 64), (16, 128, 32), (16, 128, 64), (16, 256, 32), (16, 256, 64), (32, 16, 64), (32, 32, 32), (32, 64, 32), (32, 128, 32), (64, 16, 64), (64, 32, 32), (64, 32, 64), (64, 64, 32), (64, 64, 64), (64, 128, 32), (64, 128, 64), (128, 16, 64), (128, 32, 32), (128, 64, 32), (128, 128, 32), (128, 128, 64), (128, 256, 32), (256, 16, 64), (256, 32, 64), (256, 128, 32), (256, 256, 32)],
    (2, 'fp16'): [(16, 16, 32), (16, 16, 64), (16, 32, 64), (16, 64, 64), (16, 128, 32), (16, 128, 64), (16, 256, 32), (16, 256, 64), (32, 16, 64), (32, 32, 32), (32, 64, 32), (32, 128, 32), (64, 16, 64), (64, 32, 32), (64, 32, 64), (64, 64, 32), (64, 64, 64), (64, 128, 32), (64, 128, 64), (128, 16, 64), (128, 32, 32), (128, 64, 32), (128, 128, 32), (128, 128, 64), (128, 256, 32), (256, 16, 64), (256, 32, 64), (256, 128, 32), (256, 256, 32)],
    (2, 'fp32'): [(16, 16, 32), (16, 32, 64), (32, 16, 64), (32, 64, 16), (32, 128, 16), (64, 16, 64), (64, 32, 16), (64, 64, 16), (64, 64, 64), (64, 128, 16), (128, 32, 16), (128, 32, 32), (128, 64, 16), (128, 128, 16), (128, 256, 16), (256, 128, 16)],
    (3, 'bf16'): [(16, 16, 32), (16, 16, 64), (16, 32, 64), (16, 64, 64), (16, 128, 32), (16, 128, 64), (16, 256, 32), (16, 256, 64), (32, 16, 64), (32, 32, 32), (32, 64, 32), (32, 128, 32), (64, 16, 64), (64, 32, 32), (64, 64, 32), (64, 64, 64), (64, 128, 32), (128, 16, 64), (128, 32, 32), (128, 64, 32), (128, 128, 32), (128, 128, 64), (128, 256, 32), (256, 16, 64), (256, 128, 32), (256, 256, 32)],
    (3, 'fp16'): [(16, 16, 32), (16, 16, 64), (16, 32, 64), (16, 64, 64), (16, 128, 32), (16, 128, 64), (16, 256, 32), (16, 256, 64), (32, 16, 64), (32, 32, 32), (32, 64, 32), (32, 128, 32), (64, 16, 64), (64, 32, 32), (64, 64, 32), (64, 64, 64), (64, 128, 32), (128, 16, 64), (128, 32, 32), (128, 64, 32), (128, 128, 32), (128, 128, 64), (128, 256, 32), (256, 16, 64), (256, 128, 32), (256, 256, 32)],
    (3, 'fp32'): [(16, 16, 32), (16, 32, 64), (32, 16, 64), (32, 64, 16), (32, 128, 16), (64, 16, 64), (64, 32, 16), (64, 64, 16), (64, 64, 64), (64, 128, 16), (128, 32, 16), (128, 32, 32), (128, 64, 16), (128, 128, 16), (128, 256, 16), (256, 128, 16)],
}

_TILES_PER_VARIANT: Dict[str, Dict[Tuple[int, str], List[Tuple[int, int, int]]]] = {
    "forward": _FWD_TILES,
    "bwd_data": _BWD_DATA_TILES,
    "bwd_weight": _BWD_WEIGHT_TILES,
}

# Override the tile sizes for split-image feature.
_SPLIT_IMAGE_TILES: List[Tuple[int, int, int]] = [
    (64, 64, 16), (64, 64, 32),
    (256, 128, 16), (256, 128, 32),
]

# Tiles that support compv4 pipeline
# compv4 has stricter requirements due to double buffering and LDS constraints
COMPV4_COMPATIBLE_TILES: List[Tuple[int, int, int]] = [
    # warp_tile [16,16,16] - all work with compv4
    (16, 64, 64),
    (32, 64, 64),
    (64, 64, 64),
    # warp_tile [16,16,32] - all work with compv4
    (16, 64, 128),
    (32, 64, 128),
    (64, 64, 128),
]

def get_tiles_for_variant(variant: str) -> List[Tuple[int, int, int]]:
    """Return all tiles available for the given conv variant.

    Returns the union of tiles across all (ndim, dtype) keys, sorted.
    """
    all_tiles: set = set()
    for tiles in _TILES_PER_VARIANT.get(variant, {}).values():
        all_tiles.update(tiles)
    return sorted(all_tiles)

def get_tiles(variant: str, ndim: int, dtype: str) -> List[Tuple[int, int, int]]:
    """Return tiles for a (variant, ndim, concrete-dtype).

    ``dtype`` is the concrete "fp16"/"bf16"/"fp32" string. Returns the exact
    tile list observed in the corresponding profiler JSON, or [] if none.
    """
    return list(_TILES_PER_VARIANT.get(variant, {}).get((ndim, dtype), []))


# Warp (m, n) shapes excluded from code generation.
# These are possible shapes, but old CK doesn't
# use them. However, they can potentially be useful.
# (4, 64) / (64, 4) are asymmetric MFMA shapes that only ever appear as
# hand-written native instances; warp_tile_k is no longer part of the key.
_EXCLUDED_WARP_SHAPES: Set[Tuple[int, int]] = {
    (4, 64), (64, 4),
}

# =============================================================================
# Wave (block-to-wave split) strategies
# =============================================================================
#
# A macro tile is partitioned among the waves of a thread block along M, N and K. 
# The split is described by a WaveStrategy that yields the wave *counts*
# (wave_m, wave_n, wave_k); the per-wave tile is then tile / wave_count.
#
# compute_wave_counts() turns a strategy into the (wave_m, wave_n,wave_k) triple, 
# and compute_wave_tile() derives the per-wave tile size from the
# macro tile and strategy.


class WaveStrategy(Enum):
    """Block-to-wave partitioning strategies for grouped convolution configs.

    Each strategy maps to a (wave_m, wave_n, wave_k) wave-count triple.
    """
    SINGLE = "single"        # (1, 1, 1) — one wave covers the whole tile
    SPLIT_M = "split_m"      # (2, 1, 1) — 2 waves along M
    SPLIT_N = "split_n"      # (1, 2, 1) — 2 waves along N
    SPLIT_MN = "split_mn"    # (2, 2, 1) — 2x2 waves along M and N
    SPLIT_M4 = "split_m4"    # (4, 1, 1) — 4 waves along M
    SPLIT_N4 = "split_n4"    # (1, 4, 1) — 4 waves along N
    SPLIT_MK = "split_mk"    # (2, 1, 2) — 2 waves along M and 2 along K


_WAVE_STRATEGY_COUNTS: Dict[WaveStrategy, Tuple[int, int, int]] = {
    WaveStrategy.SINGLE: (1, 1, 1),
    WaveStrategy.SPLIT_M: (2, 1, 1),
    WaveStrategy.SPLIT_N: (1, 2, 1),
    WaveStrategy.SPLIT_MN: (2, 2, 1),
    WaveStrategy.SPLIT_M4: (4, 1, 1),
    WaveStrategy.SPLIT_N4: (1, 4, 1),
    WaveStrategy.SPLIT_MK: (2, 1, 2),
}


def compute_wave_counts(strategy: WaveStrategy) -> Tuple[int, int, int]:
    """Return the (wave_m, wave_n, wave_k) wave-count triple for a strategy."""
    return _WAVE_STRATEGY_COUNTS[strategy]


def compute_wave_tile(
    strategy: WaveStrategy, tile_m: int, tile_n: int, tile_k: int,
) -> Tuple[int, int, int]:
    """Return the per-wave tile (macro tile / wave counts) for a strategy."""
    wm, wn, wk = compute_wave_counts(strategy)
    return (tile_m // wm, tile_n // wn, tile_k // wk)


# Curated wave strategies as a shared base table plus small per-variant
# override tables. Most tiles use the same strategy across the variants that
# use them (_BASE_TILE_WAVE_STRATEGIES); the handful that genuinely differ are
# captured in _WAVE_STRATEGY_OVERRIDES. Tiles absent from both fall back to
# [WaveStrategy.SINGLE] in get_wave_strategies().

_BASE_TILE_WAVE_STRATEGIES: Dict[Tuple[int, int, int], List[WaveStrategy]] = {
    (16, 16, 32): [WaveStrategy.SINGLE],
    (16, 16, 64): [WaveStrategy.SINGLE],
    (16, 16, 128): [WaveStrategy.SINGLE],
    (16, 32, 64): [WaveStrategy.SPLIT_N],
    (16, 64, 32): [WaveStrategy.SINGLE],
    (16, 64, 64): [WaveStrategy.SPLIT_N],
    (16, 128, 32): [WaveStrategy.SINGLE],
    (16, 128, 64): [WaveStrategy.SPLIT_N],
    (16, 256, 32): [WaveStrategy.SINGLE],
    (16, 256, 64): [WaveStrategy.SPLIT_N4],
    (32, 16, 64): [WaveStrategy.SPLIT_M, WaveStrategy.SPLIT_MK],
    (32, 32, 32): [WaveStrategy.SINGLE],
    (32, 64, 16): [WaveStrategy.SINGLE],
    (32, 64, 32): [WaveStrategy.SINGLE],
    (32, 64, 64): [WaveStrategy.SPLIT_N],
    (32, 128, 16): [WaveStrategy.SPLIT_N],
    (32, 128, 32): [WaveStrategy.SPLIT_N],
    (32, 128, 64): [WaveStrategy.SPLIT_N],
    (32, 256, 64): [WaveStrategy.SPLIT_N4],
    (64, 16, 16): [WaveStrategy.SINGLE],
    (64, 16, 32): [WaveStrategy.SINGLE, WaveStrategy.SPLIT_M4],
    (64, 16, 64): [WaveStrategy.SPLIT_M, WaveStrategy.SPLIT_MK],
    (64, 32, 16): [WaveStrategy.SINGLE],
    (64, 32, 32): [WaveStrategy.SINGLE],
    (64, 32, 64): [WaveStrategy.SPLIT_M],
    (64, 64, 8): [WaveStrategy.SPLIT_M],
    (64, 64, 16): [WaveStrategy.SINGLE],
    (64, 64, 32): [WaveStrategy.SINGLE],
    (64, 64, 64): [WaveStrategy.SPLIT_MN],
    (64, 128, 16): [WaveStrategy.SPLIT_N, WaveStrategy.SPLIT_MN],
    (64, 128, 32): [WaveStrategy.SPLIT_N, WaveStrategy.SPLIT_MN],
    (64, 128, 64): [WaveStrategy.SPLIT_MN],
    (128, 16, 64): [WaveStrategy.SPLIT_M, WaveStrategy.SPLIT_MK],
    (128, 32, 16): [WaveStrategy.SPLIT_M],
    (128, 32, 32): [WaveStrategy.SPLIT_M, WaveStrategy.SPLIT_MK],
    (128, 32, 64): [WaveStrategy.SPLIT_M],
    (128, 64, 8): [WaveStrategy.SPLIT_M, WaveStrategy.SPLIT_MN],
    (128, 64, 16): [WaveStrategy.SPLIT_M, WaveStrategy.SPLIT_MN],
    (128, 64, 32): [WaveStrategy.SPLIT_M, WaveStrategy.SPLIT_MN],
    (128, 64, 64): [WaveStrategy.SPLIT_MN],
    (128, 128, 16): [WaveStrategy.SPLIT_N, WaveStrategy.SPLIT_MN],
    (128, 128, 32): [WaveStrategy.SPLIT_N, WaveStrategy.SPLIT_MN],
    (128, 128, 64): [WaveStrategy.SPLIT_MN],
    (128, 192, 16): [WaveStrategy.SPLIT_MN],
    (128, 256, 16): [WaveStrategy.SPLIT_MN],
    (128, 256, 32): [WaveStrategy.SPLIT_MN],
    (224, 256, 64): [WaveStrategy.SPLIT_MN],
    (256, 16, 64): [WaveStrategy.SPLIT_M4],
    (256, 32, 64): [WaveStrategy.SPLIT_M4],
    (256, 64, 8): [WaveStrategy.SPLIT_MN],
    (256, 128, 16): [WaveStrategy.SPLIT_MN],
    (256, 128, 32): [WaveStrategy.SPLIT_MN],
    (256, 128, 64): [WaveStrategy.SPLIT_MN],
    (256, 224, 64): [WaveStrategy.SPLIT_MN],
    (256, 256, 32): [WaveStrategy.SPLIT_MN],
}

# Per-variant deviations from _BASE_TILE_WAVE_STRATEGIES. An entry here
# fully replaces the base strategy list for that (variant, tile). Only the
# tiles where a variant genuinely differs appear; bwd_data is the main
# deviator (it prefers SPLIT_M4 over SPLIT_M for several skinny-N tiles).
_WAVE_STRATEGY_OVERRIDES: Dict[str, Dict[Tuple[int, int, int], List[WaveStrategy]]] = {
    "forward": {
        (64, 64, 32): [WaveStrategy.SINGLE, WaveStrategy.SPLIT_MN],
    },
    "bwd_data": {
        (64, 16, 16): [WaveStrategy.SPLIT_M4],
        (64, 16, 64): [WaveStrategy.SPLIT_M4],
        (128, 32, 16): [WaveStrategy.SPLIT_M4],
        (128, 32, 32): [WaveStrategy.SPLIT_M, WaveStrategy.SPLIT_M4],
        (128, 32, 64): [WaveStrategy.SPLIT_M4],
    },
    "bwd_weight": {
        (16, 64, 64): [WaveStrategy.SPLIT_N, WaveStrategy.SPLIT_N4],
        (32, 128, 32): [WaveStrategy.SINGLE, WaveStrategy.SPLIT_N],
        (64, 32, 64): [WaveStrategy.SPLIT_M, WaveStrategy.SPLIT_N],
        (128, 32, 32): [WaveStrategy.SINGLE, WaveStrategy.SPLIT_M],
    },
}

# Borderline (variant, tile, wave) pairs whose warp (m, n) selection deviates
# from _default_warp_mn(). These are the per-wave-tile == 32x32 cases, where the
# reference data is not consistent with the size threshold:
#   - forward (64, 64, 32)/(2,2,1) uses 16x16 despite a 32x32 per-wave tile;
#   - forward (256, 256, 32)/(2,2,1) and bwd_weight (64, 64, 64)/(2,2,1) use
#     BOTH 16x16 and 32x32.
_WARP_MN_EXCEPTIONS: Dict[
    Tuple[str, Tuple[int, int, int], Tuple[int, int, int]], List[Tuple[int, int]]
] = {
    ("forward", (64, 64, 32), (2, 2, 1)): [(16, 16)],
    ("forward", (64, 64, 64), (2, 2, 1)): [(16, 16), (32, 32)],
    ("forward", (128, 128, 64), (2, 2, 1)): [(16, 16), (32, 32)],
    ("forward", (256, 256, 32), (2, 2, 1)): [(16, 16), (32, 32)],
    ("bwd_weight", (64, 64, 64), (2, 2, 1)): [(16, 16), (32, 32)],
    # bwd_data wavelet tiles: wavelet instances use a 16x16 MFMA
    # warp tile (warp_tile_k=32 for half) even though the per-wave tile is 32x32.
    ("bwd_data", (64, 64, 64), (2, 2, 1)): [(16, 16)],
    ("bwd_data", (128, 64, 64), (2, 2, 1)): [(16, 16)],
    ("bwd_data", (256, 32, 64), (4, 1, 1)): [(16, 16)],
    ("bwd_data", (256, 128, 32), (2, 2, 1)): [(16, 16), (32, 32)],
    ("bwd_data", (256, 128, 64), (2, 2, 1)): [(16, 16)],
}


def _default_warp_mn(tile_m: int, tile_n: int, wave_m: int, wave_n: int) -> Tuple[int, int]:
    """Return the default warp (m, n) MFMA shape for a (tile, wave).

    A 32x32 warp tile is used when the per-wave tile (tile / wave) is at least
    32 and a multiple of 32 in both M and N; otherwise a 16x16 warp tile is used.
    Borderline cases are overridden via _WARP_MN_EXCEPTIONS.
    """
    pm, pn = tile_m // wave_m, tile_n // wave_n
    if pm >= 32 and pn >= 32 and pm % 32 == 0 and pn % 32 == 0:
        return (32, 32)
    return (16, 16)


def get_warp_mn_candidates(
    tile_m: int, tile_n: int, tile_k: int,
    wave_m: int, wave_n: int, wave_k: int,
    variant: str = "forward",
) -> List[Tuple[int, int]]:
    """Return the curated warp (m, n) candidates for a (variant, tile, wave).

    Uses _WARP_MN_EXCEPTIONS when present, otherwise _default_warp_mn().
    """
    key = (variant, (tile_m, tile_n, tile_k), (wave_m, wave_n, wave_k))
    if key in _WARP_MN_EXCEPTIONS:
        return list(_WARP_MN_EXCEPTIONS[key])
    return [_default_warp_mn(tile_m, tile_n, wave_m, wave_n)]


def get_wave_strategies(
    tile_m: int, tile_n: int, tile_k: int, variant: str,
) -> List[WaveStrategy]:
    """Return the curated WaveStrategy list for a (variant, macro tile).

    A per-variant override (_WAVE_STRATEGY_OVERRIDES) takes precedence over the
    shared base table (_BASE_TILE_WAVE_STRATEGIES). Tiles absent from both fall
    back to [WaveStrategy.SINGLE].
    """
    tile = (tile_m, tile_n, tile_k)
    override = _WAVE_STRATEGY_OVERRIDES.get(variant, {}).get(tile)
    if override is not None:
        return list(override)
    entry = _BASE_TILE_WAVE_STRATEGIES.get(tile)
    if entry is None:
        logging.warning(
            "No curated wave strategies for variant=%s tile=(%d,%d,%d); "
            "falling back to [WaveStrategy.SINGLE]",
            variant, tile_m, tile_n, tile_k,
        )
        return [WaveStrategy.SINGLE]
    return list(entry)


def get_wave_configs(
    tile_m: int, tile_n: int, tile_k: int, variant: str,
) -> List[Tuple[int, int, int]]:
    """Return wave-count configs for a tile+variant.

    Each curated WaveStrategy is turned into its (wave_m, wave_n, wave_k) triple.
    Falls back to generic [(1, 1, 1)] for unknown tiles.
    """
    return [
        compute_wave_counts(strategy)
        for strategy in get_wave_strategies(tile_m, tile_n, tile_k, variant)
    ]


def get_all_valid_vector_sizes(
    tile_m: int, tile_n: int, tile_k: int,
    wave_m: int, wave_n: int, wave_k: int,
    wt_m: int, wt_n: int, wt_k: int,
    dtype_key: str,
) -> Set[Tuple[int, int, int]]:
    """Return the set of (vec_a, vec_b, vec_c) triples tile_math considers valid.

    Thin wrapper around tile_math.get_valid_vec_sizes for use as a hard gate:
    curated/strategy vec triples are kept only if present in this set.
    """
    return set(_tm_get_valid_vec_sizes(
        tile_m, tile_n, tile_k,
        wave_m, wave_n, wave_k,
        wt_m, wt_n, wt_k,
        dtype_key,
    ))


def get_k_mfma(dtype_key: str, m_per_xdl: int) -> int:
    """Return the MFMA K dimension for a square warp tile (m_per_xdl == n_per_xdl).

    Single authority mirroring generate_instances.get_k_mfma (the CK Builder).
    Arch-independent (reflects the current CDNA4 MFMA shapes):
      - half (fp16/bf16): 32x32 -> 16, 16x16 -> 32
      - float (fp32):     32x32 -> 2,  16x16 -> 4
    """
    if dtype_key.startswith("fp32"):
        return 2 if m_per_xdl == 32 else 4
    return 16 if m_per_xdl == 32 else 32


def compute_warp_tile_k(
    dtype_key: str, warp_tile_m: int, tile_k: int, streamk: bool = False,
    use_legacy: bool = False,
) -> int:
    """Derive warp_tile_k (the MFMA K) for a config.

    Mirrors the CK Builder formula (generate_instances.py)
        k_per_xdl = min(max(k1, get_k_mfma(dtype, m_per_xdl, n_per_xdl)), k_per_block)
    where k1 is the per-instance AK1. For half precision get_k_mfma dominates k1,
    so warp_tile_k = min(get_k_mfma, tile_k). For fp32, k1 dominates; the reference
    instances use k1 = 4 when tile_k <= 16 else 8.

    Hand-written StreamK *native* instances bypass get_k_mfma and carry the legacy
    (gfx942) warp_tile_k verbatim: half 32x32 -> 8, 16x16 -> 16, clamped to tile_k
    (fp32 native instances clamp to tile_k // 4 instead).

    ``use_legacy`` applies the same legacy formula for non-streamk contexts that
    nevertheless use the gfx942 warp_tile_k convention (e.g. some wavelet bwd_weight
    instances).
    """
    if streamk or use_legacy:
        legacy_half_k = 8 if warp_tile_m == 32 else 16
        cap = tile_k // 4 if dtype_key.startswith("fp32") else tile_k
        return min(legacy_half_k, cap)
    k1 = 4 if tile_k <= 16 else 8
    return min(max(k1, get_k_mfma(dtype_key, warp_tile_m)), tile_k)


def get_warp_configs_for_tile_and_wave(
    tile_m: int, tile_n: int, tile_k: int,
    wave_m: int, wave_n: int, wave_k: int,
    dtype_key: str, arch: str = "gfx942", variant: str = "forward",
) -> List[Tuple[int, int]]:
    """Return curated (warp_tile_m, warp_tile_n) shapes for a (variant, wave).

    The (warp_tile_m, warp_tile_n) shape is derived by ``get_warp_mn_candidates``
    (a size-threshold function plus a small exception table); warp_tile_k is
    derived later by ``compute_warp_tile_k``.

    A (warp_tile_m, warp_tile_n) shape is kept only when it:
      - is arch/dtype-supported (WARP_TILE_SUPPORTED_COMBINATIONS[arch][dtype_key]),
      - is not in _EXCLUDED_WARP_SHAPES,
      - divides the macro tile: tile_m % (wave_m*warp_tile_m) == 0 and
        tile_n % (wave_n*warp_tile_n) == 0.
    """
    tile = (tile_m, tile_n, tile_k)
    wave = (wave_m, wave_n, wave_k)
    curated_waves = {
        compute_wave_counts(strategy)
        for strategy in get_wave_strategies(tile_m, tile_n, tile_k, variant)
    }

    if wave not in curated_waves:
        raise ValueError(f"No curated warp tiles for variant={variant} tile={tile} wave={wave}")
    curated = get_warp_mn_candidates(tile_m, tile_n, tile_k, wave_m, wave_n, wave_k, variant)

    supported_mn = {
        (wt[0], wt[1])
        for wt in WARP_TILE_SUPPORTED_COMBINATIONS.get(arch, {}).get(dtype_key, [])
    }

    result: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for wt in curated:
        mn = (wt[0], wt[1])
        if mn in _EXCLUDED_WARP_SHAPES:
            continue
        if mn not in supported_mn:
            continue
        if tile_m % (wave_m * mn[0]) != 0 or tile_n % (wave_n * mn[1]) != 0:
            continue
        if mn not in seen:
            seen.add(mn)
            result.append(mn)
    return result


def get_wave_warp_pairs(
    tile_m: int, tile_n: int, tile_k: int,
    variant: str, dtype_key: str, arch: str = "gfx942",
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int]]]:
    """Return (wave, warp_tile_mn) pairs: curated waves x curated warp (m, n) shapes.

    Combines curated wave configs with the curated per-variant wave->warp map
    (both from profiler JSON), filtered by arch/dtype support and divisibility.
    warp_tile_k is not part of the pair; it is derived later via
    ``compute_warp_tile_k``.

    The curated pairs are gated against _tm_get_valid_wave_warp_pairs (imported from
    tile_math.py) on the (wave, warp_m, warp_n) granularity, and any curated pair
    rejected is dropped with a warning.
    """
    tm_pairs = _tm_get_valid_wave_warp_pairs(tile_m, tile_n, tile_k, dtype_key, arch)
    tm_pairs_mn = {(wave, (wt[0], wt[1])) for (wave, wt) in tm_pairs}
    result = []
    for wave in get_wave_configs(tile_m, tile_n, tile_k, variant):
        for mn in get_warp_configs_for_tile_and_wave(
            tile_m, tile_n, tile_k, *wave, dtype_key, arch, variant,
        ):
            if (wave, mn) not in tm_pairs_mn:
                logging.warning(
                    "Dropping curated wave/warp pair %s rejected by tile_math: "
                    "tile=(%d,%d,%d) variant=%s dtype_key=%s",
                    (wave, mn), tile_m, tile_n, tile_k, variant, dtype_key,
                )
                continue
            result.append((wave, mn))

    # If no curated pairs survived (e.g. for architectures whose supported wave
    # combos don't match the CDNA-derived curated strategies, such as rdna4/gfx1250),
    # fall back to all tile_math-valid pairs so those arches still get kernels.
    if not result and tm_pairs:
        for wave, wt in tm_pairs:
            result.append((wave, (wt[0], wt[1])))

    return result


# =============================================================================
# Vector Size Strategies (tile + dtype based, wave/warp independent)
# =============================================================================
#
# Convolution GEMM tensors vectorize along different logical dimensions:
#   Forward:    vec_a, vec_b -> C (channels), vec_c -> K (output channels)
#   BWD data:   vec_a -> K, vec_b, vec_c -> C
#   BWD weight: vec_a -> K, vec_b, vec_c -> C
#
# Each strategy produces one (vec_a, vec_b, vec_c) triple from the dtype-determined max.


class VecStrategy(Enum):
    """Vectorization strategies for grouped convolution GEMM configs.

    Each strategy produces a single (vec_a, vec_b, vec_c) triple derived
    from the dtype-determined maximum vector sizes.
    """
    GENERIC = "generic"                          # (1, 1, 1) — minimum fallback
    UNIFORM_MAX = "uniform_max"                  # (max, max, max) — balanced throughput
    MAX_AB_HALF_C = "max_ab_half_c"              # (max, max, max/2) — fwd (8,8,4) pattern
    MAX_A_MIN_BC = "max_a_min_bc"                # (max, 1, 1) — bwd (4,1,1)/(8,1,1) pattern
    MIN_A_MAX_BC = "min_a_max_bc"                # (1, max, max) — bwd (1,4,4)/(1,8,8) pattern
    HALF_UNIFORM = "half_uniform"                # (max/2, max/2, max/2) — (2,2,2) pattern
    QUARTER_AB_MAX_C = "quarter_ab_max_c"        # (max/2, max/2, max) — fwd (4,4,8)/(1,1,8) pattern
    MAX_AB_QUARTER_C = "max_ab_quarter_c"        # (max, max, max/4) — half (8,8,2), fp32 (4,4,1)
    MAX_A_QUARTER_BC = "max_a_quarter_bc"        # (max, max/4, max/4) — half (8,2,2)
    MIN_AB_MAX_C = "min_ab_max_c"                # (1, 1, max) — half (1,1,8), fp32 (1,1,4)
    MAX_A_HALF_BC = "max_a_half_bc"              # (max, max/2, max/2) — fp32 (4,2,2)
    MAX_AB_MIN_C = "max_ab_min_c"                # (max, max, 1) — half (8,8,1)
    MIN_AB_QUARTER_C = "min_ab_quarter_c"        # (1, 1, max/4) — half (1,1,2)
    MIN_A_MAX_B_HALF_C = "min_a_max_b_half_c"    # (1, max, max/2) — half (1,8,4), fp32 (1,4,2) - TODO: Is this valid?
    HALF_A_MAX_BC = "half_a_max_bc"              # (max/2, max, max) — fp32 (2,4,4)
    HALF_A_MIN_BC = "half_a_min_bc"              # (max/2, 1, 1) — fp32 (2,1,1)


def _max_vec(dtype_class: str) -> int:
    """Return the maximum vector width for a dtype class."""
    if dtype_class == "float":
        return 4
    elif dtype_class in ("half", "fp16", "bf16"):
        return 8
    else:
        raise ValueError(f"Unknown dtype class: {dtype_class}")


def compute_vector_size(
    strategy: VecStrategy,
    dtype_class: str,
) -> Tuple[int, int, int]:
    """Compute a (vec_a, vec_b, vec_c) triple for a given strategy and dtype.

    Args:
        strategy: Which vectorization pattern to use.
        dtype_class: "float" (fp32) or "half" (fp16/bf16).

    Returns:
        A single (vec_a, vec_b, vec_c) triple.
    """
    m = _max_vec(dtype_class)
    h = max(1, m // 2)
    q = max(1, m // 4)

    if strategy == VecStrategy.GENERIC:
        return (1, 1, 1)
    elif strategy == VecStrategy.UNIFORM_MAX:
        return (m, m, m)
    elif strategy == VecStrategy.MAX_AB_HALF_C:
        return (m, m, h)
    elif strategy == VecStrategy.MAX_A_MIN_BC:
        return (m, 1, 1)
    elif strategy == VecStrategy.MIN_A_MAX_BC:
        return (1, m, m)
    elif strategy == VecStrategy.HALF_UNIFORM:
        return (h, h, h)
    elif strategy == VecStrategy.QUARTER_AB_MAX_C:
        return (h, h, m)
    elif strategy == VecStrategy.MAX_AB_QUARTER_C:
        return (m, m, q)
    elif strategy == VecStrategy.MAX_A_QUARTER_BC:
        return (m, q, q)
    elif strategy == VecStrategy.MIN_AB_MAX_C:
        return (1, 1, m)
    elif strategy == VecStrategy.MAX_A_HALF_BC:
        return (m, h, h)
    elif strategy == VecStrategy.MAX_AB_MIN_C:
        return (m, m, 1)
    elif strategy == VecStrategy.MIN_AB_QUARTER_C:
        return (1, 1, q)
    elif strategy == VecStrategy.MIN_A_MAX_B_HALF_C:
        return (1, m, h)
    elif strategy == VecStrategy.HALF_A_MAX_BC:
        return (h, m, m)
    elif strategy == VecStrategy.HALF_A_MIN_BC:
        return (h, 1, 1)
    else:
        return (1, 1, 1)


# =============================================================================
# Per-Tile VecStrategy Tables
# =============================================================================
# Per-variant mapping: tile -> {dtype_class -> list[VecStrategy]}.
# Derived per (variant, tile, dtype_class) via greedy set-cover over VecStrategy.
# Tiles not in the table fall back to [GENERIC] (dtype-independent).
#

_FWD_TILE_STRATEGIES: Dict[Tuple[int, int, int], Dict[str, List[VecStrategy]]] = {
    (16, 16, 64): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (16, 16, 128): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (16, 32, 64): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (16, 64, 64): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (16, 128, 64): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (16, 256, 64): {"half": [VecStrategy.MAX_AB_HALF_C]},
    (32, 16, 64): {"half": [VecStrategy.MAX_AB_QUARTER_C], "float": [VecStrategy.MAX_AB_HALF_C]},
    (32, 64, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (32, 64, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (32, 64, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (32, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (32, 128, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (32, 128, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (32, 256, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (64, 16, 16): {"half": [VecStrategy.HALF_A_MIN_BC], "float": [VecStrategy.MAX_A_MIN_BC]},
    (64, 16, 64): {"half": [VecStrategy.MAX_AB_QUARTER_C], "float": [VecStrategy.MAX_AB_HALF_C]},
    (64, 32, 16): {"float": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_AB_QUARTER_C]},
    (64, 32, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C, VecStrategy.MAX_AB_MIN_C]},
    (64, 32, 64): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (64, 64, 8): {"half": [VecStrategy.MIN_AB_MAX_C]},
    (64, 64, 16): {"float": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX]},
    (64, 64, 32): {"half": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX, VecStrategy.HALF_UNIFORM, VecStrategy.MIN_AB_MAX_C], "float": [VecStrategy.UNIFORM_MAX]},
    (64, 64, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (64, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (64, 128, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (64, 128, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (128, 16, 64): {"half": [VecStrategy.MAX_AB_QUARTER_C], "float": [VecStrategy.MAX_AB_HALF_C]},
    (128, 32, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (128, 32, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.HALF_UNIFORM, VecStrategy.MIN_AB_MAX_C]},
    (128, 32, 64): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (128, 64, 8): {"half": [VecStrategy.MIN_AB_MAX_C]},
    (128, 64, 16): {"half": [VecStrategy.MIN_AB_MAX_C], "float": [VecStrategy.UNIFORM_MAX]},
    (128, 64, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (128, 64, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (128, 128, 16): {"float": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (128, 128, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (128, 128, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (128, 192, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (128, 256, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (128, 256, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (224, 256, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (256, 16, 64): {"half": [VecStrategy.MAX_AB_QUARTER_C]},
    (256, 32, 64): {"half": [VecStrategy.MAX_AB_HALF_C]},
    (256, 64, 8): {"half": [VecStrategy.MIN_AB_MAX_C]},
    (256, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (256, 128, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (256, 224, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (256, 256, 32): {"half": [VecStrategy.UNIFORM_MAX]},
}


_BWD_DATA_TILE_STRATEGIES: Dict[Tuple[int, int, int], Dict[str, List[VecStrategy]]] = {
    (16, 64, 32): {"half": [VecStrategy.MAX_AB_HALF_C, VecStrategy.MAX_A_MIN_BC, VecStrategy.MIN_A_MAX_B_HALF_C], "float": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_A_MIN_BC, VecStrategy.MIN_A_MAX_BC]},
    (32, 64, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
    (32, 128, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
    (64, 16, 16): {"half": [VecStrategy.HALF_A_MIN_BC], "float": [VecStrategy.MAX_A_MIN_BC]},
    (64, 16, 32): {"half": [VecStrategy.MAX_AB_HALF_C, VecStrategy.MAX_A_MIN_BC, VecStrategy.MAX_A_QUARTER_BC, VecStrategy.MIN_A_MAX_B_HALF_C], "float": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_A_MIN_BC, VecStrategy.MIN_A_MAX_BC]},
    (64, 32, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
    (64, 64, 32): {"half": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX], "float": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX]},
    (64, 128, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_A_MAX_BC], "float": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_A_MAX_BC]},
    (128, 32, 16): {"half": [VecStrategy.HALF_A_MIN_BC], "float": [VecStrategy.MAX_A_MIN_BC, VecStrategy.MAX_A_HALF_BC]},
    (128, 32, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_A_MIN_BC, VecStrategy.MAX_A_QUARTER_BC], "float": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_A_MIN_BC]},
    (128, 64, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
    (128, 128, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
    (128, 256, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
    (256, 128, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
}


_BWD_WEIGHT_TILE_STRATEGIES: Dict[Tuple[int, int, int], Dict[str, List[VecStrategy]]] = {
    (16, 16, 32): {"half": [VecStrategy.GENERIC, VecStrategy.MIN_AB_QUARTER_C]},
    (16, 16, 64): {"half": [VecStrategy.GENERIC, VecStrategy.HALF_UNIFORM, VecStrategy.HALF_A_MIN_BC]},
    (16, 32, 64): {"half": [VecStrategy.GENERIC, VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.HALF_A_MAX_BC]},
    (16, 64, 64): {"half": [VecStrategy.GENERIC, VecStrategy.MAX_AB_HALF_C, VecStrategy.MIN_A_MAX_B_HALF_C]},
    (16, 128, 64): {"half": [VecStrategy.MIN_A_MAX_B_HALF_C]},
    (16, 256, 32): {"half": [VecStrategy.MAX_AB_MIN_C]},
    (16, 256, 64): {"half": [VecStrategy.MIN_A_MAX_B_HALF_C]},
    (32, 16, 64): {"half": [VecStrategy.GENERIC, VecStrategy.HALF_A_MIN_BC], "float": [VecStrategy.MAX_A_HALF_BC]},
    (32, 64, 16): {"float": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_A_MAX_BC]},
    (32, 64, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (32, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (32, 128, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_AB_QUARTER_C, VecStrategy.MAX_AB_MIN_C]},
    (64, 16, 64): {"half": [VecStrategy.GENERIC, VecStrategy.MAX_A_MIN_BC, VecStrategy.MAX_A_QUARTER_BC]},
    (64, 32, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (64, 32, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (64, 64, 16): {"float": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX]},
    (64, 64, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (64, 64, 64): {"half": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX, VecStrategy.MAX_AB_HALF_C, VecStrategy.MAX_AB_QUARTER_C, VecStrategy.HALF_A_MIN_BC], "float": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX, VecStrategy.MAX_A_MIN_BC, VecStrategy.MIN_A_MAX_BC]},
    (64, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (64, 128, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (64, 128, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (128, 16, 64): {"half": [VecStrategy.MAX_A_MIN_BC, VecStrategy.MAX_A_QUARTER_BC]},
    (128, 32, 16): {"float": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_A_MIN_BC]},
    (128, 32, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_AB_QUARTER_C, VecStrategy.MAX_A_QUARTER_BC, VecStrategy.MAX_AB_MIN_C]},
    (128, 64, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (128, 64, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (128, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (128, 128, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (128, 128, 64): {"half": [VecStrategy.MAX_AB_HALF_C, VecStrategy.HALF_UNIFORM]},
    (128, 256, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (128, 256, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (256, 16, 64): {"half": [VecStrategy.MAX_A_MIN_BC, VecStrategy.MAX_A_QUARTER_BC]},
    (256, 32, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (256, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (256, 128, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (256, 256, 32): {"half": [VecStrategy.MAX_AB_HALF_C, VecStrategy.HALF_UNIFORM]},
}


def get_vec_strategies(
    tile_m: int, tile_n: int, tile_k: int, variant: str,
    dtype_class: Optional[str] = None,
) -> List[VecStrategy]:
    """Return curated VecStrategy list for a tile+variant (from profiler JSON).

    Tables are dtype-class-keyed. If ``dtype_class`` ("half"/"float") is given,
    return only that dtype's strategies. If None, return the union across dtype
    classes (preserving order, deduplicated). Falls back to [GENERIC] for tiles
    not in the table.
    """
    table = {
        "forward": _FWD_TILE_STRATEGIES,
        "bwd_data": _BWD_DATA_TILE_STRATEGIES,
        "bwd_weight": _BWD_WEIGHT_TILE_STRATEGIES,
    }.get(variant, {})
    entry = table.get((tile_m, tile_n, tile_k))
    if entry is None:
        return [VecStrategy.GENERIC]
    if dtype_class is not None:
        return entry.get(dtype_class, [])
    # Union across dtype classes, deduplicated, order-preserving.
    out: List[VecStrategy] = []
    seen = set()
    for dc in ("half", "float"):
        for s in entry.get(dc, []):
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


# =============================================================================
# Explicit Per-Tile Vec Overrides (don't fit any strategy family)
# =============================================================================
# A small set of (variant, tile) -> extra vec triples used by the JSON profiler
# configs that no VecStrategy formula produces. These are added on top of the
# strategy-derived vecs (superset semantics). Includes bwd_data's vec_a=16
# (A vectorizes along K, beyond the dtype-class max) and a handful of
# idiosyncratic per-tile triples.

_EXTRA_VEC_TRIPLES: Dict[str, Dict[Tuple[int, int, int], Dict[str, List[Tuple[int, int, int]]]]] = {
    "forward": {
        (32, 64, 64): {"float": [(4, 4, 8)]},
        (32, 128, 64): {"float": [(4, 4, 8)]},
        (64, 64, 32): {"half": [(1, 2, 1), (2, 1, 2)], "float": [(1, 2, 1), (2, 1, 2)]},
        (64, 64, 64): {"half": [(8, 8, 1), (8, 8, 4)]},
        (128, 128, 32): {"float": [(4, 4, 8)]},
        (128, 128, 64): {"half": [(8, 8, 1), (8, 8, 8)], "float": [(4, 4, 8)]},
        (256, 128, 32): {"half": [(2, 2, 2)]},
    },
    "bwd_data": {
        (64, 16, 32): {"float": [(8, 1, 1), (8, 2, 2)]},
        (64, 16, 64): {"half": [(16, 1, 1), (16, 2, 2)]},
        (64, 64, 64): {"half": [(8, 8, 4), (8, 8, 8)]},
        (128, 32, 16): {"half": [(4, 2, 2)]},
        (128, 32, 32): {"float": [(8, 1, 1), (8, 2, 2)]},
        (128, 32, 64): {"half": [(16, 1, 1), (16, 2, 2), (16, 8, 8)]},
        (128, 64, 64): {"half": [(8, 8, 8)]},
        (128, 256, 32): {"half": [(8, 4, 8)]},
        (256, 32, 64): {"half": [(8, 8, 8)]},
        (256, 128, 32): {"half": [(4, 8, 8)]},
        (256, 128, 64): {"half": [(8, 8, 4), (8, 8, 8)]},
    },
    "bwd_weight": {
        (16, 16, 32): {"float": [(1, 1, 2)]},
        (16, 16, 64): {"half": [(1, 4, 4)]},
        (16, 32, 64): {"half": [(1, 2, 4), (1, 4, 4), (2, 1, 1), (2, 2, 4), (2, 4, 4)]},
        (16, 64, 64): {"half": [(2, 1, 1), (2, 8, 4), (4, 8, 8)]},
        (16, 128, 32): {"half": [(4, 4, 1)]},
        (16, 128, 64): {"half": [(2, 8, 4)]},
        (16, 256, 64): {"half": [(2, 8, 4)]},
        (32, 16, 64): {"half": [(1, 2, 2), (2, 1, 1), (2, 2, 2), (4, 2, 2)]},
        (32, 32, 32): {"half": [(2, 2, 1), (2, 2, 2)]},
        (32, 64, 32): {"half": [(2, 2, 1), (2, 8, 8), (4, 4, 1), (4, 4, 2)]},
        (64, 16, 64): {"half": [(1, 2, 2)], "float": [(8, 2, 2)]},
        (64, 32, 32): {"half": [(4, 4, 1), (4, 4, 2)]},
        (64, 32, 64): {"half": [(8, 8, 8)]},
        (64, 64, 32): {"half": [(2, 2, 2)]},
        (64, 64, 64): {"half": [(1, 4, 4), (2, 2, 2), (2, 2, 4), (4, 8, 8)]},
        (128, 32, 32): {"float": [(8, 2, 2)]},
    },
}


def get_extra_vec_triples(
    tile_m: int, tile_n: int, tile_k: int, variant: str,
    dtype_class: Optional[str] = None,
) -> List[Tuple[int, int, int]]:
    """Return explicit per-tile vec triples not produced by any VecStrategy.

    Dtype-class-keyed. If ``dtype_class`` is given, return only that
    dtype's extra triples; if None, return the union across dtype classes.
    Empty for tiles whose vecs are fully covered by strategies.
    """
    per_dc = _EXTRA_VEC_TRIPLES.get(variant, {}).get((tile_m, tile_n, tile_k))
    if not per_dc:
        return []
    if dtype_class is not None:
        return per_dc.get(dtype_class, [])
    out: List[Tuple[int, int, int]] = []
    seen = set()
    for dc in ("half", "float"):
        for tr in per_dc.get(dc, []):
            if tr not in seen:
                seen.add(tr)
                out.append(tr)
    return out


# bwd_weight wavelet tiles that use the legacy gfx942 warp_tile_k formula
# (same as streamk native instances) rather than the new get_k_mfma formula.
# Derived from the CK Builder conf files: these tiles carry warp_tile_k=16 for
# half (warp_tile_m=16) instead of the new formula value of 32.
_BWD_WEIGHT_WAVELET_LEGACY_WARP_K_TILES: Set[Tuple[int, int, int]] = {
    (64, 32, 64),
    (64, 64, 64),
}

# =============================================================================
# Pipeline / Scheduler Rules (per-tile, replaces cross-product)
# =============================================================================

_COMPV4_SET: Set[Tuple[int, int, int]] = set(COMPV4_COMPATIBLE_TILES)


# =============================================================================
# Curated per-tile (pipeline, scheduler) map
# =============================================================================
# Per-variant mapping: tile -> list of (pipeline, scheduler).
# Preferred over the rule-based computation below; tiles absent here fall back to
# the shape-based rules.

_FWD_TILE_PIPELINES: Dict[Tuple[int, int, int], List[Tuple[str, str]]] = {
    (16, 16, 64): [('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 16, 128): [('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 32, 64): [('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 64, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 128, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 256, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (32, 16, 64): [('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (32, 64, 16): [('compv1', 'intrawave')],
    (32, 64, 32): [('compv1', 'intrawave')],
    (32, 64, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (32, 128, 16): [('compv1', 'intrawave')],
    (32, 128, 32): [('compv1', 'intrawave')],
    (32, 128, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (32, 256, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (64, 16, 16): [('compv1', 'intrawave')],
    (64, 16, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (64, 32, 16): [('compv1', 'intrawave')],
    (64, 32, 32): [('compv1', 'intrawave')],
    (64, 32, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (64, 64, 8): [('compv1', 'intrawave')],
    (64, 64, 16): [('compv1', 'intrawave')],
    (64, 64, 32): [('compv1', 'intrawave')],
    (64, 64, 64): [('compv3', 'intrawave')],
    (64, 128, 16): [('compv1', 'intrawave')],
    (64, 128, 32): [('compv1', 'intrawave')],
    (64, 128, 64): [('compv3', 'intrawave')],
    (128, 16, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (128, 32, 16): [('compv1', 'intrawave')],
    (128, 32, 32): [('compv1', 'interwave'), ('compv1', 'intrawave')],
    (128, 32, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (128, 64, 8): [('compv1', 'intrawave')],
    (128, 64, 16): [('compv1', 'intrawave')],
    (128, 64, 32): [('compv1', 'intrawave')],
    (128, 64, 64): [('compv3', 'intrawave')],
    (128, 128, 16): [('compv1', 'intrawave')],
    (128, 128, 32): [('compv1', 'intrawave'), ('compv4', 'intrawave')],
    (128, 128, 64): [('compv1', 'interwave'), ('compv3', 'intrawave'), ('compv4', 'intrawave'), ('compv6', 'intrawave')],
    (128, 192, 16): [('compv1', 'intrawave')],
    (128, 256, 16): [('compv1', 'intrawave')],
    (128, 256, 32): [('compv1', 'interwave'), ('compv1', 'intrawave')],
    (224, 256, 64): [('compv3', 'intrawave')],
    (256, 16, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (256, 32, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (256, 64, 8): [('compv1', 'intrawave')],
    (256, 128, 16): [('compv1', 'intrawave')],
    (256, 128, 32): [('compv1', 'interwave'), ('compv1', 'intrawave')],
    (256, 224, 64): [('compv3', 'intrawave')],
    (256, 256, 32): [('compv3', 'intrawave'), ('compv4', 'intrawave'), ('compv6', 'intrawave')],
}

_BWD_DATA_TILE_PIPELINES: Dict[Tuple[int, int, int], List[Tuple[str, str]]] = {
    (16, 64, 32): [('compv1', 'intrawave')],
    (32, 64, 32): [('compv1', 'intrawave')],
    (32, 128, 32): [('compv1', 'intrawave')],
    (64, 16, 16): [('compv1', 'intrawave')],
    (64, 16, 32): [('compv1', 'intrawave')],
    (64, 16, 64): [('compv1', 'intrawave')],
    (64, 32, 32): [('compv1', 'intrawave')],
    (64, 64, 32): [('compv1', 'intrawave')],
    (64, 64, 64): [('wavelet', 'intrawave')],
    (64, 128, 32): [('compv1', 'intrawave')],
    (128, 32, 16): [('compv1', 'intrawave')],
    (128, 32, 32): [('compv1', 'intrawave')],
    (128, 32, 64): [('compv1', 'intrawave')],
    (128, 64, 32): [('compv1', 'intrawave')],
    (128, 64, 64): [('wavelet', 'intrawave')],
    (128, 128, 32): [('compv1', 'intrawave')],
    (128, 256, 32): [('compv1', 'intrawave')],
    (256, 32, 64): [('wavelet', 'intrawave')],
    (256, 128, 32): [('compv1', 'intrawave'), ('wavelet', 'intrawave')],
    (256, 128, 64): [('wavelet', 'intrawave')],
}

_BWD_WEIGHT_TILE_PIPELINES: Dict[Tuple[int, int, int], List[Tuple[str, str]]] = {
    (16, 16, 32): [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')],
    (16, 16, 64): [('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 32, 64): [('basic_async_v1', 'intrawave'), ('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 64, 64): [('basic_async_v1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 128, 32): [('compv1', 'intrawave')],
    (16, 128, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 256, 32): [('compv1', 'intrawave')],
    (16, 256, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (32, 16, 64): [('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (32, 32, 32): [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')],
    (32, 64, 16): [('compv1', 'intrawave')],
    (32, 64, 32): [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')],
    (32, 128, 16): [('compv1', 'intrawave')],
    (32, 128, 32): [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')],
    (64, 16, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (64, 32, 16): [('compv1', 'intrawave')],
    (64, 32, 32): [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')],
    (64, 64, 16): [('compv1', 'intrawave')],
    (64, 64, 32): [('compv1', 'intrawave')],
    (64, 64, 64): [('basic_async_v1', 'intrawave'), ('compv1', 'intrawave')],
    (64, 128, 16): [('compv1', 'intrawave')],
    (64, 128, 32): [('compv1', 'intrawave')],
    (64, 128, 64): [('basic_async_v1', 'intrawave')],
    (128, 16, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (128, 32, 16): [('compv1', 'intrawave')],
    (128, 32, 32): [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')],
    (128, 64, 16): [('compv1', 'intrawave')],
    (128, 64, 32): [('compv1', 'intrawave')],
    (128, 128, 16): [('compv1', 'intrawave')],
    (128, 128, 32): [('compv1', 'intrawave')],
    (128, 128, 64): [('compv1', 'interwave'), ('compv3', 'intrawave'), ('compv4', 'intrawave'), ('compv6', 'intrawave')],
    (128, 256, 16): [('compv1', 'intrawave')],
    (128, 256, 32): [('compv1', 'intrawave')],
    (256, 16, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (256, 32, 64): [('basic_async_v1', 'intrawave')],
    (256, 128, 16): [('compv1', 'intrawave')],
    (256, 128, 32): [('compv1', 'intrawave')],
    (256, 256, 32): [('compv3', 'intrawave'), ('compv4', 'intrawave'), ('compv6', 'intrawave')],
}


# Dtype-class-keyed pipeline overrides for tiles whose (pipeline, scheduler)
# set differs between half (bf16/fp16) and float (fp32). Consulted first when a
# dtype_class is provided. Tiles used by only one dtype-class are handled by the
# ndim+dtype tile tables, so only tiles present in BOTH classes appear here.
# half = union(bf16, fp16) pipes; float = fp32 pipes (from profiler JSON).
_BWD_WEIGHT_TILE_PIPELINES_DCLASS: Dict[Tuple[int, int, int], Dict[str, List[Tuple[str, str]]]] = {
    (16, 16, 32): {'half': [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')], 'float': [('compv6', 'intrawave'), ('mem', 'intrawave')]},
    (16, 32, 64): {'half': [('basic_async_v1', 'intrawave'), ('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')], 'float': [('compv1', 'intrawave')]},
    (16, 64, 64): {'half': [('basic_async_v1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave'), ('wavelet', 'intrawave')], 'float': []},
    (32, 16, 64): {'half': [('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')], 'float': [('compv1', 'intrawave')]},
    (64, 16, 64): {'half': [('mem', 'interwave'), ('mem', 'intrawave')], 'float': [('mem', 'intrawave')]},
    (64, 32, 64): {'half': [('wavelet', 'intrawave')], 'float': []},
    (64, 64, 64): {'half': [('basic_async_v1', 'intrawave'), ('compv1', 'intrawave'), ('wavelet', 'intrawave')], 'float': [('compv1', 'intrawave')]},
    (128, 32, 32): {'half': [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')], 'float': [('compv1', 'intrawave')]},
}

_FWD_TILE_PIPELINES_DCLASS: Dict[Tuple[int, int, int], Dict[str, List[Tuple[str, str]]]] = {
    (64, 64, 64): {'half': [('compv3', 'intrawave'), ('wavelet', 'intrawave')], 'float': [('compv3', 'intrawave')]},
    (128, 128, 32): {'half': [('compv1', 'intrawave'), ('compv4', 'intrawave')], 'float': [('compv4', 'intrawave')]},
    (128, 128, 64): {'half': [('compv1', 'interwave'), ('compv3', 'intrawave'), ('compv4', 'intrawave'), ('compv6', 'intrawave'), ('wavelet', 'intrawave')], 'float': [('compv1', 'interwave'), ('compv3', 'intrawave'), ('compv6', 'intrawave')]},
}


def get_pipelines_for_tile(
    tile_m: int, tile_n: int, tile_k: int, variant: str,
    dtype_class: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Return list of (pipeline, scheduler) pairs for a tile shape and variant.

    When ``dtype_class`` ("half"/"float") is given and a dtype-class override
    exists for this (variant, tile), that override is returned (trims half-only
    pipelines from fp32 and vice versa). Otherwise prefers the curated per-tile
    map (from profiler JSON) when the tile is present, falling back to the
    shape-based rules below.
    """
    tile_key = (tile_m, tile_n, tile_k)

    if dtype_class is not None:
        dclass_table = {
            "forward": _FWD_TILE_PIPELINES_DCLASS,
            "bwd_weight": _BWD_WEIGHT_TILE_PIPELINES_DCLASS,
        }.get(variant, {})
        override = dclass_table.get(tile_key)
        if override is not None and dtype_class in override:
            return list(override[dtype_class])

    curated = {
        "forward": _FWD_TILE_PIPELINES,
        "bwd_data": _BWD_DATA_TILE_PIPELINES,
        "bwd_weight": _BWD_WEIGHT_TILE_PIPELINES,
    }.get(variant, {})
    if tile_key in curated:
        return list(curated[tile_key])

    tile_area = tile_m * tile_n
    min_dim = min(tile_m, tile_n)

    if variant == "forward":
        pipes: List[Tuple[str, str]] = [("compv1", "intrawave")]
        if tile_k >= 32:
            pipes.append(("compv1", "interwave"))
        if min_dim <= 32 and tile_k >= 64:
            pipes.append(("mem", "intrawave"))
            pipes.append(("mem", "interwave"))
        if tile_area >= 4096 and tile_k >= 32:
            pipes.append(("compv3", "intrawave"))
        if tile_key in _COMPV4_SET or (tile_area >= 16384 and tile_k >= 32):
            pipes.append(("compv4", "intrawave"))
        if tile_area >= 4096 and tile_k >= 32:
            pipes.append(("compv6", "intrawave"))
        return pipes

    elif variant == "bwd_data":
        return [("compv1", "intrawave")]

    elif variant == "bwd_weight":
        pipes = [("compv1", "intrawave")]
        if tile_k >= 64:
            pipes.append(("compv1", "interwave"))
        if tile_k >= 32:
            pipes.append(("mem", "intrawave"))
        if tile_k >= 64:
            pipes.append(("mem", "interwave"))
        if tile_area >= 4096 and tile_k >= 32:
            pipes.append(("compv3", "intrawave"))
        if tile_key in _COMPV4_SET or (tile_area >= 16384 and tile_k >= 32):
            pipes.append(("compv4", "intrawave"))
        if tile_k >= 32:
            pipes.append(("compv6", "intrawave"))
        if tile_k == 64:
            pipes.append(("basic_async_v1", "intrawave"))
        return pipes

    return [("compv1", "intrawave")]


# =============================================================================
# Specialization Rules (per-tile, replaces cross-product)
# =============================================================================


def get_specs_for_tile(
    tile_m: int, tile_n: int, tile_k: int, variant: str,
) -> List[str]:
    """Return list of specialization strings for a tile shape and variant.

    Rule-based specialization assignment — assigns only the specializations
    that make sense for the given tile dimensions.
    """
    if variant == "forward":
        if tile_k == 8:
            return ["default"]
        # tile_k >= 16
        specs = ["default", "filter1x1_pad0", "filter1x1_stride1_pad0"]
        if tile_m * tile_n <= 4096:
            specs.append("filter3x3")
        return specs
    elif variant in ("bwd_data", "bwd_weight"):
        return ["default", "filter1x1_stride1_pad0"]

    return ["default"]


# =============================================================================
# Pipeline / Scheduler Rules (variant-level, kept for backward compat)
# =============================================================================

# Valid (pipeline, scheduler) pairs per variant
# Derived from JSON profiler configs (union of bf16 + fp32).
VARIANT_PIPELINE_SCHEDULER: Dict[str, List[Tuple[str, str]]] = {
    "forward": [
        ("compv1", "intrawave"),
        ("compv1", "interwave"),
        ("compv3", "intrawave"),
        ("compv4", "intrawave"),
        ("compv6", "intrawave"),
        ("mem", "intrawave"),
        ("mem", "interwave"),
    ],
    "bwd_data": [
        ("compv1", "intrawave"),
    ],
    "bwd_weight": [
        ("compv1", "intrawave"),
        ("compv1", "interwave"),
        ("compv3", "intrawave"),
        ("compv4", "intrawave"),
        ("compv6", "intrawave"),
        ("mem", "intrawave"),
        ("mem", "interwave"),
        ("basic_async_v1", "intrawave"),
    ],
}

# Specializations per variant
VARIANT_SPECIALIZATIONS: Dict[str, List[str]] = {
    "forward":     ["default", "filter1x1_pad0", "filter1x1_stride1_pad0", "filter3x3"],
    "bwd_data":    ["default", "filter1x1_stride1_pad0"],
    "bwd_weight":  ["default", "filter1x1_stride1_pad0"],
}

# =============================================================================
# Feature Flag Rules (Phase 5)
# =============================================================================


@dataclass
class StreamKSpec:
    """StreamK parameters for a feature config."""
    strategy: str = "TREE"          # "TREE" | "LINEAR"
    persistent: bool = False        # non-persistent by default


@dataclass
class FeatureSpec:
    """A single feature-flag variant rule.

    Fields that are False/1/None represent 'off'.
    tile_override / pipeline_override = None means use variant defaults.
    """
    split_image: bool = False
    explicit_gemm: bool = False
    two_stage: bool = False
    double_smem_buffer: bool = False
    num_groups_to_merge: int = 1
    streamk_config: Optional[StreamKSpec] = None

    # Optional overrides (None = use variant defaults)
    tile_override: Optional[List[Tuple[int, int, int]]] = None
    pipeline_override: Optional[List[Tuple[str, str]]] = None

    # Optional restrictions (None = applies to all). When set, the feature is
    # only emitted for the listed dtype classes ("half"/"float") and/or ndims.
    dtype_classes: Optional[List[str]] = None
    ndims: Optional[List[int]] = None


# Tiles used per feature (derived from JSON profiler analysis).
# Restricting features to specific tiles prevents config explosion.
_FWD_GM_TILES: List[Tuple[int, int, int]] = [
    (64, 16, 16), (128, 32, 32),
]
_BWD_EG_TILES: List[Tuple[int, int, int]] = [
    (16, 16, 64), (16, 32, 64), (16, 64, 64), (16, 128, 64), (16, 256, 64),
    (32, 16, 64), (64, 16, 64), (128, 16, 64), (128, 128, 64),
    (256, 16, 64), (256, 256, 32),
]
_BWD_EG_2S_TILES: List[Tuple[int, int, int]] = [
    (16, 16, 64), (16, 32, 64), (16, 64, 64),
    (32, 16, 64), (64, 16, 64), (128, 16, 64), (256, 16, 64),
]
_BWD_2S_TILES: List[Tuple[int, int, int]] = [
    (16, 16, 32), (64, 64, 64),
]
_BWD_GM2_2S_TILES: List[Tuple[int, int, int]] = [
    (32, 32, 32), (32, 64, 32),
]
_BWD_GM4_2S_TILES: List[Tuple[int, int, int]] = [
    (16, 128, 32), (32, 64, 32), (64, 32, 32),
]
_BWD_GM8_2S_TILES: List[Tuple[int, int, int]] = [
    (16, 256, 32), (32, 128, 32), (128, 32, 32),
]
_BWD_SK_TILES: List[Tuple[int, int, int]] = [
    (16, 16, 32), (16, 32, 64), (32, 16, 64),
    (64, 16, 64), (64, 64, 64), (128, 32, 32),
]

VARIANT_FEATURES: Dict[str, List[FeatureSpec]] = {
    "forward": [
        # split_image: small tile subset, compv1/intrawave only
        FeatureSpec(
            split_image=True,
            tile_override=_SPLIT_IMAGE_TILES,
            pipeline_override=[("compv1", "intrawave")],
        ),
        # num_groups_to_merge (restricted to tiles that benefit from merging)
        FeatureSpec(num_groups_to_merge=8,  tile_override=_FWD_GM_TILES),
        FeatureSpec(num_groups_to_merge=16, tile_override=[(64, 16, 16)]),
        FeatureSpec(num_groups_to_merge=32, tile_override=[(64, 16, 16)]),
    ],
    "bwd_data": [
        # num_groups_to_merge=32 wavelet instance for the 256x32x64 tile
        FeatureSpec(
            num_groups_to_merge=32,
            tile_override=[(256, 32, 64)],
            pipeline_override=[("wavelet", "intrawave")],
            dtype_classes=["half"],
        ),
    ],
    "bwd_weight": [
        # explicit_gemm / two_stage / num_groups_to_merge are half-only:
        # the fp32 bwd_weight profiler JSON contains none of these features.
        # explicit_gemm only: tiles with tile_k=64 (larger internal GEMM)
        FeatureSpec(explicit_gemm=True, tile_override=_BWD_EG_TILES, dtype_classes=["half"]),
        # two_stage only
        FeatureSpec(two_stage=True, tile_override=_BWD_2S_TILES, dtype_classes=["half"]),
        # two_stage + explicit_gemm
        FeatureSpec(two_stage=True, explicit_gemm=True, tile_override=_BWD_EG_2S_TILES, dtype_classes=["half"]),
        # num_groups_to_merge + two_stage combinations
        FeatureSpec(num_groups_to_merge=2, two_stage=True, tile_override=_BWD_GM2_2S_TILES, dtype_classes=["half"]),
        FeatureSpec(num_groups_to_merge=4, two_stage=True, tile_override=_BWD_GM4_2S_TILES, dtype_classes=["half"]),
        FeatureSpec(num_groups_to_merge=8, two_stage=True, tile_override=_BWD_GM8_2S_TILES, dtype_classes=["half"]),
        # basic_async_v1 + num_groups_to_merge=2
        FeatureSpec(
            num_groups_to_merge=2,
            tile_override=[(16, 32, 64), (16, 64, 64), (64, 128, 64)],
            pipeline_override=[("basic_async_v1", "intrawave")],
            dtype_classes=["half"],
        ),
        # StreamK non-persistent
        FeatureSpec(
            streamk_config=StreamKSpec(strategy="TREE", persistent=False),
            tile_override=_BWD_SK_TILES,
            pipeline_override=[
                ("compv1", "intrawave"),
                ("mem", "intrawave"),
                ("basic_async_v1", "intrawave"),
            ],
        ),
        # StreamK persistent
        FeatureSpec(
            streamk_config=StreamKSpec(strategy="TREE", persistent=True),
            tile_override=_BWD_SK_TILES,
            pipeline_override=[
                ("compv1", "intrawave"),
                ("mem", "intrawave"),
                ("basic_async_v1", "intrawave"),
            ],
        ),
    ],
}

# =============================================================================
# Pipeline Variant Suffixes (single source of truth — kept for backward compat)
# =============================================================================
# Empirically verified valid (pipeline, wave_mode, has_dsb, has_si) combinations
# observed in the 2D and 3D bf16 gfx950 benchmark CSVs. 30 entries total per ndim.
# Each tuple: (pipeline, wave_mode, has_dsb, has_si)
#   wave_mode: "intrawave" | "interwave"
#   has_dsb:   1 if "_dsb" suffix present (double smem buffer), else 0
#   has_si:    1 if "_si"  suffix present (store immediate),    else 0
PIPELINE_VARIANTS: List[Tuple[str, str, int, int]] = [
    # basic_v1: both intra/inter × {∅, dsb, si, dsb_si} = 8 combos
    ("basic_v1", "intrawave", 0, 0),
    ("basic_v1", "intrawave", 1, 0),
    ("basic_v1", "intrawave", 0, 1),
    ("basic_v1", "intrawave", 1, 1),
    ("basic_v1", "interwave", 0, 0),
    ("basic_v1", "interwave", 1, 0),
    ("basic_v1", "interwave", 0, 1),
    ("basic_v1", "interwave", 1, 1),
    # compv3: intrawave × {∅, dsb, si, dsb_si} = 4 combos
    ("compv3", "intrawave", 0, 0),
    ("compv3", "intrawave", 1, 0),
    ("compv3", "intrawave", 0, 1),
    ("compv3", "intrawave", 1, 1),
    # compv4: intrawave × {dsb, dsb_si} only = 2 combos
    ("compv4", "intrawave", 1, 0),
    ("compv4", "intrawave", 1, 1),
    # compv5: intrawave × {∅, dsb, si, dsb_si} = 4 combos
    ("compv5", "intrawave", 0, 0),
    ("compv5", "intrawave", 1, 0),
    ("compv5", "intrawave", 0, 1),
    ("compv5", "intrawave", 1, 1),
    # compv6: intrawave × {∅, dsb, si, dsb_si} = 4 combos
    ("compv6", "intrawave", 0, 0),
    ("compv6", "intrawave", 1, 0),
    ("compv6", "intrawave", 0, 1),
    ("compv6", "intrawave", 1, 1),
    # mem: both intra/inter × {∅, dsb, si, dsb_si} = 8 combos
    ("mem", "intrawave", 0, 0),
    ("mem", "intrawave", 1, 0),
    ("mem", "intrawave", 0, 1),
    ("mem", "intrawave", 1, 1),
    ("mem", "interwave", 0, 0),
    ("mem", "interwave", 1, 0),
    ("mem", "interwave", 0, 1),
    ("mem", "interwave", 1, 1),
]


def iter_pipeline_variants(pipelines: List[str] = None):
    """Iterate (pipeline, wave_mode, has_dsb, has_si) tuples, optionally filtered.

    Args:
        pipelines: optional list of pipeline names to keep. If None, yield all.
    """
    if pipelines is None:
        for entry in PIPELINE_VARIANTS:
            yield entry
        return
    keep = set(pipelines)
    for entry in PIPELINE_VARIANTS:
        if entry[0] in keep:
            yield entry


# Valid pipelines per variant (kept for backward compat; full list)
VARIANT_PIPELINES: Dict[str, List[str]] = {
    "forward": [
        "basic_v1",
        "mem",
        "compv3",
        "compv4",
        "compv5",
        "compv6",
        "comp_async",
        "basic_async_v1",
    ],
    "bwd_data": [
        "basic_v1",
        "mem",
        "compv3",
        "compv4",
        "compv5",
        "compv6",
        "comp_async",
        "basic_async_v1",
    ],
    "bwd_weight": [
        "basic_v1",
        "mem",
        "compv3",
        "compv4",
        "compv5",
        "compv6",
        "comp_async",
        "basic_async_v1",
    ],
}

# =============================================================================
# Shared Validation Rules
# =============================================================================
# These functions are the single source of truth for validation rules
# for convolution code generation.

# --- Vector size validation ---


def is_valid_vector_size(vec: int) -> bool:
    """AMD GPUs only support vector widths 1, 2, 4, 8, 16."""
    return vec == 1 or vec % 2 == 0


def check_vectors(vec_a: int, vec_b: int, vec_c: int) -> bool:
    """Check all three vector sizes are valid (1 or even)."""
    return all(is_valid_vector_size(v) for v in (vec_a, vec_b, vec_c))



# --- Pipeline-variant restrictions ---

UNSUPPORTED_VARIANT_PIPELINES = {
    "bwd_weight": {"compv5"},
    "bwd_data": {"compv5"},
}


def is_valid_pipeline_for_variant(pipeline: str, variant: str) -> bool:
    """Check pipeline is supported for the given conv variant.

    Backward weight and backward data reject compv5 due to transpose_tile2d /
    get_length issues.
    """
    blocked = UNSUPPORTED_VARIANT_PIPELINES.get(variant, set())
    return pipeline not in blocked


# --- Stream-K restrictions ---


def is_streamk_valid_for_variant(variant: str) -> bool:
    """Stream-K is only supported for backward weight."""
    return variant == "bwd_weight"


# =============================================================================
# Depthwise Convolution Parameter Space
# =============================================================================

DEPTHWISE_TILE_SIZES: List[Tuple[int, int]] = [
    (8, 8), (14, 28), (16, 16), (28, 28), (32, 32),
]

DEPTHWISE_FILTER_SIZES: List[int] = [3, 5]

DEPTHWISE_STRIDES: List[Tuple[int, int]] = [(1, 1), (2, 2)]

# Curated depthwise configs matching the JSON profiler set.
# Each tuple: (tile_h, tile_w, filt, str_h, str_w, sub_h, sub_w, nbatch, in_vec, out_vec)
# Padding is derived: pad = (filt - 1) // 2.
# Validated by is_valid_depthwise_config() at module load time.
DEPTHWISE_PARAMS: List[Tuple[int, ...]] = [
    # Filter 3, Stride (1,1)
    (8,  8,  3, 1, 1, 2, 2, 8, 2, 2),
    (16, 16, 3, 1, 1, 1, 4, 8, 8, 8),
    (16, 16, 3, 1, 1, 2, 2, 1, 2, 2),
    (28, 28, 3, 1, 1, 4, 4, 1, 8, 8),
    (32, 32, 3, 1, 1, 4, 4, 1, 8, 8),
    # Filter 3, Stride (2,2)
    (14, 28, 3, 2, 2, 2, 4, 1, 8, 8),
    (16, 16, 3, 2, 2, 1, 4, 1, 8, 8),
    (16, 16, 3, 2, 2, 1, 4, 2, 8, 8),
    (16, 16, 3, 2, 2, 2, 2, 1, 2, 2),
    (16, 16, 3, 2, 2, 2, 2, 1, 8, 8),
    (32, 32, 3, 2, 2, 2, 8, 1, 8, 8),
    (32, 32, 3, 2, 2, 4, 4, 1, 4, 4),
    (32, 32, 3, 2, 2, 4, 4, 1, 8, 8),
    (32, 32, 3, 2, 2, 4, 4, 2, 8, 8),
    # Filter 5, Stride (1,1)
    (8,  8,  5, 1, 1, 1, 1, 1, 1, 1),
    (8,  8,  5, 1, 1, 2, 2, 8, 2, 2),
    (16, 16, 5, 1, 1, 1, 4, 1, 8, 8),
    (16, 16, 5, 1, 1, 1, 4, 8, 8, 8),
    (28, 28, 5, 1, 1, 4, 4, 8, 8, 8),
    (32, 32, 5, 1, 1, 4, 4, 4, 8, 8),
]


def get_depthwise_configs():
    """Get curated depthwise convolution configurations.

    Returns the profiler config set, with each entry validated by
    tile_math.is_valid_depthwise_config().

    Returns:
        List of tile_math.DepthwiseConfig objects.
    """
    from .tile_math import DepthwiseConfig, is_valid_depthwise_config

    configs = []
    for params in DEPTHWISE_PARAMS:
        th, tw, filt, sh, sw, sub_h, sub_w, nb, iv, ov = params
        pad = (filt - 1) // 2
        cfg = DepthwiseConfig(th, tw, filt, sh, sw, pad, pad, nb, sub_h, sub_w, iv, ov)
        assert is_valid_depthwise_config(cfg), f"Invalid depthwise config: {params}"
        configs.append(cfg)
    return configs


log = logging.getLogger(__name__)


def get_configs(
    arch: str,
    variants: List,
    ndims: List[int],
    datatypes: List[str],
) -> List:
    """Build all available configs for the "profiler" (JSON-derived) rule set.

    Unified rule-set entry point used by
    ``unified_grouped_conv_codegen.get_default_configs``. Iterates over all tile
    shapes per (variant, ndim, datatype), selecting wave/warp pairs, vector
    sizes, pipelines, and specializations via this module's rule helpers; each
    emitted config is tagged with its concrete ``datatype``.
    """
    from unified_grouped_conv_codegen import (
        GroupedConvVariant,
        GroupedConvTraitConfig,
        GroupedConvKernelConfig,
        TileConfig,
        StreamKConfig,
        StreamKReductionStrategy,
        DepthwiseConvKernelConfig,
    )

    seen: set = set()
    configs: List[Union[GroupedConvKernelConfig, DepthwiseConvKernelConfig]] = []

    def _add_config(config: "GroupedConvKernelConfig") -> None:
        """Deduplicate and add config if arch-valid."""
        if not config.is_valid_for_arch():
            return
        key = (
            config.variant,
            config.ndim_spatial,
            config.tile.tile_m, config.tile.tile_n, config.tile.tile_k,
            config.tile.warp_m, config.tile.warp_n, config.tile.warp_k,
            config.tile.warp_tile_m, config.tile.warp_tile_n, config.tile.warp_tile_k,
            config.trait.pipeline, config.trait.scheduler, config.trait.epilogue,
            config.vector_size_a, config.vector_size_b, config.vector_size_c,
            config.trait.double_smem_buffer, config.trait.two_stage,
            config.trait.explicit_gemm, config.trait.split_image,
            config.trait.num_groups_to_merge, config.trait.specialization,
            getattr(config.trait.streamk_config, "streamk_enabled", False),
            getattr(config.trait.streamk_config, "streamk_persistent", False),
            config.datatype,
        )
        if key in seen:
            return
        seen.add(key)
        configs.append(config)

    def _make_streamk_config(spec_sk) -> "StreamKConfig":
        """Convert a StreamKSpec from this module into a StreamKConfig."""
        return StreamKConfig(
            streamk_enabled=True,
            strategy=StreamKReductionStrategy.TREE if spec_sk.strategy == "TREE"
                     else StreamKReductionStrategy.LINEAR,
            streamk_persistent=spec_sk.persistent,
        )

    def _emit_configs_for_tile(
        tile_m: int, tile_n: int, tile_k: int,
        pipelines: List[Tuple[str, str]],
        specializations: List[str],
        variant,
        ndim: int,
        datatype: str,
        dtype_class: str,
        dtype_key: str,
        feat: Optional["FeatureSpec"] = None,
    ) -> None:
        """Emit all valid configs for a single tile shape + pipeline list, for
        one concrete datatype. Wave/warp pairs and vec triples are selected for
        this dtype only; each emitted config is tagged with ``datatype``."""
        
        var_str = {
            GroupedConvVariant.FORWARD: "forward",
            GroupedConvVariant.BACKWARD_DATA: "bwd_data",
            GroupedConvVariant.BACKWARD_WEIGHT: "bwd_weight",
        }.get(variant, "forward")

        pairs = get_wave_warp_pairs(tile_m, tile_n, tile_k, var_str, dtype_key, arch)
        if not pairs:
            return

        # Candidate vec triples for this dtype class (strategies + long-tail
        # extras). Gated per (wave, warp) below against tile_math, the authority.
        vec_set: set = set()
        for strategy in get_vec_strategies(tile_m, tile_n, tile_k, var_str, dtype_class):
            vec_set.add(compute_vector_size(strategy, dtype_class))
        for triple in get_extra_vec_triples(tile_m, tile_n, tile_k, var_str, dtype_class):
            vec_set.add(triple)
        vec_list = sorted(vec_set)

        # bwd_weight wavelet tiles that use legacy warp_tile_k (gfx942 convention)
        _bww_legacy = (
            var_str == "bwd_weight"
            and (tile_m, tile_n, tile_k) in _BWD_WEIGHT_WAVELET_LEGACY_WARP_K_TILES
        )

        for (wave_m, wave_n, wave_k), (warp_tile_m, warp_tile_n) in pairs:
            # Derive warp_tile_k (the MFMA K) for the non-streamk case; this is
            # the value used both for the vec gate and for non-streamk configs.
            # StreamK configs re-derive it per-config below.
            # Some bwd_weight wavelet tiles use the legacy gfx942 formula.
            warp_tile_k = compute_warp_tile_k(
                dtype_key, warp_tile_m, tile_k, streamk=False, use_legacy=_bww_legacy,
            )
            # tile_math gate: a candidate vec triple survives only if it is valid
            # for this (tile, wave, warp) under this dtype_key.
            tm_valid = get_all_valid_vector_sizes(
                tile_m, tile_n, tile_k,
                wave_m, wave_n, wave_k,
                warp_tile_m, warp_tile_n, warp_tile_k,
                dtype_key,
            )
            gated_vecs = []
            for triple in vec_list:
                if triple in tm_valid:
                    gated_vecs.append(triple)
                else:
                    log.debug(
                        "tile_math rejects vec %s for tile=(%d,%d,%d) "
                        "wave=(%d,%d,%d) warp=(%d,%d,%d)",
                        triple, tile_m, tile_n, tile_k,
                        wave_m, wave_n, wave_k,
                        warp_tile_m, warp_tile_n, warp_tile_k,
                    )

            for vec_a, vec_b, vec_c in gated_vecs:
                for pipeline, scheduler in pipelines:
                    # The wavelet pipeline uses its own direct kernel path and
                    # does not implement the UniversalGemmKernel interface that
                    # the explicit_gemm path routes through (its operator() takes
                    # 4 arguments, not the 6 UniversalGemmKernel::RunGemm passes).
                    # The two are therefore incompatible.
                    if pipeline == "wavelet" and feat and feat.explicit_gemm:
                        continue

                    # KNOWN LIMITATION (bwd_weight, half precision):
                    # The interwave pipelines -- "compv1" (runtime BASIC_V1) and
                    # "mem" (runtime MEMORY) -- produce INCORRECT results for the
                    # bwd_weight Default specialization when the macro tile is
                    # split across multiple waves (wave_m * wave_n > 1) in fp16/bf16
                    # on grouped convolutions with a small per-group GEMM-M
                    # (per-group K). Observed failing tiles: 64x64x64 (2,2,1),
                    # 64x16x64 (2,1,1), 16x32x64 (1,2,1) -- ~50% of the weight
                    # output (whole groups) comes back zero.
                    #
                    # This problems shows up when the CK Tile convolution integration
                    # tests are executed against the instances generated with "full" rule set.
                    # Rule set "tests" doesn't create the problematic instances.

                    # compv4 forces double_smem_buffer
                    dsb = (pipeline == "compv4") or (feat.double_smem_buffer if feat else False)

                    for spec in specializations:
                        # Build StreamKConfig
                        if feat and feat.streamk_config:
                            sk_cfg = _make_streamk_config(feat.streamk_config)
                        else:
                            sk_cfg = StreamKConfig()  # disabled by default

                        trait = GroupedConvTraitConfig(
                            pipeline=pipeline,
                            scheduler=scheduler,
                            epilogue="cshuffle",
                            double_smem_buffer=dsb,
                            pad_m=True,
                            pad_n=True,
                            pad_k=True,
                            two_stage=(feat.two_stage if feat else False),
                            explicit_gemm=(feat.explicit_gemm if feat else False),
                            split_image=(feat.split_image if feat else False),
                            num_groups_to_merge=(feat.num_groups_to_merge if feat else 1),
                            specialization=spec,
                            streamk_config=sk_cfg,
                        )

                        if not trait.is_valid():
                            continue

                        # Derive warp_tile_k for this config. Hand-written StreamK
                        # native instances bypass get_k_mfma and carry the legacy
                        # warp_tile_k, handled inside compute_warp_tile_k.
                        # Same legacy convention applies to some bwd_weight wavelet tiles.
                        eff_warp_tile_k = compute_warp_tile_k(
                            dtype_key, warp_tile_m, tile_k,
                            streamk=sk_cfg.streamk_enabled,
                            use_legacy=_bww_legacy and pipeline == "wavelet",
                        )

                        tile_cfg = TileConfig(
                            tile_m=tile_m,
                            tile_n=tile_n,
                            tile_k=tile_k,
                            warp_m=wave_m,
                            warp_n=wave_n,
                            warp_k=wave_k,
                            warp_tile_m=warp_tile_m,
                            warp_tile_n=warp_tile_n,
                            warp_tile_k=eff_warp_tile_k,
                        )

                        if not tile_cfg.is_valid():
                            continue

                        config = GroupedConvKernelConfig(
                            tile=tile_cfg,
                            trait=trait,
                            variant=variant,
                            ndim_spatial=ndim,
                            arch=arch,
                            vector_size_a=vec_a,
                            vector_size_b=vec_b,
                            vector_size_c=vec_c,
                            datatype=datatype,
                        )
                        _add_config(config)

    for variant in variants:
        # FORWARD_DEPTHWISE has no GEMM tile loop; its configs are emitted by the
        # depthwise block below (tied to FORWARD). Skip the GEMM loop for it.
        variant_str = {
            GroupedConvVariant.FORWARD: "forward",
            GroupedConvVariant.BACKWARD_DATA: "bwd_data",
            GroupedConvVariant.BACKWARD_WEIGHT: "bwd_weight",
            GroupedConvVariant.FORWARD_DEPTHWISE: None,
        }.get(variant)

        if variant_str is not None:
            features = VARIANT_FEATURES.get(variant_str, [])

            for ndim in ndims:
                for datatype in datatypes:
                    dtype_key = DTYPE_TO_DTYPE_KEY.get(datatype)
                    if dtype_key is None:
                        continue
                    dclass = "float" if datatype == "fp32" else "half"

                    # Tiles for this exact (variant, ndim, dtype).
                    base_tiles = get_tiles(variant_str, ndim, datatype)

                    # --- Base configs (no feature flags) ---
                    for tile_m, tile_n, tile_k in base_tiles:
                        tile_pipelines = get_pipelines_for_tile(
                            tile_m, tile_n, tile_k, variant_str, dclass
                        )
                        tile_specs = get_specs_for_tile(tile_m, tile_n, tile_k, variant_str)
                        _emit_configs_for_tile(
                            tile_m, tile_n, tile_k,
                            tile_pipelines, tile_specs,
                            variant, ndim,
                            datatype, dclass, dtype_key,
                            feat=None,
                        )

                    # --- Feature-flag configs ---
                    for feat in features:
                        if feat.dtype_classes is not None and dclass not in feat.dtype_classes:
                            continue
                        if feat.ndims is not None and ndim not in feat.ndims:
                            continue
                        feat_tiles = feat.tile_override if feat.tile_override is not None else base_tiles
                        for tile_m, tile_n, tile_k in feat_tiles:
                            if feat.pipeline_override is not None:
                                feat_pipes = feat.pipeline_override
                            else:
                                feat_pipes = get_pipelines_for_tile(
                                    tile_m, tile_n, tile_k, variant_str, dclass
                                )
                            tile_specs = get_specs_for_tile(tile_m, tile_n, tile_k, variant_str)
                            _emit_configs_for_tile(
                                tile_m, tile_n, tile_k,
                                feat_pipes, tile_specs,
                                variant, ndim,
                                datatype, dclass, dtype_key,
                                feat=feat,
                            )

        # --- Depthwise configs (forward only, 2D only) ---
        if variant == GroupedConvVariant.FORWARD and 2 in ndims:
            dw_seen: set = set()
            for dw_cfg in get_depthwise_configs():
                for dt in (datatypes or ["fp16"]):
                    dw_key = (dw_cfg.tile_h, dw_cfg.tile_w, dw_cfg.filt,
                              dw_cfg.str_h, dw_cfg.str_w, dw_cfg.pad_h,
                              dw_cfg.pad_w, dw_cfg.nbatch, dw_cfg.sub_h,
                              dw_cfg.sub_w, dw_cfg.in_vec, dw_cfg.out_vec, dt)
                    if dw_key in dw_seen:
                        continue
                    dw_seen.add(dw_key)
                    configs.append(DepthwiseConvKernelConfig(
                        tile_h=dw_cfg.tile_h, tile_w=dw_cfg.tile_w,
                        filt=dw_cfg.filt,
                        str_h=dw_cfg.str_h, str_w=dw_cfg.str_w,
                        pad_h=dw_cfg.pad_h, pad_w=dw_cfg.pad_w,
                        nbatch=dw_cfg.nbatch,
                        sub_h=dw_cfg.sub_h, sub_w=dw_cfg.sub_w,
                        in_vec=dw_cfg.in_vec, out_vec=dw_cfg.out_vec,
                        ndim_spatial=2,
                        arch=arch,
                        layout="ngchw",
                        datatype=dt,
                    ))

    return configs


if __name__ == "__main__":
    all_tiles = sorted(
        set(get_tiles_for_variant("forward"))
        | set(get_tiles_for_variant("bwd_data"))
        | set(get_tiles_for_variant("bwd_weight"))
    )
    print(f"Total unique tiles: {len(all_tiles)}")
    for variant in ("forward", "bwd_data", "bwd_weight"):
        print(f"  {variant}: {len(get_tiles_for_variant(variant))}")
