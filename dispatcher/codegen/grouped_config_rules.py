#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Single Source of Truth for Grouped Convolution Tile Configurations

This module defines all valid tile configurations for grouped convolution kernels.
Both codegen and instance_builder import from here to ensure consistency.

Architecture:
  grouped_conv_tile_configs.py  (SOURCE OF TRUTH)
      ├── Used by unified_grouped_conv_codegen.py
      └── Used by grouped_conv_instance_builder.py
"""

from typing import Dict, List, Tuple

# =============================================================================
# Tile Configurations (Single Source of Truth)
# =============================================================================

# Common tile configurations used across variants
# Format: (tile_m, tile_n, tile_k)
# CRITICAL: tile_m MUST equal wave_m × warp_tile_m (TileGemmShape constraint)
# Only tiles that successfully compile are included
COMMON_TILES: List[Tuple[int, int, int]] = [
    # Using warp_tile [16,16,16]: tile_m = wave_m × 16
    (16, 64, 64),  # 1 × 16 = 16, wave=(1,4,1)
    (32, 64, 64),  # 2 × 16 = 32, wave=(2,2,1)
    (64, 64, 64),  # 4 × 16 = 64, wave=(4,1,1)
    # (128, 64, 64),  # 8 × 16 = 128, wave=(8,2,1) - EXCLUDED: Compile error
    # Using warp_tile [32,32,16]: tile_m = wave_m × 32
    (32, 128, 64),  # 1 × 32 = 32, wave=(1,4,1)
    (64, 128, 64),  # 2 × 32 = 64, wave=(2,2,1)
    (128, 128, 64),  # 4 × 32 = 128, wave=(4,4,1) - NEW!
    # Note: 256x64x64 excluded - compilation issues
    # Using warp_tile [16,16,32]: tile_m = wave_m × 16
    (16, 64, 128),  # 1 × 16 = 16, wave=(1,4,1)
    (32, 64, 128),  # 2 × 16 = 32, wave=(2,2,1)
    (64, 64, 128),  # 4 × 16 = 64, wave=(4,1,1)
    (128, 64, 128),  # 8 × 16 = 128, wave=(8,2,1) - NEW!
    # Note: Excluded tiles:
    # - 128x64x64: wave=8x2x1, warp=16x16x16 - compile error
    # - 32x128x128, 64x128x128, 128x128x128, 256x128x128 (warp_tile 32x32x32) - compv4 issues
    # - 256x64x64, 256x128x128 - arch filter rejection
]

# Wave configurations per tile
# Key: (tile_m, tile_n, tile_k) -> (wave_m, wave_n, wave_k)
# Constraint: tile_m == wave_m × warp_tile_m
# Only use approved wave configs from arch_specs.json: [1,4,1], [2,2,1], [4,1,1], [8,2,1], [4,4,1]
TILE_TO_WAVE: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {
    # warp_tile [16,16,16]
    (16, 64, 64): (1, 4, 1),
    (32, 64, 64): (2, 2, 1),
    (64, 64, 64): (4, 1, 1),
    # warp_tile [32,32,16]
    (32, 128, 64): (1, 4, 1),
    (64, 128, 64): (2, 2, 1),
    (128, 128, 64): (4, 4, 1),  # NEW - balanced 4x4 wave
    # warp_tile [16,16,32]
    (16, 64, 128): (1, 4, 1),
    (32, 64, 128): (2, 2, 1),
    (64, 64, 128): (4, 1, 1),
    (128, 64, 128): (8, 2, 1),  # NEW
}

# Warp tile configurations (must match arch_specs.json gfx950 bf16 approved list)
# Key: (tile_m, tile_n, tile_k) -> (warp_m, warp_n, warp_k)
TILE_TO_WARP: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {
    # warp_tile [16,16,16]
    (16, 64, 64): (16, 16, 16),
    (32, 64, 64): (16, 16, 16),
    (64, 64, 64): (16, 16, 16),
    # warp_tile [32,32,16]
    (32, 128, 64): (32, 32, 16),
    (64, 128, 64): (32, 32, 16),
    (128, 128, 64): (32, 32, 16),  # NEW
    # warp_tile [16,16,32]
    (16, 64, 128): (16, 16, 32),
    (32, 64, 128): (16, 16, 32),
    (64, 64, 128): (16, 16, 32),
    (128, 64, 128): (16, 16, 32),  # NEW
}

# Vector sizes per tile (for memory operations)
# Key: (tile_m, tile_n, tile_k) -> (vec_a, vec_b, vec_c)
TILE_TO_VECTOR: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {
    (16, 64, 64): (4, 8, 8),
    (32, 64, 64): (4, 8, 8),
    (64, 64, 64): (4, 8, 8),
    (32, 128, 64): (4, 8, 8),
    (64, 128, 64): (4, 8, 8),
    (128, 128, 64): (4, 8, 8),
    (16, 64, 128): (4, 8, 8),
    (32, 64, 128): (4, 8, 8),
    (64, 64, 128): (4, 8, 8),
    (128, 64, 128): (4, 8, 8),
}

# =============================================================================
# Pipeline Variant Suffixes (single source of truth)
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


# Valid pipelines per variant
# All 8 pipelines (basic_v1, mem, compv3-6, comp_async, basic_async_v1) successfully
# build and run for all variants in both 2D and 3D (verified via 10_test_all_pipelines.py)
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

# Tiles that support compv4 pipeline
# compv4 has stricter requirements due to double buffering and LDS constraints
# Pattern: only warp_tile [16,16,16] or [16,16,32] work with compv4
# Large warp_tile [32,32,16] and wave [8,2,1] fail arch validation for compv4
COMPV4_COMPATIBLE_TILES: List[Tuple[int, int, int]] = [
    # warp_tile [16,16,16] - all work with compv4
    (16, 64, 64),
    (32, 64, 64),
    (64, 64, 64),
    # (128, 64, 64),  # Excluded: wave=8x2x1 fails for compv4
    # warp_tile [16,16,32] - all work with compv4
    (16, 64, 128),
    (32, 64, 128),
    (64, 64, 128),
    # (128, 64, 128),  # Excluded: wave=8x2x1 fails for compv4
]

# Backward weight tiles (very restricted due to transpose_tile2d constraints)
# Testing all tiles to verify which ones actually work
BWD_WEIGHT_TILES: List[Tuple[int, int, int]] = [
    # warp_tile [16,16,16]
    (16, 64, 64),  # Known working config
    (32, 64, 64),  # Test
    (64, 64, 64),  # Test
    # warp_tile [32,32,16]
    (32, 128, 64),  # Test
    (64, 128, 64),  # Test
    (128, 128, 64),  # Test
    # warp_tile [16,16,32]
    (16, 64, 128),  # Test
    (32, 64, 128),  # Test
    (64, 64, 128),  # Test
    (128, 64, 128),  # Test
]

# =============================================================================
# Shared Validation Rules
# =============================================================================
# These functions are the single source of truth for validation rules
# for onvolution code generation.

# --- Vector size validation ---

WARP_SIZE = 64


def is_valid_vector_size(vec: int) -> bool:
    """AMD GPUs only support vector widths 1, 2, 4, 8, 16."""
    return vec == 1 or vec % 2 == 0


def check_vectors(vec_a: int, vec_b: int, vec_c: int) -> bool:
    """Check all three vector sizes are valid (1 or even)."""
    return all(is_valid_vector_size(v) for v in (vec_a, vec_b, vec_c))


# --- Tile coverage validation ---


def check_warp_coverage(
    tile_m: int, tile_n: int, tile_k: int,
    vec_a: int, vec_b: int,
    variant: str = "forward",
) -> bool:
    """Check tile dims don't exceed single-warp vector load coverage.

    The A-tile dimension is direction-aware:
      Forward / bwd_weight: tile_m is the A-tile dim
      Backward data:        tile_k is the A-tile dim
    """
    a_tile_dim = tile_k if variant == "bwd_data" else tile_m
    if a_tile_dim > WARP_SIZE * vec_a:
        return False
    if tile_n > WARP_SIZE * vec_b:
        return False
    return True


def check_bwd_data_vec_coverage(
    tile_m: int, tile_n: int, tile_k: int,
    warp_m: int, warp_n: int, warp_k: int,
    vec_a: int, vec_b: int,
) -> bool:
    """Bwd_data: vector width must not exceed elements per thread per tile slice."""
    block_size = WARP_SIZE * warp_m * warp_n * warp_k
    if vec_a > (tile_m * tile_k) // block_size:
        return False
    if vec_b > (tile_n * tile_k) // block_size:
        return False
    return True


# --- Pipeline-scheduler restrictions ---

INTERWAVE_PIPELINES = {"basic_v1", "mem"}  # Only these support interwave


def is_valid_pipeline_scheduler(pipeline: str, scheduler: str) -> bool:
    """Check pipeline+scheduler combo is valid.

    Only 'mem' and 'basic_v1' pipelines support interwave; all compute
    pipelines (compv3/v4/v5/v6/async) only support intrawave.
    """
    if scheduler == "interwave" and pipeline not in INTERWAVE_PIPELINES:
        return False
    return True


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
# Tile Registration Validation
# =============================================================================


def validate_tile_config(tile_m: int, tile_n: int, tile_k: int) -> bool:
    """Check if a tile configuration is valid and registered."""
    tile_key = (tile_m, tile_n, tile_k)
    return (
        tile_key in TILE_TO_WAVE
        and tile_key in TILE_TO_WARP
        and tile_key in TILE_TO_VECTOR
    )


def get_tile_full_config(tile_m: int, tile_n: int, tile_k: int) -> dict:
    """Get complete configuration for a tile size.

    Returns:
        dict with keys: wave_m, wave_n, wave_k, warp_m, warp_n, warp_k, vec_a, vec_b, vec_c
        or None if tile not found
    """
    tile_key = (tile_m, tile_n, tile_k)
    if not validate_tile_config(tile_m, tile_n, tile_k):
        return None

    wave_m, wave_n, wave_k = TILE_TO_WAVE[tile_key]
    warp_m, warp_n, warp_k = TILE_TO_WARP[tile_key]
    vec_a, vec_b, vec_c = TILE_TO_VECTOR[tile_key]

    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "wave_m": wave_m,
        "wave_n": wave_n,
        "wave_k": wave_k,
        "warp_m": warp_m,
        "warp_n": warp_n,
        "warp_k": warp_k,
        "vec_a": vec_a,
        "vec_b": vec_b,
        "vec_c": vec_c,
    }


# =============================================================================
# Summary Statistics
# =============================================================================


def print_summary():
    """Print summary of available tile configurations."""
    print("=" * 80)
    print("Grouped Convolution Tile Configurations (Single Source of Truth)")
    print("=" * 80)
    print(f"Total tiles: {len(COMMON_TILES)}")
    print(f"Backward weight tiles: {len(BWD_WEIGHT_TILES)}")
    print()
    print("Tile sizes (M×N×K):")
    for tile in COMMON_TILES:
        m, n, k = tile
        wave = TILE_TO_WAVE[tile]
        warp = TILE_TO_WARP[tile]
        print(
            f"  {m:3}×{n:3}×{k:3}  wave={wave[0]}×{wave[1]}×{wave[2]}  warp={warp[0]}×{warp[1]}×{warp[2]}"
        )
    print("=" * 80)


if __name__ == "__main__":
    print_summary()
