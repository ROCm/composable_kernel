# SPDX-License-Identifier: MIT

"""
AUTO-GENERATED FILE - DO NOT EDIT DIRECTLY!

Generated from: arch_specs.json
Generated at: 2026-06-01T10:50:14.618422

To update this file:
1. Edit arch_specs.json
2. Run: python generate_arch_specs.py

This module provides architecture-specific configurations for kernel filtering.
"""

from typing import Dict, List, Set, Tuple

# =============================================================================
# Architecture Data (Generated from arch_specs.json)
# =============================================================================

# GPU architecture to family mapping
ARCH_FAMILY_MAP: Dict[str, str] = {
    "gfx908": "cdna1",
    "gfx90a": "cdna2",
    "gfx942": "cdna3",
    "gfx950": "cdna4",
    "gfx1100": "rdna3",
    "gfx1200": "rdna4",
    "gfx1201": "rdna4",
    "gfx1250": "rdna4",
}

# Element size in bytes for each data type
ELEMENT_SIZE_MAP: Dict[str, float] = {'fp16': 2, 'bf16': 2, 'fp32': 4, 'fp64': 8, 'fp8': 1, 'bf8': 1, 'int8': 1, 'int4': 0.5, 'pk_fp4': 0.5, 'int32': 4}

# Supported warp configurations per architecture [warp_m, warp_n, warp_k]
WARP_SUPPORTED_COMBINATIONS: Dict[str, List[List[int]]] = {
    "gfx908": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
    "gfx90a": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
    "gfx942": [[1, 1, 1], [1, 2, 1], [1, 4, 1], [2, 1, 1], [2, 1, 2], [2, 2, 1], [4, 1, 1]],
    "gfx950": [[1, 1, 1], [1, 2, 1], [1, 4, 1], [2, 1, 1], [2, 1, 2], [2, 2, 1], [4, 1, 1], [8, 2, 1], [4, 4, 1]],
    "gfx1100": [[2, 4, 1], [1, 8, 1], [8, 1, 1], [4, 2, 1]],
    "gfx1200": [[2, 4, 1], [1, 8, 1], [8, 1, 1], [4, 2, 1]],
    "gfx1201": [[2, 4, 1], [1, 8, 1], [8, 1, 1], [4, 2, 1]],
    "gfx1250": [[2, 4, 1], [1, 8, 1], [8, 1, 1], [4, 2, 1], [2, 1, 1], [1, 2, 2], [4, 1, 1], [1, 4, 1], [2, 2, 1]],
}

# Supported warp tile combinations: arch -> dtype_key -> [[warp_tile_m, n, k], ...]
WARP_TILE_SUPPORTED_COMBINATIONS: Dict[str, Dict[str, List[List[int]]]] = {
    "gfx908": {
        "fp32_fp32_fp32": [[16, 16, 4], [16, 16, 8], [16, 16, 16], [32, 32, 4], [32, 32, 8]],
        "fp16_fp16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32]],
        "bf16_bf16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32]],
        "int8_int8_int32": [[32, 32, 16], [16, 16, 32]],
    },
    "gfx90a": {
        "fp32_fp32_fp32": [[16, 16, 4], [16, 16, 8], [16, 16, 16], [32, 32, 4], [32, 32, 8]],
        "fp16_fp16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32], [4, 64, 16], [64, 4, 16]],
        "bf16_bf16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32], [4, 64, 16], [64, 4, 16]],
        "fp8_fp8_fp32": [[32, 32, 16], [32, 32, 32]],
        "bf8_bf8_fp32": [[32, 32, 16], [32, 32, 32]],
        "int8_int8_int32": [[32, 32, 16], [16, 16, 32]],
    },
    "gfx942": {
        "fp32_fp32_fp32": [[16, 16, 4], [16, 16, 8], [16, 16, 16], [32, 32, 4], [32, 32, 8]],
        "fp16_fp16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32], [4, 64, 16], [64, 4, 16]],
        "bf16_bf16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32], [4, 64, 16], [64, 4, 16]],
        "fp8_fp8_fp32": [[32, 32, 16], [32, 32, 32], [16, 16, 32], [16, 16, 64]],
        "fp8_bf8_fp32": [[32, 32, 16], [16, 16, 32], [32, 32, 32]],
        "bf8_fp8_fp32": [[32, 32, 16]],
        "bf8_bf8_fp32": [[32, 32, 16], [32, 32, 32], [16, 16, 32], [16, 16, 64]],
        "int8_int8_int32": [[32, 32, 16], [16, 16, 32]],
    },
    "gfx950": {
        "fp32_fp32_fp32": [[16, 16, 4], [16, 16, 8], [16, 16, 16], [32, 32, 4], [32, 32, 8]],
        "fp16_fp16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32], [4, 64, 16], [64, 4, 16]],
        "bf16_bf16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32], [4, 64, 16], [64, 4, 16]],
        "fp8_fp8_fp32": [[32, 32, 16], [32, 32, 32], [16, 16, 32], [16, 16, 64], [16, 16, 128], [32, 32, 64]],
        "fp8_bf8_fp32": [[32, 32, 16], [32, 32, 32], [16, 16, 32], [16, 16, 128], [32, 32, 64]],
        "bf8_fp8_fp32": [[32, 32, 16], [16, 16, 128], [32, 32, 64]],
        "bf8_bf8_fp32": [[32, 32, 16], [32, 32, 32], [16, 16, 32], [16, 16, 64], [16, 16, 128], [32, 32, 64]],
        "int8_int8_int32": [[32, 32, 16], [16, 16, 32]],
        "pk_fp4_pk_fp4_fp32": [[16, 16, 128]],
    },
    "gfx1100": {
        "fp16_fp16_fp32": [[16, 16, 16]],
        "bf16_bf16_fp32": [[16, 16, 16]],
        "int8_int8_int32": [[16, 16, 16]],
    },
    "gfx1200": {
        "fp16_fp16_fp32": [[16, 16, 16]],
        "bf16_bf16_fp32": [[16, 16, 16]],
        "fp8_fp8_fp32": [[16, 16, 16]],
        "bf8_bf8_fp32": [[16, 16, 16]],
        "fp8_bf8_fp32": [[16, 16, 16]],
        "bf8_fp8_fp32": [[16, 16, 16]],
        "int8_int8_int32": [[16, 16, 16]],
    },
    "gfx1201": {
        "fp16_fp16_fp32": [[16, 16, 16]],
        "bf16_bf16_fp32": [[16, 16, 16]],
        "fp8_fp8_fp32": [[16, 16, 16]],
        "bf8_bf8_fp32": [[16, 16, 16]],
        "fp8_bf8_fp32": [[16, 16, 16]],
        "bf8_fp8_fp32": [[16, 16, 16]],
        "int8_int8_int32": [[16, 16, 16]],
    },
    "gfx1250": {
        "fp16_fp16_fp32": [[16, 16, 32]],
        "bf16_bf16_fp32": [[16, 16, 32]],
    },
}

# Preshuffle-specific warp tile combinations (subset of standard GEMM)
PRESHUFFLE_WARP_TILE_SUPPORTED_COMBINATIONS: Dict[str, Dict[str, List[List[int]]]] = {
    "gfx90a": {
        "fp16_fp16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32], [64, 4, 16]],
        "bf16_bf16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32], [64, 4, 16]],
        "fp8_fp8_fp32": [[32, 32, 16], [32, 32, 32]],
        "bf8_bf8_fp32": [[32, 32, 16], [32, 32, 32]],
    },
    "gfx942": {
        "fp16_fp16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32], [64, 4, 16]],
        "bf16_bf16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32], [64, 4, 16]],
        "fp8_fp8_fp32": [[32, 32, 16], [32, 32, 32], [16, 16, 32], [16, 16, 64]],
        "bf8_bf8_fp32": [[32, 32, 16], [32, 32, 32], [16, 16, 64], [16, 16, 32]],
        "int8_int8_int32": [[16, 16, 32], [32, 32, 16]],
    },
    "gfx950": {
        "fp16_fp16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32], [64, 4, 16], [32, 32, 32], [16, 16, 64]],
        "bf16_bf16_fp32": [[32, 32, 8], [16, 16, 16], [32, 32, 16], [16, 16, 32], [64, 4, 16], [32, 32, 32], [16, 16, 64]],
        "fp8_fp8_fp32": [[32, 32, 16], [32, 32, 32], [16, 16, 32], [16, 16, 64], [16, 16, 128], [32, 32, 64]],
        "bf8_bf8_fp32": [[32, 32, 16], [32, 32, 32], [16, 16, 64], [16, 16, 32], [16, 16, 128], [32, 32, 64]],
    },
}

# Preshuffle-supported pipelines
PRESHUFFLE_PIPELINES: List[str] = ['preshufflev2']

# LDS capacity limits per pipeline type (in bytes)
LDS_CAPACITY_LIMITS: Dict[str, int] = {'mem': 65536, 'compv1': 65536, 'compv2': 65536, 'compv3': 65536, 'compv4': 32768, 'compv5': 65536, 'compv6': 32768, 'preshufflev1': 32768, 'preshufflev2': 32768, 'default': 65536}

# Unsupported trait combinations: (pipeline, epilogue, scheduler)
TRAIT_UNSUPPORTED_COMBINATIONS: Set[Tuple[str, str, str]] = {
    ("compv3", "cshuffle", "interwave"),
    ("compv3", "default", "interwave"),
    ("compv4", "cshuffle", "interwave"),
    ("compv4", "default", "interwave"),
    ("compv5", "cshuffle", "interwave"),
    ("compv5", "default", "interwave"),
    ("compv6", "cshuffle", "interwave"),
    ("compv6", "default", "interwave"),
    ("comp_async", "cshuffle", "interwave"),
    ("comp_async", "default", "interwave"),
}

# Valid dtype combinations: (A_dtype, B_dtype) -> acc_dtype and notes
DTYPE_COMBINATIONS: Dict[str, Dict[str, str]] = {
    "fp32_fp32": {"acc": "fp32", "notes": "Full precision"},
    "fp16_fp16": {"acc": "fp32", "notes": "Standard half precision"},
    "bf16_bf16": {"acc": "fp32", "notes": "Brain float 16"},
    "fp8_fp8": {"acc": "fp32", "notes": "FP8 E4M3"},
    "fp8_bf8": {"acc": "fp32", "notes": "Mixed FP8/BF8"},
    "bf8_fp8": {"acc": "fp32", "notes": "Mixed BF8/FP8"},
    "bf8_bf8": {"acc": "fp32", "notes": "BF8 E5M2"},
    "int8_int8": {"acc": "int32", "notes": "Integer GEMM"},
    "pk_fp4_pk_fp4": {"acc": "fp32", "notes": "Packed 4-bit float"},
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_supported_archs() -> List[str]:
    """Get list of all supported GPU architectures."""
    return list(ARCH_FAMILY_MAP.keys())


def get_arch_family(gpu_arch: str) -> str:
    """Get the GPU family for an architecture."""
    return ARCH_FAMILY_MAP.get(gpu_arch.lower(), "unknown")


def get_element_size(dtype: str) -> float:
    """Get element size in bytes for a data type."""
    return ELEMENT_SIZE_MAP.get(dtype.lower(), 2.0)


def get_warp_configs(gpu_arch: str) -> List[List[int]]:
    """Get supported warp configurations for an architecture."""
    return WARP_SUPPORTED_COMBINATIONS.get(gpu_arch.lower(), [])


def get_warp_tile_combos(gpu_arch: str, dtype_key: str) -> List[List[int]]:
    """Get supported warp tile combinations for arch and data types."""
    gpu_combos = WARP_TILE_SUPPORTED_COMBINATIONS.get(gpu_arch.lower(), {})
    return gpu_combos.get(dtype_key.lower(), [])


def get_lds_limit(pipeline: str) -> int:
    """Get LDS capacity limit for a pipeline type."""
    return LDS_CAPACITY_LIMITS.get(pipeline.lower(), LDS_CAPACITY_LIMITS["default"])


def is_trait_combo_unsupported(pipeline: str, epilogue: str, scheduler: str) -> bool:
    """Check if a trait combination is unsupported."""
    return (pipeline.lower(), epilogue.lower(), scheduler.lower()) in TRAIT_UNSUPPORTED_COMBINATIONS


def get_dtype_info(dtype_a: str, dtype_b: str) -> Dict[str, str]:
    """Get accumulator type and notes for a dtype combination."""
    key = f"{dtype_a.lower()}_{dtype_b.lower()}"
    return DTYPE_COMBINATIONS.get(key, {"acc": "fp32", "notes": "unknown"})


def is_dtype_combo_valid(dtype_a: str, dtype_b: str) -> bool:
    """Check if a dtype combination is valid."""
    key = f"{dtype_a.lower()}_{dtype_b.lower()}"
    return key in DTYPE_COMBINATIONS


def get_valid_dtype_combos() -> List[str]:
    """Get list of all valid dtype combinations."""
    return list(DTYPE_COMBINATIONS.keys())
