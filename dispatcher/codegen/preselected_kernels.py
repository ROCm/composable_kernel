#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Preselected, Benchmarked Kernel Configurations

Curated kernel sets optimized for different workload characteristics:
- Compute-friendly: Large tiles, high arithmetic intensity
- Memory-friendly: Smaller tiles, better memory access patterns
- Latency-friendly: Minimal tiles, low latency for small problems
"""

from functools import partial, lru_cache
from typing import List
from unified_gemm_codegen import KernelConfig, TileConfig, TraitConfig, GemmVariant


# ============================================================================
# Base Configurations
# ============================================================================


def _base_fp16_rcr_compute() -> partial:
    """Base configuration for compute-intensive FP16 RCR kernels"""
    return partial(
        KernelConfig,
        tile=None,  # Will be overridden
        trait=TraitConfig(
            pipeline="compv4",
            epilogue="cshuffle",
            scheduler="intrawave",
            pad_m=True,
            pad_n=True,
            pad_k=True,
            persistent=False,
        ),
        variant=GemmVariant.STANDARD,
        block_size=256,
        k_block_per_cu=1,
        num_wave_groups=1,
    )


def _base_fp16_rcr_memory() -> partial:
    """Base configuration for memory-intensive FP16 RCR kernels"""
    # Note: Use 'mem' pipeline for interwave scheduler (compv3/compv4/compv5/compv6 only support intrawave)
    return partial(
        KernelConfig,
        tile=None,  # Will be overridden
        trait=TraitConfig(
            pipeline="mem",
            epilogue="cshuffle",
            scheduler="interwave",
            pad_m=True,
            pad_n=True,
            pad_k=True,
            persistent=False,
        ),
        variant=GemmVariant.STANDARD,
        block_size=128,
        k_block_per_cu=1,
        num_wave_groups=1,
    )


def _base_fp16_rcr_latency() -> partial:
    """Base configuration for latency-sensitive FP16 RCR kernels"""
    return partial(
        KernelConfig,
        tile=None,  # Will be overridden
        trait=TraitConfig(
            pipeline="mem",
            epilogue="default",
            scheduler="intrawave",
            pad_m=True,
            pad_n=True,
            pad_k=True,
            persistent=False,
        ),
        variant=GemmVariant.STANDARD,
        block_size=128,
        k_block_per_cu=1,
        num_wave_groups=1,
    )


# ============================================================================
# Preselected FP16 RCR Kernels
# ============================================================================


@lru_cache(None)
def preselected_fp16_rcr_compute() -> List[KernelConfig]:
    """
    Compute-friendly FP16 RCR kernels

    Optimized for:
    - Large M, N dimensions (>= 128)
    - High arithmetic intensity
    - Good occupancy
    - Maximum throughput
    """
    base = _base_fp16_rcr_compute()

    return [
        # Large tiles for maximum compute
        base(tile=TileConfig(256, 256, 32, 4, 4, 1, 32, 32, 16)),
        base(tile=TileConfig(256, 256, 64, 4, 4, 1, 32, 32, 16)),
        base(tile=TileConfig(256, 128, 32, 4, 2, 1, 32, 32, 16)),
        base(tile=TileConfig(128, 256, 32, 2, 4, 1, 32, 32, 16)),
        # Balanced tiles
        base(tile=TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)),
        base(tile=TileConfig(128, 128, 64, 2, 2, 1, 32, 32, 16)),
        # With persistent kernel for large batches
        base(
            tile=TileConfig(256, 256, 32, 4, 4, 1, 32, 32, 16),
            trait=TraitConfig(
                pipeline="compv4",
                epilogue="cshuffle",
                scheduler="intrawave",
                pad_m=False,
                pad_n=False,
                pad_k=False,
                persistent=True,
            ),
        ),
    ]


@lru_cache(None)
def preselected_fp16_rcr_memory() -> List[KernelConfig]:
    """
    Memory-friendly FP16 RCR kernels

    Optimized for:
    - Small to medium M, N dimensions
    - Memory-bound workloads
    - Better cache utilization
    - Lower register pressure
    """
    base = _base_fp16_rcr_memory()

    return [
        # Small tiles for memory efficiency
        base(tile=TileConfig(16, 32, 32, 1, 1, 1, 16, 16, 16)),
        base(tile=TileConfig(32, 16, 32, 1, 1, 1, 16, 16, 16)),
        base(tile=TileConfig(16, 64, 32, 1, 2, 1, 16, 16, 16)),
        base(tile=TileConfig(64, 16, 32, 2, 1, 1, 16, 16, 16)),
        # Medium tiles
        base(tile=TileConfig(32, 64, 32, 1, 1, 1, 32, 32, 16)),
        base(tile=TileConfig(64, 32, 32, 1, 1, 1, 32, 32, 16)),
        base(tile=TileConfig(32, 128, 32, 1, 2, 1, 32, 32, 16)),
        base(tile=TileConfig(128, 32, 32, 2, 1, 1, 32, 32, 16)),
    ]


@lru_cache(None)
def preselected_fp16_rcr_latency() -> List[KernelConfig]:
    """
    Latency-friendly FP16 RCR kernels

    Optimized for:
    - Very small M, N dimensions (< 64)
    - Minimal launch overhead
    - Low latency
    - Quick execution
    """
    base = _base_fp16_rcr_latency()

    return [
        # Minimal tiles for low latency
        base(tile=TileConfig(16, 32, 32, 1, 1, 1, 16, 16, 16)),
        base(tile=TileConfig(32, 16, 32, 1, 1, 1, 16, 16, 16)),
    ]


# ============================================================================
# Preselected Multi-D Kernels
# ============================================================================


@lru_cache(None)
def preselected_fp16_rcr_multi_d() -> List[KernelConfig]:
    """
    Multi-D GEMM kernels with element-wise fusion

    Common fusions:
    - MultiDAdd: E = C + D0 + D1
    - Relu: E = max(C, 0)
    - Gelu: E = gelu(C)
    """
    base = _base_fp16_rcr_compute()

    configs = []

    # Best-performing tile for fused operations
    tile = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)

    # Common element-wise operations
    for ew_op in ["MultiDAdd", "Relu", "Gelu", "FastGelu"]:
        for num_d in [1, 2]:
            configs.append(
                base(
                    tile=tile,
                    variant=GemmVariant.MULTI_D,
                    elementwise_op=ew_op,
                    num_d_tensors=num_d,
                )
            )

    return configs


@lru_cache(None)
def preselected_fp16_rcr_preshuffle() -> List[KernelConfig]:
    """
    Preshuffle GEMM kernels for weight optimization

    Best for:
    - Repeated use of same weights
    - Inference workloads
    - Batch size > 1
    """
    base = _base_fp16_rcr_compute()

    return [
        base(
            tile=TileConfig(256, 256, 32, 4, 4, 1, 32, 32, 16),
            variant=GemmVariant.PRESHUFFLE,
            preshuffle=True,
        ),
        base(
            tile=TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16),
            variant=GemmVariant.PRESHUFFLE,
            preshuffle=True,
        ),
    ]


# ============================================================================
# Unified Preselected Sets
# ============================================================================


@lru_cache(None)
def preselected_fp16_rcr_all() -> List[KernelConfig]:
    """All preselected FP16 RCR kernels"""
    return (
        preselected_fp16_rcr_compute()
        + preselected_fp16_rcr_memory()
        + preselected_fp16_rcr_latency()
        + preselected_fp16_rcr_multi_d()
        + preselected_fp16_rcr_preshuffle()
    )


@lru_cache(None)
def preselected_fp16_rcr_essential() -> List[KernelConfig]:
    """
    Essential FP16 RCR kernels - minimal set for most workloads

    Covers:
    - 90% of common GEMM sizes
    - Key fusion operations
    - Balanced performance
    """
    base_compute = _base_fp16_rcr_compute()
    base_memory = _base_fp16_rcr_memory()

    return [
        # Top compute kernels
        base_compute(tile=TileConfig(256, 256, 32, 4, 4, 1, 32, 32, 16)),
        base_compute(tile=TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)),
        # Top memory kernels
        base_memory(tile=TileConfig(32, 64, 32, 1, 1, 1, 32, 32, 16)),
        base_memory(tile=TileConfig(64, 32, 32, 1, 1, 1, 32, 32, 16)),
        # Essential fusions
        base_compute(
            tile=TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16),
            variant=GemmVariant.MULTI_D,
            elementwise_op="Relu",
            num_d_tensors=1,
        ),
        base_compute(
            tile=TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16),
            variant=GemmVariant.MULTI_D,
            elementwise_op="Gelu",
            num_d_tensors=1,
        ),
    ]


# ============================================================================
# Default Fallback
# ============================================================================


def default_kernel() -> KernelConfig:
    """
    Default fallback kernel - guaranteed to work

    Known-good configuration tested on gfx942
    """
    return KernelConfig(
        tile=TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16),
        trait=TraitConfig(
            pipeline="compv4",
            epilogue="cshuffle",
            scheduler="intrawave",
            pad_m=True,
            pad_n=True,
            pad_k=True,
            persistent=False,
        ),
        variant=GemmVariant.STANDARD,
        block_size=256,
        k_block_per_cu=1,
        num_wave_groups=1,
    )


# ============================================================================
# BF16 Preselected Sets
# ============================================================================


@lru_cache(None)
def preselected_bf16_rcr_essential() -> List[KernelConfig]:
    """Essential BF16 RCR kernels"""
    base_compute = partial(
        KernelConfig,
        tile=None,
        trait=TraitConfig(
            pipeline="compv4",
            epilogue="cshuffle",
            scheduler="intrawave",
            pad_m=True,
            pad_n=True,
            pad_k=True,
            persistent=False,
        ),
        variant=GemmVariant.STANDARD,
        block_size=256,
    )

    return [
        base_compute(tile=TileConfig(256, 256, 32, 4, 4, 1, 32, 32, 16)),
        base_compute(tile=TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)),
    ]


# ============================================================================
# INT8 Preselected Sets
# ============================================================================


@lru_cache(None)
def preselected_int8_rcr_essential() -> List[KernelConfig]:
    """Essential INT8 RCR kernels for quantized inference"""
    base = partial(
        KernelConfig,
        tile=None,
        trait=TraitConfig(
            pipeline="compv4",
            epilogue="cshuffle",
            scheduler="intrawave",
            pad_m=True,
            pad_n=True,
            pad_k=True,
            persistent=False,
        ),
        variant=GemmVariant.STANDARD,
        block_size=256,
    )

    return [
        base(tile=TileConfig(256, 256, 64, 4, 4, 1, 32, 32, 16)),
        base(tile=TileConfig(128, 128, 64, 2, 2, 1, 32, 32, 16)),
    ]


# ============================================================================
# FP8 Preselected Sets
# ============================================================================


@lru_cache(None)
def preselected_fp8_rcr_essential() -> List[KernelConfig]:
    """Essential FP8 RCR kernels for AI training"""
    base = partial(
        KernelConfig,
        tile=None,
        trait=TraitConfig(
            pipeline="compv4",
            epilogue="cshuffle",
            scheduler="intrawave",
            pad_m=True,
            pad_n=True,
            pad_k=True,
            persistent=False,
        ),
        variant=GemmVariant.STANDARD,
        block_size=256,
    )

    return [
        base(tile=TileConfig(256, 256, 64, 4, 4, 1, 32, 32, 16)),
        base(tile=TileConfig(128, 128, 64, 2, 2, 1, 32, 32, 16)),
    ]


# ============================================================================
# Mixed Precision Preselected Sets
# ============================================================================


@lru_cache(None)
def preselected_mixed_precision() -> List[KernelConfig]:
    """Mixed-precision kernels (FP16 inputs, FP32 output)"""
    base = partial(
        KernelConfig,
        tile=None,
        trait=TraitConfig(
            pipeline="compv4",
            epilogue="cshuffle",
            scheduler="intrawave",
            pad_m=True,
            pad_n=True,
            pad_k=True,
            persistent=False,
        ),
        variant=GemmVariant.STANDARD,
        block_size=256,
    )

    return [
        base(tile=TileConfig(256, 256, 32, 4, 4, 1, 32, 32, 16)),
        base(tile=TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)),
    ]


# ============================================================================
# Registry
# ============================================================================

PRESELECTED_SETS = {
    # FP16 sets
    "fp16_rcr_compute": preselected_fp16_rcr_compute,
    "fp16_rcr_memory": preselected_fp16_rcr_memory,
    "fp16_rcr_latency": preselected_fp16_rcr_latency,
    "fp16_rcr_multi_d": preselected_fp16_rcr_multi_d,
    "fp16_rcr_preshuffle": preselected_fp16_rcr_preshuffle,
    "fp16_rcr_all": preselected_fp16_rcr_all,
    "fp16_rcr_essential": preselected_fp16_rcr_essential,
    # BF16 sets
    "bf16_rcr_essential": preselected_bf16_rcr_essential,
    # INT8 sets
    "int8_rcr_essential": preselected_int8_rcr_essential,
    # FP8 sets
    "fp8_rcr_essential": preselected_fp8_rcr_essential,
    # Mixed precision
    "mixed_precision": preselected_mixed_precision,
}


def get_preselected_set(name: str) -> List[KernelConfig]:
    """Get a preselected kernel set by name"""
    if name not in PRESELECTED_SETS:
        raise ValueError(
            f"Unknown preselected set: {name}. Available: {list(PRESELECTED_SETS.keys())}"
        )
    return PRESELECTED_SETS[name]()


def list_preselected_sets() -> List[str]:
    """List all available preselected sets"""
    return list(PRESELECTED_SETS.keys())


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="List preselected kernel configurations"
    )
    parser.add_argument(
        "--set",
        type=str,
        default="fp16_rcr_essential",
        choices=list_preselected_sets(),
        help="Preselected set to display",
    )
    parser.add_argument("--count-only", action="store_true", help="Only show count")

    args = parser.parse_args()

    configs = get_preselected_set(args.set)

    if args.count_only:
        print(f"{args.set}: {len(configs)} kernels")
    else:
        print(f"Preselected set: {args.set}")
        print(f"Total kernels: {len(configs)}\n")
        for i, cfg in enumerate(configs, 1):
            print(f"{i}. {cfg.variant.value}")
            print(f"   Tile: {cfg.tile.tile_m}x{cfg.tile.tile_n}x{cfg.tile.tile_k}")
            print(f"   Pipeline: {cfg.trait.pipeline}, Epilogue: {cfg.trait.epilogue}")
            if cfg.variant == GemmVariant.MULTI_D:
                print(
                    f"   Element-wise: {cfg.elementwise_op}, D tensors: {cfg.num_d_tensors}"
                )
            print()
