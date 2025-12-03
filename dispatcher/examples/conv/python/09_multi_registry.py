#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 09: Multiple Convolution Registries

Demonstrates using multiple registries for different workload types,
with kernel configuration validation.

Usage:
    python3 09_multi_registry.py
    python3 09_multi_registry.py --arch gfx942
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from conv_utils import (
    ConvSignature,
    ConvAlgorithm,
    ArchInfo,
    ConvKernelConfig,
    ConvProblem,
    ConvRegistry,
    ConvDispatcher,
    GpuConvRunner,
    validate_conv_config,
    auto_correct_conv_config,
    reset_for_conv_example,
    cleanup_conv,
)
import numpy as np


def create_validated_kernel(dtype, tile_k, tile_c, pipeline, scheduler, arch_name):
    """Create a validated kernel configuration."""
    sig = ConvSignature()
    sig.dtype(dtype)
    sig.layout = "nhwgc"
    sig.direction = "forward"

    algo = ConvAlgorithm()
    algo.tile(1, tile_k, tile_c)
    algo.wave(2, 2, 1)
    algo.warp(32, 32, 16)
    algo.pipeline = pipeline
    algo.scheduler = scheduler

    arch = ArchInfo(name=arch_name)

    # Validate
    validation = validate_conv_config(
        pipeline=algo.pipeline,
        scheduler=algo.scheduler,
        epilogue=algo.epilogue,
        wave_m=algo.wave_m,
        wave_n=algo.wave_n,
        wave_k=algo.wave_k,
        warp_m=algo.warp_m,
        warp_n=algo.warp_n,
        warp_k=algo.warp_k,
        dtype=sig.dtype_in,
        arch=arch.name,
    )

    if not validation.is_valid:
        # Auto-correct
        corrected, was_modified, _ = auto_correct_conv_config(
            pipeline=algo.pipeline,
            scheduler=algo.scheduler,
            epilogue=algo.epilogue,
            wave_m=algo.wave_m,
            wave_n=algo.wave_n,
            wave_k=algo.wave_k,
            warp_m=algo.warp_m,
            warp_n=algo.warp_n,
            warp_k=algo.warp_k,
            dtype=sig.dtype_in,
            arch=arch.name,
        )
        if was_modified:
            algo.scheduler = corrected["scheduler"]
            algo.wave_m = corrected["wave_m"]
            algo.wave_n = corrected["wave_n"]
            algo.warp_m = corrected["warp_m"]
            algo.warp_n = corrected["warp_n"]
            algo.warp_k = corrected["warp_k"]

    return ConvKernelConfig(signature=sig, algorithm=algo, arch=arch)


def create_compute_bound_registry(arch_name: str) -> ConvRegistry:
    """
    Create registry for compute-bound problems.

    Compute-bound: High arithmetic intensity, benefit from larger tiles.
    Examples: Large feature maps, many channels.
    """
    registry = ConvRegistry(name="compute_bound")

    # Large tile configurations for compute-bound
    for tile_k, tile_c in [(256, 256), (256, 128), (128, 256)]:
        config = create_validated_kernel(
            dtype="fp16",
            tile_k=tile_k,
            tile_c=tile_c,
            pipeline="compv4",
            scheduler="intrawave",
            arch_name=arch_name,
        )
        registry.register_kernel(config)

    return registry


def create_memory_bound_registry(arch_name: str) -> ConvRegistry:
    """
    Create registry for memory-bound problems.

    Memory-bound: Lower arithmetic intensity, need efficient memory access.
    Examples: Depthwise conv, small feature maps, 1x1 convolutions.
    """
    registry = ConvRegistry(name="memory_bound")

    # Smaller tiles but more memory-efficient configurations
    for tile_k, tile_c in [(128, 128), (64, 128), (128, 64)]:
        config = create_validated_kernel(
            dtype="fp16",
            tile_k=tile_k,
            tile_c=tile_c,
            pipeline="compv3",
            scheduler="interwave",
            arch_name=arch_name,
        )
        registry.register_kernel(config)

    return registry


def create_latency_optimized_registry(arch_name: str) -> ConvRegistry:
    """
    Create registry for latency-optimized problems.

    Latency-optimized: Small problems where kernel launch overhead matters.
    Examples: Inference with batch=1, small spatial dimensions.
    """
    registry = ConvRegistry(name="latency_optimized")

    # Small tile configurations for low latency
    for tile_k, tile_c in [(64, 64), (32, 64), (64, 32)]:
        config = create_validated_kernel(
            dtype="fp16",
            tile_k=tile_k,
            tile_c=tile_c,
            pipeline="compv3",
            scheduler="intrawave",
            arch_name=arch_name,
        )
        registry.register_kernel(config)

    return registry


def classify_problem(problem: ConvProblem) -> str:
    """Classify a problem as compute-bound, memory-bound, or latency-optimized."""
    # Simple heuristics based on problem characteristics
    if problem.is_pointwise():
        return "memory_bound"

    if problem.Hi <= 7 and problem.Wi <= 7:
        return "latency_optimized"

    if problem.C >= 256 and problem.K >= 256:
        return "compute_bound"

    if problem.Y == 1 and problem.X == 1:
        return "memory_bound"

    return "compute_bound"


def main():
    parser = argparse.ArgumentParser(description="Multiple Convolution Registries")
    parser.add_argument(
        "--arch", type=str, default="gfx942", help="Target architecture"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 09: Multiple Convolution Registries")
    print("=" * 70)
    print()

    # =========================================================================
    # Step 0: Reset state for clean example run
    # =========================================================================
    reset_for_conv_example(verbose=True)

    # =========================================================================
    # Step 1: Create specialized registries
    # =========================================================================
    print("\nCREATING SPECIALIZED REGISTRIES")
    print("=" * 60)

    compute_registry = create_compute_bound_registry(args.arch)
    memory_registry = create_memory_bound_registry(args.arch)
    latency_registry = create_latency_optimized_registry(args.arch)

    print(f"\nCompute-bound registry: {compute_registry.kernel_count} kernels")
    for cfg in compute_registry.get_kernels()[:3]:
        print(f"  - {cfg.name()}")
    print()

    print(f"Memory-bound registry: {memory_registry.kernel_count} kernels")
    for cfg in memory_registry.get_kernels()[:3]:
        print(f"  - {cfg.name()}")
    print()

    print(f"Latency-optimized registry: {latency_registry.kernel_count} kernels")
    for cfg in latency_registry.get_kernels()[:3]:
        print(f"  - {cfg.name()}")
    print()

    # =========================================================================
    # Step 2: Create dispatchers
    # =========================================================================
    print("CREATING DISPATCHERS")
    print("=" * 60)

    compute_dispatcher = ConvDispatcher(compute_registry)
    memory_dispatcher = ConvDispatcher(memory_registry)
    latency_dispatcher = ConvDispatcher(latency_registry)

    print(f"Compute dispatcher: {compute_dispatcher}")
    print(f"Memory dispatcher: {memory_dispatcher}")
    print(f"Latency dispatcher: {latency_dispatcher}")
    print()

    # =========================================================================
    # Step 3: Test problem classification
    # =========================================================================
    print("PROBLEM CLASSIFICATION")
    print("=" * 60)

    problems = [
        # Compute-bound: large channels
        ConvProblem(N=1, C=512, K=512, Hi=14, Wi=14, Y=3, X=3, pad_h=1, pad_w=1),
        # Memory-bound: 1x1 convolution
        ConvProblem(N=1, C=256, K=256, Hi=28, Wi=28, Y=1, X=1),
        # Latency-optimized: small spatial
        ConvProblem(N=1, C=512, K=512, Hi=7, Wi=7, Y=3, X=3, pad_h=1, pad_w=1),
        # Compute-bound: large feature map
        ConvProblem(N=1, C=64, K=128, Hi=56, Wi=56, Y=3, X=3, pad_h=1, pad_w=1),
        # Memory-bound: depthwise-like
        ConvProblem(N=1, C=64, K=64, Hi=28, Wi=28, Y=3, X=3, pad_h=1, pad_w=1, G=64),
    ]

    print(f"\n{'Problem Description':<50} | {'Classification':<20}")
    print("-" * 75)

    for prob in problems:
        classification = classify_problem(prob)
        desc = f"C={prob.C} K={prob.K} {prob.Hi}x{prob.Wi} {prob.Y}x{prob.X}"
        print(f"{desc:<50} | {classification:<20}")

    print()

    # =========================================================================
    # Step 4: Select appropriate dispatcher
    # =========================================================================
    print("DISPATCHER SELECTION")
    print("=" * 60)
    print()

    dispatchers = {
        "compute_bound": compute_dispatcher,
        "memory_bound": memory_dispatcher,
        "latency_optimized": latency_dispatcher,
    }

    for prob in problems:
        classification = classify_problem(prob)
        dispatcher = dispatchers[classification]

        kernel = dispatcher.select_kernel(prob)

        print(f"Problem: C={prob.C} K={prob.K} {prob.Hi}x{prob.Wi}")
        print(f"  Classification: {classification}")
        print(f"  Selected kernel: {kernel or 'None'}")
        print()

    # =========================================================================
    # Step 5: Registry merging
    # =========================================================================
    print("REGISTRY MERGING")
    print("=" * 60)
    print()

    # Create a combined registry
    combined_registry = ConvRegistry(name="combined")

    # Add all kernels from all registries
    for cfg in compute_registry.get_kernels():
        combined_registry.register_kernel(cfg)
    for cfg in memory_registry.get_kernels():
        combined_registry.register_kernel(cfg)
    for cfg in latency_registry.get_kernels():
        combined_registry.register_kernel(cfg)

    print(f"Combined registry: {combined_registry.kernel_count} kernels")
    print()

    # =========================================================================
    # Step 6: GPU Execution with different registries
    # =========================================================================
    print("GPU EXECUTION TEST")
    print("=" * 60)
    print()

    runner = GpuConvRunner()
    if runner.is_available():
        print(f"Library: {runner.library_path}")
        print()

        # Test with compute-bound problem
        prob = problems[0]  # C=512 K=512 14x14
        np_dtype = np.float16
        input_np = np.random.uniform(
            -0.5, 0.5, (prob.N, prob.Hi, prob.Wi, prob.G, prob.C)
        ).astype(np_dtype)
        weight_np = np.random.uniform(
            -0.5, 0.5, (prob.G, prob.K, prob.Y, prob.X, prob.C)
        ).astype(np_dtype)

        result = runner.run(input_np, weight_np, prob)

        if result.get("success"):
            print("  *** GPU EXECUTION SUCCESSFUL ***")
            print(f"  Problem: C={prob.C} K={prob.K} {prob.Hi}x{prob.Wi}")
            print(f"  Time:   {result['time_ms']:.4f} ms")
            print(f"  TFLOPS: {result['tflops']:.2f}")
        else:
            print(f"  GPU execution: {result.get('error', 'failed')}")

        runner.cleanup()
    else:
        print("  GPU library not available")
    print()

    # =========================================================================
    # Cleanup and Summary
    # =========================================================================
    cleanup_conv()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Multiple registries allow specialized kernel selection:")
    print()
    print("  1. COMPUTE-BOUND: Large tiles (256x256), intrawave scheduler")
    print("     Use for: Many channels, large feature maps")
    print()
    print("  2. MEMORY-BOUND: Medium tiles (128x128), interwave scheduler")
    print("     Use for: 1x1 convolutions, depthwise, low channel count")
    print()
    print("  3. LATENCY-OPTIMIZED: Small tiles (64x64), small block size")
    print("     Use for: Batch=1 inference, small spatial dimensions")
    print()
    print("Benefits:")
    print("  - Better performance through workload-specific optimization")
    print("  - Reduced kernel search time (smaller registry per workload)")
    print("  - Flexibility to combine or separate registries as needed")
    print("=" * 70)


if __name__ == "__main__":
    main()
