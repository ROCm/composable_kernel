#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 09: Multiple Registries

Demonstrates creating multiple registries with different kernel configurations
for different optimization targets (compute, memory, latency).

Complexity: ★★★★★

Usage:
    python3 09_multi_registry.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    CodegenRunner,
    DispatcherLib,
    Registry,
    Dispatcher,
)


def main():
    print("=" * 60)
    print("Example 09: Multiple Registries")
    print("=" * 60)

    # =========================================================================
    # Step 1: Define kernel configs for different optimization targets
    # =========================================================================
    print("\nStep 1: Define Kernel Configurations")

    # Compute-optimized: Large tiles for maximum throughput
    compute_config = KernelConfig(
        tile_m=256,
        tile_n=256,
        tile_k=64,
        wave_m=4,
        wave_n=4,
        wave_k=1,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        block_size=256,
        pipeline="compv4",
    )
    print("\n  compute_config (large matrices):")
    print(f"    Tile: {compute_config.tile_str}")
    print("    Best for: M*N >= 4096*4096")

    # Memory-optimized: Medium tiles for balanced workloads
    memory_config = KernelConfig(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        wave_m=2,
        wave_n=2,
        wave_k=1,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        block_size=256,
        pipeline="compv4",
    )
    print("\n  memory_config (medium matrices):")
    print(f"    Tile: {memory_config.tile_str}")
    print("    Best for: 1024*1024 <= M*N < 4096*4096")

    # Latency-optimized: Small tiles for quick response
    latency_config = KernelConfig(
        tile_m=64,
        tile_n=64,
        tile_k=32,
        wave_m=1,
        wave_n=1,
        wave_k=1,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        block_size=64,
        pipeline="compv3",
    )
    print("\n  latency_config (small matrices):")
    print(f"    Tile: {latency_config.tile_str}")
    print("    Best for: M*N < 1024*1024")

    # =========================================================================
    # Step 2: Create registries for each optimization target
    # =========================================================================
    print("\nStep 2: Create Registries")

    compute_registry = Registry(name="compute")
    compute_registry.register_kernel(compute_config)
    print(f"  {compute_registry}")

    memory_registry = Registry(name="memory")
    memory_registry.register_kernel(memory_config)
    print(f"  {memory_registry}")

    latency_registry = Registry(name="latency")
    latency_registry.register_kernel(latency_config)
    print(f"  {latency_registry}")

    # =========================================================================
    # Step 3: Generate kernels and load library
    # =========================================================================
    print("\nStep 3: Generate Kernels")

    codegen = CodegenRunner()
    result = codegen.generate("standard")
    print(f"  Generated {result.kernel_count} kernels")

    lib = DispatcherLib.auto()
    if lib is None:
        print("  ERROR: Could not load library")
        return 1

    # Bind library to all registries
    compute_registry.bind_library(lib)
    memory_registry.bind_library(lib)
    latency_registry.bind_library(lib)

    # =========================================================================
    # Step 4: Create dispatchers for each registry
    # =========================================================================
    print("\nStep 4: Create Dispatchers")

    compute_dispatcher = Dispatcher(registry=compute_registry, lib=lib)
    memory_dispatcher = Dispatcher(registry=memory_registry, lib=lib)
    latency_dispatcher = Dispatcher(registry=latency_registry, lib=lib)

    print(f"  {compute_dispatcher}")
    print(f"  {memory_dispatcher}")
    print(f"  {latency_dispatcher}")

    # =========================================================================
    # Step 5: Smart dispatcher selection based on problem size
    # =========================================================================
    print("\nStep 5: Smart Dispatcher Selection")

    def select_dispatcher(M: int, N: int, K: int) -> Dispatcher:
        """Select best dispatcher based on problem size."""
        elements = M * N
        if elements >= 4096 * 4096:
            return compute_dispatcher
        elif elements >= 1024 * 1024:
            return memory_dispatcher
        else:
            return latency_dispatcher

    test_sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    print(f"\n  {'Size':<20} {'Elements':>12} {'Registry':>12}")
    print("  " + "-" * 50)

    for M, N, K in test_sizes:
        dispatcher = select_dispatcher(M, N, K)
        print(f"  {M}x{N}x{K:<10} {M * N:>12,} {dispatcher.registry.name:>12}")

    # =========================================================================
    # Step 6: Run GEMM with auto-selected dispatcher
    # =========================================================================
    print("\nStep 6: Run GEMM with Smart Selection")

    print(f"\n  {'Size':<20} {'Registry':>10} {'Time (ms)':>12} {'TFLOPS':>10}")
    print("  " + "-" * 55)

    for M, N, K in test_sizes:
        # Select best dispatcher for this problem
        dispatcher = select_dispatcher(M, N, K)

        if not dispatcher.is_supported(M, N, K):
            continue

        # Create inputs
        A = np.random.randn(M, K).astype(np.float16) * 0.1
        B = np.random.randn(K, N).astype(np.float16) * 0.1

        # Run with selected dispatcher
        result = dispatcher.run(A, B, M, N, K)

        if result.success:
            print(
                f"  {M}x{N}x{K:<10} {dispatcher.registry.name:>10} "
                f"{result.time_ms:>12.4f} {result.tflops:>10.2f}"
            )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Multi-Registry Pattern:")
    print("=" * 60)
    print("  1. Define KernelConfig for each optimization target")
    print("  2. Create Registry for each target")
    print("  3. Register configs to appropriate registries")
    print("  4. Create Dispatcher for each registry")
    print("  5. Select dispatcher based on problem characteristics")
    print("  6. Run GEMM with selected dispatcher")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
