#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 09: Multiple Registries

Demonstrates multiple registries for different optimization targets.


Usage:
    python3 09_multi_registry.py
    python3 09_multi_registry.py --help
    python3 09_multi_registry.py --dtype bf16
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    Registry,
    Dispatcher,
    setup_gemm_dispatcher,
    cleanup_gemm,
    reset_for_example,
    detect_gpu_arch,
)


def main():
    parser = argparse.ArgumentParser(
        description="Multiple Registries Example - optimization-specific registries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 09_multi_registry.py                # Default FP16
  python3 09_multi_registry.py --dtype bf16   # BF16 mode
        """,
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--arch",
        default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo)",
    )
    args = parser.parse_args()

    reset_for_example()

    print("=" * 60)
    print("Example 09: Multiple Registries")
    print("=" * 60)

    # =========================================================================
    # Step 1: Setup base dispatcher
    # =========================================================================
    print("\nStep 1: Setup Base Dispatcher")

    base_config = KernelConfig(
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        tile_m=128,
        tile_n=128,
        tile_k=32,
        gfx_arch=args.arch,
    )

    setup = setup_gemm_dispatcher(base_config, registry_name="base", verbose=True)
    if not setup.success:
        print(f"  ERROR: {setup.error}")
        return 1

    lib = setup.lib
    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    # =========================================================================
    # Step 2: Define configs for different optimization targets
    # =========================================================================
    print("\nStep 2: Define Optimization Targets")

    compute_config = KernelConfig(
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        tile_m=256,
        tile_n=256,
        tile_k=64,
        wave_m=4,
        wave_n=4,
        pipeline="compv4",
        gfx_arch=args.arch,
    )
    memory_config = KernelConfig(
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        tile_m=128,
        tile_n=128,
        tile_k=32,
        wave_m=2,
        wave_n=2,
        pipeline="compv4",
        gfx_arch=args.arch,
    )
    latency_config = KernelConfig(
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        tile_m=64,
        tile_n=64,
        tile_k=32,
        wave_m=1,
        wave_n=1,
        pipeline="compv3",
        gfx_arch=args.arch,
    )

    print(f"  Compute: {compute_config.tile_str} (large matrices)")
    print(f"  Memory:  {memory_config.tile_str} (medium matrices)")
    print(f"  Latency: {latency_config.tile_str} (small matrices)")

    # =========================================================================
    # Step 3: Create registries
    # =========================================================================
    print("\nStep 3: Create Registries")

    compute_registry = Registry(name="compute", lib=lib)
    compute_registry.register_kernel(compute_config)

    memory_registry = Registry(name="memory", lib=lib)
    memory_registry.register_kernel(memory_config)

    latency_registry = Registry(name="latency", lib=lib)
    latency_registry.register_kernel(latency_config)

    # =========================================================================
    # Step 4: Create dispatchers
    # =========================================================================
    print("\nStep 4: Create Dispatchers")

    compute_dispatcher = Dispatcher(registry=compute_registry, lib=lib)
    memory_dispatcher = Dispatcher(registry=memory_registry, lib=lib)
    latency_dispatcher = Dispatcher(registry=latency_registry, lib=lib)

    print(f"  {compute_dispatcher}")
    print(f"  {memory_dispatcher}")
    print(f"  {latency_dispatcher}")

    # =========================================================================
    # Step 5: Smart dispatcher selection
    # =========================================================================
    print("\nStep 5: Smart Dispatcher Selection")

    def select_dispatcher(M: int, N: int, K: int) -> Dispatcher:
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

    print(f"\n  {'Size':<20} {'Registry':>10} {'Time (ms)':>12} {'TFLOPS':>10}")
    print("  " + "-" * 55)

    for M, N, K in test_sizes:
        dispatcher = select_dispatcher(M, N, K)

        if not dispatcher.is_supported(M, N, K):
            continue

        A = np.random.randn(M, K).astype(np_dtype) * 0.1
        B = np.random.randn(K, N).astype(np_dtype) * 0.1

        result = dispatcher.run(A, B, M, N, K)

        if result.success:
            print(
                f"  {M}x{N}x{K:<10} {dispatcher.registry.name:>10} "
                f"{result.time_ms:>12.4f} {result.tflops:>10.2f}"
            )

    # Cleanup
    cleanup_gemm()

    # Summary
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
