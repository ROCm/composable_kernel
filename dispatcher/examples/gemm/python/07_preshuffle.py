#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 07: Preshuffle GEMM for Inference

Demonstrates weight matrix preshuffling for optimized inference workloads.

Preshuffle transforms the B (weight) matrix layout on the HOST before
sending to GPU. This allows the kernel to use optimized memory access
patterns, reducing bank conflicts and improving throughput.

Benefits:
- Weights are fixed during inference, so shuffle once, use many times
- Optimized warp-level memory access patterns
- Reduced LDS bank conflicts
- Best for large matrices (2048+)

Complexity: ★★★★☆

Usage:
    python3 07_preshuffle.py
    python3 07_preshuffle.py --help
    python3 07_preshuffle.py --dtype bf16
    python3 07_preshuffle.py --preshuffle    # Enable preshuffle transformation
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
    reset_for_example,
    preshuffle_weight_matrix,
    is_preshuffle_supported,
)


def main():
    parser = argparse.ArgumentParser(
        description="Preshuffle GEMM - weight matrix pre-transformation for inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 07_preshuffle.py                    # Standard GEMM (no preshuffle)
  python3 07_preshuffle.py --preshuffle       # Enable preshuffle transformation
  python3 07_preshuffle.py --dtype bf16       # BF16 mode
  python3 07_preshuffle.py --max-size 8192    # Test larger sizes

Preshuffle transforms the B matrix layout for optimized memory access.
The transformation is done ONCE on the host, then the shuffled weights
can be reused for many inference calls.
        """,
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=4096,
        help="Maximum problem size (default: 4096)",
    )
    parser.add_argument(
        "--arch", default="gfx942", help="Target architecture (default: gfx942)"
    )
    parser.add_argument(
        "--preshuffle",
        action="store_true",
        help="Enable preshuffle transformation (demonstrates the concept)",
    )
    args = parser.parse_args()

    reset_for_example()

    print("=" * 70)
    print("Example 07: Preshuffle GEMM for Inference")
    print("=" * 70)

    # =========================================================================
    # Step 1: Setup dispatcher
    # =========================================================================
    print("\nStep 1: Setup Dispatcher")

    # Configuration for inference workloads
    config = KernelConfig()
    config.dtype_a = args.dtype
    config.dtype_b = args.dtype
    config.dtype_c = args.dtype
    config.tile_m = 128
    config.tile_n = 128
    config.tile_k = 32
    config.warp_m = 32
    config.warp_n = 32
    config.warp_k = 16
    config.gfx_arch = args.arch

    # Use preshuffle variant and pipeline if preshuffle is requested
    # Note: actual preshuffle kernels require preshufflev2 pipeline
    # For demonstration, we use standard pipeline but show the preshuffle
    # transformation concept which can be applied to any kernel
    if args.preshuffle:
        config.variant = "preshuffle"  # Enable preshuffle-specific validation
        # Note: Real preshuffle kernels would use:
        # config.pipeline = "preshufflev2"
        # For this demo, we use compv4 with host-side preshuffle transformation
        config.pipeline = "compv4"
    else:
        config.variant = "standard"
        config.pipeline = "compv4"

    setup = setup_gemm_dispatcher(config, registry_name="preshuffle_demo", verbose=True)
    if not setup.success:
        print(f"  ERROR: {setup.error}")
        return 1

    dispatcher = setup.dispatcher
    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    # Check preshuffle support
    preshuffle_enabled = args.preshuffle and is_preshuffle_supported(args.arch)
    if args.preshuffle and not is_preshuffle_supported(args.arch):
        print(f"\n  WARNING: Preshuffle not supported on {args.arch}")
        preshuffle_enabled = False

    print(f"\n  Preshuffle Mode: {'ENABLED' if preshuffle_enabled else 'DISABLED'}")
    print(f"  Warp Tile: {config.warp_m}x{config.warp_n}x{config.warp_k}")

    if preshuffle_enabled:
        print("\n  Preshuffle Transformation:")
        print("    - B matrix will be transformed on host before GPU copy")
        print("    - Layout optimized for warp-level coalesced access")
        print("    - Transform once, reuse for many inference calls")

    # =========================================================================
    # Step 2: Demonstrate preshuffle transformation
    # =========================================================================
    if preshuffle_enabled:
        print("\nStep 2: Demonstrate Preshuffle Transformation")
        print("-" * 50)

        # Small example to show the transformation
        K_demo, N_demo = 64, 64
        B_demo = np.arange(K_demo * N_demo, dtype=np_dtype).reshape(K_demo, N_demo)

        print(f"  Original B shape: {B_demo.shape}")
        print(f"  Original B[0:4, 0:4]:\n{B_demo[0:4, 0:4]}")

        B_shuffled_demo = preshuffle_weight_matrix(
            B_demo,
            warp_tile_n=config.warp_n,
            warp_tile_k=config.warp_k,
            arch=args.arch,
        )

        print(f"\n  Shuffled B shape: {B_shuffled_demo.shape}")
        print(f"  Shuffled B[0:4, 0:4]:\n{B_shuffled_demo[0:4, 0:4]}")
        print("  (Data is reordered for optimized warp-level access)")

    # =========================================================================
    # Step 3: Run GEMM with optional preshuffle
    # =========================================================================
    print(
        f"\nStep 3: Run GEMM {'with' if preshuffle_enabled else 'without'} Preshuffle"
    )

    all_sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    sizes = [(m, n, k) for m, n, k in all_sizes if max(m, n, k) <= args.max_size]

    print(f"\n  {'Size':<20} {'Time (ms)':>12} {'TFLOPS':>10} {'Mode':<12}")
    print("  " + "-" * 58)

    for M, N, K in sizes:
        if not dispatcher.is_supported(M, N, K):
            continue

        # Create input matrices
        A = np.random.randn(M, K).astype(np_dtype) * 0.1
        B = np.random.randn(K, N).astype(np_dtype) * 0.1

        # Apply preshuffle transformation if enabled
        if preshuffle_enabled:
            B_input = preshuffle_weight_matrix(
                B,
                warp_tile_n=config.warp_n,
                warp_tile_k=config.warp_k,
                arch=args.arch,
            )
            mode = "preshuffle"
        else:
            B_input = B
            mode = "standard"

        result = dispatcher.run(A, B_input, M, N, K)

        if result.success:
            print(
                f"  {M}x{N}x{K:<10} {result.time_ms:>12.4f} {result.tflops:>10.2f} {mode:<12}"
            )

    # =========================================================================
    # Step 4: Inference pattern demonstration
    # =========================================================================
    if preshuffle_enabled:
        print("\nStep 4: Inference Pattern (shuffle once, use many times)")
        print("-" * 50)

        M, N, K = 2048, 2048, 2048
        if dispatcher.is_supported(M, N, K):
            # Simulate inference: weights are fixed, only activations change
            B_weights = np.random.randn(K, N).astype(np_dtype) * 0.1

            # Preshuffle weights ONCE (offline, during model loading)
            print("  Preshuffling weights (one-time cost)...")
            import time

            t0 = time.time()
            B_shuffled = preshuffle_weight_matrix(
                B_weights,
                warp_tile_n=config.warp_n,
                warp_tile_k=config.warp_k,
                arch=args.arch,
            )
            shuffle_time = (time.time() - t0) * 1000
            print(f"  Preshuffle time: {shuffle_time:.2f} ms")

            # Run multiple inference calls with same shuffled weights
            print("\n  Running 5 inference calls with pre-shuffled weights:")
            for i in range(5):
                A_batch = np.random.randn(M, K).astype(np_dtype) * 0.1
                result = dispatcher.run(A_batch, B_shuffled, M, N, K)
                if result.success:
                    print(f"    Inference {i + 1}: {result.time_ms:.4f} ms")

    # Cleanup
    cleanup_gemm()

    # Summary
    print("\n" + "=" * 70)
    print("Preshuffle Summary:")
    print("=" * 70)
    print("  Preshuffle transforms B matrix layout for optimized memory access.")
    print()
    print("  Inference Pattern:")
    print("    1. Load model weights (B matrix)")
    print(
        "    2. Preshuffle weights ONCE: B_shuffled = preshuffle_weight_matrix(B, ...)"
    )
    print("    3. For each inference batch:")
    print("       - Create activation matrix A")
    print("       - Run GEMM with pre-shuffled B: C = A @ B_shuffled")
    print()
    print("  Benefits:")
    print("    - Shuffle cost amortized over many inference calls")
    print("    - Optimized warp-level memory access patterns")
    print("    - Reduced LDS bank conflicts")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
