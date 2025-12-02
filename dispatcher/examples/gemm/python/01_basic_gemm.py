#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 01: Basic GEMM

The most explicit example - shows the complete manual workflow:
1. Define KernelConfig with all parameters
2. Setup dispatcher (validates, generates, loads library)
3. Run GEMM
4. Cleanup

The system validates your kernel config against arch_specs_generated.py
and automatically corrects invalid configurations (e.g., unsupported
scheduler/pipeline combinations).

Complexity: ★☆☆☆☆

Usage:
    python3 01_basic_gemm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from ctypes_utils import (
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
    reset_for_example,
)


def main():
    # Reset state for clean example run
    reset_for_example()

    print("=" * 60)
    print("Example 01: Basic GEMM")
    print("=" * 60)

    # =========================================================================
    # Step 1: Define KernelConfig with all parameters
    # =========================================================================
    print("\nStep 1: Define KernelConfig")

    # Define your desired kernel configuration
    # Invalid configs will be auto-corrected
    kernel_config = KernelConfig(
        # Data types
        dtype_a="bf16",
        dtype_b="bf16",
        dtype_c="bf16",
        dtype_acc="fp32",
        # Layouts (RCR = Row-Column-Row)
        layout_a="row",
        layout_b="col",
        layout_c="row",
        # Tile shape
        tile_m=128,
        tile_n=128,
        tile_k=32,
        # Wave shape
        wave_m=2,
        wave_n=2,
        wave_k=1,
        # Warp tile
        warp_m=16,
        warp_n=16,
        warp_k=16,
        # Pipeline
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="cshuffle",
        # Target
        gfx_arch="gfx942",
    )

    kernel_config.print_config()

    # =========================================================================
    # Step 2: Setup dispatcher (validates, generates kernel, loads library)
    # =========================================================================
    print("\nStep 2: Setup Dispatcher")

    setup = setup_gemm_dispatcher(
        config=kernel_config,
        registry_name="basic_gemm",
        verbose=True,
        auto_rebuild=True,  # Rebuild library if dtype mismatch
    )

    if not setup.success:
        print(f"  ERROR: {setup.error}")
        return 1

    dispatcher = setup.dispatcher
    print(f"  Dispatcher: {dispatcher}")

    # =========================================================================
    # Step 3: Run GEMM
    # =========================================================================
    print("\nStep 3: Run GEMM")

    M, N, K = 1024, 1024, 1024
    print(f"  Problem: {M}x{N}x{K}")

    # Create inputs
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float16) * 0.1
    B = np.random.randn(K, N).astype(np.float16) * 0.1

    # Run GEMM
    result = dispatcher.run(A, B, M, N, K)

    print(f"  Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Time:   {result.time_ms:.4f} ms")
    print(f"  TFLOPS: {result.tflops:.2f}")

    # =========================================================================
    # Step 4: Verify and cleanup
    # =========================================================================
    print("\nStep 4: Verify Output")

    C = result.output
    print(f"  C[0,0] = {C[0, 0]:.6f}")
    print(f"  C.sum() = {np.sum(C):.2f}")
    print(f"  C.shape = {C.shape}")

    # Cleanup
    cleanup_gemm()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Data Flow:")
    print("=" * 60)
    print("  KernelConfig ──> setup_gemm_dispatcher() ──> Dispatcher")
    print("                                                  │")
    print("  Inputs (A, B) ─────────────────────────────────>│──> C = A @ B")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
