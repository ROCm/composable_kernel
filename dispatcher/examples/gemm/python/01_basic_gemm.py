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

This example clearly prints the EXACT kernel configuration requested
and verifies the correct kernel is selected/compiled.

Complexity: ★☆☆☆☆

Usage:
    python3 01_basic_gemm.py
    python3 01_basic_gemm.py --help
    python3 01_basic_gemm.py --dtype bf16
    python3 01_basic_gemm.py --dtype fp16 --pipeline compv3
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
    print_kernel_config,
    print_auto_correction,
)


def main():
    parser = argparse.ArgumentParser(
        description="Basic GEMM Example - demonstrates complete workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 01_basic_gemm.py                    # Default FP16 GEMM
  python3 01_basic_gemm.py --dtype bf16       # BF16 GEMM
  python3 01_basic_gemm.py --dtype fp32       # FP32 GEMM
  python3 01_basic_gemm.py --pipeline compv3  # Use compv3 pipeline
  python3 01_basic_gemm.py --size 2048        # Larger problem size
        """,
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32", "fp8", "int8"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--pipeline",
        default="compv4",
        choices=["compv3", "compv4", "mem"],
        help="Pipeline version (default: compv4)",
    )
    parser.add_argument(
        "--scheduler",
        default="intrawave",
        choices=["intrawave", "interwave"],
        help="Scheduler (default: intrawave)",
    )
    parser.add_argument(
        "--tile-m", type=int, default=128, help="Tile M size (default: 128)"
    )
    parser.add_argument(
        "--tile-n", type=int, default=128, help="Tile N size (default: 128)"
    )
    parser.add_argument(
        "--tile-k", type=int, default=32, help="Tile K size (default: 32)"
    )
    parser.add_argument(
        "--arch", default="gfx942", help="Target architecture (default: gfx942)"
    )
    parser.add_argument(
        "--size", type=int, default=1024, help="Problem size MxNxK (default: 1024)"
    )
    args = parser.parse_args()

    # Reset state for clean example run
    reset_for_example()

    print("=" * 70)
    print("Example 01: Basic GEMM")
    print("=" * 70)

    # =========================================================================
    # Step 1: Define KernelConfig with all parameters
    # =========================================================================
    print("\nStep 1: Define KernelConfig")

    # Determine accumulator type based on dtype
    if args.dtype in ["fp16", "bf16", "fp32", "fp8"]:
        acc_dtype = "fp32"
    elif args.dtype == "int8":
        acc_dtype = "int32"
    else:
        acc_dtype = "fp32"

    # Determine warp tile based on dtype
    if args.dtype == "fp32":
        warp_m, warp_n, warp_k = 16, 16, 4
    elif args.dtype in ["fp8", "int8"]:
        warp_m, warp_n, warp_k = 32, 32, 16
    else:  # fp16, bf16
        warp_m, warp_n, warp_k = 16, 16, 16

    # Define your desired kernel configuration
    # Invalid configs will be auto-corrected
    kernel_config = KernelConfig(
        # Data types
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        dtype_acc=acc_dtype,
        # Layouts (RCR = Row-Column-Row)
        layout_a="row",
        layout_b="row",
        layout_c="row",
        # Tile shape
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        # Wave shape
        wave_m=1,
        wave_n=1,
        wave_k=1,
        # Warp tile
        warp_m=warp_m,
        warp_n=warp_n,
        warp_k=warp_k,
        # Pipeline
        pipeline=args.pipeline,
        scheduler=args.scheduler,
        epilogue="cshuffle",
        # Target
        gfx_arch=args.arch,
    )

    # Print the EXACT configuration requested
    print_kernel_config(kernel_config, "REQUESTED KERNEL CONFIGURATION")

    # =========================================================================
    # Step 2: Setup dispatcher (validates, generates kernel, loads library)
    # =========================================================================
    print("Step 2: Setup Dispatcher")
    print("-" * 50)

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

    # Print the ACTUAL configuration after any auto-correction
    if setup.config != kernel_config:
        # Show what was corrected
        if hasattr(setup, "corrections") and setup.corrections:
            print_auto_correction(kernel_config, setup.config, setup.corrections)
        print_kernel_config(
            setup.config, "ACTUAL KERNEL CONFIGURATION (after auto-correction)"
        )

    print(f"\n  ✓ Dispatcher ready: {dispatcher}")
    print(f"  ✓ Library kernel: {setup.lib.get_kernel_name()}")

    # =========================================================================
    # Step 3: Run GEMM
    # =========================================================================
    print("\nStep 3: Run GEMM")
    print("-" * 50)

    M, N, K = args.size, args.size, args.size
    print(f"  Problem: {M}x{N}x{K}")
    print(f"  Data type: {args.dtype}")

    # Create inputs with appropriate dtype
    np.random.seed(42)
    if args.dtype in ["fp16", "bf16"]:
        np_dtype = np.float16
    elif args.dtype == "fp32":
        np_dtype = np.float32
    elif args.dtype in ["fp8", "int8"]:
        np_dtype = np.float16  # Use fp16 for storage
    else:
        np_dtype = np.float16

    A = np.random.randn(M, K).astype(np_dtype) * 0.1
    B = np.random.randn(K, N).astype(np_dtype) * 0.1

    # Run GEMM
    result = dispatcher.run(A, B, M, N, K)

    print(f"\n  *** GEMM EXECUTION {'SUCCESSFUL' if result.success else 'FAILED'} ***")
    print(f"  Time:   {result.time_ms:.4f} ms")
    print(f"  TFLOPS: {result.tflops:.2f}")

    # =========================================================================
    # Step 4: Verify and cleanup
    # =========================================================================
    print("\nStep 4: Verify Output")
    print("-" * 50)

    C = result.output
    print(f"  C[0,0] = {C[0, 0]:.6f}")
    print(f"  C.sum() = {np.sum(C):.2f}")
    print(f"  C.shape = {C.shape}")

    # Cleanup
    cleanup_gemm()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Kernel:   {args.dtype} GEMM with {args.pipeline} pipeline")
    print(f"  Config:   tile={args.tile_m}x{args.tile_n}x{args.tile_k}")
    print(f"  Problem:  {M}x{N}x{K}")
    print(f"  Result:   {'SUCCESS' if result.success else 'FAILED'}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
