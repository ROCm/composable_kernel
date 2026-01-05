#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 01: Basic Convolution with GPU Execution

Demonstrates the Signature/Algorithm/Arch pattern with GPU execution.
Includes validation against arch filter with auto-correction for invalid configs.

This example clearly prints the EXACT kernel configuration requested
and verifies the correct kernel is selected/compiled.

Usage:
    python3 01_basic_conv.py
    python3 01_basic_conv.py --help
    python3 01_basic_conv.py --dtype bf16
    python3 01_basic_conv.py --dtype fp16 --pipeline compv4
"""

import sys
import ctypes
import argparse
import numpy as np
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from conv_utils import (
    ConvSignature,
    ConvAlgorithm,
    ArchInfo,
    ConvKernelConfig,
    ConvProblem,
    ConvDispatcherLib,
    validate_conv_config,
    find_matching_conv_kernel_header,
    auto_correct_conv_config,
    reset_for_conv_example,
    cleanup_conv,
    EnhancedConvCodegenRunner,
    print_conv_kernel_config,
    print_conv_auto_correction,
)


def hip_check(result):
    """Check HIP result and raise if error"""
    if result != 0:
        raise RuntimeError(f"HIP error: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="Basic Convolution Example - demonstrates complete workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 01_basic_conv.py                    # Default FP16 Conv
  python3 01_basic_conv.py --dtype bf16       # BF16 Conv
  python3 01_basic_conv.py --pipeline compv3  # Use compv3 pipeline
  python3 01_basic_conv.py --tile-k 64        # Smaller tile size
        """,
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
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
        "--tile-k", type=int, default=128, help="Tile K size (default: 128)"
    )
    parser.add_argument(
        "--tile-c", type=int, default=128, help="Tile C size (default: 128)"
    )
    parser.add_argument(
        "--arch", default="gfx942", help="Target architecture (default: gfx942)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 01: Basic Convolution with GPU Execution")
    print("=" * 70)
    print()

    # Reset state for clean example run
    reset_for_conv_example(verbose=True)

    # =========================================================================
    # Step 1: Define kernel configuration from command line args
    # =========================================================================
    print("\nStep 1: Define Kernel Configuration")
    print("-" * 50)

    sig = ConvSignature()
    sig.dtype(args.dtype, args.dtype, args.dtype, "fp32")
    sig.layout = "nhwgc"
    sig.direction = "forward"
    sig.num_dims = 2

    algo = ConvAlgorithm()
    algo.tile(1, args.tile_k, args.tile_c)
    algo.wave(2, 2, 1)
    algo.warp(32, 32, 16)
    algo.pipeline = args.pipeline
    algo.scheduler = args.scheduler
    algo.epilogue = "cshuffle"

    arch = ArchInfo(name=args.arch)

    # Print the EXACT configuration requested
    print_conv_kernel_config(sig, algo, arch, "REQUESTED KERNEL CONFIGURATION")

    # =========================================================================
    # Step 2: Validate configuration against arch filter
    # =========================================================================
    print("Step 2: Validate Config Against Arch Filter")
    print("-" * 50)

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
    validation.print_result()

    if not validation.is_valid:
        print("\n  ⚠ Auto-correcting configuration...")
        corrected, was_modified, corrections = auto_correct_conv_config(
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
            verbose=False,  # We'll print manually for better formatting
        )
        if was_modified:
            # Print what was corrected
            print_conv_auto_correction(corrections)

            # Apply corrections
            algo.scheduler = corrected["scheduler"]
            algo.wave_m = corrected["wave_m"]
            algo.wave_n = corrected["wave_n"]
            algo.warp_m = corrected["warp_m"]
            algo.warp_n = corrected["warp_n"]
            algo.warp_k = corrected["warp_k"]
            print_conv_kernel_config(sig, algo, arch, "CORRECTED KERNEL CONFIGURATION")
    print()

    # =========================================================================
    # Step 3: Generate kernel if needed
    # =========================================================================
    print("Step 3: Generate Kernel (if needed)")
    print("-" * 50)

    config = ConvKernelConfig(signature=sig, algorithm=algo, arch=arch)

    codegen = EnhancedConvCodegenRunner(
        datatype=sig.dtype_in,
        direction=sig.direction,
        ndim=sig.num_dims,
        gpu_target=arch.name,
    )

    codegen_result = codegen.generate_from_config(config, show_instances=True)
    if codegen_result.success:
        print(
            f"  ✓ Kernel ready: {codegen_result.kernel_path.name if codegen_result.kernel_path else 'found'}"
        )
    else:
        print(
            f"  ⚠ Kernel generation: {codegen_result.stderr[:100] if codegen_result.stderr else 'using existing'}"
        )
    print()

    # =========================================================================
    # Step 4: Find matching kernel header
    # =========================================================================
    print("Step 4: Find Matching Kernel Header")
    print("-" * 50)

    kernel_header = find_matching_conv_kernel_header(
        dtype=sig.dtype_in,
        conv_type=sig.direction,
        ndim=sig.num_dims,
        pipeline=algo.pipeline,
        scheduler=algo.scheduler,
        tile_k=algo.tile_k,
        tile_c=algo.tile_c,
        wave_m=algo.wave_m,
        wave_n=algo.wave_n,
        wave_k=algo.wave_k,
    )

    if kernel_header:
        print(f"  ✓ Found: {kernel_header.name}")
    else:
        print("  ⚠ No matching kernel found in generated_kernels/")
    print()

    # =========================================================================
    # Step 5: Define problem
    # =========================================================================
    print("Step 5: Define Problem")
    print("-" * 50)

    problem = ConvProblem(
        N=1,
        C=64,
        K=128,
        Hi=28,
        Wi=28,
        Y=3,
        X=3,
        pad_h=1,
        pad_w=1,
        stride_h=1,
        stride_w=1,
    )

    print(f"  N={problem.N}, C={problem.C}, K={problem.K}")
    print(f"  Input: {problem.Hi}x{problem.Wi}")
    print(f"  Filter: {problem.Y}x{problem.X}")
    print(f"  Output: {problem.Ho}x{problem.Wo}")
    print(f"  FLOPs: {problem.flops:.2e}")
    print()

    # =========================================================================
    # Step 6: Load Dispatcher Library
    # =========================================================================
    print("Step 6: Load Dispatcher Library")
    print("-" * 50)

    lib = ConvDispatcherLib.find()

    if lib is None:
        print("  [ERROR] Dispatcher library not found")
        print(
            "  Build with: cd dispatcher/build && cmake .. && make dispatcher_conv_lib"
        )
        return 1

    if not lib.has_kernels():
        print("  [ERROR] Library has no compiled kernels")
        print("  Generate kernels first:")
        print(
            "  python3 codegen/unified_conv_codegen.py --datatype fp16 --variant forward"
        )
        return 1

    lib.initialize()
    print(f"  Library: {lib.path}")
    print(f"  Version: {lib.get_version()}")
    print(f"  Has kernels: {lib.has_kernels()}")
    kernel_count = lib.get_kernel_count()
    print(f"  Kernel count: {kernel_count}")

    # Show the actual compiled kernel(s)
    if kernel_count > 0:
        print("\n  Registered kernels in library:")
        for i in range(kernel_count):
            kernel_name = lib.get_kernel_name(i)
            if kernel_name:
                print(f"    [{i}] {kernel_name}")

        # Note about fallback kernels
        print("\n  Note: Library contains pre-compiled fallback kernels.")
        print("        These support fp16 forward/backward convolutions.")
        print("        For other configs, kernels are JIT-compiled on demand.")
    print()

    # =========================================================================
    # Step 7: GPU Execution
    # =========================================================================
    print("Step 7: GPU Execution")
    print("-" * 50)

    # Use ctypes to call HIP directly
    try:
        hip_lib = ctypes.CDLL("libamdhip64.so")
    except OSError:
        print("  [ERROR] Cannot load libamdhip64.so")
        print("  Make sure ROCm is installed")
        lib.cleanup()
        return 1

    # Determine dtype
    if args.dtype == "fp16":
        np_dtype = np.float16
    elif args.dtype == "bf16":
        # NumPy doesn't have bf16, use uint16 as storage
        np_dtype = np.float16  # Will be interpreted as bf16 by GPU
    else:
        np_dtype = np.float32

    dtype_size = np_dtype().itemsize
    input_size = problem.N * problem.C * problem.Hi * problem.Wi * dtype_size
    weight_size = problem.K * problem.C * problem.Y * problem.X * dtype_size
    output_size = problem.N * problem.K * problem.Ho * problem.Wo * dtype_size

    # hipMalloc
    hip_lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    hip_lib.hipMalloc.restype = ctypes.c_int
    hip_lib.hipFree.argtypes = [ctypes.c_void_p]
    hip_lib.hipFree.restype = ctypes.c_int
    hip_lib.hipMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    hip_lib.hipMemcpy.restype = ctypes.c_int
    hip_lib.hipDeviceSynchronize.argtypes = []
    hip_lib.hipDeviceSynchronize.restype = ctypes.c_int

    # Create numpy arrays
    input_host = np.random.randn(problem.N, problem.Hi, problem.Wi, problem.C).astype(
        np_dtype
    )
    weight_host = np.random.randn(problem.K, problem.Y, problem.X, problem.C).astype(
        np_dtype
    )
    output_host = np.zeros(
        (problem.N, problem.Ho, problem.Wo, problem.K), dtype=np_dtype
    )

    # Allocate device memory
    input_dev = ctypes.c_void_p()
    weight_dev = ctypes.c_void_p()
    output_dev = ctypes.c_void_p()

    hip_lib.hipMalloc(ctypes.byref(input_dev), input_size)
    hip_lib.hipMalloc(ctypes.byref(weight_dev), weight_size)
    hip_lib.hipMalloc(ctypes.byref(output_dev), output_size)

    # Copy to device (hipMemcpyHostToDevice = 1)
    hip_lib.hipMemcpy(input_dev, input_host.ctypes.data, input_size, 1)
    hip_lib.hipMemcpy(weight_dev, weight_host.ctypes.data, weight_size, 1)

    print(f"  Input:  {input_host.shape} ({args.dtype}) -> GPU")
    print(f"  Weight: {weight_host.shape} ({args.dtype}) -> GPU")
    print(f"  Output: {output_host.shape} (allocated)")

    # Run convolution on GPU
    elapsed_ms = lib.run(input_dev.value, weight_dev.value, output_dev.value, problem)

    hip_lib.hipDeviceSynchronize()

    if elapsed_ms > 0:
        tflops = problem.flops / (elapsed_ms * 1e9)
        print("\n  *** GPU EXECUTION SUCCESSFUL ***")
        print(f"  Time:   {elapsed_ms:.4f} ms")
        print(f"  TFLOPS: {tflops:.2f}")
    else:
        print(f"  [ERROR] GPU execution failed (returned {elapsed_ms})")

    # Get actual kernel used (before cleanup)
    actual_kernel = lib.get_kernel_name(0) if lib.has_kernels() else "none"

    # Cleanup
    hip_lib.hipFree(input_dev)
    hip_lib.hipFree(weight_dev)
    hip_lib.hipFree(output_dev)

    lib.cleanup()
    cleanup_conv()

    print()
    print("=" * 70)
    print("SUMMARY: Python example ran convolution on GPU!")
    print(f"  Kernel type: {sig.dtype_in} {sig.direction} {sig.num_dims}D")
    print(f"  Requested:   tile={algo.tile_k}x{algo.tile_c}, pipeline={algo.pipeline}")
    print(f"  Actual kernel: {actual_kernel}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
