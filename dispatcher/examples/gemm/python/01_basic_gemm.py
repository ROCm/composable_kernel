#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 01: Basic GEMM

The most explicit example - shows the complete manual workflow:
1. Define KernelConfig with all parameters
2. Generate the kernel code from config
3. Create Registry and register kernel
4. Build dispatcher library
5. Create Dispatcher with registry
6. Define problem and run GEMM

Complexity: ★☆☆☆☆

Usage:
    python3 01_basic_gemm.py
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
    print("Example 01: Basic GEMM (Manual Workflow)")
    print("=" * 60)

    # =========================================================================
    # Step 1: Define KernelConfig with all parameters
    # =========================================================================
    print("\nStep 1: Define KernelConfig")

    kernel_config = KernelConfig(
        # Data types
        dtype_a="fp16",  # Input A: FP16
        dtype_b="fp16",  # Input B: FP16
        dtype_c="fp16",  # Output C: FP16
        dtype_acc="fp32",  # Accumulator: FP32
        # Layouts (RCR = Row-Column-Row)
        layout_a="row",  # A is row-major
        layout_b="col",  # B is column-major
        layout_c="row",  # C is row-major
        # Tile shape
        tile_m=128,
        tile_n=128,
        tile_k=32,
        # Wave shape
        wave_m=2,
        wave_n=2,
        wave_k=1,
        # Warp tile
        warp_m=32,
        warp_n=32,
        warp_k=16,
        # Block and pipeline
        block_size=256,
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="cshuffle",
        # Padding and target
        pad_m=True,
        pad_n=True,
        pad_k=True,
        gfx_arch="gfx942",
    )

    kernel_config.print_config()

    # =========================================================================
    # Step 2: Generate kernel code from config
    # =========================================================================
    print("\nStep 2: Generate Kernel Code")

    codegen = CodegenRunner(
        datatype=kernel_config.dtype_a,
        layout=kernel_config.layout,
        gpu_target=kernel_config.gfx_arch,
    )

    codegen_result = codegen.generate_from_config(kernel_config)

    print(f"  Input:  kernel_config (tile={kernel_config.tile_str})")
    print(f"  Output: {codegen.output_dir}")
    print(f"  Status: {'OK' if codegen_result.success else 'FAILED'}")

    # =========================================================================
    # Step 3: Create Registry and register kernel
    # =========================================================================
    print("\nStep 3: Create Registry")

    registry = Registry(name="basic_gemm_registry")

    # Register our kernel config
    registry.register_kernel(kernel_config)

    print(f"  Registry: {registry}")
    print(f"  Registered: {kernel_config.tile_str}")

    # =========================================================================
    # Step 4: Build/Load dispatcher library
    # =========================================================================
    print("\nStep 4: Load Dispatcher Library")

    lib = DispatcherLib.auto()
    if lib is None:
        print("  ERROR: Could not load dispatcher library")
        return 1

    # Bind library to registry
    registry.bind_library(lib)

    print(f"  Library: {lib.path.name}")
    print(f"  Kernel:  {lib.get_kernel_name()}")

    # =========================================================================
    # Step 5: Create Dispatcher with registry
    # =========================================================================
    print("\nStep 5: Create Dispatcher")

    dispatcher = Dispatcher(registry=registry, lib=lib)

    print(f"  Input:  registry ({registry.name})")
    print(f"  Output: {dispatcher}")

    # =========================================================================
    # Step 6: Define problem dimensions
    # =========================================================================
    print("\nStep 6: Define Problem")

    M, N, K = 1024, 1024, 1024

    print(f"  M = {M}")
    print(f"  N = {N}")
    print(f"  K = {K}")

    # Check support via dispatcher
    is_supported = dispatcher.is_supported(M, N, K)
    print(f"  Supported: {is_supported}")

    if not is_supported:
        print("  ERROR: Problem not supported")
        return 1

    # Select kernel
    selected = dispatcher.select_kernel(M, N, K)
    print(f"  Selected kernel: {selected}")

    # =========================================================================
    # Step 7: Create input matrices
    # =========================================================================
    print("\nStep 7: Create Inputs")

    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float16) * 0.1
    B = np.random.randn(K, N).astype(np.float16) * 0.1

    print(f"  A: shape={A.shape}, dtype={A.dtype}")
    print(f"  B: shape={B.shape}, dtype={B.dtype}")

    # =========================================================================
    # Step 8: Run GEMM via Dispatcher
    # =========================================================================
    print("\nStep 8: Run GEMM")

    # Explicit call: dispatcher.run(A, B, M, N, K)
    result = dispatcher.run(A, B, M, N, K)

    print(f"  Input:  A ({M}x{K}), B ({K}x{N})")
    print(f"  Output: C ({M}x{N})")
    print(f"  Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Time:   {result.time_ms:.4f} ms")
    print(f"  TFLOPS: {result.tflops:.2f}")

    # =========================================================================
    # Step 9: Verify output
    # =========================================================================
    print("\nStep 9: Verify Output")

    C = result.output
    print(f"  C[0,0] = {C[0, 0]:.6f}")
    print(f"  C.sum() = {np.sum(C):.2f}")
    print(f"  C.shape = {C.shape}")

    # =========================================================================
    # Summary: Data flow
    # =========================================================================
    print("\n" + "=" * 60)
    print("Data Flow:")
    print("=" * 60)
    print("  KernelConfig ──┬──> CodegenRunner ──> kernel.hpp")
    print("                 │")
    print("                 └──> Registry ──> Dispatcher")
    print("                                       │")
    print("  Problem (M,N,K) ────────────────────>│")
    print("                                       │")
    print("  Inputs (A, B) ──────────────────────>│──> C = A @ B")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
