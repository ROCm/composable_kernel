#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example usage of CK Tile Dispatcher Python API
"""

try:
    from ck_tile.dispatcher import (
        Dispatcher,
        Registry,
        Problem,
        KernelKey,
        DataType,
        LayoutTag,
        Pipeline,
        Scheduler,
        Epilogue,
    )
except ImportError:
    print("Error: Dispatcher Python bindings not built")
    print("Build with: cmake -DBUILD_DISPATCHER_PYTHON=ON")
    exit(1)


def example_query_registry():
    """Example: Query the kernel registry"""
    print("=== Query Registry Example ===")
    
    registry = Registry.instance()
    print(f"Total registered kernels: {len(registry)}")
    
    # Get all kernels
    all_kernels = registry.get_all()
    for kernel in all_kernels:
        print(f"  - {kernel.get_name()}")
        key = kernel.get_key()
        print(f"    Identifier: {key.encode_identifier()}")
        print(f"    Tile: {key.algorithm.tile_shape.m}x{key.algorithm.tile_shape.n}x{key.algorithm.tile_shape.k}")
        print(f"    Persistent: {key.algorithm.persistent}")


def example_create_problem():
    """Example: Create and configure a Problem"""
    print("\n=== Create Problem Example ===")
    
    # Create problem with dimensions
    problem = Problem(M=1024, N=1024, K=1024)
    print(f"Problem: {problem}")
    print(f"  Valid: {problem.is_valid()}")
    print(f"  Operations: {problem.num_ops()}")
    
    # Configure preferences
    problem.prefer_persistent = True
    problem.enable_validation = False
    problem.k_batch = 1
    
    print(f"  Prefer persistent: {problem.prefer_persistent}")


def example_kernel_selection():
    """Example: Select kernels based on problem"""
    print("\n=== Kernel Selection Example ===")
    
    dispatcher = Dispatcher()
    problem = Problem(M=2048, N=2048, K=1024)
    
    # Select kernel automatically
    kernel = dispatcher.select_kernel(problem)
    if kernel:
        print(f"Selected kernel: {kernel.get_name()}")
        print(f"  Supports problem: {kernel.supports(problem)}")
    else:
        print("No suitable kernel found")


def example_filter_kernels():
    """Example: Filter kernels by criteria"""
    print("\n=== Filter Kernels Example ===")
    
    registry = Registry.instance()
    
    # Filter for persistent kernels
    persistent_kernels = registry.filter(
        lambda k: k.get_key().algorithm.persistent
    )
    print(f"Persistent kernels: {len(persistent_kernels)}")
    
    # Filter for large tile sizes
    large_tile_kernels = registry.filter(
        lambda k: k.get_key().algorithm.tile_shape.m >= 256
    )
    print(f"Large tile (>=256) kernels: {len(large_tile_kernels)}")


def example_kernel_key():
    """Example: Work with KernelKey"""
    print("\n=== KernelKey Example ===")
    
    # Create a KernelKey
    key = KernelKey()
    
    # Configure signature
    key.signature.dtype_a = DataType.FP16
    key.signature.dtype_b = DataType.FP16
    key.signature.dtype_c = DataType.FP16
    key.signature.dtype_acc = DataType.FP32
    key.signature.layout_a = LayoutTag.RowMajor
    key.signature.layout_b = LayoutTag.ColMajor
    key.signature.layout_c = LayoutTag.RowMajor
    key.signature.elementwise_op = "PassThrough"
    key.signature.num_d_tensors = 0
    
    # Configure algorithm
    key.algorithm.tile_shape.m = 256
    key.algorithm.tile_shape.n = 256
    key.algorithm.tile_shape.k = 32
    key.algorithm.wave_shape.m = 2
    key.algorithm.wave_shape.n = 2
    key.algorithm.wave_shape.k = 1
    key.algorithm.warp_tile_shape.m = 32
    key.algorithm.warp_tile_shape.n = 32
    key.algorithm.warp_tile_shape.k = 16
    key.algorithm.pipeline = Pipeline.CompV4
    key.algorithm.scheduler = Scheduler.Intrawave
    key.algorithm.epilogue = Epilogue.CShuffle
    key.algorithm.block_size = 256
    key.algorithm.persistent = True
    
    key.gfx_arch = 942
    
    print(f"KernelKey: {key}")
    print(f"  Identifier: {key.encode_identifier()}")
    
    # Lookup kernel by key
    registry = Registry.instance()
    kernel = registry.lookup(key)
    if kernel:
        print(f"  Found kernel: {kernel.get_name()}")
    else:
        print("  Kernel not found in registry")


def example_heuristics():
    """Example: Use heuristics for kernel selection"""
    print("\n=== Heuristics Example ===")
    
    def my_heuristic(problem):
        """Simple heuristic: prefer larger tiles for larger problems"""
        candidates = []
        
        if problem.M >= 2048 and problem.N >= 2048:
            # Large problem
            candidates.append("256x256x32_2x2x1_32x32x16_persist")
            candidates.append("256x256x64_2x2x1_32x32x16_persist")
        else:
            # Smaller problem
            candidates.append("128x128x32_2x2x1_32x32x16_persist")
            candidates.append("128x128x64_2x2x1_32x32x16_persist")
        
        return candidates
    
    dispatcher = Dispatcher()
    dispatcher.set_heuristic(my_heuristic)
    
    # Test with different problem sizes
    for M, N, K in [(1024, 1024, 1024), (4096, 4096, 2048)]:
        problem = Problem(M, N, K)
        kernel = dispatcher.select_kernel(problem)
        if kernel:
            print(f"Problem {M}x{N}x{K} -> {kernel.get_name()}")
        else:
            print(f"Problem {M}x{N}x{K} -> No kernel found")


def main():
    """Run all examples"""
    print("CK Tile Dispatcher Python API Examples\n")
    
    # Note: These examples assume kernels are registered
    # In practice, you would register kernels first
    
    example_create_problem()
    example_kernel_key()
    example_query_registry()
    example_filter_kernels()
    example_kernel_selection()
    example_heuristics()
    
    print("\n=== Examples Complete ===")


if __name__ == "__main__":
    main()

