#!/usr/bin/env python3
"""
CK Tile Dispatcher - Complete Python Workflow Example

Demonstrates the full end-to-end workflow:
1. Generate CK Tile kernels from Python
2. Build C++ executable with kernels
3. Execute on GPU
4. All from simple Python API

This shows the vision from DISPATCHER.md Appendix A.14-A.15
"""

import sys
import os
from pathlib import Path

# Add Python module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from dispatcher_api import (
    Dispatcher,
    SimpleGemmAPI,
    generate_kernels,
    quick_gemm,
    list_available_presets,
    info as api_info
)

def demo_1_manual_workflow():
    """Demo 1: Manual step-by-step workflow"""
    print("\n" + "="*70)
    print("Demo 1: Manual Workflow")
    print("="*70 + "\n")
    
    dispatcher = Dispatcher(gpu_arch='gfx942')
    
    # Step 1: Generate kernels
    print("Step 1: Generating kernels...")
    result = dispatcher.generate_kernels(
        datatype='fp16',
        layout='rcr',
        preset='essential'
    )
    print(f"  ✓ Generated {result['num_kernels']} kernels\n")
    
    # Step 2: Load kernels
    print("Step 2: Loading kernel metadata...")
    kernels_dir = dispatcher.load_generated_kernels()
    print(f"  ✓ Kernels loaded from {kernels_dir}\n")
    
    # Step 3: Build executable
    print("Step 3: Building GPU executable...")
    try:
        executable = dispatcher.build_gpu_executable()
        print(f"  ✓ Executable built: {executable}\n")
    except Exception as e:
        print(f"  Note: Build requires CMake and ROCm")
        print(f"  Error: {e}\n")
        return
    
    # Step 4: Execute
    print("Step 4: Executing on GPU...")
    try:
        result = dispatcher.run_gpu_gemm(M=1024, N=1024, K=1024, executable=executable)
        
        if result['success']:
            print("  ✓ GPU execution successful!")
            print("\n  Output:")
            for line in result['output'].split('\n'):
                if line.strip() and ('✓' in line or 'GFLOPS' in line or 'Kernel' in line):
                    print(f"    {line}")
        else:
            print("  ✗ Execution failed")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n✓ Manual workflow complete!\n")


def demo_2_simple_api():
    """Demo 2: Simplified API"""
    print("\n" + "="*70)
    print("Demo 2: Simple GEMM API")
    print("="*70 + "\n")
    
    gemm = SimpleGemmAPI(gpu_arch='gfx942')
    
    # All-in-one method
    try:
        result = gemm.run_workflow(
            M=1024,
            N=1024,
            K=1024,
            datatype='fp16',
            layout='rcr'
        )
        
        if result['success']:
            print("✓ Simple API workflow complete!")
        
    except Exception as e:
        print(f"Note: This requires CMake and GPU. Error: {e}")
    
    print()


def demo_3_kernel_generation_only():
    """Demo 3: Just generate kernels (no GPU execution)"""
    print("\n" + "="*70)
    print("Demo 3: Kernel Generation Only")
    print("="*70 + "\n")
    
    print("Generating FP16 RCR essential kernels...")
    
    result = generate_kernels(
        datatype='fp16',
        layout='rcr',
        preset='essential',
        gpu_target='gfx942',
        verbose=True
    )
    
    print(f"\n✓ Generated {result['num_kernels']} kernels")
    print(f"  Output: {result['output_dir']}")
    print(f"  Datatype: {result['datatype']}")
    print(f"  Layout: {result['layout']}\n")
    
    # List generated files
    output_dir = Path(result['output_dir'])
    kernel_files = list(output_dir.glob("gemm_*.hpp"))
    
    if kernel_files:
        print(f"Generated kernel files ({len(kernel_files)}):")
        for kf in kernel_files[:5]:  # Show first 5
            print(f"  - {kf.name}")
        if len(kernel_files) > 5:
            print(f"  ... and {len(kernel_files) - 5} more")
    
    print()


def demo_4_cpp_extension_api():
    """Demo 4: Low-level C++ extension API"""
    print("\n" + "="*70)
    print("Demo 4: C++ Extension API (Low-Level)")
    print("="*70 + "\n")
    
    try:
        import _dispatcher_native as cpp
        print("✓ C++ extension loaded\n")
        
        # Create objects
        print("Creating dispatcher objects...")
        problem = cpp.Problem(1024, 1024, 1024)
        print(f"  Problem: {problem}")
        print(f"  Valid: {problem.is_valid()}")
        print(f"  Ops: {problem.num_ops():,}\n")
        
        # Create kernel key
        print("Creating kernel key...")
        key = cpp.KernelKey()
        key.signature.dtype_a = cpp.DataType.FP16
        key.algorithm.tile_shape.m = 256
        key.algorithm.tile_shape.n = 256
        key.algorithm.tile_shape.k = 32
        print(f"  Kernel ID: {key.encode_identifier()}\n")
        
        # Registry
        print("Accessing registry...")
        registry = cpp.Registry.instance()
        print(f"  Registry size: {len(registry)}\n")
        
        # Dispatcher
        print("Creating dispatcher...")
        dispatcher = cpp.Dispatcher()
        dispatcher.set_strategy(cpp.SelectionStrategy.FirstFit)
        print(f"  Dispatcher: {dispatcher}\n")
        
        print("✓ C++ extension API working!\n")
        
    except ImportError:
        print("✗ C++ extension not available")
        print("  Build with: cmake -DBUILD_DISPATCHER_PYTHON=ON\n")


def demo_5_available_presets():
    """Demo 5: Show available presets"""
    print("\n" + "="*70)
    print("Demo 5: Available Kernel Presets")
    print("="*70 + "\n")
    
    presets = list_available_presets()
    
    print("Available kernel preset combinations:\n")
    for dtype_layout, preset_list in presets.items():
        print(f"  {dtype_layout}:")
        for preset in preset_list:
            print(f"    - {preset}")
    
    print("\nUsage:")
    print("  generate_kernels(datatype='fp16', layout='rcr', preset='essential')")
    print()


def main():
    """Run all demos"""
    print("="*70)
    print("CK Tile Dispatcher - Complete Python API Demo")
    print("="*70)
    
    # Show API info
    api_info()
    
    # Run demos
    demo_1_manual_workflow()
    demo_2_simple_api()
    demo_3_kernel_generation_only()
    demo_4_cpp_extension_api()
    demo_5_available_presets()
    
    # Final summary
    print("="*70)
    print("Summary")
    print("="*70 + "\n")
    
    print("✓ All Python API demos complete!")
    print("\nThe Python API provides:")
    print("  1. Kernel generation (generate_kernels)")
    print("  2. Automatic build (Dispatcher.build_gpu_executable)")
    print("  3. GPU execution (Dispatcher.run_gpu_gemm)")
    print("  4. Simple one-liner (quick_gemm)")
    print("  5. Low-level C++ access (_dispatcher_native)")
    print("\nFor production use:")
    print("  from ck_tile_dispatcher.dispatcher_api import SimpleGemmAPI")
    print("  gemm = SimpleGemmAPI()")
    print("  gemm.ensure_kernels_ready()")
    print("  result = gemm.execute(M=2048, N=2048, K=2048)")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

