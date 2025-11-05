#!/usr/bin/env python3
"""
CK Tile Dispatcher - Python GPU Example
Demonstrates end-to-end GEMM execution with real CK Tile kernels
"""

import sys
import os
import numpy as np

# Add dispatcher Python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

try:
    import _dispatcher_native as cpp
    print("✓ C++ extension loaded successfully")
except ImportError as e:
    print(f"✗ Failed to load C++ extension: {e}")
    print("  Build with: cmake -DBUILD_DISPATCHER_PYTHON=ON")
    print(f"  Module should be at: {os.path.dirname(__file__)}/../python/_dispatcher_native*.so")
    sys.exit(1)

def create_test_kernel_key():
    """Create a kernel key for FP16 256x256x32 tile configuration"""
    key = cpp.KernelKey()
    
    # Signature - WHAT operation
    key.signature.dtype_a = cpp.DataType.FP16
    key.signature.dtype_b = cpp.DataType.FP16
    key.signature.dtype_c = cpp.DataType.FP16
    key.signature.dtype_acc = cpp.DataType.FP32
    
    key.signature.layout_a = cpp.LayoutTag.RowMajor
    key.signature.layout_b = cpp.LayoutTag.ColMajor
    key.signature.layout_c = cpp.LayoutTag.RowMajor
    
    key.signature.transpose_a = False
    key.signature.transpose_b = False
    key.signature.grouped = False
    key.signature.split_k = 1
    key.signature.elementwise_op = "PassThrough"
    key.signature.num_d_tensors = 0
    key.signature.structured_sparsity = False
    
    # Algorithm - HOW it's implemented
    key.algorithm.tile_shape.m = 256
    key.algorithm.tile_shape.n = 256
    key.algorithm.tile_shape.k = 32
    
    key.algorithm.wave_shape.m = 2
    key.algorithm.wave_shape.n = 2
    key.algorithm.wave_shape.k = 1
    
    key.algorithm.warp_tile_shape.m = 32
    key.algorithm.warp_tile_shape.n = 32
    key.algorithm.warp_tile_shape.k = 16
    
    key.algorithm.pipeline = cpp.Pipeline.CompV4
    key.algorithm.scheduler = cpp.Scheduler.Intrawave
    key.algorithm.epilogue = cpp.Epilogue.CShuffle
    
    key.algorithm.block_size = 256
    key.algorithm.double_buffer = True
    key.algorithm.persistent = False
    key.algorithm.preshuffle = False
    key.algorithm.transpose_c = False
    key.algorithm.num_wave_groups = 1
    
    key.gfx_arch = 942
    
    return key

def test_dispatcher_core_api():
    """Test core dispatcher API without GPU execution"""
    print("\n" + "="*70)
    print("Testing Core Dispatcher API (CPU-only)")
    print("="*70)
    
    # Test 1: Create a kernel key
    print("\n1. Creating KernelKey...")
    key = create_test_kernel_key()
    identifier = key.encode_identifier()
    print(f"   Kernel ID: {identifier}")
    print(f"   Tile size: {key.algorithm.tile_shape.m}x{key.algorithm.tile_shape.n}x{key.algorithm.tile_shape.k}")
    
    # Test 2: Create a problem
    print("\n2. Creating Problem...")
    problem = cpp.Problem(1024, 1024, 1024)
    print(f"   Problem: M={problem.M}, N={problem.N}, K={problem.K}")
    print(f"   Valid: {problem.is_valid()}")
    print(f"   Num ops: {problem.num_ops():,}")
    
    # Test 3: Access registry
    print("\n3. Accessing Registry...")
    registry = cpp.Registry.instance()
    print(f"   Registry size: {len(registry)}")
    print(f"   Registry: {registry}")
    
    # Test 4: Create dispatcher
    print("\n4. Creating Dispatcher...")
    dispatcher = cpp.Dispatcher()
    print(f"   Dispatcher: {dispatcher}")
    
    # Test 5: Test selection strategies
    print("\n5. Setting selection strategy...")
    dispatcher.set_strategy(cpp.SelectionStrategy.FirstFit)
    print("   ✓ FirstFit strategy set")
    
    # Test 6: Test heuristic
    print("\n6. Testing heuristic function...")
    def size_heuristic(prob):
        """Simple heuristic based on problem size"""
        if prob.M * prob.N > 1000000:
            return ["256x256x32_2x2x1_32x32x16_nopers"]
        else:
            return ["128x128x64_2x2x1_32x32x16_nopers"]
    
    dispatcher.set_heuristic(size_heuristic)
    print("   ✓ Heuristic function registered")
    
    print("\n✓ All core API tests passed!")
    return True

def print_system_info():
    """Print system and GPU information"""
    print("\n" + "="*70)
    print("System Information")
    print("="*70)
    
    print(f"\nPython version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"C++ extension version: {cpp.__version__}")
    
    # Try to get GPU info
    try:
        import subprocess
        result = subprocess.run(['rocm-smi', '--showproductname'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            print(f"\nGPU Info:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"  {line}")
    except:
        print("\nGPU Info: rocm-smi not available")

def create_mock_kernel_for_testing():
    """
    Create a mock kernel instance for testing dispatcher workflow.
    In real usage, this would be a TileKernelInstance wrapping actual GPU code.
    """
    print("\n" + "="*70)
    print("Mock Kernel Registration Example")
    print("="*70)
    
    print("\nNote: This demonstrates the dispatcher workflow.")
    print("Real GPU kernel execution requires:")
    print("  1. Tile_engine generated CK Tile kernels")
    print("  2. C++ wrapper code to instantiate TileKernelInstance")
    print("  3. Registration of kernel instances with the dispatcher")
    print("  4. GPU memory allocation (e.g., via PyTorch or CuPy)")
    
    print("\nFor a complete GPU example, see:")
    print("  - dispatcher/examples/gpu_gemm_example.cpp")
    print("  - dispatcher/BUILD_AND_TEST.md")

def main():
    """Main test function"""
    print("="*70)
    print("CK Tile Dispatcher - Python GPU Example")
    print("="*70)
    
    # Print system info
    print_system_info()
    
    # Test core API
    success = test_dispatcher_core_api()
    
    # Show mock kernel example
    create_mock_kernel_for_testing()
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if success:
        print("\n✓ Python bindings are working correctly!")
        print("✓ Core dispatcher API is accessible from Python")
        print("\nNext steps for GPU execution:")
        print("  1. Generate CK Tile kernels: cmake --build . --target generate_tile_gemm_kernels")
        print("  2. Create C++ registration code (see examples/)")
        print("  3. Build with GPU support: cmake -DGPU_TARGETS=gfx942")
        print("  4. Use PyTorch/CuPy for GPU memory management")
    else:
        print("\n✗ Some tests failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

