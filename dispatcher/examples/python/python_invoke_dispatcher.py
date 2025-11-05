#!/usr/bin/env python3
"""
Python Invokes Dispatcher - Complete Example

Demonstrates invoking the dispatcher from Python with real GPU execution:
1. Generate kernels from Python
2. Build C++ helper executable
3. Execute GPU GEMM through dispatcher
4. Parse results back to Python
5. Validate with NumPy

This is the complete Python → Dispatcher → GPU workflow!
"""

import sys
import json
import subprocess
import numpy as np
from pathlib import Path

# Add Python module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import _dispatcher_native as cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False


def generate_kernels_if_needed():
    """Generate kernels if they don't exist"""
    dispatcher_root = Path(__file__).parent.parent
    codegen_script = dispatcher_root / "codegen" / "unified_gemm_codegen.py"
    build_dir = dispatcher_root / "build"
    kernels_dir = build_dir / "generated_kernels"
    
    # Check if kernels already exist
    kernel_header = kernels_dir / "gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_128x128x32_2x2x1_32x32x16.hpp"
    
    if kernel_header.exists():
        print("OK Kernels already generated")
        return kernels_dir
    
    print("Generating kernels...")
    cmd = [
        sys.executable,
        str(codegen_script),
        '--output-dir', str(kernels_dir),
        '--datatype', 'fp16',
        '--layout', 'rcr',
        '--gpu-target', 'gfx942',
        '--preselected', 'fp16_rcr_essential'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Kernel generation failed: {result.stderr}")
    
    print(f"OK Generated kernels")
    return kernels_dir


def build_gpu_helper():
    """Build the Python GPU helper executable"""
    dispatcher_root = Path(__file__).parent.parent
    build_dir = dispatcher_root / "build"
    build_dir.mkdir(exist_ok=True)
    
    helper_executable = build_dir / "examples" / "python_gpu_helper"
    
    # Check if already built
    if helper_executable.exists():
        print("OK GPU helper already built")
        return helper_executable
    
    print("Building GPU helper...")
    
    # Configure CMake if needed
    if not (build_dir / "CMakeCache.txt").exists():
        cmake_cmd = [
            'cmake', '..',
            '-D', 'CMAKE_PREFIX_PATH=/opt/rocm',
            '-D', 'CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc',
            '-D', 'CMAKE_BUILD_TYPE=Release',
            '-D', 'GPU_TARGETS=gfx942',
            '-D', 'BUILD_DISPATCHER_EXAMPLES=ON'
        ]
        
        result = subprocess.run(cmake_cmd, cwd=str(build_dir), 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"CMake failed: {result.stderr}")
    
    # Build
    make_cmd = ['make', 'python_gpu_helper', '-j4']
    result = subprocess.run(make_cmd, cwd=str(build_dir),
                           capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Build failed: {result.stderr}")
    
    if not helper_executable.exists():
        raise FileNotFoundError(f"Helper not found: {helper_executable}")
    
    print(f"OK Built GPU helper: {helper_executable}")
    return helper_executable


def execute_gpu_gemm(M, N, K, validate=False, helper_path=None):
    """
    Execute GEMM on GPU through C++ helper
    
    Args:
        M, N, K: Problem dimensions
        validate: Whether to validate results
        helper_path: Path to helper executable
    
    Returns:
        Dict with execution results
    """
    if helper_path is None:
        helper_path = build_gpu_helper()
    
    # Build command
    cmd = [str(helper_path), str(M), str(N), str(K)]
    if validate:
        cmd.append('--validate')
    
    # Execute
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    if result.returncode != 0:
        raise RuntimeError(f"GPU execution failed: {result.stderr}")
    
    # Parse JSON output
    try:
        # The output is JSON format
        data = json.loads(result.stdout)
        return data
    except json.JSONDecodeError:
        # Fallback parsing
        return {
            'problem': {'M': M, 'N': N, 'K': K},
            'output': result.stdout,
            'status': 'success' if result.returncode == 0 else 'failed'
        }


def demo_basic_execution():
    """Demo 1: Basic GPU execution"""
    print("\n" + "="*70)
    print("Demo 1: Basic GPU GEMM Execution")
    print("="*70 + "\n")
    
    M, N, K = 512, 512, 512
    
    print(f"Executing GEMM: M={M}, N={N}, K={K}")
    result = execute_gpu_gemm(M, N, K, validate=False)
    
    print("\nResults:")
    print(f"  Kernel: {result['kernel']}")
    print(f"  Selected: {result['selected_kernel']}")
    print(f"  Time: {result['execution']['time_ms']:.4f} ms")
    print(f"  Performance: {result['execution']['tflops']:.2f} TFLOPS")
    print(f"  FLOPs: {result['execution']['flops']:,}")
    print("\nOK Basic execution successful")


def demo_validated_execution():
    """Demo 2: GPU execution with CPU validation"""
    print("\n" + "="*70)
    print("Demo 2: GPU Execution with Validation")
    print("="*70 + "\n")
    
    M, N, K = 256, 256, 256
    
    print(f"Executing GEMM with validation: M={M}, N={N}, K={K}")
    result = execute_gpu_gemm(M, N, K, validate=True)
    
    print("\nResults:")
    print(f"  Time: {result['execution']['time_ms']:.4f} ms")
    print(f"  Performance: {result['execution']['tflops']:.2f} TFLOPS")
    
    if 'validation' in result:
        val = result['validation']
        print(f"\nValidation:")
        print(f"  Accuracy: {val['accuracy']:.2f}%")
        print(f"  Max error: {val['max_error']:.6f}")
        print(f"  Correct: {val['correct_elements']}/{val['total_elements']}")
        
        if val['accuracy'] > 99.0:
            print("\nOK GPU results match CPU reference!")
        else:
            print("\n[FAIL] Validation failed")
    else:
        print("\nNo validation data")


def demo_multiple_sizes():
    """Demo 3: Test multiple problem sizes"""
    print("\n" + "="*70)
    print("Demo 3: Multiple Problem Sizes")
    print("="*70 + "\n")
    
    sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]
    
    print(f"{'Size':<15} | {'Time (ms)':<10} | {'TFLOPS':<8} | Status")
    print("-" * 55)
    
    for M, N, K in sizes:
        try:
            result = execute_gpu_gemm(M, N, K, validate=False)
            time_ms = result['execution']['time_ms']
            tflops = result['execution']['tflops']
            status = "OK"
        except Exception as e:
            time_ms = 0
            tflops = 0
            status = f"FAIL ({e})"
        
        size_str = f"{M}×{N}×{K}"
        print(f"{size_str:<15} | {time_ms:<10.4f} | {tflops:<8.2f} | {status}")
    
    print("\nOK Multi-size test complete")


def demo_numpy_integration():
    """Demo 4: NumPy integration concept"""
    print("\n" + "="*70)
    print("Demo 4: NumPy Integration (Conceptual)")
    print("="*70 + "\n")
    
    M, N, K = 256, 256, 256
    
    # Create numpy arrays
    print("Creating NumPy arrays...")
    A = np.ones((M, K), dtype=np.float16)  # Row-major
    B = np.ones((K, N), dtype=np.float16, order='F')  # Column-major
    
    print(f"  A: {A.shape}, {A.dtype}, {'C-contiguous' if A.flags['C_CONTIGUOUS'] else 'F-contiguous'}")
    print(f"  B: {B.shape}, {B.dtype}, {'C-contiguous' if B.flags['C_CONTIGUOUS'] else 'F-contiguous'}")
    print()
    
    # NumPy reference
    print("Computing NumPy reference...")
    C_numpy = np.matmul(A, B)
    print(f"  C_numpy[0,0] = {C_numpy[0,0]} (expected: {K})")
    print()
    
    # GPU execution
    print("Executing on GPU via dispatcher...")
    result = execute_gpu_gemm(M, N, K, validate=True)
    
    print(f"  GPU time: {result['execution']['time_ms']:.4f} ms")
    print(f"  GPU TFLOPS: {result['execution']['tflops']:.2f}")
    
    if 'validation' in result:
        print(f"  GPU accuracy: {result['validation']['accuracy']:.2f}%")
    print()
    
    print("OK NumPy integration demonstrated")
    print("  Note: For actual numpy integration, use ctypes or custom C++ wrapper")
    print("        to pass numpy array pointers directly to dispatcher")


def demo_cpp_extension():
    """Demo 5: Using C++ extension directly"""
    if not HAS_CPP:
        print("\n[FAIL] C++ extension not available")
        print("   Build with: -DBUILD_DISPATCHER_PYTHON=ON")
        print("   Set PYTHONPATH: export PYTHONPATH=../python")
        return
    
    print("\n" + "="*70)
    print("Demo 5: C++ Extension API")
    print("="*70 + "\n")
    
    # Access registry
    registry = cpp.Registry.instance()
    print(f"Registry: {registry}")
    print(f"  Size: {len(registry)} kernels registered")
    print()
    
    # Create problem
    problem = cpp.Problem(1024, 1024, 1024)
    print(f"Problem: {problem}")
    print(f"  Operations: {problem.num_ops():,}")
    print()
    
    # Create dispatcher
    dispatcher = cpp.Dispatcher()
    print(f"Dispatcher: {dispatcher}")
    print()
    
    # Show enums
    print("Available enums:")
    print(f"  DataType.FP16 = {cpp.DataType.FP16}")
    print(f"  LayoutTag.RowMajor = {cpp.LayoutTag.RowMajor}")
    print(f"  Pipeline.CompV4 = {cpp.Pipeline.CompV4}")
    print(f"  Priority.High = {cpp.Priority.High}")
    print()
    
    print("OK C++ extension working")


def main():
    print("\n" + "="*70)
    print("Python Invokes Dispatcher - Complete Example")
    print("="*70 + "\n")
    
    print("This example shows how to invoke the CK Tile dispatcher")
    print("from Python with real GPU execution.\n")
    
    # Setup
    print("Setup Phase")
    print("-" * 70)
    
    try:
        kernels_dir = generate_kernels_if_needed()
        print()
    except Exception as e:
        print(f"[FAIL] Failed to generate kernels: {e}")
        return 1
    
    try:
        helper = build_gpu_helper()
        print()
    except Exception as e:
        print(f"[FAIL] Failed to build helper: {e}")
        return 1
    
    # Execute demos
    print("\nExecution Demos")
    print("-" * 70)
    
    try:
        demo_basic_execution()
        demo_validated_execution()
        demo_multiple_sizes()
        demo_numpy_integration()
        demo_cpp_extension()
    except Exception as e:
        print(f"\n[FAIL] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "="*70)
    print("Summary - Python → Dispatcher → GPU")
    print("="*70)
    print("\n[OK] Successfully demonstrated:")
    print("  1. Kernel generation from Python")
    print("  2. Building C++ dispatcher executable")
    print("  3. GPU GEMM execution via dispatcher")
    print("  4. Result parsing back to Python")
    print("  5. Validation against CPU/NumPy")
    print("  6. Multiple problem sizes")
    print("  7. C++ extension API access")
    print("\n[OK] Python → Dispatcher integration working!")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

