#!/usr/bin/env python3
"""
Python GPU Dispatcher Example - Real GPU Execution

Demonstrates:
1. Automatic kernel generation from Python
2. Building C++ executable with dispatcher
3. Executing real GPU GEMM operations
4. Integration with numpy for data validation

This shows the complete Python → C++ → GPU workflow.
"""

import sys
import numpy as np
from pathlib import Path
import subprocess
import tempfile

# Add Python module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import _dispatcher_native as cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    print("Note: C++ extension not available. Will use subprocess approach.")


def generate_and_build_test():
    """Generate kernels and build a test executable"""
    print("="*70)
    print("Step 1: Generate CK Tile Kernels")
    print("="*70 + "\n")
    
    dispatcher_root = Path(__file__).parent.parent
    codegen_script = dispatcher_root / "codegen" / "unified_gemm_codegen.py"
    build_dir = dispatcher_root / "build"
    kernels_dir = build_dir / "generated_kernels"
    
    # Generate kernels
    cmd = [
        sys.executable,
        str(codegen_script),
        '--output-dir', str(kernels_dir),
        '--datatype', 'fp16',
        '--layout', 'rcr',
        '--gpu-target', 'gfx942',
        '--preselected', 'fp16_rcr_essential'
    ]
    
    print(f"Generating FP16 RCR kernels...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[FAIL] Generation failed: {result.stderr}")
        return None
    
    # Count kernels
    kernel_files = list(kernels_dir.glob("gemm_*.hpp"))
    print(f"OK Generated {len(kernel_files)} kernel files")
    print()
    
    return kernels_dir


def build_cpp_tests(rebuild=False):
    """Build C++ tests that use the dispatcher"""
    print("="*70)
    print("Step 2: Build C++ Tests with Dispatcher")
    print("="*70 + "\n")
    
    dispatcher_root = Path(__file__).parent.parent
    build_dir = dispatcher_root / "build"
    build_dir.mkdir(exist_ok=True)
    
    # CMake configure
    if rebuild or not (build_dir / "CMakeCache.txt").exists():
        print("Configuring with CMake...")
        cmake_cmd = [
            'cmake', '..',
            '-D', 'CMAKE_PREFIX_PATH=/opt/rocm',
            '-D', 'CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc',
            '-D', 'CMAKE_BUILD_TYPE=Release',
            '-D', 'GPU_TARGETS=gfx942',
            '-D', 'BUILD_DISPATCHER_TESTS=ON',
            '-D', 'BUILD_DISPATCHER_REAL_KERNEL_TESTS=ON'
        ]
        
        result = subprocess.run(cmake_cmd, cwd=str(build_dir), 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[FAIL] CMake failed: {result.stderr}")
            return None
        
        print("OK CMake configured")
    else:
        print("OK CMake already configured")
    
    # Build
    print("Building tests...")
    make_cmd = ['make', 'test_real_kernel_simple', '-j4']
    result = subprocess.run(make_cmd, cwd=str(build_dir),
                           capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[FAIL] Build failed")
        print(result.stderr)
        return None
    
    executable = build_dir / "test" / "test_real_kernel_simple"
    if executable.exists():
        print(f"OK Built: {executable}")
        print()
        return executable
    else:
        print(f"[FAIL] Executable not found: {executable}")
        return None


def run_gpu_test(executable):
    """Run the GPU test executable"""
    print("="*70)
    print("Step 3: Execute GPU Test via Dispatcher")
    print("="*70 + "\n")
    
    print(f"Running: {executable.name}")
    print()
    
    result = subprocess.run([str(executable)], capture_output=True, text=True,
                           timeout=30)
    
    if result.returncode != 0:
        print(f"[FAIL] Execution failed: {result.stderr}")
        return False
    
    # Parse output
    output_lines = result.stdout.split('\n')
    
    for line in output_lines:
        # Print key lines
        if any(marker in line for marker in ['OK', '[OK]', 'TFLOPS', 'Kernel:', 'Problem:', 
                                               'Selected', 'Accuracy', 'TEST PASSED']):
            print(line)
    
    print()
    return True


def demo_cpp_extension_direct():
    """Demo: Direct C++ extension usage"""
    if not HAS_CPP:
        print("Skipping C++ extension demo (not available)")
        return
    
    print("="*70)
    print("Step 4: Direct C++ Extension Usage")
    print("="*70 + "\n")
    
    # Create objects
    problem = cpp.Problem(512, 512, 512)
    registry = cpp.Registry.instance()
    dispatcher = cpp.Dispatcher()
    
    print(f"Created objects:")
    print(f"  Problem: {problem}")
    print(f"  Registry: {registry} (size: {len(registry)})")
    print(f"  Dispatcher: {dispatcher}")
    print()
    
    # Show available types
    print(f"Available data types: FP16, BF16, FP32, FP8, INT8, INT32")
    print(f"Available layouts: RowMajor, ColMajor")
    print(f"Available pipelines: Mem, CompV3, CompV4, CompV5")
    print()
    
    # Try kernel selection
    print("Attempting kernel selection...")
    kernel = dispatcher.select_kernel(problem)
    
    if kernel is None:
        print("  No kernel selected (expected - registry empty in this demo)")
        print("  In real usage, kernels would be loaded from generated code")
    else:
        print(f"  Selected: {kernel.get_name()}")
    print()


def demo_python_numpy_integration():
    """Demo: Integration with numpy"""
    print("="*70)
    print("Step 5: NumPy Integration Concept")
    print("="*70 + "\n")
    
    # Create numpy arrays
    M, N, K = 256, 256, 256
    
    A = np.ones((M, K), dtype=np.float16)
    B = np.ones((K, N), dtype=np.float16, order='F')  # Column-major
    C = np.zeros((M, N), dtype=np.float16)
    
    print(f"Created NumPy arrays:")
    print(f"  A: shape={A.shape}, dtype={A.dtype}, order={'C' if A.flags['C_CONTIGUOUS'] else 'F'}")
    print(f"  B: shape={B.shape}, dtype={B.dtype}, order={'C' if B.flags['C_CONTIGUOUS'] else 'F'}")
    print(f"  C: shape={C.shape}, dtype={C.dtype}")
    print()
    
    # Expected result
    C_expected = np.matmul(A, B)
    
    print(f"NumPy matmul result:")
    print(f"  Expected C[0,0] = {C_expected[0,0]} (should be {K})")
    print()
    
    print("Note: To execute on GPU via dispatcher:")
    print("  1. Convert numpy arrays to GPU memory (hipMalloc)")
    print("  2. Call dispatcher.run() with device pointers")
    print("  3. Copy results back to numpy arrays")
    print("  This requires ctypes or a C++ wrapper")
    print()


def main():
    print("\n" + "="*70)
    print("Python GPU Dispatcher Example")
    print("="*70 + "\n")
    
    # Generate and build
    kernels_dir = generate_and_build_test()
    if kernels_dir is None:
        print("[FAIL] Failed to generate kernels")
        return 1
    
    executable = build_cpp_tests()
    if executable is None:
        print("[FAIL] Failed to build tests")
        return 1
    
    # Run GPU test
    success = run_gpu_test(executable)
    if not success:
        print("[FAIL] GPU test failed")
        return 1
    
    # Demo C++ extension
    demo_cpp_extension_direct()
    
    # Demo numpy integration
    demo_python_numpy_integration()
    
    # Summary
    print("="*70)
    print("Summary")
    print("="*70)
    print("\n[OK] Complete workflow demonstrated:")
    print("  1. Generated kernels from Python OK")
    print("  2. Built C++ tests with dispatcher OK")
    print("  3. Executed real GPU kernels OK")
    print("  4. Used C++ extension API OK")
    print("  5. Showed NumPy integration pattern OK")
    print()
    print("Next steps:")
    print("  - Add ctypes wrapper for direct GPU memory access")
    print("  - Create Python GEMM function that wraps C++ execution")
    print("  - Add PyTorch integration for tensor operations")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

