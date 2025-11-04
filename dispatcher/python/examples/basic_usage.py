"""
Basic usage examples for CK Tile Dispatcher
"""

import numpy as np
import ck_tile_dispatcher as ckd


def example_1_simple_gemm():
    """Example 1: Simple GEMM"""
    print("=" * 80)
    print("Example 1: Simple GEMM")
    print("=" * 80)
    
    # Create matrices
    M, N, K = 1024, 1024, 1024
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)
    
    # Perform GEMM
    C = ckd.gemm(A, B)
    
    print(f"✓ Computed C = A @ B")
    print(f"  A shape: {A.shape}")
    print(f"  B shape: {B.shape}")
    print(f"  C shape: {C.shape}")
    print()


def example_2_dispatcher_api():
    """Example 2: Using Dispatcher API"""
    print("=" * 80)
    print("Example 2: Dispatcher API")
    print("=" * 80)
    
    # Create dispatcher
    dispatcher = ckd.Dispatcher(gpu_arch="gfx942")
    
    # Register kernels
    dispatcher.register_kernels("fp16_rcr_essential")
    
    # Create problem
    M, N, K = 2048, 2048, 2048
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)
    
    # Dispatch
    C = dispatcher.gemm(A, B)
    
    print(f"✓ Dispatched GEMM using {dispatcher}")
    print(f"  Problem size: {M}x{N}x{K}")
    print(f"  Registered kernels: {dispatcher.get_registered_kernels()}")
    print()


def example_3_with_scaling():
    """Example 3: GEMM with alpha/beta scaling"""
    print("=" * 80)
    print("Example 3: GEMM with Scaling")
    print("=" * 80)
    
    # Create matrices
    M, N, K = 512, 512, 512
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)
    C = np.random.randn(M, N).astype(np.float16)
    
    # Compute: C = 2.0 * A @ B + 0.5 * C
    alpha, beta = 2.0, 0.5
    C_result = ckd.gemm(A, B, C, alpha=alpha, beta=beta)
    
    print(f"✓ Computed C = {alpha} * A @ B + {beta} * C")
    print(f"  Result shape: {C_result.shape}")
    print()


def example_4_batched_gemm():
    """Example 4: Batched GEMM"""
    print("=" * 80)
    print("Example 4: Batched GEMM")
    print("=" * 80)
    
    # Create batched matrices
    batch_size = 8
    M, N, K = 256, 256, 256
    A = np.random.randn(batch_size, M, K).astype(np.float16)
    B = np.random.randn(batch_size, K, N).astype(np.float16)
    
    # Batched GEMM
    C = ckd.batched_gemm(A, B)
    
    print(f"✓ Computed batched GEMM")
    print(f"  Batch size: {batch_size}")
    print(f"  Problem size: {M}x{N}x{K}")
    print(f"  Output shape: {C.shape}")
    print()


def example_5_benchmarking():
    """Example 5: Benchmarking"""
    print("=" * 80)
    print("Example 5: Benchmarking")
    print("=" * 80)
    
    # Create dispatcher
    dispatcher = ckd.Dispatcher()
    dispatcher.register_kernels("fp16_rcr_essential")
    
    # Benchmark single problem size
    result = ckd.benchmark_kernel(
        dispatcher,
        M=1024, N=1024, K=1024,
        dtype=np.float16,
        num_iterations=100
    )
    
    print(f"✓ Benchmark result:")
    print(f"  Problem size: {result.problem_size}")
    print(f"  Kernel: {result.kernel_name}")
    print(f"  Time: {result.execution_time_ms:.3f} ms")
    print(f"  Performance: {result.gflops:.2f} GFLOPS")
    print(f"  Bandwidth: {result.bandwidth_gb_s:.2f} GB/s")
    print()


def example_6_validation():
    """Example 6: Validation"""
    print("=" * 80)
    print("Example 6: Validation")
    print("=" * 80)
    
    # Create dispatcher
    dispatcher = ckd.Dispatcher()
    dispatcher.register_kernels("fp16_rcr_essential")
    
    # Run validation tests
    results = ckd.validate_dispatcher(dispatcher, num_tests=5)
    
    print(f"✓ Validation complete:")
    print(f"  Tests run: {results['num_tests']}")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print()


def example_7_profiling():
    """Example 7: Profiling"""
    print("=" * 80)
    print("Example 7: Profiling")
    print("=" * 80)
    
    # Create profiler
    profiler = ckd.Profiler()
    
    # Create dispatcher
    dispatcher = ckd.Dispatcher()
    dispatcher.register_kernels("fp16_rcr_essential")
    
    # Profile multiple GEMMs
    with profiler:
        for size in [256, 512, 1024]:
            A = np.random.randn(size, size).astype(np.float16)
            B = np.random.randn(size, size).astype(np.float16)
            C = dispatcher.gemm(A, B)
            
            # Record profile
            profiler.record(
                kernel_name=f"gemm_{size}",
                problem_size=(size, size, size),
                execution_time_ms=1.0,  # Placeholder
                gflops=100.0,  # Placeholder
                bandwidth_gb_s=50.0  # Placeholder
            )
    
    # Print summary
    profiler.print_summary()
    print()


def example_8_system_info():
    """Example 8: System Information"""
    print("=" * 80)
    print("Example 8: System Information")
    print("=" * 80)
    
    # Print dispatcher info
    ckd.info()
    print()
    
    # Print system info
    ckd.print_system_info()
    print()
    
    # Available kernels
    print("Available kernel sets:")
    for kernel_set in ckd.get_available_kernels():
        print(f"  - {kernel_set}")
    print()


def main():
    """Run all examples"""
    examples = [
        example_1_simple_gemm,
        example_2_dispatcher_api,
        example_3_with_scaling,
        example_4_batched_gemm,
        example_5_benchmarking,
        example_6_validation,
        example_7_profiling,
        example_8_system_info,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"✗ Example failed: {e}")
            print()


if __name__ == "__main__":
    main()

