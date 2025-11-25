"""
Utility functions for CK Tile Dispatcher
"""

import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np


# ============================================================================
# Kernel Information
# ============================================================================


def get_available_kernels() -> List[str]:
    """
    Get list of available kernel sets

    Returns:
        List of kernel set names
    """
    return [
        # FP16 kernels
        "fp16_rcr_essential",
        "fp16_rcr_compute",
        "fp16_rcr_memory",
        "fp16_rcr_latency",
        "fp16_rcr_multi_d",
        "fp16_rcr_preshuffle",
        # BF16 kernels
        "bf16_rcr_essential",
        "bf16_rcr_compute",
        "bf16_rcr_memory",
        # INT8 kernels
        "int8_rcr_essential",
        "int8_rcr_compute",
        # FP8 kernels
        "fp8_rcr_essential",
        "fp8_rcr_compute",
        # Mixed precision
        "mixed_precision",
    ]


def get_kernel_info(kernel_name: str) -> Dict:
    """
    Get detailed information about a kernel

    Args:
        kernel_name: Name of kernel

    Returns:
        Dictionary with kernel metadata
    """
    # This would query the C++ registry
    # For now, return placeholder
    return {
        "name": kernel_name,
        "dtype": "fp16",
        "tile_size": (256, 256, 32),
        "block_size": 256,
        "pipeline": "default",
    }


# ============================================================================
# Benchmarking
# ============================================================================


@dataclass
class BenchmarkResult:
    """Result of a benchmark run"""

    problem_size: tuple  # (M, N, K)
    kernel_name: str
    execution_time_ms: float
    gflops: float
    bandwidth_gb_s: float
    num_iterations: int

    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)

    def __repr__(self):
        return (
            f"BenchmarkResult({self.problem_size}, "
            f"{self.kernel_name}, {self.gflops:.2f} GFLOPS)"
        )


def benchmark_kernel(
    dispatcher,
    M: int,
    N: int,
    K: int,
    dtype=np.float16,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> BenchmarkResult:
    """
    Benchmark a single kernel configuration

    Args:
        dispatcher: Dispatcher instance
        M, N, K: Problem dimensions
        dtype: Data type
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations

    Returns:
        BenchmarkResult
    """
    from .core import Problem, DataType, LayoutTag

    # Allocate tensors
    A = np.random.randn(M, K).astype(dtype)
    B = np.random.randn(K, N).astype(dtype)
    C = np.zeros((M, N), dtype=dtype)

    # Create problem
    problem = Problem(
        M=M,
        N=N,
        K=K,
        A=A,
        B=B,
        C=C,
        dtype_a=DataType.from_numpy(dtype),
        dtype_b=DataType.from_numpy(dtype),
        dtype_c=DataType.from_numpy(dtype),
        layout_a=LayoutTag.ROW_MAJOR,
        layout_b=LayoutTag.COL_MAJOR,
        layout_c=LayoutTag.ROW_MAJOR,
    )

    # Warmup
    for _ in range(num_warmup):
        dispatcher.dispatch(problem)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = dispatcher.dispatch(problem)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Calculate statistics
    avg_time = np.mean(times)

    # Calculate GFLOPS
    flops = 2.0 * M * N * K
    gflops = flops / (avg_time * 1e6)

    # Calculate bandwidth (GB/s)
    bytes_transferred = (M * K + K * N + M * N) * np.dtype(dtype).itemsize
    bandwidth = bytes_transferred / (avg_time * 1e6)

    return BenchmarkResult(
        problem_size=(M, N, K),
        kernel_name=result.kernel_name if result.success else "failed",
        execution_time_ms=avg_time,
        gflops=gflops,
        bandwidth_gb_s=bandwidth,
        num_iterations=num_iterations,
    )


def benchmark_suite(
    dispatcher,
    problem_sizes: Optional[List[tuple]] = None,
    dtype=np.float16,
    output_file: Optional[str] = None,
) -> List[BenchmarkResult]:
    """
    Run a suite of benchmarks

    Args:
        dispatcher: Dispatcher instance
        problem_sizes: List of (M, N, K) tuples
        dtype: Data type
        output_file: Optional JSON file to save results

    Returns:
        List of BenchmarkResults
    """
    if problem_sizes is None:
        # Default problem sizes
        problem_sizes = [
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ]

    results = []

    print(f"Running benchmark suite with {len(problem_sizes)} problem sizes...")

    for i, (M, N, K) in enumerate(problem_sizes):
        print(f"  [{i + 1}/{len(problem_sizes)}] Benchmarking {M}x{N}x{K}...", end=" ")

        try:
            result = benchmark_kernel(dispatcher, M, N, K, dtype)
            results.append(result)
            print(f"✓ {result.gflops:.2f} GFLOPS")
        except Exception as e:
            print(f"✗ Failed: {e}")

    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\n✓ Results saved to {output_file}")

    return results


# ============================================================================
# Profiling
# ============================================================================


def profile_dispatch(dispatcher, problem, num_iterations: int = 100) -> Dict:
    """
    Profile a single dispatch call

    Args:
        dispatcher: Dispatcher instance
        problem: Problem specification
        num_iterations: Number of iterations

    Returns:
        Dictionary with profiling info
    """
    import cProfile
    import pstats
    from io import StringIO

    # Create profiler
    profiler = cProfile.Profile()

    # Profile dispatch
    profiler.enable()
    for _ in range(num_iterations):
        dispatcher.dispatch(problem)
    profiler.disable()

    # Get statistics
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    return {
        "profile_output": stream.getvalue(),
        "num_iterations": num_iterations,
    }


# ============================================================================
# Validation
# ============================================================================


def validate_gemm(
    A: np.ndarray,
    B: np.ndarray,
    C_actual: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    C_initial: Optional[np.ndarray] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> tuple:
    """
    Validate GEMM result against reference

    Args:
        A, B: Input matrices
        C_actual: Actual output
        alpha, beta: GEMM scalars
        C_initial: Initial C value (for beta != 0)
        rtol, atol: Relative and absolute tolerance

    Returns:
        (is_correct, max_error, mean_error)
    """
    # Compute reference
    C_ref = alpha * (A @ B)
    if beta != 0.0 and C_initial is not None:
        C_ref += beta * C_initial

    # Compute errors
    diff = np.abs(C_actual - C_ref)
    max_error = np.max(diff)
    mean_error = np.mean(diff)

    # Check tolerance
    is_correct = np.allclose(C_actual, C_ref, rtol=rtol, atol=atol)

    return is_correct, max_error, mean_error


def validate_dispatcher(dispatcher, num_tests: int = 10) -> Dict:
    """
    Validate dispatcher with random tests

    Args:
        dispatcher: Dispatcher instance
        num_tests: Number of random tests

    Returns:
        Dictionary with validation results
    """
    from .core import Problem, DataType, LayoutTag

    results = {
        "num_tests": num_tests,
        "passed": 0,
        "failed": 0,
        "errors": [],
    }

    print(f"Running {num_tests} validation tests...")

    for i in range(num_tests):
        # Random problem size
        M = np.random.randint(64, 2048)
        N = np.random.randint(64, 2048)
        K = np.random.randint(64, 2048)

        # Random data
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(K, N).astype(np.float16)
        C = np.zeros((M, N), dtype=np.float16)

        # Create problem
        problem = Problem(
            M=M,
            N=N,
            K=K,
            A=A,
            B=B,
            C=C,
            dtype_a=DataType.FP16,
            dtype_b=DataType.FP16,
            dtype_c=DataType.FP16,
            layout_a=LayoutTag.ROW_MAJOR,
            layout_b=LayoutTag.COL_MAJOR,
            layout_c=LayoutTag.ROW_MAJOR,
        )

        # Dispatch
        result = dispatcher.dispatch(problem)

        if result.success:
            # Validate result
            is_correct, max_err, mean_err = validate_gemm(A, B, C)

            if is_correct:
                results["passed"] += 1
                print(f"  [{i + 1}/{num_tests}] ✓ {M}x{N}x{K} (max_err={max_err:.2e})")
            else:
                results["failed"] += 1
                error_msg = f"Validation failed for {M}x{N}x{K}: max_err={max_err:.2e}"
                results["errors"].append(error_msg)
                print(f"  [{i + 1}/{num_tests}] ✗ {error_msg}")
        else:
            results["failed"] += 1
            error_msg = f"Dispatch failed for {M}x{N}x{K}: {result.error_message}"
            results["errors"].append(error_msg)
            print(f"  [{i + 1}/{num_tests}] ✗ {error_msg}")

    print(f"\nValidation complete: {results['passed']}/{num_tests} passed")

    return results


# ============================================================================
# Visualization
# ============================================================================


def plot_benchmark_results(
    results: List[BenchmarkResult], output_file: Optional[str] = None
):
    """
    Plot benchmark results

    Args:
        results: List of BenchmarkResults
        output_file: Optional file to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    # Extract data
    problem_sizes = [f"{r.problem_size[0]}" for r in results]
    gflops = [r.gflops for r in results]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(problem_sizes, gflops)
    ax.set_xlabel("Problem Size (M=N=K)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title("CK Tile GEMM Performance")
    ax.grid(True, alpha=0.3)

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to {output_file}")
    else:
        plt.show()


# ============================================================================
# Configuration Management
# ============================================================================


def save_config(config: Dict, filename: str):
    """Save configuration to JSON file"""
    with open(filename, "w") as f:
        json.dump(config, f, indent=2)


def load_config(filename: str) -> Dict:
    """Load configuration from JSON file"""
    with open(filename, "r") as f:
        return json.load(f)


# ============================================================================
# System Information
# ============================================================================


def get_system_info() -> Dict:
    """Get system information"""
    import platform

    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }

    # Try to get GPU info
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    return info


def print_system_info():
    """Print system information"""
    info = get_system_info()

    print("System Information:")
    print("=" * 50)
    for key, value in info.items():
        print(f"  {key:20s}: {value}")
    print("=" * 50)
