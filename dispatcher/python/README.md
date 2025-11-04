# CK Tile Dispatcher - Python Interface

High-level Python bindings for the CK Tile GEMM dispatcher with PyTorch integration.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core API](#core-api)
- [PyTorch Integration](#pytorch-integration)
- [Advanced Features](#advanced-features)
- [Examples](#examples)
- [API Reference](#api-reference)

## Installation

### From Source

```bash
cd dispatcher
mkdir build && cd build
cmake .. -DBUILD_PYTHON=ON
make -j
pip install -e ../python
```

### Requirements

- Python >= 3.8
- NumPy >= 1.19
- PyTorch >= 2.0 (optional, for PyTorch integration)
- ROCm >= 5.7 (for GPU support)

## Quick Start

### Basic GEMM

```python
import numpy as np
import ck_tile_dispatcher as ckd

# Create matrices
A = np.random.randn(1024, 1024).astype(np.float16)
B = np.random.randn(1024, 1024).astype(np.float16)

# Compute C = A @ B
C = ckd.gemm(A, B)
```

### With PyTorch

```python
import torch
from ck_tile_dispatcher import ck_gemm

# Create tensors
A = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
B = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)

# Compute C = A @ B
C = ck_gemm(A, B)
```

## Core API

### Dispatcher Class

The main dispatcher class for kernel selection and execution.

```python
from ck_tile_dispatcher import Dispatcher

# Create dispatcher
dispatcher = Dispatcher(gpu_arch="gfx942")

# Register kernels
dispatcher.register_kernels("fp16_rcr_essential")

# Perform GEMM
C = dispatcher.gemm(A, B)
```

### Problem Specification

```python
from ck_tile_dispatcher import Problem, DataType, LayoutTag

problem = Problem(
    M=1024, N=1024, K=1024,
    A=A, B=B, C=C,
    dtype_a=DataType.FP16,
    dtype_b=DataType.FP16,
    dtype_c=DataType.FP16,
    layout_a=LayoutTag.ROW_MAJOR,
    layout_b=LayoutTag.COL_MAJOR,
    layout_c=LayoutTag.ROW_MAJOR,
    alpha=1.0,
    beta=0.0
)

result = dispatcher.dispatch(problem)
```

### Kernel Selection

```python
# Available kernel sets
kernels = ckd.get_available_kernels()
print(kernels)
# ['fp16_rcr_essential', 'fp16_rcr_compute', 'bf16_rcr_essential', ...]

# Register specific kernel set
dispatcher.register_kernels("fp16_rcr_compute")
```

## PyTorch Integration

### CKLinear Layer

Drop-in replacement for `torch.nn.Linear`:

```python
from ck_tile_dispatcher import CKLinear

# Create layer
layer = CKLinear(1024, 2048).cuda().half()

# Forward pass
output = layer(input)
```

### CK MLP

Multi-layer perceptron using CK Tile:

```python
from ck_tile_dispatcher import CKMLP

# Create MLP
mlp = CKMLP([1024, 2048, 4096, 2048], activation='gelu').cuda().half()

# Forward pass
output = mlp(input)
```

### Model Conversion

Convert existing models to use CK Tile:

```python
from ck_tile_dispatcher import convert_linear_to_ck
import torch.nn as nn

# Original model
model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024)
)

# Convert to CK Tile
model_ck = convert_linear_to_ck(model)
```

### Autograd Support

Full support for automatic differentiation:

```python
from ck_tile_dispatcher import ck_gemm

A = torch.randn(512, 512, device='cuda', requires_grad=True)
B = torch.randn(512, 512, device='cuda', requires_grad=True)

# Forward
C = ck_gemm(A, B)
loss = C.sum()

# Backward
loss.backward()
print(A.grad.shape)  # (512, 512)
```

## Advanced Features

### Benchmarking

```python
from ck_tile_dispatcher import benchmark_kernel, benchmark_suite

# Single benchmark
result = benchmark_kernel(
    dispatcher,
    M=1024, N=1024, K=1024,
    num_iterations=100
)
print(f"Performance: {result.gflops:.2f} GFLOPS")

# Benchmark suite
results = benchmark_suite(
    dispatcher,
    problem_sizes=[(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)],
    output_file="benchmark_results.json"
)
```

### Profiling

```python
from ck_tile_dispatcher import Profiler

# Profile execution
profiler = Profiler()
with profiler:
    C = dispatcher.gemm(A, B)

# Print summary
profiler.print_summary()

# Save report
profiler.save("profile_report.json")
```

### Validation

```python
from ck_tile_dispatcher import validate_dispatcher, validate_gemm

# Validate dispatcher
results = validate_dispatcher(dispatcher, num_tests=10)
print(f"Passed: {results['passed']}/{results['num_tests']}")

# Validate single GEMM
is_correct, max_err, mean_err = validate_gemm(A, B, C)
print(f"Correct: {is_correct}, Max error: {max_err:.2e}")
```

### Comparative Profiling

```python
from ck_tile_dispatcher import ComparativeProfiler
import torch

cp = ComparativeProfiler()
cp.add_implementation("ck_tile", lambda: ck_gemm(A, B))
cp.add_implementation("pytorch", lambda: torch.matmul(A, B))

results = cp.run(num_iterations=100)
cp.print_comparison()
cp.plot_comparison("comparison.png")
```

### Benchmark vs PyTorch

```python
from ck_tile_dispatcher import benchmark_vs_pytorch

results = benchmark_vs_pytorch(
    M=2048, N=2048, K=2048,
    num_iterations=100
)

print(f"CK Tile: {results['ck_tile_gflops']:.2f} GFLOPS")
print(f"PyTorch: {results['pytorch_gflops']:.2f} GFLOPS")
print(f"Speedup: {results['speedup']:.2f}x")
```

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.py` - Core API examples
- `pytorch_examples.py` - PyTorch integration examples

Run examples:

```bash
python examples/basic_usage.py
python examples/pytorch_examples.py
```

## API Reference

### Core Classes

#### `Dispatcher`

Main dispatcher class.

**Constructor:**
```python
Dispatcher(gpu_arch: str = "gfx942")
```

**Methods:**
- `register_kernels(kernel_set: str)` - Register a kernel set
- `dispatch(problem: Problem) -> DispatchResult` - Dispatch a problem
- `gemm(A, B, C=None, alpha=1.0, beta=0.0, transpose_a=False, transpose_b=False) -> ndarray` - High-level GEMM
- `get_registered_kernels() -> List[str]` - Get registered kernel sets
- `clear_cache()` - Clear kernel cache

#### `Problem`

GEMM problem specification.

**Fields:**
- `M, N, K: int` - Problem dimensions
- `A, B, C: ndarray | int` - Input/output matrices or device pointers
- `dtype_a, dtype_b, dtype_c: DataType` - Data types
- `layout_a, layout_b, layout_c: LayoutTag` - Memory layouts
- `batch_size: int` - Batch size (default: 1)
- `alpha, beta: float` - Scaling factors

**Methods:**
- `validate() -> Tuple[bool, str]` - Validate problem

#### `DispatchResult`

Result of kernel dispatch.

**Fields:**
- `success: bool` - Whether dispatch succeeded
- `kernel_name: str` - Name of selected kernel
- `execution_time_ms: float` - Execution time
- `gflops: float` - Performance in GFLOPS
- `error_message: str` - Error message (if failed)

### PyTorch Classes

#### `CKLinear`

Linear layer using CK Tile.

**Constructor:**
```python
CKLinear(in_features: int, out_features: int, bias: bool = True)
```

**Methods:**
- `forward(input: Tensor) -> Tensor` - Forward pass

#### `CKMLP`

Multi-layer perceptron using CK Tile.

**Constructor:**
```python
CKMLP(layer_sizes: List[int], activation: str = 'relu', dropout: float = 0.0)
```

**Methods:**
- `forward(x: Tensor) -> Tensor` - Forward pass

### Utility Functions

#### `get_available_kernels() -> List[str]`

Get list of available kernel sets.

#### `benchmark_kernel(dispatcher, M, N, K, dtype, num_iterations) -> BenchmarkResult`

Benchmark a single kernel configuration.

#### `benchmark_suite(dispatcher, problem_sizes, dtype, output_file) -> List[BenchmarkResult]`

Run a suite of benchmarks.

#### `validate_dispatcher(dispatcher, num_tests) -> Dict`

Validate dispatcher with random tests.

#### `validate_gemm(A, B, C_actual, alpha, beta, rtol, atol) -> Tuple[bool, float, float]`

Validate GEMM result against reference.

### Profiling Classes

#### `Profiler`

Advanced profiler for dispatcher.

**Constructor:**
```python
Profiler(enabled: bool = True)
```

**Methods:**
- `start()` - Start profiling
- `stop()` - Stop profiling
- `record(kernel_name, problem_size, execution_time_ms, gflops, bandwidth_gb_s)` - Record execution
- `reset()` - Reset profiler
- `print_summary()` - Print summary
- `save(filename)` - Save report

#### `ComparativeProfiler`

Compare performance of different implementations.

**Methods:**
- `add_implementation(name, func)` - Add implementation
- `run(num_warmup, num_iterations) -> Dict` - Run benchmarks
- `print_comparison()` - Print comparison table
- `plot_comparison(output_file)` - Plot comparison

### Enums

#### `DataType`

- `FP32` - 32-bit floating point
- `FP16` - 16-bit floating point
- `BF16` - BFloat16
- `FP8_E4M3` - FP8 E4M3
- `FP8_E5M2` - FP8 E5M2
- `BF8` - BFloat8
- `INT8` - 8-bit integer
- `INT32` - 32-bit integer

#### `LayoutTag`

- `ROW_MAJOR` - Row-major layout
- `COL_MAJOR` - Column-major layout

## Performance Tips

1. **Use FP16 for best performance** on modern AMD GPUs
2. **Register only needed kernel sets** to reduce overhead
3. **Reuse dispatcher instances** to benefit from caching
4. **Use batched operations** when possible
5. **Profile your workload** to identify bottlenecks

## Troubleshooting

### Import Error

If you get an import error:

```python
ImportError: cannot import name '_ck_dispatcher_cpp'
```

Make sure the C++ extension is built:

```bash
cd dispatcher/build
cmake .. -DBUILD_PYTHON=ON
make -j
```

### CUDA/ROCm Not Available

If CUDA/ROCm is not available, the dispatcher will fall back to NumPy:

```python
import ck_tile_dispatcher as ckd
ckd.info()  # Check if C++ extension is loaded
```

### Performance Issues

If performance is lower than expected:

1. Check that you're using the right kernel set (e.g., `fp16_rcr_compute` for compute-bound)
2. Verify problem size is large enough to saturate GPU
3. Use profiler to identify bottlenecks
4. Check for memory layout mismatches

## Contributing

Contributions are welcome! Please see the main CK repository for contribution guidelines.

## License

MIT License. See LICENSE file for details.

## Citation

If you use CK Tile Dispatcher in your research, please cite:

```bibtex
@software{ck_tile_dispatcher,
  title = {CK Tile Dispatcher},
  author = {AMD CK Tile Team},
  year = {2025},
  url = {https://github.com/ROCm/composable_kernel}
}
```

