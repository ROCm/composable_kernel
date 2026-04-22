# GEMM Python Examples

CK Tile Dispatcher Python examples for GEMM (General Matrix Multiplication) operations.

> **Main Documentation**: [Dispatcher README](../../../README.md) | [Examples Overview](../../README.md)

## Quick Start

### Build Library

```bash
cd /path/to/composable_kernel/dispatcher
mkdir -p build && cd build

cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DBUILD_DISPATCHER_EXAMPLES=ON

# Build Python library (kernels generated automatically)
make dispatcher_gemm_lib -j$(nproc)
```

### Run Examples

```bash
cd /path/to/composable_kernel/dispatcher

python3 examples/gemm/python/01_basic_gemm.py
python3 examples/gemm/python/04_validation.py
python3 examples/gemm/python/07_stress_test.py
python3 examples/gemm/python/08_heuristics.py
```

## Examples

| Example | Description |
|---------|-------------|
| [01_basic_gemm.py](01_basic_gemm.py) | Basic GEMM with multi-kernel support |
| [02_batch_gemm.py](02_batch_gemm.py) | Batched GEMM operations |
| [03_benchmark.py](03_benchmark.py) | Performance benchmarking |
| [04_validation.py](04_validation.py) | CPU reference validation |
| [05_numpy_integration.py](05_numpy_integration.py) | NumPy array integration |
| [06_json_export.py](06_json_export.py) | Registry JSON export |
| [07_stress_test.py](07_stress_test.py) | Multi-kernel stress testing |
| [08_heuristics.py](08_heuristics.py) | Heuristic-based kernel selection |
| [09_multi_registry.py](09_multi_registry.py) | Multiple registries |
| [10_advanced_benchmark.py](10_advanced_benchmark.py) | Advanced benchmark with full control |
| [11_json_import.py](11_json_import.py) | Import kernels from JSON |

## Example Details

### 01_basic_gemm.py - Basic GEMM
Demonstrates the Python API with multi-kernel support:

```python
from ctypes_utils import KernelConfig, setup_gemm_dispatcher, print_kernel_config_table

# Define multiple kernel configurations
kernels = [
    KernelConfig(
        tile_m=128, tile_n=128, tile_k=32,
        wave_m=2, wave_n=2, wave_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        pipeline="compv3", scheduler="intrawave"
    ),
    KernelConfig(
        tile_m=256, tile_n=256, tile_k=32,
        wave_m=2, wave_n=2, wave_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        pipeline="compv4", scheduler="intrawave"
    ),
]

# Display configurations
print_kernel_config_table(kernels)

# Set up dispatcher with all kernels
lib, dispatcher, registry = setup_gemm_dispatcher(kernels)

# Run GEMM
elapsed_ms = run_gemm(lib, M, N, K, ...)
```

### 02_batch_gemm.py - Batch GEMM
Batched matrix multiplication:
- Multiple independent GEMM operations
- Batch dimension handling

### 03_benchmark.py - Benchmarking
Performance measurement:
- GPU timing
- TFLOPS calculation
- Multiple iterations

### 04_validation.py - Validation
Correctness verification:
- NumPy reference implementation
- Tolerance-based validation
- Error reporting

### 05_numpy_integration.py - NumPy Integration
Seamless NumPy integration:
- NumPy arrays to GPU buffers
- Results back to NumPy
- Automatic type conversion

### 06_json_export.py - JSON Export
Registry serialization for tool integration:
- Export kernel configurations
- Machine-readable format

### 07_stress_test.py - Stress Testing
Comprehensive multi-kernel stress testing:

```python
from ctypes_utils import KernelConfig, setup_gemm_dispatcher, print_kernel_config_table

# Define 48 unique kernel configurations
kernels = [
    KernelConfig(tile_m=128, tile_n=128, tile_k=32, pipeline="compv3", ...),
    KernelConfig(tile_m=256, tile_n=256, tile_k=32, pipeline="compv4", ...),
    KernelConfig(tile_m=128, tile_n=256, tile_k=64, pipeline="compv3", ...),
    # ... many more configurations
]

# Test each kernel
for i, kernel in enumerate(kernels):
    lib, dispatcher, registry = setup_gemm_dispatcher([kernel])
    result = run_and_validate(lib, M, N, K, seed=42 + i)  # Different seed per kernel
    print(f"Kernel {i}: {result.max_err:.6e} {'PASS' if result.passed else 'FAIL'}")
```

**Features:**
- 48 unique kernel configurations
- Various tile sizes, pipelines, and schedulers
- Per-kernel validation with unique random seeds
- Performance reporting

### 08_heuristics.py - Heuristic Selection
Custom kernel selection based on problem characteristics:

```python
# Define kernel pools for different strategies
SMALL_KERNELS = [KernelConfig(tile_m=64, tile_n=64, ...), ...]
LARGE_KERNELS = [KernelConfig(tile_m=256, tile_n=256, ...), ...]
COMPUTE_KERNELS = [KernelConfig(pipeline="compv4", ...), ...]
MEMORY_KERNELS = [KernelConfig(pipeline="compv3", ...), ...]

# Size-based heuristic
def size_based_heuristic(M, N, K):
    if M * N < 512 * 512:
        return SMALL_KERNELS
    else:
        return LARGE_KERNELS

# Strategy-based selection
def compute_strategy():
    return COMPUTE_KERNELS  # Optimized for compute-bound problems

def memory_strategy():
    return MEMORY_KERNELS   # Optimized for memory-bound problems

# Test different strategies
for strategy in [size_based_heuristic, compute_strategy, memory_strategy]:
    kernels = strategy(M, N, K)
    lib, dispatcher, registry = setup_gemm_dispatcher(kernels)
    elapsed_ms = run_gemm(lib, M, N, K, ...)
```

**Features:**
- 24 kernel configurations across 6 categories
- Size-based heuristic (small vs large)
- Optimization strategies (compute, memory, latency)
- Performance comparison across strategies

### 09_multi_registry.py - Multiple Registries
Separate registries for different workloads:
- Compute-optimized registry
- Latency-optimized registry
- Dynamic registry selection

### 10_advanced_benchmark.py - Advanced Benchmark
Full control over benchmark parameters:
- Warmup iterations
- Benchmark iterations
- Statistical analysis

### 11_json_import.py - JSON Import
Import kernel configurations from JSON:
- External configuration files
- Dynamic kernel loading

## Utility Module: ctypes_utils.py

```python
from ctypes_utils import (
    KernelConfig,              # Single kernel configuration
    setup_gemm_dispatcher,     # Set up dispatcher with kernels
    print_kernel_config_table, # Display kernel configurations
    Dispatcher,                # High-level dispatcher
    Registry,                  # Kernel registry
    Validator,                 # Validation utilities
)
```

### KernelConfig

```python
config = KernelConfig(
    # Tile sizes
    tile_m=256, tile_n=256, tile_k=32,
    # Wave configuration
    wave_m=2, wave_n=2, wave_k=1,
    # Warp tile sizes
    warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
    # Pipeline and scheduler
    pipeline="compv4",      # "compv3" or "compv4"
    scheduler="intrawave",  # "intrawave" or "interwave"
    # Optional
    epilogue="default",
    padding=True,
    double_buffer=True,
)
```

### setup_gemm_dispatcher

```python
# Single kernel
lib, dispatcher, registry = setup_gemm_dispatcher(config)

# Multiple kernels
lib, dispatcher, registry = setup_gemm_dispatcher([config1, config2, ...])

# With auto-rebuild
lib, dispatcher, registry = setup_gemm_dispatcher(config, auto_rebuild=True)
```

### print_kernel_config_table

```python
kernels = [config1, config2, config3]
print_kernel_config_table(kernels)
# Output:
# +----+-------+-------+-------+--------+-----------+
# | #  | Tile  | Wave  | Warp  | Pipe   | Scheduler |
# +----+-------+-------+-------+--------+-----------+
# | 1  | 128x128x32 | 2x2x1 | 32x32x16 | compv3 | intrawave |
# | 2  | 256x256x32 | 2x2x1 | 32x32x16 | compv4 | intrawave |
# | 3  | 128x256x64 | 2x2x1 | 32x32x16 | compv3 | interwave |
# +----+-------+-------+-------+--------+-----------+
```

### GPU Memory Management

```python
import ctypes
import numpy as np

# Load HIP library
hip = ctypes.CDLL("libamdhip64.so")

# Allocate GPU memory
gpu_ptr = ctypes.c_void_p()
hip.hipMalloc(ctypes.byref(gpu_ptr), size_in_bytes)

# Copy to GPU (1 = hipMemcpyHostToDevice)
hip.hipMemcpy(gpu_ptr, host_array.ctypes.data, size, 1)

# Copy back (2 = hipMemcpyDeviceToHost)
hip.hipMemcpy(host_array.ctypes.data, gpu_ptr, size, 2)

# Free
hip.hipFree(gpu_ptr)
```

## Performance Testing

Test compilation performance with different kernel counts:

```bash
# Test with 10 kernels (~15s compile time)
python3 01_basic_gemm.py --num-kernels 10

# Test with 20 kernels (~25s compile time)
python3 01_basic_gemm.py --num-kernels 20

# Test with 48 kernels (~50s compile time)
python3 01_basic_gemm.py --num-kernels 48
```

Compilation time scales roughly linearly with kernel count.

## Related Documentation

- [C++ GEMM Examples](../cpp/README.md)
- [Python Utilities](../../../python/README.md)
- [Main Dispatcher README](../../../README.md)
