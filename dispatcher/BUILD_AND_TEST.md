# CK Tile Dispatcher - Build and Test Guide

This guide provides step-by-step instructions for building, testing, and using the CK Tile Dispatcher.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Building the Dispatcher](#building-the-dispatcher)
3. [Running Tests](#running-tests)
4. [Python Bindings](#python-bindings)
5. [Usage Examples](#usage-examples)
6. [Integration with Tile Engine](#integration-with-tile-engine)

## Prerequisites

### Required

- **CMake** >= 3.16
- **C++ Compiler** with C++17 support (GCC 7+, Clang 5+, MSVC 2017+)
- **ROCm** / **HIP** for GPU support
- **CK Tile headers** (from parent directory)

### Optional (for full functionality)

- **Google Test** (for C++ tests) - will be fetched automatically if not found
- **Python** 3.8+ with development headers (for Python bindings)
- **pybind11** (for Python bindings) - will be fetched if not found
- **pytest** (for Python tests)

## Building the Dispatcher

### Basic Build (C++ Only)

```bash
cd dispatcher
mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_DISPATCHER_TESTS=ON

make -j$(nproc)
```

This builds:
- `libck_tile_dispatcher.a` - Core dispatcher library
- C++ unit tests (if `BUILD_DISPATCHER_TESTS=ON`)

### Build with Python Bindings

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_DISPATCHER_TESTS=ON \
    -DBUILD_DISPATCHER_PYTHON=ON

make -j$(nproc)
```

This additionally builds:
- `_ck_dispatcher_cpp.so` - Python C++ extension module

### Build with Auto-Generated Wrappers (for Tile Engine Integration)

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_DISPATCHER_TESTS=ON \
    -DDISPATCHER_AUTO_GENERATE_WRAPPERS=ON \
    -DTILE_ENGINE_DIR=../tile_engine/ops/gemm

make -j$(nproc)
```

This enables automatic wrapper generation from tile_engine generated kernels.

## Running Tests

### C++ Tests

Run all C++ tests:

```bash
cd build
ctest --output-on-failure
```

Run individual test suites:

```bash
# Kernel key tests
./test/test_kernel_key

# Problem tests
./test/test_problem

# Registry tests
./test/test_registry

# Dispatcher tests
./test/test_dispatcher

# Tile backend tests
./test/test_tile_backend

# End-to-end integration tests
./test/test_integration_e2e
```

Run tests with verbose output:

```bash
./test/test_dispatcher --gtest_filter="*" --gtest_print_time=1
```

### Python Tests

Install Python package in development mode:

```bash
cd dispatcher/python
pip install -e .
```

Run Python tests:

```bash
# All tests
pytest -v

# Specific test file
pytest tests/test_cpp_bindings.py -v

# Specific test class
pytest tests/test_core.py::TestDispatcher -v

# With coverage
pytest --cov=ck_tile_dispatcher --cov-report=html
```

## Python Bindings

### Installation

```bash
cd dispatcher/python
pip install -e .
```

### Verification

```python
import _ck_dispatcher_cpp as cpp

# Check module loaded
print(f"C++ extension: {cpp}")

# Test basic functionality
problem = cpp.Problem(1024, 1024, 1024)
print(f"Problem: M={problem.M}, N={problem.N}, K={problem.K}")
print(f"Num ops: {problem.num_ops()}")

# Check registry
registry = cpp.Registry.instance()
print(f"Registry size: {registry.size()}")
```

## Usage Examples

### C++ Example: Basic Dispatch

```cpp
#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/tile_backend.hpp"

using namespace ck_tile::dispatcher;

int main() {
    // 1. Create kernel key
    KernelKey key;
    key.signature.dtype_a = DataType::FP16;
    key.signature.dtype_b = DataType::FP16;
    key.signature.dtype_c = DataType::FP16;
    key.signature.dtype_acc = DataType::FP32;
    key.algorithm.tile_shape = {256, 256, 32};
    key.gfx_arch = 942;
    
    // 2. Create and register kernel (assuming TileKernel is a generated kernel type)
    // auto kernel = std::make_shared<TileKernelInstance<TileKernel>>(key, "my_kernel");
    // Registry::instance().register_kernel(kernel);
    
    // 3. Create dispatcher
    Dispatcher dispatcher;
    
    // 4. Define problem
    Problem problem(1024, 1024, 1024);
    
    // 5. Dispatch and execute
    // float time = dispatcher.run(a_dev, b_dev, c_dev, problem);
    // printf("Execution time: %.3f ms\n", time);
    
    return 0;
}
```

### Python Example: Basic Dispatch

```python
import ck_tile_dispatcher as ckd
import numpy as np

# Create dispatcher
dispatcher = ckd.Dispatcher()

# Register kernel set
dispatcher.register_kernels("fp16_rcr_essential")

# Prepare data
M, N, K = 1024, 1024, 1024
A = np.random.randn(M, K).astype(np.float16)
B = np.random.randn(K, N).astype(np.float16)

# Execute GEMM
C = ckd.gemm(A, B)

print(f"Result shape: {C.shape}")
print(f"Result dtype: {C.dtype}")
```

### C++ Example: Heuristic-Based Selection

```cpp
#include "ck_tile/dispatcher/dispatcher.hpp"

using namespace ck_tile::dispatcher;

int main() {
    // Create dispatcher
    Dispatcher dispatcher;
    
    // Define heuristic function
    auto heuristic = [](const Problem& p) -> std::vector<std::string> {
        // For large problems, prefer larger tiles
        if (p.M >= 2048 && p.N >= 2048) {
            return {
                "256x256x64_4x2x1_32x32x32_persist",
                "256x256x32_2x2x1_32x32x16_nopers"
            };
        }
        // For small problems, prefer smaller tiles
        return {
            "128x128x32_2x2x1_32x32x16_nopers",
            "64x64x64_2x2x1_16x16x16_nopers"
        };
    };
    
    // Set heuristic
    dispatcher.set_heuristic(heuristic);
    
    // Problem dimensions
    Problem problem(2048, 2048, 2048);
    
    // Dispatcher will use heuristic to select best kernel
    auto kernel = dispatcher.select_kernel(problem);
    if (kernel) {
        printf("Selected kernel: %s\n", kernel->get_name().c_str());
    }
    
    return 0;
}
```

## Integration with Tile Engine

The dispatcher integrates with tile_engine generated kernels through a wrapper generation system.

### Step 1: Generate Tile Engine Kernels

```bash
cd tile_engine/ops/gemm
python gemm_instance_builder.py \
    --config default_config.json \
    --output build/generated \
    --parallel 8
```

### Step 2: Build Dispatcher with Auto-Generated Wrappers

```bash
cd dispatcher
mkdir build && cd build

cmake .. \
    -DDISPATCHER_AUTO_GENERATE_WRAPPERS=ON \
    -DTILE_ENGINE_DIR=../../tile_engine/ops/gemm \
    -DBUILD_DISPATCHER_TESTS=ON

make -j$(nproc)
```

### Step 3: Use Generated Kernels

The generated wrappers are automatically included and registered. You can then use them via the dispatcher:

```cpp
#include "ck_tile/dispatcher/dispatcher.hpp"

// Kernels are automatically registered during initialization
Dispatcher dispatcher;

// Define problem
Problem problem(1024, 1024, 1024);

// Dispatch executes using registered tile_engine kernels
float time = dispatcher.run(a_dev, b_dev, c_dev, problem);
```

## Performance Profiling

### C++ Profiling

```cpp
#include "ck_tile/dispatcher/dispatcher.hpp"
#include <chrono>

// Execute kernel multiple times for accurate timing
const int warmup_iters = 10;
const int bench_iters = 100;

Dispatcher dispatcher;
Problem problem(2048, 2048, 2048);

// Warmup
for (int i = 0; i < warmup_iters; i++) {
    dispatcher.run(a_dev, b_dev, c_dev, problem);
}

// Benchmark
auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < bench_iters; i++) {
    dispatcher.run(a_dev, b_dev, c_dev, problem);
}
auto end = std::chrono::high_resolution_clock::now();

float avg_time = std::chrono::duration<float, std::milli>(end - start).count() / bench_iters;
float gflops = (2.0f * problem.M * problem.N * problem.K) / (avg_time * 1e6);

printf("Average time: %.3f ms\n", avg_time);
printf("Performance: %.2f GFLOPS\n", gflops);
```

### Python Profiling

```python
import ck_tile_dispatcher as ckd
from ck_tile_dispatcher import Profiler

# Create profiler
profiler = Profiler()

# Profile GEMM operation
result = profiler.profile_gemm(
    M=2048, N=2048, K=2048,
    dtype=ckd.DataType.FP16,
    num_warmup=10,
    num_iterations=100
)

# Print report
profiler.print_report()

# Get detailed statistics
print(f"Average time: {result.avg_time_ms:.3f} ms")
print(f"Min time: {result.min_time_ms:.3f} ms")
print(f"Max time: {result.max_time_ms:.3f} ms")
print(f"Performance: {result.gflops:.2f} GFLOPS")
```

## Troubleshooting

### Build Issues

**Issue**: CMake can't find CK Tile headers

**Solution**: Ensure the parent directory contains `include/ck_tile/` or specify the path:
```bash
cmake .. -DCK_TILE_INCLUDE_DIR=/path/to/ck_tile/include
```

**Issue**: Google Test not found

**Solution**: The build will automatically fetch Google Test from GitHub. Ensure internet connectivity or install locally:
```bash
sudo apt install libgtest-dev  # Ubuntu/Debian
```

### Runtime Issues

**Issue**: No suitable kernel found

**Solution**: 
1. Verify kernels are registered
2. Check problem dimensions match kernel tile sizes
3. Enable validation: `problem.enable_validation = true`

**Issue**: Python module not found

**Solution**:
```bash
cd dispatcher/python
pip install -e .
```

### Test Failures

**Issue**: Tests fail with "No GPU device"

**Solution**: Most tests use mock kernels and don't require GPU. Tests requiring GPU are marked `DISABLED_`. Run without GPU tests:
```bash
ctest -E "DISABLED"
```

## Next Steps

- See [DISPATCHER.md](../DISPATCHER.md) for complete design documentation
- See [examples/](examples/) for more usage examples
- See [codegen/README.md](codegen/README.md) for codegen documentation
- See [python/README.md](python/README.md) for Python API reference

## Contributing

When contributing tests:

1. C++ tests: Add to `test/` directory following Google Test conventions
2. Python tests: Add to `python/tests/` directory following pytest conventions
3. Update CMakeLists.txt to include new test files
4. Ensure tests pass: `ctest` for C++, `pytest` for Python

## License

MIT License - Copyright (c) 2025, Advanced Micro Devices, Inc.

