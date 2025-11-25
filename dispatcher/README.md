# CK Tile Dispatcher

A unified kernel dispatch system for AMD GPUs with C++ and Python frontends.

**Validated Platform:** AMD Instinct MI300 series (gfx942)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Build Options](#build-options)
4. [Python Usage](#python-usage)
5. [C++ Usage](#c-usage)
6. [Testing](#testing)
7. [Kernel Generation](#kernel-generation)
8. [JSON Export](#json-export)
9. [Multiple Registries](#multiple-registries)
10. [Troubleshooting](#troubleshooting)
11. [File Structure](#file-structure)

---

## Quick Start

### Fastest Path to Running GEMM on GPU

**From the repository root:**

```bash
# 1. Navigate to dispatcher
cd dispatcher

# 2. Create build directory and configure
mkdir -p build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx942" \
  -DBUILD_DISPATCHER_EXAMPLES=ON

# 3. Build
make -j$(nproc)

# 4. Run performance example
./examples/single_tile_kernel_example
```

**Expected output:**
```
Problem 1024x1024x1024: 0.0186 ms, 115.5 TFLOPS
```

---

## Installation

### Prerequisites

| Requirement | Version | How to Check |
|-------------|---------|--------------|
| ROCm | 6.0+ | `rocminfo` |
| CMake | 3.16+ | `cmake --version` |
| Python | 3.8+ | `python3 --version` |
| NumPy | Any | `pip show numpy` |

### Check Your GPU Architecture

```bash
# Find your GPU's GFX architecture
rocminfo | grep "Name:" | head -1
# Example output: "Name: gfx942" → use GPU_TARGETS="gfx942"
```

Common architectures:
- **gfx942** - MI300X, MI300A (Instinct MI300 series)
- **gfx90a** - MI200 series (MI250, MI250X)
- **gfx908** - MI100

---

## Build Options

### Option 1: Basic Build (Library Only)

Use this when you only need the dispatcher library for integration into your own project.

**What it builds:** `libck_tile_dispatcher.a` static library

**When to use:** Integrating dispatcher into an existing application

```bash
cd dispatcher
mkdir -p build && cd build

cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx942"

make -j$(nproc)
```

**Output:** `build/libck_tile_dispatcher.a`

---

### Option 2: Full Build (Tests + Examples + Python)

Use this for development, testing, or to run the included examples.

**What it builds:**
- Static library
- 11 unit/integration tests
- 7 C++ example executables
- Python bindings (optional)

**When to use:** Development, learning the API, running benchmarks

```bash
cd dispatcher
mkdir -p build && cd build

cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx942" \
  -DBUILD_DISPATCHER_TESTS=ON \
  -DBUILD_DISPATCHER_EXAMPLES=ON \
  -DBUILD_DISPATCHER_PYTHON=ON

make -j$(nproc)
```

**Output:**
```
build/
├── libck_tile_dispatcher.a          # Library
├── test/
│   ├── test_kernel_key              # Unit tests
│   ├── test_registry
│   ├── test_dispatcher
│   ├── test_real_kernel_simple      # GPU tests
│   └── ...
├── examples/
│   ├── single_tile_kernel_example   # Performance demo
│   ├── verify_correctness           # Validation
│   └── ...
└── python/
    └── _dispatcher_native.so        # Python extension
```

---

### Build Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Debug | **Must be `Release` for performance** |
| `GPU_TARGETS` | None | GPU architecture(s): `"gfx942"`, `"gfx90a;gfx942"` |
| `BUILD_DISPATCHER_TESTS` | OFF | Build unit and GPU tests |
| `BUILD_DISPATCHER_EXAMPLES` | OFF | Build example executables |
| `BUILD_DISPATCHER_PYTHON` | OFF | Build Python bindings |

**Important:** Always use `-DCMAKE_BUILD_TYPE=Release`. Debug builds are ~45,000x slower!

---

## Python Usage

### Setup

**Step 1: Set Python path**

```bash
# From the dispatcher directory
export PYTHONPATH=$PWD/python:$PYTHONPATH

# Or add to ~/.bashrc for persistence
echo 'export PYTHONPATH=/path/to/composable_kernel/dispatcher/python:$PYTHONPATH' >> ~/.bashrc
```

**Step 2: Install NumPy**

```bash
pip install numpy
```

**Step 3: Make scripts executable (optional)**

```bash
chmod +x examples/python/*.py
```

### Run Python Examples

**From the `dispatcher` directory:**

```bash
# Basic NumPy → GPU workflow
python3 examples/python/numpy_to_gpu_complete.py

# Advanced benchmarks (multiple sizes)
python3 examples/python/numpy_dispatcher_advanced.py
```

### Python API Example

```python
import numpy as np

# Create matrices
A = np.random.randn(1024, 1024).astype(np.float16)
B = np.random.randn(1024, 1024).astype(np.float16)

# Load dispatcher and run GEMM on GPU
from dispatcher_api import Dispatcher

dispatcher = Dispatcher(gpu_arch='gfx942')
C = dispatcher.gemm(A, B)

# Results: ~110 TFLOPS, 100% accuracy vs NumPy
```

### Automatic Dimension Inference

The dispatcher can automatically infer M, N, K from tensor shapes:

```python
from core import Problem

# Automatic inference from NumPy arrays
problem = Problem.from_arrays(A, B, C)

# Or from dimensions
problem = Problem.from_ab(
    a_rows=1024, a_cols=512,
    b_rows=512, b_cols=2048,
    transpose_a=False, transpose_b=False
)
# Infers: M=1024, N=2048, K=512
```

---

## C++ Usage

### Include Headers

```cpp
#include "ck_tile/dispatcher.hpp"  // Main header (includes all components)

// Or include individual components:
#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/problem.hpp"
```

### Basic Example

```cpp
#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;

int main() {
    // 1. Register a kernel (usually done at startup)
    auto kernel = std::make_shared<TileKernelInstance>(/* ... */);
    Registry::instance().register_kernel(kernel, Priority::High);

    // 2. Create problem specification
    Problem problem(1024, 1024, 1024);  // M, N, K

    // 3. Create dispatcher and run
    Dispatcher dispatcher;
    float time_ms = dispatcher.run(a_ptr, b_ptr, c_ptr, problem);

    std::cout << "Time: " << time_ms << " ms\n";
    return 0;
}
```

### Automatic Dimension Inference (C++)

```cpp
#include "ck_tile/dispatcher/problem.hpp"

// From matrix dimensions
auto problem = Problem::from_ab(
    1024, 512,   // A: 1024 rows, 512 cols
    512, 2048,   // B: 512 rows, 2048 cols
    false, false // transpose_a, transpose_b
);
// Infers: M=1024, N=2048, K=512

// From shapes
auto problem2 = Problem::from_shapes(
    TensorShape{1024, 512, false},   // A
    TensorShape{512, 2048, false},   // B
    TensorShape{1024, 2048, false}   // C (optional)
);
```

### Selection Strategies

```cpp
Dispatcher dispatcher;

// Strategy 1: First matching kernel (fastest selection)
dispatcher.set_strategy(SelectionStrategy::FirstFit);

// Strategy 2: Use heuristic function
dispatcher.set_heuristic([](const Problem& p) -> std::vector<std::string> {
    if (p.M >= 2048) return {"256x256x32_4x4x1_32x32x16"};
    return {"128x128x64_2x2x1_32x32x16"};
});
dispatcher.set_strategy(SelectionStrategy::Heuristic);

// Strategy 3: Explicit kernel selection
float time = dispatcher.run_explicit("my_kernel_id", a, b, c, nullptr, problem);
```

---

## Testing

### Run All Tests

**From the `dispatcher/build` directory:**

```bash
# Run all tests
ctest --output-on-failure

# Expected: 11/11 tests passed
```

### Test Categories

| Test | Description | Runtime |
|------|-------------|---------|
| `test_kernel_key` | KernelKey serialization | < 1s |
| `test_problem` | Problem specification | < 1s |
| `test_registry` | Kernel registry operations | < 1s |
| `test_dispatcher` | Dispatcher logic | < 1s |
| `test_tile_backend` | Backend interface | < 1s |
| `test_integration_e2e` | End-to-end integration | < 1s |
| `test_minimal` | Smoke test | < 1s |
| `test_real_kernel_simple` | Basic GPU execution | ~18s |
| `test_real_kernel_multi_size` | Multiple problem sizes | ~15s |
| `test_real_kernel_performance` | Performance metrics | ~17s |
| `test_real_kernel_correctness` | GPU vs CPU validation | ~16s |

### Run Specific Tests

```bash
# Run only unit tests (fast, no GPU)
ctest -R "test_kernel|test_problem|test_registry|test_dispatcher"

# Run only GPU tests
ctest -R "test_real"

# Verbose output for debugging
ctest -V -R test_real_kernel_simple
```

---

## Kernel Generation

The dispatcher uses kernels generated by `unified_gemm_codegen.py`. Kernels are auto-generated when building tests/examples, but you can generate them manually.

### Generate Kernels Manually

**From the `dispatcher/codegen` directory:**

```bash
cd codegen

python3 unified_gemm_codegen.py \
  --output-dir ../build/generated_kernels \
  --datatype fp16 \
  --layout rcr \
  --gpu-target gfx942 \
  --preselected fp16_rcr_essential
```

### Generation Options

| Option | Values | Description |
|--------|--------|-------------|
| `--datatype` | `fp16`, `bf16`, `fp32`, `int8` | Data type |
| `--layout` | `rcr`, `rrr`, `crr`, `ccr` | Matrix layouts (A, B, C) |
| `--gpu-target` | `gfx942`, `gfx90a`, `gfx908` | Target GPU |
| `--preselected` | `fp16_rcr_essential`, etc. | Predefined kernel set |

### Layout Notation

- `R` = Row-major
- `C` = Column-major
- Order: A, B, C (e.g., `rcr` = A row-major, B column-major, C row-major)

---

## JSON Export

### Enable Auto-Export

The registry can automatically export kernel metadata to JSON:

**C++:**
```cpp
auto& registry = Registry::instance();
registry.enable_auto_export("kernels.json");

// Every kernel registration now auto-exports
registry.register_kernel(kernel, Priority::High);  // → writes to kernels.json
```

**Python:**
```python
from json_export import enable_auto_export

enable_auto_export("kernels.json")
```

### Manual Export

```cpp
// Export to string
std::string json = registry.export_json(true);  // true = include statistics

// Export to file
registry.export_json_to_file("kernels.json", true);
```

### JSON Format

```json
{
  "metadata": {
    "timestamp": "2025-11-25T10:30:45",
    "registry_name": "global_singleton",
    "total_kernels": 6
  },
  "statistics": {
    "by_datatype": {"fp16_fp16_fp16": 6},
    "by_pipeline": {"compv4": 2, "compv3": 2, "mem": 2}
  },
  "kernels": [
    {
      "name": "gemm_fp16_rcr_...",
      "identifier": "256x256x32_4x4x1_32x32x16_nopers",
      "signature": { /* data types, layouts */ },
      "algorithm": { /* tile shapes, pipeline */ }
    }
  ]
}
```

---

## Multiple Registries

Create separate registries for different kernel sets:

```cpp
// Create separate registries
Registry fp16_registry;
fp16_registry.set_name("fp16_kernels");

Registry production_registry;
production_registry.set_name("production_kernels");

// Register to specific registries
fp16_registry.register_kernel(fp16_kernel, Priority::High);
production_registry.register_kernel(prod_kernel, Priority::High);

// Create dispatchers with specific registries
Dispatcher fp16_dispatcher(&fp16_registry);
Dispatcher prod_dispatcher(&production_registry);

// Merge registries
Registry combined;
combined.merge_from(fp16_registry, Priority::High);
combined.merge_from(production_registry, Priority::Normal);
```

The global singleton `Registry::instance()` remains available for simple use cases.

---

## Troubleshooting

### Build Issues

| Problem | Solution |
|---------|----------|
| Performance is slow (>100ms) | Use `-DCMAKE_BUILD_TYPE=Release` |
| CMake can't find HIP | Set `-DCMAKE_PREFIX_PATH=/opt/rocm` |
| Wrong GPU targeted | Set `-DGPU_TARGETS` to your GPU (check with `rocminfo`) |
| Tests not building | Add `-DBUILD_DISPATCHER_TESTS=ON` |

### Python Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Set `PYTHONPATH` to include `dispatcher/python` |
| `ImportError: _dispatcher_native` | Build with `-DBUILD_DISPATCHER_PYTHON=ON` |
| NumPy not found | Run `pip install numpy` |
| Permission denied | Run `chmod +x examples/python/*.py` |

### Runtime Issues

| Problem | Solution |
|---------|----------|
| No kernels found | Generate kernels first (see [Kernel Generation](#kernel-generation)) |
| GPU not detected | Check ROCm installation with `rocminfo` |
| Out of memory | Reduce problem size or batch size |

### Debug Commands

```bash
# Check ROCm installation
rocminfo | head -20

# Check GPU architecture
rocminfo | grep "Name:"

# Verify Python extension
python3 -c "import sys; sys.path.insert(0, 'python'); import _dispatcher_native; print('OK')"

# Verbose test output
cd build && ctest -V --output-on-failure

# Check generated kernels
ls build/generated_kernels/
```

---

## File Structure

```
dispatcher/
├── include/ck_tile/dispatcher/     # C++ headers
│   ├── dispatcher.hpp              # Main dispatcher class
│   ├── registry.hpp                # Kernel registry
│   ├── kernel_key.hpp              # Kernel configuration
│   ├── problem.hpp                 # Problem specification
│   ├── kernel_instance.hpp         # Kernel interface
│   ├── arch_filter.hpp             # GPU architecture filtering
│   └── backends/
│       └── tile_backend.hpp        # CK Tile backend
│
├── src/                            # C++ implementation
│   ├── dispatcher.cpp
│   └── registry.cpp
│
├── python/                         # Python API
│   ├── __init__.py
│   ├── core.py                     # Core types (Problem, KernelKey)
│   ├── dispatcher_api.py           # High-level API
│   └── bindings.cpp                # pybind11 bindings
│
├── codegen/                        # Kernel generation
│   ├── unified_gemm_codegen.py     # Main generator
│   ├── arch_specs.json             # GPU specifications
│   └── ADDING_NEW_GPU.md           # Guide for new GPU support
│
├── test/                           # Tests (11 total)
│   ├── test_*.cpp                  # Unit tests
│   └── test_real_kernel_*.cpp      # GPU tests
│
├── examples/
│   ├── cpp/                        # C++ examples
│   │   ├── single_tile_kernel_example.cpp
│   │   └── ...
│   └── python/                     # Python examples
│       ├── numpy_to_gpu_complete.py
│       └── ...
│
└── CMakeLists.txt                  # Build configuration
```

---

## Performance Reference

| Problem Size | Time | TFLOPS | Environment |
|--------------|------|--------|-------------|
| 512³ | 0.011 ms | 23.5 | MI300X |
| 1024³ | 0.019 ms | 115.5 | MI300X |
| 2048³ | 0.054 ms | 319.0 | MI300X |

---

## License

MIT License - Copyright (c) 2025, Advanced Micro Devices, Inc.
