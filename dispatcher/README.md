# CK Tile Dispatcher

A unified kernel dispatch system for AMD GPUs with C++ and Python frontends.

**Validated Platform:** AMD Instinct MI300 series (gfx942)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Build Options](#build-options)
4. [Core Concepts](#core-concepts)
5. [Python Usage](#python-usage)
6. [C++ Usage](#c-usage)
7. [Examples](#examples)
8. [Kernel Generation](#kernel-generation)
9. [Testing](#testing)
10. [Adding New GPU Support](#adding-new-gpu-support)
11. [Troubleshooting](#troubleshooting)
12. [File Structure](#file-structure)
13. [Performance Reference](#performance-reference)

---

## Quick Start

### Fastest Path to Running GEMM on GPU

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

# 4. Run example
./examples/example_01_basic_gemm
```

**Expected output:**
```
Problem 1024x1024x1024: 0.028 ms, 76 TFLOPS
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
rocminfo | grep "Name:" | head -1
# Example: "Name: gfx942" → use GPU_TARGETS="gfx942"
```

**Supported architectures:**
- **gfx942** - MI300X, MI300A (Instinct MI300 series)
- **gfx950** - MI350 series
- **gfx90a** - MI200 series (MI250, MI250X)
- **gfx1201** - RDNA4 series

---

## Build Options

### Option 1: Basic Build (Library Only)

```bash
cd dispatcher && mkdir -p build && cd build

cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx942"

make -j$(nproc)
```

**Output:** `build/libck_tile_dispatcher.a`

### Option 2: Full Build (Tests + Examples)

```bash
cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx942" \
  -DBUILD_DISPATCHER_TESTS=ON \
  -DBUILD_DISPATCHER_EXAMPLES=ON

make -j$(nproc)
```

### Build Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Debug | **Must be `Release` for performance** |
| `GPU_TARGETS` | None | GPU architecture: `"gfx942"`, `"gfx90a"` |
| `BUILD_DISPATCHER_TESTS` | OFF | Build unit and GPU tests |
| `BUILD_DISPATCHER_EXAMPLES` | OFF | Build example executables |

⚠️ **Always use `-DCMAKE_BUILD_TYPE=Release`**. Debug builds are ~45,000x slower!

---

## Core Concepts

The dispatcher uses an explicit data flow pattern:

```
KernelConfig → Registry → Dispatcher → run()
```

### KernelConfig

Defines all kernel parameters:

```python
from ctypes_utils import KernelConfig

config = KernelConfig(
    # Data types
    dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32",
    
    # Layouts (row/col)
    layout_a="row", layout_b="col", layout_c="row",
    
    # Tile shape (work per thread block)
    tile_m=128, tile_n=128, tile_k=32,
    
    # Wave shape (warps per block)
    wave_m=2, wave_n=2, wave_k=1,
    
    # Pipeline
    pipeline="compv4", scheduler="intrawave",
    
    # Padding (enables arbitrary sizes)
    pad_m=True, pad_n=True, pad_k=True,
    
    # Target GPU
    gfx_arch="gfx942",
)
```

### Registry

Stores and manages kernel instances:

```python
from ctypes_utils import Registry

registry = Registry(name="my_registry")
registry.register_kernel(config)
```

### Dispatcher

Selects and runs kernels:

```python
from ctypes_utils import Dispatcher

dispatcher = Dispatcher(registry=registry, lib=lib)
result = dispatcher.run(A, B, M, N, K)
```

---

## Python Usage

### Setup

```bash
# Set Python path (from dispatcher directory)
export PYTHONPATH=$PWD/python:$PYTHONPATH

# Install NumPy
pip install numpy
```

### Complete Example

```python
import numpy as np
from ctypes_utils import (
    KernelConfig, CodegenRunner, DispatcherLib, Registry, Dispatcher
)

# 1. Define kernel configuration
config = KernelConfig(
    tile_m=128, tile_n=128, tile_k=32,
    pad_m=True, pad_n=True, pad_k=True,
)

# 2. Generate kernel code
codegen = CodegenRunner()
codegen.generate_from_config(config)

# 3. Load library
lib = DispatcherLib.auto()

# 4. Create registry and register kernel
registry = Registry(name="example", lib=lib)
registry.register_kernel(config)

# 5. Create dispatcher
dispatcher = Dispatcher(registry=registry, lib=lib)

# 6. Run GEMM
A = np.random.randn(1024, 1024).astype(np.float16)
B = np.random.randn(1024, 1024).astype(np.float16)
result = dispatcher.run(A, B, 1024, 1024, 1024)

print(f"Time: {result.time_ms:.4f} ms, TFLOPS: {result.tflops:.2f}")
```

### Python Utilities (`python/ctypes_utils.py`)

| Class | Purpose |
|-------|---------|
| `KernelConfig` | Define kernel parameters |
| `CodegenRunner` | Generate kernel code |
| `DispatcherLib` | Load compiled library |
| `Registry` | Store kernel configurations |
| `Dispatcher` | Select and run kernels |
| `GemmRunner` | High-level GEMM runner |
| `Validator` | Validate results |

See [python/README.md](python/README.md) for full API reference.

---

## C++ Usage

### Include Headers

```cpp
#include "ck_tile/dispatcher.hpp"  // All-in-one include

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;
```

### Complete Example

```cpp
#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;

int main() {
    // 1. Build kernel key
    KernelKeyBuilder builder = KernelKeyBuilder::fp16_rcr();
    builder.tile_m = 128;
    builder.tile_n = 128;
    builder.tile_k = 32;
    KernelKey key = builder.build();

    // 2. Create kernel instance
    auto kernel = create_generated_tile_kernel<
        SelectedKernel, ADataType, BDataType, CDataType, AccDataType
    >(key, "my_kernel");

    // 3. Register to registry
    Registry::instance().register_kernel(kernel, Priority::High);

    // 4. Create dispatcher and problem
    Dispatcher dispatcher;
    Problem problem(1024, 1024, 1024);

    // 5. Run GEMM
    float time_ms = dispatcher.run(a_ptr, b_ptr, c_ptr, problem, nullptr);
    
    std::cout << "Time: " << time_ms << " ms\n";
    return 0;
}
```

### C++ Utilities (`include/ck_tile/dispatcher/utils.hpp`)

| Utility | Description |
|---------|-------------|
| `GpuBuffer<T>` | GPU memory management |
| `GpuTimer` | Kernel timing |
| `create_fp16_rcr_key()` | Quick key creation |
| `calculate_tflops()` | Performance calculation |
| `validate_result()` | Result validation |

See [include/ck_tile/dispatcher/README.md](include/ck_tile/dispatcher/README.md) for header documentation.

---

## Examples

### C++ Examples (`examples/cpp/`)

| Example | Description | Complexity |
|---------|-------------|------------|
| `01_basic_gemm.cpp` | Complete explicit workflow | ★☆☆☆☆ |
| `02_multi_size.cpp` | Multiple problem sizes | ★★☆☆☆ |
| `03_benchmark.cpp` | Performance testing | ★★★☆☆ |
| `04_validation.cpp` | Correctness vs CPU | ★★★☆☆ |
| `05_heuristics.cpp` | Kernel selection strategies | ★★★★☆ |
| `06_json_export.cpp` | Export registry to JSON | ★★☆☆☆ |
| `07_preshuffle.cpp` | PreShuffle pipeline | ★★★★☆ |
| `08_multi_d.cpp` | Multi-D GEMM with fusion | ★★★★★ |
| `09_multi_registry.cpp` | Multiple registries | ★★★★★ |

```bash
# Run C++ examples
cd build/examples
./example_01_basic_gemm
./example_03_benchmark 2048 2048 2048
```

### Python Examples (`examples/python/`)

| Example | Description | Complexity |
|---------|-------------|------------|
| `01_basic_gemm.py` | Complete explicit workflow | ★☆☆☆☆ |
| `02_batch_gemm.py` | Multiple sizes | ★★☆☆☆ |
| `03_benchmark.py` | Performance testing | ★★★☆☆ |
| `04_validation.py` | Correctness vs NumPy | ★★★☆☆ |
| `05_numpy_integration.py` | NumPy workflow | ★★☆☆☆ |
| `06_json_export.py` | Export registry to JSON | ★★☆☆☆ |
| `07_preshuffle.py` | PreShuffle kernels | ★★★★☆ |
| `08_multi_d.py` | Multi-D GEMM | ★★★★★ |
| `09_multi_registry.py` | Multiple registries | ★★★★★ |

```bash
# Run Python examples
cd examples/python
python3 01_basic_gemm.py
python3 09_multi_registry.py
```

See [examples/README.md](examples/README.md) for detailed example documentation.

---

## Kernel Generation

### Using CodegenRunner (Python)

```python
from ctypes_utils import CodegenRunner, KernelConfig

# Generate from config
config = KernelConfig(tile_m=256, tile_n=256, tile_k=64)
codegen = CodegenRunner()
result = codegen.generate_from_config(config)

# Generate variant
result = codegen.generate("preshuffle")
result = codegen.generate("multi_d")

# Generate all variants
results = codegen.generate_all()
```

### Using Command Line

```bash
cd codegen

# Generate standard kernels
python3 unified_gemm_codegen.py \
  --output-dir ../build/generated_kernels \
  --datatype fp16 \
  --layout rcr \
  --gpu-target gfx942 \
  --variants standard

# Generate all variants
python3 unified_gemm_codegen.py \
  --output-dir ../build/generated_kernels \
  --variants standard preshuffle multi_d
```

### Generation Options

| Option | Values | Description |
|--------|--------|-------------|
| `--datatype` | `fp16`, `bf16`, `fp32`, `int8` | Data type |
| `--layout` | `rcr`, `rrr`, `crr`, `ccr` | Matrix layouts |
| `--gpu-target` | `gfx942`, `gfx90a`, `gfx950` | Target GPU |
| `--variants` | `standard`, `preshuffle`, `multi_d` | Kernel variants |

See [codegen/README.md](codegen/README.md) for full codegen documentation.

---

## Testing

### Run All Tests

```bash
cd build
ctest --output-on-failure
```

### Test Categories

| Test | Description | GPU Required |
|------|-------------|--------------|
| `test_kernel_key*` | KernelKey serialization | No |
| `test_problem*` | Problem specification | No |
| `test_registry*` | Registry operations | No |
| `test_dispatcher*` | Dispatcher logic | No |
| `test_sanity_ck_tile` | GPU sanity check | Yes |
| `test_regression` | Regression tests | No |

### Run Specific Tests

```bash
# Unit tests only (fast, no GPU)
ctest -R "test_kernel|test_problem|test_registry"

# GPU tests only
ctest -R "test_sanity"

# Verbose output
ctest -V -R test_kernel_key
```

---

## Adding New GPU Support

The dispatcher uses `arch_specs.json` as the single source of truth for GPU specifications.

### Quick Steps

1. Edit `codegen/arch_specs.json`
2. Run `python codegen/generate_arch_specs.py`
3. Rebuild

### Example: Adding gfx1100

```json
{
  "architectures": {
    "gfx1100": {
      "family": "rdna3",
      "description": "AMD Radeon RX 7000 series",
      "warp_size": 32,
      "lds_capacity_kb": 64,
      "warp_configs": [[2, 4, 1], [4, 2, 1]],
      "warp_tile_combos": {
        "fp16_fp16_fp16": [[16, 16, 16], [32, 32, 16]]
      }
    }
  }
}
```

See [codegen/ADDING_NEW_GPU.md](codegen/ADDING_NEW_GPU.md) for complete guide.

---

## Troubleshooting

### Build Issues

| Problem | Solution |
|---------|----------|
| Performance is slow | Use `-DCMAKE_BUILD_TYPE=Release` |
| CMake can't find HIP | Set `-DCMAKE_PREFIX_PATH=/opt/rocm` |
| Wrong GPU targeted | Set `-DGPU_TARGETS` to your GPU |

### Python Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Set `PYTHONPATH` to include `dispatcher/python` |
| Library not found | Build examples first: `make dispatcher_gemm` |
| NumPy not found | Run `pip install numpy` |

### Runtime Issues

| Problem | Solution |
|---------|----------|
| No kernels found | Generate kernels first (see [Kernel Generation](#kernel-generation)) |
| GPU not detected | Check ROCm: `rocminfo` |
| Wrong results | Check layout (RCR = A row-major, B column-major) |

### Debug Commands

```bash
# Check ROCm
rocminfo | head -20

# Check GPU architecture
rocminfo | grep "Name:"

# Check generated kernels
ls build/generated_kernels/

# Verbose test
ctest -V --output-on-failure
```

---

## File Structure

```
dispatcher/
├── README.md                           # This file
│
├── include/ck_tile/dispatcher/         # C++ headers
│   ├── dispatcher.hpp                  # Main dispatcher
│   ├── registry.hpp                    # Kernel registry
│   ├── kernel_key.hpp                  # Kernel configuration
│   ├── problem.hpp                     # Problem specification
│   ├── utils.hpp                       # Utilities
│   └── backends/                       # Backend implementations
│
├── src/                                # C++ implementation
│   ├── dispatcher.cpp
│   └── registry.cpp
│
├── python/                             # Python API
│   ├── README.md                       # Python documentation
│   ├── ctypes_utils.py                 # Core utilities
│   └── core.py                         # Core types
│
├── codegen/                            # Kernel generation
│   ├── README.md                       # Codegen documentation
│   ├── ADDING_NEW_GPU.md              # GPU addition guide
│   ├── unified_gemm_codegen.py        # Main generator
│   └── arch_specs.json                # GPU specifications
│
├── examples/                           # Examples
│   ├── README.md                       # Examples documentation
│   ├── cpp/                            # C++ examples (01-09)
│   └── python/                         # Python examples (01-09)
│
├── test/                               # Tests
│
└── CMakeLists.txt                      # Build configuration
```

---

## Performance Reference

| Problem Size | Time | TFLOPS | GPU |
|--------------|------|--------|-----|
| 512³ | 0.016 ms | 17 | MI300X |
| 1024³ | 0.028 ms | 76 | MI300X |
| 2048³ | 0.075 ms | 230 | MI300X |
| 4096³ | 0.45 ms | 305 | MI300X |

---

## Related Documentation

- [examples/README.md](examples/README.md) - Detailed example documentation
- [codegen/README.md](codegen/README.md) - Kernel generation guide
- [codegen/ADDING_NEW_GPU.md](codegen/ADDING_NEW_GPU.md) - GPU support guide
- [python/README.md](python/README.md) - Python API reference
- [include/ck_tile/dispatcher/README.md](include/ck_tile/dispatcher/README.md) - C++ header documentation

---

## License

MIT License - Copyright (c) 2025, Advanced Micro Devices, Inc.
