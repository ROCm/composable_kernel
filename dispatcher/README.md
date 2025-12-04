# CK Tile Dispatcher

A unified kernel dispatch system for AMD GPUs with C++ and Python frontends.

**Validated Platform:** AMD Instinct MI300 series (gfx942)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Build Guide](#step-by-step-build-guide)
4. [Running Examples](#running-examples)
5. [External Integration](#external-integration)
6. [Core Concepts](#core-concepts)
7. [Troubleshooting](#troubleshooting)
8. [File Structure](#file-structure)

---

## Quick Start

**Complete setup from scratch (5 minutes):**

```bash
# From the composable_kernel root directory
cd dispatcher

# Step 1: Create build directory
mkdir -p build && cd build

# Step 2: Configure CMake
cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx942" \
  -DBUILD_DISPATCHER_EXAMPLES=ON

# Step 3: Generate kernels and build (CMake handles this automatically)
make -j$(nproc)

# Step 4: Run C++ examples
./examples/gemm_01_basic
./examples/conv_01_forward

# Step 5: Run Python examples (from dispatcher directory)
cd ..
python3 examples/gemm/python/01_basic_gemm.py
python3 examples/conv/python/01_basic_conv.py
```

---

## Prerequisites

### Required Software

| Software | Minimum Version | Check Command |
|----------|-----------------|---------------|
| ROCm | 6.4+ | `rocminfo` |
| CMake | 3.16+ | `cmake --version` |
| Python | 3.8+ | `python3 --version` |
| NumPy | 1.20+ | `pip show numpy` |
| hipcc | (from ROCm) | `/opt/rocm/bin/hipcc --version` |

> **Note:** Newer GPU targets (gfx950, gfx1201) require ROCm 6.3+. For ROCm 6.4+, you can also use `amdclang++` instead of `hipcc`.

### Check Your GPU Architecture

```bash
# Find your GPU architecture
rocminfo | grep -i "gfx"
# Example output: "gfx942"
```

**Supported architectures:**
- **gfx942** - MI300X, MI300A, MI308, MI325 (Instinct MI300 series)
- **gfx90a** - MI200 series (MI250, MI250X) 
- **gfx950** - MI350 series 
- **gfx1101** - RDNA3 series 
- **gfx1201** - RDNA4 series 

### Install Dependencies

```bash
# Install NumPy using pip or uv pip (required for Python examples)
pip install numpy
```

### Supported Data Types

CK Tile supports a wide range of data types for GEMM operations:

| A dtype | B dtype | Acc dtype | Warp Tile Sizes | Notes |
|---------|---------|-----------|-----------------|-------|
| `fp32` | `fp32` | `fp32` | 16x16x4, 16x16x16 | Full precision |
| `fp16` | `fp16` | `fp32` | 32x32x8, 32x32x16, 16x16x16, 16x16x32 | Standard half |
| `bf16` | `bf16` | `fp32` | 32x32x8, 32x32x16, 16x16x16, 16x16x32 | Brain float 16 |
| `fp8` | `fp8` | `fp32` | 32x32x16, 32x32x32, 16x16x32, 16x16x64 | FP8 E4M3 |
| `fp8` | `bf8` | `fp32` | 32x32x16, 16x16x32 | Mixed FP8/BF8 |
| `bf8` | `fp8` | `fp32` | 32x32x16, 16x16x128 | Mixed BF8/FP8 |
| `bf8` | `bf8` | `fp32` | 32x32x16, 32x32x32, 16x16x32 | BF8 E5M2 |
| `int8` | `int8` | `int32` | 32x32x16, 16x16x32, 16x16x16 | Integer GEMM |
| `pk_fp4` | `pk_fp4` | `fp32` | 16x16x128 | Packed 4-bit float |

**Notes:**
- Accumulator is always `fp32` except for `int8` which uses `int32`
- FP8 types: `fp8` = E4M3, `bf8` = E5M2
- `pk_fp4` = Packed 4-bit float (2 values per byte)
- Some dtypes require specific GPU architectures (e.g., FP8 requires MI300+)

---

## Step-by-Step Build Guide

### Step 1: Navigate to Dispatcher Directory

```bash
# From composable_kernel root
cd dispatcher

# Verify you're in the right place
ls CMakeLists.txt  # Should exist
```

### Step 2: Create Build Directory

```bash
mkdir -p build
cd build
```

### Step 3: Configure CMake

**Basic configuration (library only):**
```bash
cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx942"
```

**Full configuration (with examples and tests):**
```bash
cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx942" \
  -DBUILD_DISPATCHER_EXAMPLES=ON \
  -DBUILD_DISPATCHER_TESTS=ON
```

**Expected output:**
```
-- Found hip: /opt/rocm (found suitable version "6.x.x")
-- Generating GEMM kernels...
-- Generating Conv kernels...
-- Built: gemm_01 through gemm_09, dispatcher_gemm_lib.so
-- Built: conv_01 through conv_11, dispatcher_conv_lib.so
-- Configuring done
```

### Step 4: Build

```bash
# Build all targets (generates kernels automatically, then compiles)
make -j$(nproc)

# Or build specific targets
make gemm_01_basic          # Single GEMM example
make dispatcher_gemm_lib    # GEMM shared library for Python
make dispatcher_conv_lib    # Conv shared library for Python
make dispatcher_conv_bwdw_lib # Conv backward weight library for Python

# Build ONLY Python libraries (faster if you don't need C++ examples)
make python_libs -j$(nproc)
```

### Kernel Generation Targets

Kernels are generated automatically during `make`, but you can also control generation explicitly:

```bash
# Generate all kernels only (no compilation)
make generate_all_kernels

# Generate specific kernel types
make generate_gemm_kernels      # GEMM kernels only
make generate_conv_kernels      # Conv kernels (fwd + bwd)
make generate_conv_fwd_kernels  # Conv forward only
make generate_conv_bwd_kernels  # Conv backward only

# Force regenerate (even if kernels exist)
make regenerate_all_kernels
make regenerate_gemm_kernels
make regenerate_conv_kernels

# Generate for specific GPU architecture
make generate_kernels_gfx942    # MI300X
make generate_kernels_gfx90a    # MI200
make generate_kernels_gfx1100   # RDNA3
```

### Step 5: Verify Build

```bash
# Check executables were built
ls examples/gemm_*
ls examples/conv_*

# Check shared libraries were built
ls examples/libdispatcher_gemm_lib.so
ls examples/libdispatcher_conv_lib.so
ls examples/libdispatcher_conv_bwdw_lib.so
```

### CMake Options Reference

| Flag | Default | Description |
|------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Debug | **Use `Release` for performance!** |
| `GPU_TARGETS` | None | Target GPU: `"gfx942"`, `"gfx90a"`, etc. |
| `BUILD_DISPATCHER_EXAMPLES` | OFF | Build C++ examples and Python libs |
| `BUILD_DISPATCHER_TESTS` | OFF | Build unit tests |
| `CMAKE_PREFIX_PATH` | - | ROCm installation path |
| `CMAKE_CXX_COMPILER` | - | Path to hipcc compiler |

⚠️ **Important:** Always use `-DCMAKE_BUILD_TYPE=Release` for benchmarking. Debug builds are slower.
⚠️ **Important:** Note that the current system provides single GPU target support for architecture-based kernel filtering, please do not use multiple GPU targets at a time (if necessary, please compile into different build directories).

---

## Running Examples

### C++ Examples

After building, executables are in `build/examples/`:

```bash
cd build/examples

# GEMM Examples
./gemm_01_basic          # Basic GEMM operation
./gemm_02_multi_size     # Multiple problem sizes
./gemm_03_benchmark      # Performance benchmarking
./gemm_04_validation     # CPU validation
./gemm_05_heuristics     # Custom kernel selection

# Convolution Examples
./conv_01_basic          # Basic 2D convolution
./conv_02_forward        # Forward convolution details
./conv_03_validation     # CPU validation (add --verify)
./conv_10_bwd_data       # Backward data (add --verify for validation)
./conv_11_bwd_weight     # Backward weight (add --verify for validation)
```

### Python Examples

Run from the `dispatcher` directory:

```bash
cd /path/to/composable_kernel/dispatcher

# GEMM Examples
python3 examples/gemm/python/01_basic_gemm.py
python3 examples/gemm/python/03_benchmark.py
python3 examples/gemm/python/05_numpy_integration.py

# Convolution Examples
python3 examples/conv/python/01_basic_conv.py
python3 examples/conv/python/04_conv2d_bwd_data.py --verify  # With CPU validation
python3 examples/conv/python/07_validation.py
```

### Example Output

**Expected C++ output (`gemm_01_basic`):**
```
======================================================================
Example 01: Basic GEMM with Declarative Kernel Definition
======================================================================

Step 1: Declared Kernels
------------------------
Kernel Set: fp16_gemm_kernels
  Architecture: gfx942
  Configurations: 1
    - gemm_fp16_rcr_compv4_cshuffle_intrawave_128x128x32

Step 2: Create Registry and Dispatcher
--------------------------------------
  Registered 1 kernels

Step 3: Define Problem
----------------------
  M=1024, N=1024, K=1024

Step 4: GPU Execution
---------------------
  *** GPU EXECUTION ***
  Time:   <varies> ms
  TFLOPS: <varies>
```

> **Note:** Timing values vary by GPU model and system configuration.

**Expected Python output (`01_basic_conv.py`):**
```
======================================================================
Example 01: Basic Convolution with GPU Execution
======================================================================

Step 3: Load Library
--------------------------------------------------
  Library: /path/to/build/examples/libdispatcher_conv_lib.so
  Has kernels: True

Step 4: GPU Execution
--------------------------------------------------
  Input:  (1, 28, 28, 64) -> GPU
  Weight: (128, 3, 3, 64) -> GPU
  Output: (1, 28, 28, 128) (allocated)

  *** GPU EXECUTION SUCCESSFUL ***
  Time:   <varies> ms
  TFLOPS: <varies>
```

---

## Benchmark Parameters

The dispatcher supports fine-grained control over benchmarking, matching CK Tile's `stream_config`:

### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup` | int | 5 | Warmup iterations (discarded from timing) |
| `repeat` | int | 20 | Benchmark iterations (averaged) |
| `flush_cache` | bool | false | Flush GPU L2 cache between iterations |
| `rotating_count` | int | 1 | Rotating buffer count (for cache simulation) |
| `timer` | string | "gpu" | Timer type: "gpu" (HIP events) or "cpu" |
| `init` | string | "random" | Matrix initialization: "random", "linear", "constant" |
| `split_k` | int | 1 | Split-K parallelism factor |

### Python Usage

```python
from conv_utils import GpuConvRunner

# Basic usage (default benchmark settings)
runner = GpuConvRunner()

# Advanced benchmark settings
runner = GpuConvRunner(
    warmup=10,           # More warmup iterations
    repeat=100,          # More benchmark iterations
    flush_cache=True,    # Flush L2 cache (for memory-bound analysis)
    rotating_count=4,    # 4 rotating buffers
    timer="gpu",         # Use GPU timer (most accurate)
)

result = runner.run(input_data, weight_data, problem)
print(f"Average time: {result['time_ms']:.4f} ms")
print(f"TFLOPS: {result['tflops']:.2f}")
```

### C++ Usage

```cpp
// Basic timing
ck_tile::stream_config cfg{nullptr, true};

// Advanced benchmark settings
ck_tile::stream_config cfg{
    nullptr,          // stream_id (nullptr = default stream)
    true,             // time_kernel
    1,                // log_level
    10,               // cold_niters (warmup)
    100,              // nrepeat
    true,             // is_gpu_timer
    true,             // flush_cache
    4                 // rotating_count
};

float avg_time = kernel.run(args, cfg);
```

### Command Line (Python Examples)

```bash
# Basic run
python3 examples/gemm/python/10_advanced_benchmark.py

# With benchmark parameters
python3 examples/gemm/python/10_advanced_benchmark.py \
    --warmup 10 \
    --repeat 100 \
    --flush-cache \
    --rotating-count 4 \
    --timer gpu

# For memory-bound analysis
python3 examples/conv/python/13_advanced_benchmark.py \
    --flush-cache \
    --init constant \
    -n 1 -c 256 -k 256 -hi 56 -wi 56
```

### When to Use Each Parameter

| Use Case | Recommended Settings |
|----------|---------------------|
| Quick test | `warmup=1, repeat=3` |
| Stable benchmark | `warmup=10, repeat=100` |
| Memory-bound analysis | `flush_cache=True, rotating_count=4` |
| Compute-bound analysis | `flush_cache=False` (default) |
| Debug timing | `timer="cpu"` |
| Production | `timer="gpu"` (default) |

---

## External Integration

### Using Dispatcher in Your Own Project

#### Option 1: CMake Integration (Recommended)

Add to your `CMakeLists.txt`:

```cmake
# Set path to composable_kernel
set(CK_ROOT "/path/to/composable_kernel")

# Add dispatcher subdirectory
add_subdirectory(${CK_ROOT}/dispatcher dispatcher_build)

# Link to your target
target_link_libraries(your_target PRIVATE ck_tile_dispatcher)
target_include_directories(your_target PRIVATE 
    ${CK_ROOT}/dispatcher/include
    ${CK_ROOT}/include
)
```

#### Option 2: Include as Pre-built Library

```cmake
# Find the pre-built library
find_library(CK_DISPATCHER ck_tile_dispatcher 
    PATHS /path/to/composable_kernel/dispatcher/build)

# Include directories
set(CK_INCLUDE_DIRS
    /path/to/composable_kernel/include
    /path/to/composable_kernel/dispatcher/include
)

target_link_libraries(your_target PRIVATE ${CK_DISPATCHER})
target_include_directories(your_target PRIVATE ${CK_INCLUDE_DIRS})
```

#### Option 3: Python Integration

```python
import sys
sys.path.insert(0, "/path/to/composable_kernel/dispatcher/examples/gemm/python")
sys.path.insert(0, "/path/to/composable_kernel/dispatcher/examples/conv/python")

# For GEMM
from ctypes_utils import DispatcherLib, Dispatcher, KernelConfig

# For Conv
from conv_utils import ConvDispatcherLib, GpuConvRunner, ConvProblem
```

### Required Include Paths

When integrating, you need these include paths:

```
/path/to/composable_kernel/include              # CK Tile core headers
/path/to/composable_kernel/dispatcher/include   # Dispatcher headers
/path/to/composable_kernel/dispatcher/build/generated_kernels  # Generated kernels
```

### Required Compile Flags

```bash
# Minimum flags for hipcc
-std=c++17
-D__HIP_PLATFORM_AMD__=1
--offload-arch=gfx942  # Your target GPU

# Recommended flags
-O3
-mllvm -enable-noalias-to-md-conversion=0
-Wno-undefined-func-template
-Wno-float-equal
-Wall 
-Werror
```

### Python Path Setup

For Python scripts outside the dispatcher directory:

```bash
# Option 1: Environment variable
export PYTHONPATH="/path/to/composable_kernel/dispatcher/examples/gemm/python:$PYTHONPATH"
export PYTHONPATH="/path/to/composable_kernel/dispatcher/examples/conv/python:$PYTHONPATH"

# Option 2: In your Python script
import sys
sys.path.insert(0, "/path/to/composable_kernel/dispatcher/examples/gemm/python")
sys.path.insert(0, "/path/to/composable_kernel/dispatcher/examples/conv/python")
```

### Library Search Paths

The Python utilities search for the shared library in these locations:

```python
# For GEMM (ctypes_utils.py)
SEARCH_PATHS = [
    "build/examples/libdispatcher_gemm_lib.so",
    "../build/examples/libdispatcher_gemm_lib.so",
    "../../build/examples/libdispatcher_gemm_lib.so",
]

# For Conv (conv_utils.py)
SEARCH_PATHS = [
    "build/examples/libdispatcher_conv_lib.so",
    "../build/examples/libdispatcher_conv_lib.so",
    "../../build/examples/libdispatcher_conv_lib.so",
]
```

If using from a different location, set the library path explicitly:

```python
# GEMM
from ctypes_utils import DispatcherLib
lib = DispatcherLib.load("/absolute/path/to/libdispatcher_gemm_lib.so")

# Conv
from conv_utils import ConvDispatcherLib
lib = ConvDispatcherLib.load("/absolute/path/to/libdispatcher_conv_lib.so")
```

---

## Core Concepts

### Data Flow

```
KernelConfig → Registry → Dispatcher → GPU Execution
```

1. **KernelConfig**: Defines kernel parameters (tile sizes, data types, layouts)
2. **Registry**: Stores multiple kernel configurations
3. **Dispatcher**: Selects best kernel for a given problem and executes it

### GEMM Layouts

| Layout | A | B | C | Use Case |
|--------|---|---|---|----------|
| RCR | Row | Col | Row | Most common (PyTorch default) |
| RRR | Row | Row | Row | Both inputs row-major |
| CRR | Col | Row | Row | A transposed |
| CCR | Col | Col | Row | Both inputs column-major |

### Convolution Layouts

| Layout | Input | Weight | Output | Description |
|--------|-------|--------|--------|-------------|
| NHWGC | N,H,W,G,C | G,K,Y,X,C | N,H,W,G,K | Grouped convolution |

### Split-K Support

Split-K divides the K dimension across multiple thread blocks, useful for large K dimensions.

| Operation | Split-K | Notes |
|-----------|---------|-------|
| GEMM | ✅ Yes | Runtime `k_batch` parameter |
| Conv Forward | ❌ No | Not supported in CK Tile |
| Conv Backward Data | ❌ No | Not supported in CK Tile |
| Conv Backward Weight | ✅ Yes | Automatic when beneficial |

**Usage (C++):**
```cpp
// GEMM with 4-way K split
auto problem = ProblemBuilder()
    .m(1024).n(1024).k(8192)
    .split_k(4)
    .build();
```

---

## Troubleshooting

### Build Issues

| Problem | Solution |
|---------|----------|
| `hipcc not found` | Set `-DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc` |
| `hip not found` | Set `-DCMAKE_PREFIX_PATH=/opt/rocm` |
| Very slow performance | Use `-DCMAKE_BUILD_TYPE=Release` |
| `gfx942 not supported` | Check ROCm version (need 6.0+) |
| Kernel generation fails | Ensure Python 3.8+ with NumPy installed |

### Runtime Issues

| Problem | Solution |
|---------|----------|
| `Library not found` | Build with `-DBUILD_DISPATCHER_EXAMPLES=ON` |
| `No kernel found` | Check GPU arch matches build target |
| Python `ModuleNotFoundError` | Add paths to `PYTHONPATH` (see above) |
| Wrong results | Verify layout matches your data |

### Debug Commands

```bash
# Check ROCm installation
rocminfo | head -20

# Check GPU architecture
rocminfo | grep "Name:"

# Verify library exists
ls -la build/examples/libdispatcher_*.so

# Run with verbose output
./build/examples/gemm_01_basic 2>&1

# Python: Check library loading
python3 -c "
import ctypes
lib = ctypes.CDLL('/path/to/libdispatcher_gemm_lib.so')
print('Library loaded successfully')
"
```

### Clean Rebuild

If you encounter issues, try a clean rebuild:

```bash
cd dispatcher
rm -rf build
mkdir build && cd build
cmake .. [your options]
make -j$(nproc)
```

---

## Technical Notes

### Tensor Layouts

CK Tile uses specific internal layouts for convolution operations:

**2D Convolution (NHWGC layout):**
- Input: `(N, H, W, G, C)` - Batch, Height, Width, Groups, Channels
- Weight: `(G, K, Y, X, C)` - Groups, Output channels, Filter height, Filter width, Input channels  
- Output: `(N, H, W, G, K)` - Batch, Height, Width, Groups, Output channels

The CK Tile kernel expects **2D spatial dimensions** `{H, W}` for 2D convolution, not `{D, H, W}`.

**3D Convolution (NDHWGC layout):**
- Uses all three spatial dimensions `{D, H, W}`
- Input: `(N, D, H, W, G, C)`
- Filter: `{Z, Y, X}` (depth, height, width)

**Important:** When interfacing via ctypes, the `ConvParam` must be constructed with the correct number of spatial dimensions:
- 2D: `filter_spatial = {Y, X}`, `input_spatial = {H, W}`
- 3D: `filter_spatial = {Z, Y, X}`, `input_spatial = {D, H, W}`

### Backward Weight Architecture

Backward weight is built as a **separate shared library** (`libdispatcher_conv_bwdw_lib.so`) to avoid CK Tile template conflicts that occur when combining forward/backward_data/backward_weight in the same compilation unit.

**Libraries:**
- `libdispatcher_conv_lib.so` - Forward + Backward Data
- `libdispatcher_conv_bwdw_lib.so` - Backward Weight (separate)

**Python Usage:**
```python
from conv_utils import GpuConvRunner, GpuConvBwdWeightRunner

# Forward and Backward Data use GpuConvRunner
runner_fwd = GpuConvRunner()

# Backward Weight uses separate runner
runner_bwdw = GpuConvBwdWeightRunner()
result = runner_bwdw.run(input_np, grad_output_np, problem, grad_weight_np)
```

### Convolution Support Matrix

| Operation | C++ Examples | Python ctypes | Status |
|-----------|--------------|---------------|--------|
| Forward 2D | ✅ conv_01 - conv_08 | ✅ GpuConvRunner | Full support |
| Forward 3D | ✅ conv_09 | ✅ GpuConvRunner | Full support |
| Backward Data | ✅ conv_10 | ✅ GpuConvRunner | Full support |
| Backward Weight | ✅ conv_11 | ✅ GpuConvBwdWeightRunner | Full support (separate lib) |

---

## File Structure

```
dispatcher/
├── README.md                    # This file
├── CMakeLists.txt              # Build configuration
│
├── include/ck_tile/dispatcher/  # C++ headers
│   ├── dispatcher.hpp           # GEMM dispatcher
│   ├── registry.hpp             # Kernel registry
│   ├── kernel_key.hpp          # Kernel configuration
│   └── conv_utils.hpp          # Conv utilities
│
├── src/                        # C++ implementation
│
├── codegen/                    # Kernel generation
│   ├── unified_gemm_codegen.py # GEMM kernel generator
│   ├── unified_conv_codegen.py # Conv kernel generator
│   └── arch_specs.json         # GPU specifications
│
├── bindings/ctypes/            # Python ctypes interface
│   ├── gemm_ctypes_lib.cpp     # GEMM Python library
│   └── conv_ctypes_lib.cpp     # Conv Python library
│
├── examples/                   # Examples
│   ├── gemm/
│   │   ├── cpp/                # C++ GEMM examples (01-09)
│   │   └── python/             # Python GEMM examples (01-09)
│   └── conv/
│       ├── cpp/                # C++ Conv examples (01-11)
│       └── python/             # Python Conv examples (01-12)
│
├── scripts/                    # Build scripts
│
└── test/                       # Unit tests
```

---

## Example Documentation

| Directory | README |
|-----------|--------|
| GEMM C++ | [examples/gemm/cpp/README.md](examples/gemm/cpp/README.md) |
| GEMM Python | [examples/gemm/python/README.md](examples/gemm/python/README.md) |
| Conv C++ | [examples/conv/cpp/README.md](examples/conv/cpp/README.md) |
| Conv Python | [examples/conv/python/README.md](examples/conv/python/README.md) |
| Codegen | [codegen/README.md](codegen/README.md) |

---

## License

MIT License - Copyright (c) 2025, Advanced Micro Devices, Inc.
