# CK Tile Dispatcher

A unified kernel dispatch system for AMD GPUs with C++ and Python frontends, supporting GEMM and Grouped Convolution operations.

**Validated Platform:** AMD Instinct MI300 series (gfx942)


---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Setup](#docker-setup-recommended)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Build Guide](#step-by-step-build-guide)
5. [Running Examples](#running-examples)
6. [External Integration](#external-integration)
7. [Core Concepts](#core-concepts)
8. [Operation Support Matrix](#operation-support-matrix)
9. [Troubleshooting](#troubleshooting)
10. [File Structure](#file-structure)

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

# Step 5: Build Python libraries (required for Python examples)
make python_libs

# Step 6: Run Python examples (from dispatcher directory)
cd ..
python3 examples/gemm/python/01_basic_gemm.py
```

---

## Docker Setup (Recommended)

For a reproducible build environment, use the official ROCm Docker image:

### Step 1: Pull and Run Container

```bash
# Pull the CK Docker image
docker pull rocm/composable_kernel:ck_ub24.04_rocm7.0.1

# Run container with GPU access
docker run \
  -it \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -w /root/workspace \
  -v $(pwd):/root/workspace \
  rocm/composable_kernel:ck_ub24.04_rocm7.0.1 \
  /bin/bash
```

> **Note:** Omit `--device` flags if building without GPU access.

### Step 2: Clone and Build

```bash
# Inside the container
git clone https://github.com/ROCm/composable_kernel.git
cd composable_kernel
git checkout builder-dispatch-tile-gemm

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install numpy

# Build dispatcher
cd dispatcher
mkdir -p build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx942" \
  -DBUILD_DISPATCHER_EXAMPLES=ON

make -j$(nproc)
```

### One-Liner Build (inside container)

```bash
git clone https://github.com/ROCm/composable_kernel.git && \
cd composable_kernel && git checkout builder-dispatch-tile-gemm && \
python3 -m venv .venv && source .venv/bin/activate && pip install numpy && \
cd dispatcher && mkdir -p build && cd build && \
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS="gfx942" -DBUILD_DISPATCHER_EXAMPLES=ON && \
make -j$(nproc)
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

### Install Python Dependencies

#### Core Dependencies (Required)

NumPy is required for Python examples and kernel generation. We recommend using a virtual environment:

**Option 1: Using standard venv**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install core dependencies
pip install -r python/requirements.txt
```

**Option 2: Using uv (faster alternative)**
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install core dependencies
uv pip install -r python/requirements.txt
```

**Option 3: System-wide install (not recommended)**
```bash
pip install -r python/requirements.txt
```

> **Note:** Always activate your virtual environment before running CMake or Python examples.

#### ML Heuristics Dependencies (Optional)

For ML-based kernel selection (examples 09-11), install additional dependencies:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Install ML dependencies (LightGBM, pandas, pyarrow, scikit-learn)
pip install -r requirements-ml.txt
```

**Why separate?** ML dependencies are large (especially pyarrow) and not needed for basic dispatcher usage. Install only if you need:
- ML-based kernel selection (`examples/gemm/python/09_ml_heuristic.py`)
- Model training (`heuristics/train.py`)
- Model evaluation (`heuristics/evaluate.py`)
- Automated benchmark analysis

**Core dependencies:** ~50 MB (NumPy only)
**With ML dependencies:** ~500 MB (includes LightGBM, pandas, pyarrow, scikit-learn)

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
-- Built: gemm_01 through gemm_06, dispatcher_gemm_lib.so
-- Configuring done
```

### Step 4: Build

```bash
# Build all targets (generates kernels automatically, then compiles)
make -j$(nproc)

# Or build specific targets
make gemm_01_basic          # Single GEMM example
make dispatcher_gemm_lib    # GEMM shared library for Python

# Build ONLY Python libraries (faster if you don't need C++ examples)
make python_libs -j$(nproc)
```

### Kernel Generation Targets

Kernels are generated automatically during `make`, but you can also control generation explicitly:

```bash
# Generate all kernels only (no compilation)
make generate_all_kernels

# Generate GEMM kernels only
make generate_gemm_kernels

# Force regenerate (even if kernels exist)
make regenerate_all_kernels
make regenerate_gemm_kernels

# Generate for specific GPU architecture
make generate_kernels_gfx942    # MI300X
make generate_kernels_gfx90a    # MI200
make generate_kernels_gfx1100   # RDNA3
```

### Step 5: Verify Build

```bash
# Check executables were built
ls examples/gemm_*

# Check shared libraries were built
ls examples/libdispatcher_gemm_lib.so
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

WARNING: **Important:** Always use `-DCMAKE_BUILD_TYPE=Release` for benchmarking. Debug builds are slower.
WARNING: **Important:** Note that the current system provides single GPU target support for architecture-based kernel filtering, please do not use multiple GPU targets at a time (if necessary, please compile into different build directories).

---

## Running Examples

### C++ Examples

After building, executables are in `build/examples/`:

```bash
cd build/examples

# GEMM Examples
./gemm_01_basic              # Basic GEMM with autofill/autocorrect
./gemm_02_multi_size         # Wildcard expansion
./gemm_03_benchmark_validation  # Benchmarking + validation
./gemm_04_heuristics         # Heuristic kernel selection
./gemm_05_json_export        # Registry JSON export
./gemm_06_multi_registry     # Multiple registries

# Grouped Convolution Examples
./grouped_conv_01_basic         # Declaration patterns + GPU execution
./grouped_conv_02_all_dirs      # Forward/BwdData/BwdWeight with GPU
./grouped_conv_03_bench_val     # Benchmark + CPU reference validation
./grouped_conv_04_registry_json # Heuristic selection + JSON export
./grouped_conv_05_bwd_data      # Backward data + CPU validation
./grouped_conv_06_bwd_weight    # Backward weight + CPU validation
./grouped_conv_07_benchmark     # Multi-tile ResNet benchmark
```

### Python Examples

Run from the `dispatcher` directory:

```bash
cd /path/to/composable_kernel/dispatcher

# GEMM Examples
python3 examples/gemm/python/01_basic_gemm.py     # Basic multi-kernel GEMM
python3 examples/gemm/python/04_validation.py     # CPU reference validation
python3 examples/gemm/python/07_stress_test.py    # Stress test
python3 examples/gemm/python/08_heuristics.py     # Heuristic selection

# Grouped Convolution Examples
python3 examples/grouped_conv/python/01_basic_grouped_conv.py  # Config patterns + registry + GPU
python3 examples/grouped_conv/python/02_forward.py             # Forward 2D/3D + CPU ref
python3 examples/grouped_conv/python/03_bwd_data.py            # Backward data + CPU ref
python3 examples/grouped_conv/python/04_bwd_weight.py          # Backward weight + CPU ref
python3 examples/grouped_conv/python/05_benchmark.py           # Multi-problem benchmark
python3 examples/grouped_conv/python/06_registry_json.py       # Heuristic selection + JSON

# FMHA Examples (JIT-compiled on the fly)
python3 examples/fmha/python/01_basic_fmha.py      # Basic forward attention
python3 examples/fmha/python/12_masks_fmha.py       # Causal masks
python3 examples/fmha/python/18_backward_fmha.py    # Backward pass
python3 examples/fmha/python/16_splitkv_fmha.py     # Split-KV for long sequences
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
from ctypes_utils import DispatcherLib

# Basic usage (default benchmark settings)
lib = DispatcherLib.load()

# Advanced benchmark settings via command line
python3 examples/gemm/python/10_advanced_benchmark.py \
    --warmup 10 \
    --repeat 100 \
    --flush-cache
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

## ML-Based Kernel Selection (Optional)

The dispatcher includes ML heuristics for automated kernel selection using trained LightGBM models.

**Prerequisites:** Install ML dependencies first:

```bash
pip install -r requirements-ml.txt  # ~500 MB (LightGBM, pandas, pyarrow, scikit-learn)
```

**Documentation:** See [heuristics/README.md](heuristics/README.md) for:
- Training and evaluating models
- Feature engineering (72 features)
- Using pre-trained models
- Python API reference

**Examples:**
```bash
python3 examples/gemm/python/09_ml_heuristic.py      # ML-based kernel selection
python3 examples/gemm/python/10_rank_kernels.py      # Kernel ranking
```

**Model Compression:** Trained models are stored in compressed `.lgbm.gz` format to save space (~67% size reduction). Python tools automatically decompress models on first use. For C++ examples, decompress manually:

```bash
# If you have compressed models
cd heuristics/models/gemm_universal_fp16_gfx950
gunzip model_tflops.lgbm.gz

# Then use in C++ example
cd ../../../build
./gemm_09_ml_heuristic --model ../heuristics/models/gemm_universal_fp16_gfx950/model_tflops.lgbm
```

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

# For GEMM
from ctypes_utils import DispatcherLib, Dispatcher, KernelConfig
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

# Option 2: In your Python script
import sys
sys.path.insert(0, "/path/to/composable_kernel/dispatcher/examples/gemm/python")
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
```

If using from a different location, set the library path explicitly:

```python
# GEMM
from ctypes_utils import DispatcherLib
lib = DispatcherLib.load("/absolute/path/to/libdispatcher_gemm_lib.so")
```

---

## Core Concepts

### Data Flow

```
KernelConfig -> Registry -> Dispatcher -> GPU Execution
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

### Split-K Support

Split-K divides the K dimension across multiple thread blocks, useful for large K dimensions.

**Usage (C++):**
```cpp
// GEMM with 4-way K split
auto problem = ProblemBuilder()
    .m(1024).n(1024).k(8192)
    .split_k(4)
    .build();
```

---

## Operation Support Matrix

This matrix shows all CK Tile operations with per-data-type, per-layout, and per-GPU support status. It uses a three-state convention: ✅ = supported by both CK Tile and the dispatcher, ❌ = supported by CK Tile but not yet in the dispatcher, blank = not supported by CK Tile itself.

| | | | | | **Data Types** | | | | | **Layouts** | | | | **GPU Targets** | | |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Op** | **CK Tile Kernel** | **fp16** | **fp8** | **bf16** | **bf8** | **int8** | **fp4** | **fp6** | **rcr** | **rrr** | **ccr** | **crr** | **90a** | **942** | **950** | **1201** |
| GEMM | gemm_multi_d [5]<br>engine: `dispatcher/`<br>example: `19_gemm_multi_d/` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GEMM | gemm_preshuffle [1][2]<br>engine: `dispatcher/` | ✅ | ✅ | ✅ | ✅ | ✅ |  |  | ✅ |  |  |  | ✅ | ✅ | ✅ | ❌ |
| GEMM | gemm_universal [3][4][7][8]<br>engine: `dispatcher/`<br>example: `03_gemm/` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GEMM | batched_contraction<br>example: `41_batched_contraction/` | ❌ |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | batched_gemm<br>example: `16_batched_gemm/` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | block_scale_gemm<br>example: `38_block_scale_gemm/` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | flatmm<br>example: `18_flatmm/` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | gemm_multi_abd<br>example: `22_gemm_multi_abd/` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | gemm_quant |  | ❌ | ❌ | ❌ | ❌ | ❌ |  | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | grouped_gemm<br>example: `17_grouped_gemm/` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | grouped_gemm_quant |  | ❌ | ❌ | ❌ | ❌ | ❌ |  | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | streamk_gemm<br>example: `40_streamk_gemm/` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Reduce | multi_reduce2d<br>example: `05_reduce/` | ❌ |  | ❌ |  |  |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Reduce | reduce2d<br>example: `05_reduce/` | ❌ |  | ❌ |  |  |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Attention | fmha<br>example: `01_fmha/` | ✅ | ✅ | ✅ | ✅ | ❌ |  |  |  |  |  |  | ✅ | ✅ | ✅ | ❌ |
| Attention | sparse_attn<br>example: `50_sparse_attn/` | ❌ |  | ❌ |  | ❌ |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Activation | softmax | ❌ |  | ❌ |  |  |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Activation | topk_softmax<br>example: `09_topk_softmax/` | ❌ | ❌ | ❌ |  |  |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Conv | grouped_conv [6]<br>example: `20_grouped_convolution/` | ❌ | ❌ | ❌ | ❌ |  |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Data Move | batched_transpose<br>example: `35_batched_transpose/` | ❌ | ❌ | ❌ |  |  |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Data Move | image_to_column<br>example: `04_img2col/` | ❌ |  |  |  |  |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Data Move | permute<br>example: `06_permute/` | ❌ | ❌ | ❌ |  | ❌ |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Elementwise | elementwise<br>example: `21_elementwise/` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| MoE | fused_moe<br>example: `15_fused_moe/` | ❌ | ❌ | ❌ | ❌ | ❌ |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Norm | add_rmsnorm2d_rdquant<br>example: `11_add_rmsnorm2d_rdquant/` | ❌ | ❌ | ❌ | ❌ | ❌ |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Norm | layernorm2d<br>example: `02_layernorm2d/` | ❌ | ❌ | ❌ | ❌ | ❌ |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Norm | norm_reduce | ❌ |  | ❌ |  |  |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Norm | rmsnorm2d<br>example: `10_rmsnorm2d/` | ❌ | ❌ | ❌ | ❌ | ❌ |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Pooling | pooling<br>example: `36_pooling/` | ❌ |  |  |  |  |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |
| Quant | smoothquant<br>example: `12_smoothquant/` | ❌ | ❌ | ❌ | ❌ | ❌ |  |  |  |  |  |  | ❌ | ❌ | ❌ | ❌ |

**Notes:**

- [1] **gemm_preshuffle:** Supports only `rcr` layout. Uses fixed `preshufflev2` pipeline, `Auto` scheduler, and `cshuffle` epilogue.
- [2] **gemm_preshuffle:** `int8` preshuffle support is limited to gfx942 and gfx950 (entries in `preshuffle_warp_tile_combos`).
- [3] **gemm_universal:** `fp4` (pk_fp4) support is only available on gfx950.
- [4] **gemm_universal:** `fp32` GEMM is supported by the dispatcher (`fp32_fp32_fp32` warp tile combos exist) but is omitted from matrix columns for consistency with the tile engine matrix format.
- [5] **gemm_multi_d:** Codegen supports `MultiDAdd` and `MultiDMultiply` element-wise ops. Preselected kernel sets also test `Relu`, `Gelu`, `FastGelu`.
- [6] **grouped_conv:** `arch_filter.py` defines conv operator types (`CONV_FWD`, `CONV_BWD_DATA`, `CONV_BWD_WEIGHT`, `CONV3D_*`) but dispatcher infrastructure is incomplete (ctypes bindings are stubs, `conv_utils.hpp` does not exist).
- [7] **(all dispatcher ops):** gfx908, gfx1100, and gfx1200 also have `warp_tile_combos` in `arch_specs.json` but are not shown in the matrix's 4 GPU columns.
- [8] **(all dispatcher ops):** `int4`, `fp32`, `fp64` are valid dispatcher data types (defined in `kernel_key.hpp` `DataType` enum) but have no dedicated matrix columns.

### Dispatcher GEMM Configuration Detail

#### Per-Variant Configuration

| GEMM Variant | Pipelines | Schedulers | Epilogues | Element-wise Ops | Output Dtype |
|:---:|:---:|:---:|:---:|:---:|:---:|
| gemm_universal | mem, compv3, compv4 | intrawave, interwave | cshuffle, default | PassThrough | Same as input (fp8/bf8 -> fp16) |
| gemm_preshuffle | preshufflev2 | Auto | cshuffle | PassThrough | Same as input (fp8/bf8 -> fp16) |
| gemm_multi_d | mem, compv3, compv4 | intrawave, interwave | cshuffle, default | MultiDAdd, MultiDMultiply | Same as input (fp8/bf8 -> fp16) |

#### Warp Tile Combinations per GPU

| GPU | fp16 | bf16 | fp8 | bf8 | int8 | pk_fp4 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| gfx1100 | 16x16x16 | 16x16x16 | -- | -- | 16x16x16 | -- |
| gfx1200 | 16x16x16 | 16x16x16 | 16x16x16 | 16x16x16 | 16x16x16 | -- |
| gfx1201 | 16x16x16 | 16x16x16 | 16x16x16 | 16x16x16 | 16x16x16 | -- |
| gfx908 | 32x32x8, 16x16x16, 32x32x16, 16x16x32 | 32x32x8, 16x16x16, 32x32x16, 16x16x32 | -- | -- | 32x32x16, 16x16x32 | -- |
| gfx90a | 32x32x8, 16x16x16, 32x32x16, 16x16x32, 4x64x16, 64x4x16 | 32x32x8, 16x16x16, 32x32x16, 16x16x32, 4x64x16, 64x4x16 | 32x32x16, 32x32x32 | 32x32x16, 32x32x32 | 32x32x16, 16x16x32 | -- |
| gfx942 | 32x32x8, 16x16x16, 32x32x16, 16x16x32, 4x64x16, 64x4x16 | 32x32x8, 16x16x16, 32x32x16, 16x16x32, 4x64x16, 64x4x16 | 32x32x16, 32x32x32, 16x16x32, 16x16x64 | 32x32x16, 32x32x32, 16x16x32, 16x16x64 | 32x32x16, 16x16x32 | -- |
| gfx950 | 32x32x8, 16x16x16, 32x32x16, 16x16x32, 4x64x16, 64x4x16 | 32x32x8, 16x16x16, 32x32x16, 16x16x32, 4x64x16, 64x4x16 | 32x32x16, 32x32x32, 16x16x32, 16x16x64, 16x16x128, 32x32x64 | 32x32x16, 32x32x32, 16x16x32, 16x16x64, 16x16x128, 32x32x64 | 32x32x16, 16x16x32 | 16x16x128 |

#### Preshuffle Warp Tile Combinations

| GPU | fp16 | bf16 | fp8 | bf8 | int8 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| gfx90a | 32x32x8, 16x16x16, 32x32x16, 16x16x32, 64x4x16 | 32x32x8, 16x16x16, 32x32x16, 16x16x32, 64x4x16 | 32x32x16, 32x32x32 | 32x32x16, 32x32x32 | -- |
| gfx942 | 32x32x8, 16x16x16, 32x32x16, 16x16x32, 64x4x16 | 32x32x8, 16x16x16, 32x32x16, 16x16x32, 64x4x16 | 32x32x16, 32x32x32, 16x16x32, 16x16x64 | 32x32x16, 32x32x32, 16x16x64, 16x16x32 | 16x16x32, 32x32x16 |
| gfx950 | 32x32x8, 16x16x16, 32x32x16, 16x16x32, 64x4x16 | 32x32x8, 16x16x16, 32x32x16, 16x16x32, 64x4x16 | 32x32x16, 32x32x32, 16x16x32, 16x16x64, 16x16x128, 32x32x64 | 32x32x16, 32x32x32, 16x16x64, 16x16x32, 16x16x128, 32x32x64 | -- |

**Legend:**
- **CK Tile Kernel column:** First line is the kernel name. Lines prefixed with "engine:" show the dispatcher directory. Lines prefixed with "example:" show the CK Tile example directory under `example/ck_tile/`.
- **Green cell** (✅): CK Tile implementation exists **and** the dispatcher supports it.
- **Red cell** (❌): CK Tile implementation exists **but** the dispatcher does **not** support it.
- **Grey cell** (blank): No CK Tile implementation exists for this combination.

**Layout codes:** Each 3-character layout code specifies the memory layout for tensors A, B, and C:
- `r` = row-major, `c` = column-major
- Example: `rcr` means A is row-major, B is column-major, C is row-major
- `gemm_multi_d` uses 4-character codes internally (e.g., `rcrr`) where the 4th character is the D tensor layout (always `r`). The matrix shows only the 3-character A/B/C portion.

**Data type mapping per config label:**

| Config Label | A (source) | B (source) | Acc | C (output) |
|:---:|:---:|:---:|:---:|:---:|
| fp16 | fp16 | fp16 | fp32 | fp16 |
| bf16 | bf16 | bf16 | fp32 | bf16 |
| int8 | int8 | int8 | int32 | int32 |
| fp8 | fp8 | fp8 | fp32 | fp16 |
| bf8 | bf8 | bf8 | fp32 | fp16 |
| fp6 | fp6 | fp6 | fp32 | fp32 |
| fp4 | fp16 or bf16 | fp4 | fp32 | fp16 or bf16 |

## Troubleshooting

### Build Issues

| Problem | Solution |
|---------|----------|
| `hipcc not found` | Set `-DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc` |
| `hip not found` | Set `-DCMAKE_PREFIX_PATH=/opt/rocm` |
| Very slow performance | Use `-DCMAKE_BUILD_TYPE=Release` |
| `gfx942 not supported` | Check ROCm version (need 6.0+) |
| Kernel generation fails | Ensure Python 3.8+ with NumPy installed in active venv |
| Build errors | First verify CK builds without dispatcher (see main CK README) |

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

## File Structure

```
dispatcher/
|---- README.md                    # This file
|---- CMakeLists.txt              # Build configuration
|
|---- include/ck_tile/dispatcher/  # C++ headers
|   |---- dispatcher.hpp           # Main dispatcher include
|   |---- registry.hpp             # GEMM kernel registry
|   |---- kernel_key.hpp           # Kernel configuration
|   |---- grouped_conv_config.hpp  # Grouped conv configuration
|   |---- grouped_conv_problem.hpp # Grouped conv problem (with builder)
|   |---- grouped_conv_kernel_decl.hpp  # Grouped conv kernel declarations
|   |---- grouped_conv_registry.hpp     # Grouped conv registry (thread-safe)
|   |---- grouped_conv_utils.hpp        # Grouped conv utilities
|   |---- fmha_types.hpp           # FMHA fwd/bwd args and traits structs
|   |---- fmha_problem.hpp         # FmhaProblem, FmhaProblemBuilder
|   |---- fmha_kernel_key.hpp      # FmhaKernelKey (Signature + Algorithm)
|   |---- fmha_kernel_instance.hpp # FmhaKernelInstance virtual interface
|   |---- fmha_kernel_decl.hpp     # Declarative FmhaSignature/FmhaAlgorithm
|   |---- fmha_registry.hpp        # FmhaRegistry (thread-safe)
|   +---- fmha_dispatcher.hpp      # FmhaDispatcher (plan, select, run)
|
|---- src/                        # C++ implementation
|
|---- codegen/                    # Kernel generation
|   |---- codegen_common.py       # Shared: TileConfig, TraitConfigBase, type mappings
|   |---- unified_gemm_codegen.py # GEMM kernel generator
|   |---- unified_grouped_conv_codegen.py  # Grouped conv kernel generator
|   |---- unified_fmha_codegen.py # FMHA kernel generator
|   |---- fmha_arch_specs.json    # FMHA per-arch tile/pipeline specs
|   |---- fmha_rules.py           # FMHA validation rules
|   |---- fmha_profiles.py        # FMHA named profiles/receipts
|   +---- arch_specs.json         # GPU specifications
|
|---- python/                     # Python utilities
|   |---- dispatcher_common.py    # Shared: paths, validation, Colors, phased output
|   |---- ctypes_utils.py         # GEMM ctypes utilities
|   |---- grouped_conv_utils.py   # Grouped conv utilities
|   +---- fmha_utils.py           # FMHA: JIT compile, FmhaRunner, FmhaKernelConfig
|
|---- scripts/                    # Build scripts
|   |---- compile_gemm_examples.py           # GEMM build script
|   +---- compile_grouped_conv_examples.py   # Grouped conv build script
|
|---- bindings/ctypes/            # Python ctypes interface
|   |---- gemm_ctypes_lib.cpp     # GEMM Python library
|   |---- conv_ctypes_lib.cpp     # Grouped conv Python library
|   +---- fmha_ctypes_lib.cpp     # FMHA Python library
|
|---- examples/                   # Examples
|   |---- gemm/
|   |   |---- cpp/                # C++ GEMM examples (01-07)
|   |   +---- python/             # Python GEMM examples (01-11)
|   |---- grouped_conv/
|   |   |---- cpp/                # C++ Grouped Conv examples (01-07)
|   |   +---- python/             # Python Grouped Conv examples (01-06)
|   +---- fmha/
|       |---- cpp/                # C++ FMHA examples (01-35)
|       +---- python/             # Python FMHA examples (01-38)
|
+---- tests/                      # Unit tests (C++ and Python)
```

---

## Example Documentation

| Directory | README |
|-----------|--------|
| GEMM C++ | [examples/gemm/cpp/README.md](examples/gemm/cpp/README.md) |
| GEMM Python | [examples/gemm/python/README.md](examples/gemm/python/README.md) |
| FMHA C++ | examples/fmha/cpp/ (35 examples covering all FMHA variants) |
| FMHA Python | examples/fmha/python/ (38 examples with JIT compilation) |
| Codegen | [codegen/README.md](codegen/README.md) |
| Python Utils | [python/README.md](python/README.md) |
| C++ Headers | [include/ck_tile/dispatcher/README.md](include/ck_tile/dispatcher/README.md) |

---

## Grouped Convolution Support

Grouped convolution is fully supported alongside GEMM, with shared infrastructure to eliminate duplication.

### Python

```bash
# Generate grouped conv kernels
python3 codegen/unified_grouped_conv_codegen.py \
    --output-dir build/generated_kernels \
    --datatype fp16 --variant forward --ndim-spatial 2

# Build grouped conv examples
python3 scripts/compile_grouped_conv_examples.py examples/grouped_conv/cpp/01_basic_grouped_conv.cpp
```

### Key Files

| Component | File |
|-----------|------|
| C++ Headers | `include/ck_tile/dispatcher/grouped_conv_*.hpp` |
| Python Codegen | `codegen/unified_grouped_conv_codegen.py` |
| Python Utils | `python/grouped_conv_utils.py` |
| Build Script | `scripts/compile_grouped_conv_examples.py` |
| Shared Codegen | `codegen/codegen_common.py` |
| Shared Utils | `python/dispatcher_common.py` |

### Variants

- **Forward** (`grouped_conv_fwd`) - Standard grouped convolution
- **Backward Data** (`grouped_conv_bwd_data`) - Gradient w.r.t. input
- **Backward Weight** (`grouped_conv_bwd_weight`) - Gradient w.r.t. weights

### Shared Infrastructure

GEMM and grouped convolution share common code to avoid duplication:
- `codegen/codegen_common.py` - TileConfig, TraitConfigBase, type mappings, parallel generation, arch-aware expansion
- `python/dispatcher_common.py` - Path helpers, validation, auto-correction, Colors, phased output

---

## License

MIT License - Copyright (c) 2025, Advanced Micro Devices, Inc.
