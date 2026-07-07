# CK Tile Engine GEMM Operations

## Overview

The CK Tile Engine GEMM module provides a comprehensive system for generating, building, and benchmarking GEMM (General Matrix Multiplication) kernels with various configurations. It supports multiple data types, layouts, and optimization strategies. The system has evolved from a monolithic build approach (where all kernels compile into a single executable) to a more flexible individual kernel compilation system, providing better build parallelism and targeted testing capabilities.

## Table of Contents

0. [Dispatcher Bridge Workflow](#dispatcher-bridge-workflow)
1. [Build System Architecture](#build-system-architecture)
2. [Build Instructions](#build-instructions)
3. [Running Benchmarks](#running-benchmarks)
4. [Configuration System](#configuration-system)
5. [Scripts and Tools](#scripts-and-tools)
6. [Command Line Options](#command-line-options)
7. [Understanding Kernel Names](#understanding-kernel-names)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tips](#performance-tips)

## Dispatcher Bridge Workflow

The **Dispatcher bridge** is the recommended path for sweeping and benchmarking
GEMM kernels. Instead of building monolithic or per-kernel executables through
CMake, Tile Engine expands a sweep config into shared `GemmKernelConfig` objects
and hands them to the Dispatcher, which codegens and compiles each into its own
`.so`. The kernel name produced by the bridge is byte-for-byte identical to the
codegen `KERNEL_NAME`, so the bridge runs exactly the same kernels the native
Tile Engine does — it only swaps the harness.

### Scripts

| Script | Role |
|---|---|
| `gemm_full_benchmark.py` | Driver: compile (Phase 1) → load problems (Phase 2) → benchmark across all visible GPUs (Phase 3). |
| `run_one_gemm_kernel.py` | Disposable worker: loads one `.so` in an isolated subprocess and times it. A GPU fault kills only the worker. |

### Folder layout

The bridged regular-GEMM path follows the same op-root convention as the merged
`fmha/` and `grouped_conv/` bridges — driver + worker + a flat `configs/` at the
op root:

```
gemm/
├── gemm_full_benchmark.py     # bridge driver (op root)
├── run_one_gemm_kernel.py     # disposable per-kernel worker (op root)
├── configs/                   # bridged gemm_universal sweep configs (flat)
├── gemm_instance_builder.py   # shared generator for the non-bridged variants
├── gemm_benchmark.{py,hpp}, gemm_common.hpp, gemm_profiler.hpp  # shared harness
├── gemm_multi_d/   gemm_preshuffle/   grouped_gemm/             # legacy variants
└── README.md
```

`configs/` ships example sweep configs:

- `default_ci_config.json` — small CI-sized sweep (the driver's default when no
  config is passed).
- `default_config.json` — full sweep.
- `user_provided_config.json` — scratch space for custom sweeps.
- `example_problems.json` — example M/N/K problem set (used when `--problems`
  is omitted).

> The JSON used by **nightly** tests is intended to drop into the same
> `configs/` directory and be selected with a positional config — no driver
> changes needed.

The not-yet-bridged variants (`gemm_multi_d/`, `gemm_preshuffle/`,
`grouped_gemm/`) keep their own per-variant `configs/` directories; the driver
selects them with `--variant`.

### Running

```bash
cd tile_engine/ops/gemm

# Default: gemm_universal variant, its CI sweep + example problems,
# auto-detect and use all visible GPUs.
python gemm_full_benchmark.py

# Full sweep, fp16/rcr, restricted to 4 GPUs, custom output:
python gemm_full_benchmark.py --variant gemm_universal \
    configs/default_config.json \
    --dtype fp16 --layout rcr --devices 4 --csv gemm_results.csv

# Specific GPU ids and a custom problem file:
python gemm_full_benchmark.py --devices 0,2,5 \
    --problems configs/example_problems.json

# Correctness mode: check every kernel against an fp32 numpy reference.
python gemm_full_benchmark.py --verify --max-kernels 8
```

### Liveness vs correctness (`--verify`)

By default a measurement is reported `OK` purely on **liveness** — the kernel
ran and produced a non-zero output (`ZERO` otherwise). It is *not* a correctness
check: a numerically wrong but non-zero result still reads `OK`. Pass `--verify`
to have each worker compare its output against an fp32 numpy reference
(`A @ B`) using the global relative metric `max|out - ref| / max|ref|`. With
`--verify`, results read `VERIFY` (within `--verify-tol`, default `2e-2`) or
`MISMATCH` (counted as a failure), and the `max_rel` / `verified` columns are
populated in the CSV. This gives self-contained per-kernel confidence; the
broader numeric parity against native Tile Engine remains a separate task.

### Multi-GPU parallelism

Phase 3 fans the `(kernel × problem)` work out across **every visible GPU** in
parallel. One worker thread per device pulls batches from a shared queue and
spawns a disposable subprocess pinned with `HIP_VISIBLE_DEVICES`, so an N-GPU box
benchmarks roughly N× faster while keeping per-batch fault isolation. Devices are
auto-detected (`HIP_VISIBLE_DEVICES`, then `rocm-smi`/`amd-smi`); override with
`--devices`. This supersedes the serial-GPU design inherited from grouped_conv.

### Supported surface

| Axis | Supported |
|---|---|
| dtype | `fp16`, `bf16` |
| layout | `rcr`, `rrr`, `crr`, `ccr` (row-major C only — ck_tile rejects column-major C at build) |

### Variant scope

The bridge is **one shared, variant-aware driver** (`gemm_full_benchmark.py` +
`run_one_gemm_kernel.py`), not a per-variant copy of the driver. The bridged
regular-GEMM path (`gemm_universal`) uses the op-root `configs/`; `--variant`
selects a not-yet-bridged variant's own `configs/` subdirectory.

What that means for this PR:

- **Only `gemm_universal` is wired and validated through the bridge here.** It is
  the foundation variant; the dispatcher codegen path is exercised and parity-
  checked for it alone.
- The `gemm_multi_d/`, `gemm_preshuffle/`, and `grouped_gemm/` `configs/`
  directories are **scaffolding** that follows the per-variant convention so the
  layout is ready. `--variant` will select them, but the bridge does **not** yet
  produce correct kernels for those variants on this PR — do not treat their
  presence as working support.
- Grouped GEMM and stream-K go through **separate bridge efforts** (stream-K in
  #8136, grouped GEMM on its own branch), not this PR.

### Removal note

The legacy regular-GEMM standalone build path has been **removed**, and the
`gemm_universal/` folder is gone entirely. The per-config benchmark generator and
driver (`gemm_universal_instance_builder.py`, `gemm_universal_benchmark.py`,
`gemm_universal_benchmark*.{cpp,hpp}`, and `gemm_universal/CMakeLists.txt`) no
longer exist; its sweep configs were promoted to the op-root `configs/` directory
(matching the `fmha/` and `grouped_conv/` bridge convention) and are consumed by
the bridge. Regular GEMM now runs exclusively through the Dispatcher bridge
workflow above (`gemm_full_benchmark.py` / `run_one_gemm_kernel.py`). The other
variants (`gemm_multi_d/`, `gemm_preshuffle/`, `grouped_gemm/`) still use the
shared `gemm_instance_builder.py` generator.

The build-system, build-instruction, and benchmark-execution sections below
describe that removed standalone path and are retained only as historical
reference for the non-bridged variants; the `benchmark_gemm_universal_*` targets
they mention are no longer produced.

## Build System Architecture

### Individual Kernel Compilation (New Approach)

The new tile engine benchmark system compiles each kernel configuration into a separate executable. This provides:
- Better build parallelism
- Faster incremental builds
- More targeted testing
- Easier debugging of specific configurations

Each benchmark executable follows the naming pattern:
```
benchmark_gemm_<dtype>_<layout>_<config>_<tile_sizes>
```

### Monolithic Build (Legacy Approach)

The original system compiles all kernels into a single executable (`benchmark_gemm_[Datatype]_[Layout]`), which can then be filtered at runtime using command-line arguments.

## Build Instructions

### Prerequisites
- ROCm installation
- CMake 3.16 or higher
- C++17 compatible compiler

### Basic Build

```bash
# In the root of composable kernel, create build directory
mkdir build && cd build

# Configure with specific datatypes and layouts
# Replace [Arch] with your GPU architecture (e.g., gfx90a, gfx942)
# Replace [Datatype1;Datatype2;...] with datatypes (fp8, bf8, int8, fp16, bf16, fp32, fp64)
# Replace [Layout1;Layout2;...] with layouts (rcr, rrr, crr, ccr)
../script/cmake-ck-dev.sh ../ [Arch] -DGEMM_DATATYPE="[Datatype1;Datatype2]" -DGEMM_LAYOUT="[Layout1;Layout2]"

# Build specific benchmarks
make benchmark_gemm_[Datatype1]_[Layout1] -j
```

### Configuration Options

The build system supports several configuration options:

#### Using Custom Config Files
```bash
# Method 1: CMake variable (config file must be in configs/ directory)
cmake -DGEMM_CONFIG_FILE=my_custom_config.json ...

# Method 2: Environment variable (takes precedence over CMake variable)
export GEMM_CONFIG_FILE=my_custom_config.json
cmake ...
```

#### Config File Priority Order
1. **Environment variable** `GEMM_CONFIG_FILE` (highest priority)
2. **CMake variable** `GEMM_CONFIG_FILE`
3. **Default config** (default_config.json for all layouts)

**Note**: All custom config files must be placed in the `tile_engine/ops/gemm/configs/` directory.

### Example Build Commands

```bash
# Build for gfx942 with fp8 and fp16 datatypes, rcr layout
mkdir build && cd build
../script/cmake-ck-dev.sh ../ gfx942 -DGEMM_DATATYPE="fp8;fp16" -DGEMM_LAYOUT="rcr;ccr;rrr;crr"
make benchmark_gemm_universal_fp8_rcr -j
make benchmark_gemm_universal_fp16_rcr -j
```

### Building Individual Kernels

```bash
# Build a specific kernel configuration
make benchmark_gemm_universal_fp8_rcr_compv4_default_intrawave_False_False_False_False_256x256x32_1x4x1_32x32x32

# Build all fp16 benchmarks in parallel
make -j$(nproc) $(make help | grep benchmark_gemm_fp16 | awk '{print $2}')
```

### Rebuilding After Configuration Changes

If you modify the configuration file, you must rebuild:
```bash
rm -rf tile_engine/ && make benchmark_gemm_universal_[Datatype]_[Layout] -j
```

## Running Benchmarks

### Individual Kernel Execution

```bash
cd /path/to/build/directory
./bin/benchmark_gemm_universal_fp16_rcr_compv3_default_intrawave_False_False_False_False_256x128x32_4x1x1_32x32x16 \
    -m=512 -n=512 -k=512 -verify=1
```

### Monolithic Executable (Legacy)

```bash
# Run specific pipeline/scheduler/epilogue combination
./bin/benchmark_gemm_universal_[Datatype]_[Layout] -pipeline=compv3 -scheduler=intrawave -epilogue=default
```

### Automated Testing

Use the provided test script to run multiple benchmarks:
```bash
cd /path/to/composable_kernel/tile_engine/ops/gemm
./test_benchmark.sh [build_directory]
```

## Configuration System

### Configuration Files

The system uses JSON configuration files to specify kernel parameters:

- `configs/default_config.json` - Default configurations for various datatypes
- `configs/user_provided_config.json` - User-customizable configurations

### Configuration Structure

```json
{
    "tile_config": {
        "tile_m": {"values": [256, 128]},
        "tile_n": {"values": [256, 128]},
        "tile_k": {"values": [64, 32]},
        "warp_m": {"values": [2, 4]},
        "warp_n": {"values": [2, 1]},
        "warp_k": {"values": [1]},
        "warp_tile_m": {"values": [32, 16]},
        "warp_tile_n": {"values": [32, 16]},
        "warp_tile_k": {"values": [16, 32]}
    },
    "trait_config": {
        "pipeline": {"values": ["compv3", "compv4", "mem"]},
        "scheduler": {"values": ["intrawave", "interwave"]},
        "epilogue": {"values": ["default", "cshuffle"]},
        "pad_m": {"values": [false]},
        "pad_n": {"values": [false]},
        "pad_k": {"values": [false]},
        "persistent": {"values": [false]}
    }
}
```

## Scripts and Tools

### Python Scripts

#### gemm_instance_builder.py
**Purpose**: Shared kernel instance generator used by the non-bridged variants
(`gemm_multi_d`, `gemm_preshuffle`, `grouped_gemm`). Creates C++ kernel
implementations based on configuration files.

> The regular-GEMM subclass `gemm_universal/gemm_universal_instance_builder.py`
> has been removed; regular GEMM now goes through the Dispatcher bridge.

**Key Features**:
- Generates individual kernel header files for separate compilation
- Supports multiple data types (fp16, fp8, bf16, fp32, fp64)
- Validates tile configurations for correctness
- Creates CMake integration files

#### gemm_instance_builder_parallel.py
**Purpose**: Parallel version of the instance builder for faster generation of multiple kernel configurations.

**Features**:
- Multi-threaded kernel generation
- Improved performance for large configuration spaces

#### validation_utils.py
**Purpose**: Provides comprehensive validation functions for kernel configurations.

**Key Functions**:
- `is_tile_config_valid()` - Validates tile dimensions and alignments
- `is_trait_combination_valid()` - Checks if pipeline/epilogue/scheduler combinations are supported
- `validate_warp_tile_combination()` - GPU-specific warp tile validation
- `validate_lds_capacity()` - Ensures configurations fit in LDS memory

**Validation Checks**:
- Dimension alignment (tile dimensions must be divisible by warp dimensions)
- LDS capacity constraints
- GPU-specific warp tile support
- Unsupported trait combinations

#### test_validation.py
**Purpose**: Test suite for the validation logic to ensure correctness.

**Usage**:
```bash
python test_validation.py
```

**Tests**:
- Warp tile combination validation
- Trait combination validation
- Full tile configuration validation

#### json_config.py
**Purpose**: Configuration file parsing and management.

**Features**:
- JSON configuration loading
- Default configuration handling
- Configuration validation

#### codegen_utils.py
**Purpose**: Utility functions for code generation.

**Features**:
- Template processing
- Code formatting utilities
- File generation helpers

### Shell Scripts

#### test_benchmark.sh
**Purpose**: Automated benchmark testing script that finds and runs all built benchmark executables.

**Features**:
- Automatic build directory detection
- Batch execution of multiple benchmarks
- CSV result collection
- Colored output for easy reading
- Example command generation

**Usage**:
```bash
# Auto-detect build directory
./test_benchmark.sh

# Specify build directory
./test_benchmark.sh /path/to/build/directory
```

**What it does**:
1. Finds all benchmark executables in the build directory
2. Runs each with multiple problem sizes (512, 1024, 2048)
3. Performs GPU verification
4. Saves results to timestamped CSV file
5. Provides summary statistics

## Command Line Options

All benchmark executables support the following options:

### Matrix Dimensions
- `-m=<value>` - M dimension (default: 3840)
- `-n=<value>` - N dimension (default: 4096)
- `-k=<value>` - K dimension (default: 2048)

### Strides
- `-stride_a=<value>` - Stride for matrix A (default: 0, auto-calculated)
- `-stride_b=<value>` - Stride for matrix B (default: 0, auto-calculated)
- `-stride_c=<value>` - Stride for matrix C (default: 0, auto-calculated)

### Verification
- `-verify=<0|1|2>` - Verification mode
  - 0: No verification (default)
  - 1: CPU verification
  - 2: GPU verification

### Performance Testing
- `-warmup=<value>` - Warmup iterations (default: 50)
- `-repeat=<value>` - Benchmark iterations (default: 100)
- `-timer=<true|false>` - Use GPU timer (default: true)
- `-flush_cache=<true|false>` - Flush cache between runs (default: true)
- `-rotating_count=<value>` - Cache rotation count (default: 1000)

### Initialization
- `-init=<0|1|2>` - Tensor initialization method
  - 0: Random values [-1, 1] (default)
  - 1: Linear sequence (i % 17)
  - 2: Constant value (1.0)

### Output Options
- `-log=<true|false>` - Enable verbose logging (default: false)
- `-metric=<0|1|2>` - Performance metric
  - 0: Latency in ms (default)
  - 1: TFLOPS
  - 2: Bandwidth in GB/s
- `-json_output=<true|false>` - JSON format output (default: false)
- `-csv_filename=<filename>` - Save results to CSV
- `-csv_format=<simple|comprehensive>` - CSV format (default: comprehensive)

### Advanced Options
- `-split_k=<value>` - Split-K factor (default: 1)
- `-structured_sparsity=<true|false>` - Enable structured sparsity (default: false)
- `-pipeline=<compv3|compv4|mem>` - Pipeline type (default: compv3)
- `-scheduler=<intrawave|interwave>` - Scheduler type (default: intrawave)
- `-epilogue=<cshuffle|default>` - Epilogue type (default: cshuffle)
- `-pad_m=<true|false>` - Pad M dimension (default: false)
- `-pad_n=<true|false>` - Pad N dimension (default: false)
- `-pad_k=<true|false>` - Pad K dimension (default: false)
- `-persistent=<true|false>` - Use persistent kernel (default: false)

## Understanding Kernel Names

The kernel naming convention encodes the configuration:

```
benchmark_gemm_universal_fp16_rcr_compv3_default_intrawave_False_False_False_False_256x128x32_4x1x1_32x32x16
               		 ^^^^  ^^^ ^^^^^^ ^^^^^^^ ^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^ ^^^^^^^ ^^^^^^^^^
               		 |     |   |      |       |         |                       |         |       |
               		 |     |   |      |       |         Padding & flags         |         |       Warp tile
               		 |     |   |      |       Scheduler                         |         Thread tile
               		 |     |   |      Epilogue                                  Block tile
               		 |     |   Pipeline
               		 |     Layout (Row-Column-Row)
               		 Data type
```

### Components:
- **Data type**: fp16, fp32, bf16, fp8, bf8, int8
- **Layout**: rcr (Row-Column-Row), rrr, crr, ccr
- **Pipeline**: mem, compv3, compv4
- **Epilogue**: default, cshuffle
- **Scheduler**: intrawave, interwave
- **Flags**: pad_m, pad_n, pad_k, persistent (4 boolean flags)
- **Tile sizes**: BlockTile x ThreadTile x WarpTile

## Troubleshooting

### Common Issues

1. **Kernel not found**
   - Ensure the specific benchmark executable is built
   - Check the build directory bin/ folder

2. **Verification failures**
   - Try GPU verification (-verify=2) which may be more accurate
   - Check data type compatibility
   - Verify stride calculations

3. **Build failures**
   - Check GPU architecture compatibility
   - Ensure ROCm is properly installed
   - Verify configuration file syntax

4. **Performance variations**
   - Increase warmup iterations
   - Disable CPU frequency scaling
   - Use GPU timer for accurate measurements

### Debug Options

Enable verbose logging:
```bash
./bin/benchmark_gemm_... -log=true -verify=1
```

Test validation logic:
```bash
python test_validation.py
```

## Performance Tips

1. **Optimal Problem Sizes**: Use sizes that are multiples of tile dimensions
2. **Warmup**: Use at least 50-100 warmup iterations
3. **GPU Timer**: Always use `-timer=true` for accurate measurements
4. **Cache Management**: Enable cache flushing for consistent results
5. **Thread Affinity**: Set CPU affinity to reduce variation

## Integration Examples

### Python Integration

```python
import subprocess
import json

# Run benchmark with JSON output
result = subprocess.run([
    './bin/benchmark_gemm_universal_fp16_rcr_...', 
    '-m=1024', '-n=1024', '-k=1024',
    '-json_output=true'
], capture_output=True, text=True)

# Parse results
data = json.loads(result.stdout)
print(f"Performance: {data['tflops']} TFLOPS")
```

### Batch Testing Script

```bash
#!/bin/bash
SIZES="512 1024 2048 4096"
for size in $SIZES; do
    echo "Testing ${size}x${size}x${size}"
    ./bin/benchmark_gemm_... -m=$size -n=$size -k=$size \
        -verify=2 -csv_filename=results.csv
done
```

## Contributing

When adding new features or configurations:
1. Update validation logic in `validation_utils.py`
2. Add tests to `test_validation.py`
3. Update configuration examples
4. Document new command-line options

For more information about the Composable Kernel project, visit the main repository documentation.
