# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Development build (from build directory)
mkdir build && cd build
../script/cmake-ck-dev.sh ..                    # Uses gfx908;gfx90a;gfx942 by default
../script/cmake-ck-dev.sh .. gfx90a             # Specific GPU target
make -j32                                        # Limit threads; ~2GB RAM per thread

# Manual cmake configuration
cmake -D CMAKE_PREFIX_PATH=/opt/rocm \
      -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -D CMAKE_BUILD_TYPE=Release \
      -D GPU_TARGETS="gfx90a" \
      ..

# Common build flags
-D DTYPES="fp16;fp32"           # Build only specific data types (speeds up build)
-D CK_USE_FP8_ON_UNSUPPORTED_ARCH=ON  # Enable FP8 on MI100/MI200
-D BUILD_DEV=ON                 # Development mode with verbose errors
```

## Testing

```bash
make -j check      # Build and run all tests
make -j smoke      # Quick tests only (<30s each)
make -j regression # Long-running tests (>=30s each)

# CTest direct commands
ctest --output-on-failure -L "SMOKE_TEST"
ctest --output-on-failure -L "REGRESSION_TEST"

# Run single test executable
./test/gemm/test_gemm_fp16
```

## Architecture Overview

### Two Programming Models

1. **CK (Legacy)** - `/include/ck/`: Traditional template-based approach
2. **CK Tile (Modern)** - `/include/ck_tile/`: Newer unified model with simpler component structure

CK Tile is independently maintained and preferred for new development. Include single headers like `#include "ck_tile/core.hpp"` or `#include "ck_tile/ops/fmha.hpp"`.

### Four-Layer Architecture

1. **Templated Tile Operators** - High-level tile abstractions
2. **Templated Kernel and Invoker** - Generic kernel templates
3. **Instantiated Kernel and Invoker** - Concrete implementations per data type/hardware
4. **Client API** - User-facing interfaces

### Core Concepts

- **Tile-Based Programming**: Operations work on tiles (sub-regions) of tensors
- **Tensor Coordinate Transformation**: Maps ND tensor indices to 1D memory offsets through transform primitives (merge/unmerge/embed)
- **Distributed Tensor**: Describes how threads collaboratively process a tensor tile

### CK Tile Components

- `core/` - Basic structures: array, tuple, sequence, numeric types, coordinate transforms
- `host/` - Kernel launch utilities, device buffers, reference implementations
- `ops/` - Operation implementations (gemm, fmha, reduce)
- `ref/` - CPU/GPU reference implementations for validation

### Instance Organization

`/library/src/tensor_operation_instance/gpu/` contains 100+ operation variants organized by:
- Operation type (gemm, batched_gemm, conv, etc.)
- Data type (fp16, fp32, fp8, bf16, int8)
- Instruction set (xdl, wmma, dl, dpp)

## Key Directories

- `/example/ck_tile/01_fmha/` - Flash Multi-Head Attention (main FMHA implementation)
- `/example/01_gemm/` - Foundational GEMM example
- `/profiler/` - Performance benchmarking tool (`make -j ckProfiler`)
- `/test/` - 68+ test directories with smoke/regression classification

## Supported Hardware

- **MI Series**: gfx908 (MI100), gfx90a (MI200), gfx942/gfx950 (MI300)
- **NAVI Series**: gfx1030-1032 (NAVI2x), gfx1100-1102 (NAVI3x), gfx1200-1201 (RDNA4)

## Code Style

Pre-commit hooks enforce formatting. Install with:
```bash
sudo script/install_precommit.sh
```

Bypass temporarily with `git commit --no-verify`.

## Profiling

```bash
make -j ckProfiler
./profiler/ckProfiler gemm_xdl -M 4096 -N 4096 -K 4096 -A fp16 -B fp16 -C fp16
```

## sccache (Faster Rebuilds)

```bash
sccache --start-server
cmake ... -DCMAKE_HIP_COMPILER_LAUNCHER=sccache \
          -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
```
