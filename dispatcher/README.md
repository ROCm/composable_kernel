# CK Tile Dispatcher

Unified dispatcher mechanism for CK Tile GEMM kernels providing kernel registration, selection, and execution.

## Overview

The dispatcher provides a clean abstraction layer for:
- **Kernel Registry**: Central mapping from kernel configurations to executable instances
- **Selection Engine**: Automatic kernel selection based on problem requirements
- **Unified Execution**: Common interface for running kernels regardless of backend

## Architecture

```
┌─────────────────────────────────────┐
│         Dispatcher API              │
│  (Python & C++)                     │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       │   Registry     │
       │  (Thread-safe) │
       └───────┬────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐         ┌─────▼─────┐
│CK Tile │         │CK Library │
│Backend │         │Backend    │
│        │         │(Future)   │
└────────┘         └───────────┘
```

## Core Abstractions

### KernelKey
Compile-time kernel configuration organized into:
- **Signature**: What operation is computed (data types, layouts, element-wise ops)
- **Algorithm**: How it's implemented (tile sizes, pipeline, scheduler)

### Problem
Runtime parameters for kernel invocation:
- Problem dimensions (M, N, K)
- Resource preferences
- Validation control

### KernelInstance
Uniform interface for kernel execution:
- `supports()`: Check problem compatibility
- `run()`: Execute kernel
- `validate()`: Verify output correctness

## Usage Example (C++)

```cpp
#include "ck_tile/dispatcher/dispatcher.hpp"

using namespace ck_tile::dispatcher;

// Create dispatcher
Dispatcher dispatcher;

// Define problem
Problem problem(1024, 1024, 1024);  // M, N, K

// Execute GEMM: C = A * B
float time = dispatcher.run(a_ptr, b_ptr, c_ptr, problem);

// Or with explicit kernel selection
float time2 = dispatcher.run_explicit(
    "256x256x32_2x2x1_32x32x16_persist",
    a_ptr, b_ptr, c_ptr, nullptr, problem);
```

## Building

### Basic Build
```bash
cd dispatcher
mkdir build && cd build
cmake ..
make -j
```

### With Auto-Generated Wrappers (Recommended)
```bash
cmake .. \
    -DBUILD_DISPATCHER_TESTS=ON \
    -DDISPATCHER_AUTO_GENERATE_WRAPPERS=ON \
    -DTILE_ENGINE_DIR=../tile_engine/ops/gemm
make -j
```

This automatically generates dispatcher wrappers from tile_engine kernels.

### Manual Wrapper Generation
```bash
# Generate wrappers manually
make dispatcher_generate_wrappers

# Or run Python script directly
python codegen/generate_dispatcher_wrappers.py \
    --tile-engine-dir ../tile_engine/ops/gemm \
    --output-dir build/generated
```

## Directory Structure

```
dispatcher/
├── include/ck_tile/dispatcher/  # Public headers
│   ├── kernel_key.hpp           # Kernel configuration metadata
│   ├── problem.hpp              # Problem abstraction
│   ├── kernel_instance.hpp      # Kernel interface
│   ├── registry.hpp             # Kernel registry
│   ├── dispatcher.hpp           # Main dispatcher
│   └── backends/
│       └── tile_backend.hpp     # CK Tile backend wrapper
├── src/                         # Implementation
│   ├── registry.cpp
│   └── dispatcher.cpp
├── codegen/                     # Unified codegen system
│   ├── generate_dispatcher_wrappers.py  # Main codegen script
│   ├── CMakeLists.txt           # Codegen build integration
│   ├── README.md                # Codegen documentation
│   └── example_integration.cpp  # Integration example
├── python/                      # Python bindings
│   ├── __init__.py
│   ├── bindings.cpp
│   └── example.py
├── test/                        # Unit tests
│   ├── test_kernel_key.cpp
│   ├── test_problem.cpp
│   └── test_registry.cpp
├── CMakeLists.txt
├── README.md
└── IMPLEMENTATION_SUMMARY.md
```

## Design Document

See `../DISPATCHER_DESIGN_DOC.md` for complete design rationale and implementation details.

## Status

**Current**: Core abstractions implemented (KernelKey, Problem, Registry, Dispatcher)

**Next Steps**:
1. CK Tile backend wrapper for generated kernels
2. Python bindings via pybind11
3. Unit tests
4. Integration with tile_engine
5. CK Library backend support (future)

## License

MIT License - Copyright (c) 2025, Advanced Micro Devices, Inc.

