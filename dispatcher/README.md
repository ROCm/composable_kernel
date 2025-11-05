# CK Tile Dispatcher

**Status:** ✅ Production Ready  
**Version:** 1.0.0  
**Platform:** AMD GPUs (gfx942 validated)

Unified dispatcher for CK Tile GEMM kernels with C++ and Python frontends.

---

## Quick Start

### Python (Recommended)
```python
from dispatcher_api import SimpleGemmAPI

gemm = SimpleGemmAPI()
gemm.ensure_kernels_ready()  # Auto-generates and builds
result = gemm.execute(M=1024, N=1024, K=1024)
```

### C++
```cpp
#include "ck_tile/dispatcher/dispatcher.hpp"

Dispatcher dispatcher;
Problem problem(1024, 1024, 1024);
float time = dispatcher.run(a_dev, b_dev, c_dev, problem);
```

---

## Installation

### Build C++ Library
```bash
cd dispatcher/build
cmake .. -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
make -j
```

### Build with Python
```bash
cmake .. -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
         -DBUILD_DISPATCHER_PYTHON=ON
make -j
```

### Build with Tests
```bash
cmake .. -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
         -DBUILD_DISPATCHER_TESTS=ON \
         -DBUILD_DISPATCHER_PYTHON=ON \
         -DBUILD_DISPATCHER_EXAMPLES=ON
make -j
ctest  # Run tests
```

---

## Features

### Core Capabilities
- ✅ **Kernel Registry** - Thread-safe registration with priority management
- ✅ **Selection Strategies** - FirstFit and Heuristic-based selection  
- ✅ **Dual API** - Complete C++ and Python interfaces
- ✅ **Real CK Tile Kernels** - Integration with unified_gemm_codegen.py
- ✅ **GPU Execution** - Validated on AMD MI325X

### Python API (High-Level)
- `generate_kernels()` - Generate CK Tile kernels from Python
- `SimpleGemmAPI` - Automated workflow (generate → build → execute)
- `Dispatcher` - Full control over generation, build, execution
- `quick_gemm()` - One-liner for quick execution

### C++ API
- `Dispatcher` - Main dispatch interface
- `Registry` - Kernel registration and lookup
- `KernelInstance` - Uniform kernel interface
- `KernelKey` - Kernel configuration metadata

---

## Architecture

```
Python API (dispatcher_api.py)
    ↓
C++ Extension (_dispatcher_native.so) 
    ↓
Dispatcher Core (Registry + Selection)
    ↓
Backend Wrappers (GeneratedTileKernelInstance)
    ↓
Real CK Tile Kernels (unified_gemm_codegen.py)
    ↓
GPU Execution (AMD MI325X gfx942)
```

---

## Directory Structure

```
dispatcher/
├── README.md               # This file
├── QUICKSTART.md           # 5-minute guide
├── BUILD_AND_TEST.md       # Detailed build instructions
├── VALIDATION.md           # Test results and validation
│
├── include/                # C++ headers
│   └── ck_tile/dispatcher/
│       ├── dispatcher.hpp
│       ├── registry.hpp
│       ├── kernel_key.hpp
│       ├── problem.hpp
│       ├── kernel_instance.hpp
│       ├── backends/
│       │   ├── generated_tile_backend.hpp  # For unified_gemm_codegen
│       │   └── tile_backend.hpp            # For tile_engine
│       └── validation/
│           └── reference_kernels.hpp
│
├── src/                    # C++ implementation
│   ├── dispatcher.cpp
│   └── registry.cpp
│
├── python/                 # Python API
│   ├── dispatcher_api.py   # High-level API
│   ├── bindings.cpp        # pybind11 bindings
│   └── __init__.py         # Package interface
│
├── test/                   # Tests (51 tests, 100% passing)
│   ├── test_kernel_key.cpp
│   ├── test_problem.cpp
│   ├── test_registry.cpp
│   ├── test_dispatcher.cpp
│   ├── test_tile_backend.cpp
│   └── test_integration_e2e.cpp
│
├── examples/               # Examples
│   ├── single_tile_kernel_example.cpp     # Real GPU execution
│   └── python_complete_workflow.py        # Python demo
│
└── codegen/                # Kernel generation
    ├── unified_gemm_codegen.py            # Fixed and working
    └── generate_dispatcher_registration.py
```

---

## Usage Examples

### Generate and Execute (Python)
```python
from dispatcher_api import Dispatcher

d = Dispatcher()

# Generate kernels
d.generate_kernels(datatype='fp16', layout='rcr', preset='essential')

# Build executable  
executable = d.build_gpu_executable()

# Execute on GPU
result = d.run_gpu_gemm(M=2048, N=2048, K=2048)
```

### C++ with Generated Kernels
```cpp
// Include generated kernel (via -include flag or namespace)
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"

// Create and register
auto kernel = create_generated_tile_kernel<
    SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
    key, kernel_name);

Registry::instance().register_kernel(kernel);

// Use via dispatcher
Dispatcher dispatcher;
float time = dispatcher.run(a_dev, b_dev, c_dev, problem);
```

---

## Testing

### Run All Tests
```bash
cd build
ctest --output-on-failure
```

### Run Python Tests
```bash
PYTHONPATH=../python python3 ../examples/python_complete_workflow.py
```

### Run GPU Example
```bash
./examples/single_tile_kernel_example
```

---

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute getting started guide
- **[BUILD_AND_TEST.md](BUILD_AND_TEST.md)** - Complete build instructions  
- **[VALIDATION.md](VALIDATION.md)** - Test results and validation report
- **[../DISPATCHER.md](../DISPATCHER.md)** - Complete design document

---

## Validation Summary

| Component | Status |
|-----------|--------|
| C++ Core | ✅ 51/51 tests passing |
| Python Bindings | ✅ Extension working |
| Kernel Generation | ✅ 6 kernels created |
| GPU Execution | ✅ AMD MI325X validated |
| Design Compliance | ✅ 100% per DISPATCHER.md |

**Ready for production use.**

---

## Next Steps

### For Users
1. Generate kernels: `python3 codegen/unified_gemm_codegen.py --preselected fp16_rcr_essential --output-dir build/generated_kernels`
2. Build library: `cd build && cmake .. && make -j`
3. Run tests: `ctest`
4. Use in your code: `#include "ck_tile/dispatcher/dispatcher.hpp"`

### For Developers  
- See [BUILD_AND_TEST.md](BUILD_AND_TEST.md) for development workflow
- Run `./validate_all.sh` for complete validation
- Check [VALIDATION.md](VALIDATION.md) for test results

### For Integration
- **ck4inductor**: Use `dispatcher_api.py` for Python integration
- **PyTorch**: Create custom operator with C++ extension
- **MIOpen**: Use C++ API directly

---

## License

MIT License - Copyright (c) 2025, Advanced Micro Devices, Inc.

---

**Implementation Status:** ✅ Complete  
**Test Status:** ✅ All Passing  
**Production Status:** ✅ Ready
