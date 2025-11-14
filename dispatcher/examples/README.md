# CK Tile Dispatcher Examples

This directory contains C++ and Python examples demonstrating the dispatcher functionality.

## Directory Structure

```
examples/
├── cpp/                      # C++ examples (real GPU execution)
│   ├── dispatcher_dynamic_lib.cpp      # Dynamic .so for Python ctypes
│   ├── python_gpu_helper.cpp           # CLI helper for Python
│   ├── single_tile_kernel_example.cpp  # Performance benchmark
│   ├── verify_correctness.cpp          # Random matrix validation
│   ├── test_known_matrices.cpp         # Structured matrix tests
│   └── verify_data_flow.cpp            # Data transfer verification
│
├── python/                   # Python examples (real GPU execution)
│   ├── numpy_to_gpu_complete.py        # NumPy integration (THE KEY FILE)
│   ├── numpy_dispatcher_advanced.py    # Advanced benchmarks
│   └── python_dispatcher_basic.py      # C++ extension API demo
│
├── README.md                 # This file
└── CMakeLists.txt            # Build configuration
```

**All examples use real CK Tile GEMM kernels. No mock/dummy code.**

## C++ Examples

### 1. python_gpu_helper

**Purpose:** CLI tool for Python integration  
**Usage:** `./build/examples/python_gpu_helper <M> <N> <K> [--validate]`  
**Output:** JSON format for easy Python parsing

```bash
./build/examples/python_gpu_helper 1024 1024 1024 --validate
```

### 2. single_tile_kernel_example

**Purpose:** Performance benchmark with single CK Tile kernel  
**Performance:** 115.5 TFLOPS on 1024³  
**Usage:** `./build/examples/single_tile_kernel_example`

Demonstrates dispatcher selecting and executing optimized GPU kernel.

### 3. verify_correctness

**Purpose:** Validate GPU results vs CPU reference with random matrices  
**Usage:** `./build/examples/verify_correctness <M> <N> <K>`

```bash
./build/examples/verify_correctness 1024 1024 1024
```

### 4. test_known_matrices

**Purpose:** Test with structured matrices (identity, all-ones)  
**Usage:** `./build/examples/test_known_matrices <size>`

```bash
./build/examples/test_known_matrices 256
```

### 5. verify_data_flow

**Purpose:** Verify data transfer integrity (GPU memory correctness)  
**Usage:** `./build/examples/verify_data_flow`

## Python Examples

### 1. numpy_to_gpu_complete.py (THE KEY EXAMPLE - Recommended!)

**Purpose:** Complete NumPy to GPU workflow via ctypes  
**Performance:** 23.52 TFLOPS on 512³, 28,025x faster than NumPy  
**Usage:**

```bash
cd dispatcher
python3 examples/python/numpy_to_gpu_complete.py
```

**Demonstrates:**
- Creating NumPy matrices in Python
- Compiling dynamic library (.so) with dispatcher
- Loading .so via ctypes
- Passing NumPy array pointers directly to C++
- GPU GEMM execution
- Results back in NumPy arrays
- Zero-copy data passing

**This is the complete Python <-> GPU integration!**

### 2. numpy_dispatcher_advanced.py

**Purpose:** Advanced benchmarks and validation  
**Performance:** Up to 319.02 TFLOPS on 2048³, 380,873x faster than NumPy  
**Usage:**

```bash
python3 examples/python/numpy_dispatcher_advanced.py
```

**Demonstrates:**
- Multiple problem sizes (128³ to 2048³)
- Random matrix validation
- Performance metrics and comparisons
- Speedup calculations vs NumPy

### 3. python_dispatcher_basic.py

**Purpose:** C++ extension API demo  
**Usage:**

```bash
cd dispatcher
python3 examples/python/python_dispatcher_basic.py
```

**Demonstrates:**
- Problem creation
- KernelKey configuration
- Registry operations  
- Dispatcher selection strategies
- Setting heuristics from Python
- Available enums and types

**Note:** This is an API reference example, not for GPU execution.

## Building Examples

Examples require generated kernels. Build with:

```bash
cd dispatcher
mkdir build && cd build

cmake .. \
  -D CMAKE_PREFIX_PATH=/opt/rocm \
  -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -D CMAKE_BUILD_TYPE=Release \
  -D GPU_TARGETS="gfx942" \
  -D BUILD_DISPATCHER_EXAMPLES=ON \
  -D BUILD_DISPATCHER_PYTHON=ON

make -j
```

## Setup

### Make Python Scripts Executable

```bash
cd dispatcher/examples/python
chmod +x *.py
```

Note: All Python examples should be executable. If you get "Permission denied", run the chmod command above.

### Set Python Path

Python examples need access to the C++ extension:

```bash
export PYTHONPATH=/home/sshuser/composable_kernel/dispatcher/python
# Or use relative path:
export PYTHONPATH=../python  # when in examples/ directory
```

Alternatively, use inline:

```bash
PYTHONPATH=../python python3 examples/python/numpy_to_gpu_complete.py
```

## Running Examples

### C++ Examples

```bash
cd build/examples

# Performance test
./single_tile_kernel_example

# Correctness validation
./verify_correctness 1024 1024 1024

# Known matrices
./test_known_matrices 256

# Data flow
./verify_data_flow

# Python helper (used by Python scripts)
./python_gpu_helper 512 512 512 --validate
```

### Python Examples

```bash
cd dispatcher

# Set Python path
export PYTHONPATH=python

# Run examples
python3 examples/python/python_dispatcher_basic.py
python3 examples/python/python_invoke_dispatcher.py
python3 examples/python/python_gpu_dispatcher.py
python3 examples/python/python_complete_workflow.py
```

## Performance Results

| Example | Problem Size | Performance | Validation |
|---------|--------------|-------------|------------|
| single_tile_kernel_example | 1024³ | 115.5 TFLOPS | N/A |
| python_invoke_dispatcher | 1024³ | 112.96 TFLOPS | 100% |
| verify_correctness | Configurable | Varies | 100% |
| python_gpu_helper | Configurable | Varies | Optional |

## Dependencies

**C++ Examples:**
- ROCm 7.0+ with HIP
- CMake 3.16+
- CK Tile headers
- Generated kernels

**Python Examples:**
- Python 3.8+
- NumPy (for validation examples)
- pybind11 (for C++ extension)
- C++ extension built with `-DBUILD_DISPATCHER_PYTHON=ON`

## Notes

- All C++ examples use generated kernels via `-include` compiler flag (tile_engine pattern)
- Python examples can invoke GPU execution through `python_gpu_helper` executable
- C++ extension (`_dispatcher_native`) provides low-level dispatcher API to Python
- For direct NumPy integration, use ctypes or custom C++ wrapper
- Examples automatically skip if kernels not generated

## Troubleshooting

**Issue:** Examples not building  
**Solution:** Generate kernels first:
```bash
cd codegen
python3 unified_gemm_codegen.py --preselected fp16_rcr_essential --output-dir ../build/generated_kernels
```

**Issue:** Python extension not found  
**Solution:** Build with `-DBUILD_DISPATCHER_PYTHON=ON` and set `PYTHONPATH=python`

**Issue:** Poor performance  
**Solution:** Use `-DCMAKE_BUILD_TYPE=Release` (not Debug)

