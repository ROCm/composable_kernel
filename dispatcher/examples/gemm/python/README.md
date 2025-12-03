# GEMM Python Examples

CK Tile Dispatcher Python examples for GEMM (General Matrix Multiplication) operations.

> **Main Documentation**: [Dispatcher README](../../../README.md) | [Examples Overview](../../README.md)

## Quick Start

### Build Library

```bash
cd /path/to/composable_kernel/dispatcher
mkdir -p build && cd build

cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DBUILD_DISPATCHER_EXAMPLES=ON

# Build Python library (kernels generated automatically)
make dispatcher_gemm_lib -j$(nproc)
```

### Run Examples

```bash
cd /path/to/composable_kernel/dispatcher

python3 examples/gemm/python/01_basic_gemm.py
python3 examples/gemm/python/04_validation.py
python3 examples/gemm/python/05_numpy_integration.py
```

## Examples

| Example | Description |
|---------|-------------|
| [01_basic_gemm.py](01_basic_gemm.py) | Basic GEMM with GPU execution |
| [02_batch_gemm.py](02_batch_gemm.py) | Batched GEMM operations |
| [03_benchmark.py](03_benchmark.py) | Performance benchmarking |
| [04_validation.py](04_validation.py) | CPU reference validation |
| [05_numpy_integration.py](05_numpy_integration.py) | NumPy array integration |
| [06_json_export.py](06_json_export.py) | Registry JSON export |
| [07_preshuffle.py](07_preshuffle.py) | Preshuffle optimization |
| [08_multi_d.py](08_multi_d.py) | Multi-D tensor ops |
| [09_multi_registry.py](09_multi_registry.py) | Multiple registries |
| [10_advanced_benchmark.py](10_advanced_benchmark.py) | Advanced benchmark with full control |

## Example Details

### 01_basic_gemm.py - Basic GEMM
Demonstrates the declarative Python API with GPU execution:

```python
from ctypes_utils import Signature, Algorithm, ArchInfo, KernelSet, DispatcherLib

# Define kernel configuration
sig = Signature()
sig.dtype("fp16")
sig.layout = "rcr"

algo = Algorithm()
algo.tile(128, 128, 32)
algo.pipeline = "compv3"
algo.scheduler = "intrawave"

# Create kernel set
kernel_set = KernelSet("basic_kernels")
kernel_set.add(sig, algo, ArchInfo(name="gfx942"))

# Run on GPU
lib = DispatcherLib.auto()
lib.initialize()
elapsed_ms = lib.run(a_ptr, b_ptr, c_ptr, M, N, K)
```

### 02_batch_gemm.py - Batch GEMM
Batched matrix multiplication:
- Multiple independent GEMM operations
- Batch dimension handling

### 03_benchmark.py - Benchmarking
Performance measurement:
- GPU timing
- TFLOPS calculation

### 04_validation.py - Validation
Correctness verification:
- NumPy reference implementation
- Tolerance-based validation

### 05_numpy_integration.py - NumPy Integration
Seamless NumPy integration:
- NumPy arrays to GPU buffers
- Results back to NumPy

### 06_json_export.py - JSON Export
Registry serialization for tool integration.

### 07_preshuffle.py - Preshuffle
Layout optimization for better performance.

### 08_multi_d.py - Multi-D Operations
Multi-dimensional tensor operations with bias.

### 09_multi_registry.py - Multiple Registries
Separate registries for different workloads.

## Utility Module: ctypes_utils.py

```python
from ctypes_utils import (
    Signature,        # Operation signature
    Algorithm,        # Algorithm details
    ArchInfo,         # Target GPU
    KernelConfig,     # Single kernel config
    KernelSet,        # Collection of kernels
    DispatcherLib,    # C++ library wrapper
    Dispatcher,       # High-level dispatcher
)
```

### Basic Usage

```python
from ctypes_utils import DispatcherLib, Dispatcher

# Load library
lib = DispatcherLib.auto()
lib.initialize()

# Create dispatcher
dispatcher = Dispatcher(lib)

# Run GEMM
elapsed_ms = dispatcher.run(a_ptr, b_ptr, c_ptr, M=4096, N=4096, K=4096)
print(f"TFLOPS: {2*M*N*K/elapsed_ms/1e9:.2f}")
```

### GPU Memory Management

```python
import ctypes
import numpy as np

# Load HIP library
hip = ctypes.CDLL("libamdhip64.so")

# Allocate GPU memory
gpu_ptr = ctypes.c_void_p()
hip.hipMalloc(ctypes.byref(gpu_ptr), size_in_bytes)

# Copy to GPU (1 = hipMemcpyHostToDevice)
hip.hipMemcpy(gpu_ptr, host_array.ctypes.data, size, 1)

# Copy back (2 = hipMemcpyDeviceToHost)
hip.hipMemcpy(host_array.ctypes.data, gpu_ptr, size, 2)

# Free
hip.hipFree(gpu_ptr)
```

## Related Documentation

- [C++ GEMM Examples](../cpp/README.md)
- [Python Conv Examples](../../conv/python/README.md)
- [Main Dispatcher README](../../../README.md)
