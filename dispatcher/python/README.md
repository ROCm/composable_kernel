# CK Tile Dispatcher - Python Interface

Python utilities for the CK Tile GEMM dispatcher.

> **See also:** [Main Dispatcher README](../README.md) for installation and core concepts.

## Setup

```bash
# Set Python path (from dispatcher directory)
export PYTHONPATH=$PWD/python:$PYTHONPATH

# Install NumPy
pip install numpy
```

## Quick Start

```python
from ctypes_utils import (
    KernelConfig, CodegenRunner, DispatcherLib, Registry, Dispatcher
)
import numpy as np

# 1. Define kernel config
config = KernelConfig(tile_m=128, tile_n=128, tile_k=32)

# 2. Generate kernel
codegen = CodegenRunner()
codegen.generate_from_config(config)

# 3. Load library and create registry
lib = DispatcherLib.auto()
registry = Registry(name="demo", lib=lib)
registry.register_kernel(config)

# 4. Create dispatcher and run
dispatcher = Dispatcher(registry=registry, lib=lib)
A = np.random.randn(1024, 1024).astype(np.float16)
B = np.random.randn(1024, 1024).astype(np.float16)
result = dispatcher.run(A, B, 1024, 1024, 1024)

print(f"Time: {result.time_ms:.4f} ms, TFLOPS: {result.tflops:.2f}")
```

## Core Classes (`ctypes_utils.py`)

### KernelConfig

Complete kernel configuration:

```python
config = KernelConfig(
    # Data types
    dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32",
    
    # Layouts
    layout_a="row", layout_b="col", layout_c="row",
    
    # Tile shape
    tile_m=128, tile_n=128, tile_k=32,
    
    # Wave/warp configuration
    wave_m=2, wave_n=2, wave_k=1,
    warp_m=32, warp_n=32, warp_k=16,
    
    # Pipeline
    pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
    
    # Padding
    pad_m=True, pad_n=True, pad_k=True,
    
    # Target
    gfx_arch="gfx942",
)

config.print_config()  # Pretty print
print(config.tile_str)  # "128x128x32"
```

### CodegenRunner

Generate kernels:

```python
codegen = CodegenRunner(
    datatype="fp16",
    layout="rcr",
    gpu_target="gfx942",
)

# Generate from config
result = codegen.generate_from_config(config)

# Generate variant
result = codegen.generate("standard")
result = codegen.generate("preshuffle")
result = codegen.generate("multi_d")

# Generate all
results = codegen.generate_all()

# Categorize kernels
categories = codegen.categorize_kernels()
print(f"Total: {categories['total']}")
print(f"Compute: {len(categories['compute'])}")
```

### Registry

Store kernel configurations:

```python
registry = Registry(name="my_registry")
registry.register_kernel(config)
registry.bind_library(lib)

print(registry.kernel_count)
print(registry.get_kernels())
```

### Dispatcher

Select and run kernels:

```python
dispatcher = Dispatcher(registry=registry, lib=lib)

# Check support
if dispatcher.is_supported(M, N, K):
    result = dispatcher.run(A, B, M, N, K)
    
# Select kernel
kernel_name = dispatcher.select_kernel(M, N, K)
```

### DispatcherLib

Load compiled library:

```python
# Auto-find or compile
lib = DispatcherLib.auto()

# Load specific path
lib = DispatcherLib.load("/path/to/libdispatcher_gemm.so")

# Library operations
lib.get_kernel_name()
lib.get_kernel_count()
lib.is_supported(M, N, K)
lib.export_json()
```

### GemmRunner / Validator

High-level utilities:

```python
# Run GEMM
runner = GemmRunner(lib)
result = runner.run(A, B)
print(f"TFLOPS: {result.tflops}")

# Validate
validator = Validator(rtol=1e-3, atol=1e-2)
is_correct, max_err, mean_err = validator.check(result.output, reference)
```

## Examples

See [examples/python/](../examples/python/):

| Example | Description |
|---------|-------------|
| `01_basic_gemm.py` | Complete explicit workflow |
| `02_batch_gemm.py` | Multiple sizes |
| `03_benchmark.py` | Performance testing |
| `04_validation.py` | Correctness testing |
| `05_numpy_integration.py` | GPUMatmul class |
| `06_json_export.py` | JSON export |
| `07_preshuffle.py` | PreShuffle kernels |
| `08_multi_d.py` | Multi-D GEMM |
| `09_multi_registry.py` | Multiple registries |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Set `PYTHONPATH` to `dispatcher/python` |
| Library not found | Run `make dispatcher_gemm` in build |
| NumPy not found | `pip install numpy` |

---

> **More info:** See [../README.md](../README.md) for full documentation.
