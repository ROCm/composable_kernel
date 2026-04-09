# CK Tile Dispatcher - Language Bindings

This directory contains language bindings for the CK Tile Dispatcher.

## Structure

```
bindings/
|---- ctypes/              # Python ctypes bindings (C API)
|   |---- gemm_ctypes_lib.cpp      # GEMM dispatcher C API
|   |---- conv_ctypes_lib.cpp      # Grouped conv dispatcher C API (fwd + bwd_data)
|   |---- conv_bwdw_ctypes_lib.cpp # Grouped conv backward weight C API (separate library)
|   |---- gpu_helper.cpp           # CLI helper for Python
|   +---- CMakeLists.txt
+---- README.md
```

## ctypes Bindings

The ctypes bindings provide a C API that Python can load via `ctypes.CDLL()`.

### Building

```bash
cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
make dispatcher_gemm_lib dispatcher_conv_lib gpu_helper
```

### Usage from Python

```python
import ctypes

# Load the library
lib = ctypes.CDLL("path/to/libdispatcher_gemm_lib.so")

# Initialize
lib.dispatcher_init()

# Check if problem is supported
is_supported = lib.dispatcher_is_supported(M, N, K)

# Run GEMM
time_ms = ctypes.c_float()
result = lib.dispatcher_run_gemm(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    ctypes.byref(time_ms)
)

# Cleanup
lib.dispatcher_cleanup()
```

### GEMM API

| Function | Description |
|----------|-------------|
| `dispatcher_init()` | Initialize the dispatcher |
| `dispatcher_is_supported(M, N, K)` | Check if problem size is supported |
| `dispatcher_select_kernel(M, N, K, name_buf, buf_size)` | Get kernel name for problem |
| `dispatcher_run_gemm(A, B, C, M, N, K, time_ms)` | Execute GEMM |
| `dispatcher_get_kernel_count()` | Get number of registered kernels |
| `dispatcher_export_registry_json()` | Export registry as JSON |
| `dispatcher_cleanup()` | Release resources |

### Grouped Convolution API

| Function | Description |
|----------|-------------|
| `conv_dispatcher_init()` | Initialize the dispatcher |
| `conv_dispatcher_is_supported(prob)` | Check if problem is supported |
| `conv_dispatcher_select_kernel(prob, name_buf, buf_size)` | Get kernel name |
| `conv_dispatcher_run(input, weight, output, prob, stream)` | Execute convolution |
| `conv_dispatcher_get_kernel_count()` | Get number of registered kernels |
| `conv_dispatcher_cleanup()` | Release resources |

## GPU Helper

The `gpu_helper` executable provides a CLI interface for Python:

```bash
./gpu_helper 1024 1024 1024 --validate
```

Output is JSON for easy parsing:
```json
{
  "problem": {"M": 1024, "N": 1024, "K": 1024},
  "kernel": "gemm_fp16_rcr_...",
  "execution": {
    "time_ms": 0.5,
    "tflops": 4.2
  },
  "validation": {
    "accuracy": 100.0
  },
  "status": "success"
}
```

## Examples

See the examples that use these bindings:

- **GEMM**: `dispatcher/examples/gemm/python/`

### Grouped Convolution

Grouped convolution C++ headers and Python utilities are in:
- **C++ Headers**: `dispatcher/include/ck_tile/dispatcher/grouped_conv_*.hpp`
- **Python Utils**: `dispatcher/python/grouped_conv_utils.py`
- **Build Script**: `dispatcher/scripts/compile_grouped_conv_examples.py`

