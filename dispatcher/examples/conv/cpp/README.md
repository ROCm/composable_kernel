# Convolution C++ Examples

CK Tile Dispatcher C++ examples for Convolution operations (Forward, Backward Data, Backward Weight).

> **Main Documentation**: [Dispatcher README](../../../README.md) | [Examples Overview](../../README.md)

## Quick Start

### Build and Run

```bash
cd /path/to/composable_kernel/dispatcher
mkdir -p build && cd build

cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DBUILD_DISPATCHER_EXAMPLES=ON

# Build all examples (kernels are generated automatically during make)
make -j$(nproc)

# Run examples
./conv_01_forward
./conv_02_validation
./conv_09_bwd_data --verify
./conv_10_bwd_weight --verify
```

### Build Targets

```bash
# Build everything (auto-generates kernels)
make

# Generate kernels only (no compilation)
make generate_all_kernels

# Force regenerate kernels
make regenerate_all_kernels

# Build only Python libraries
make python_libs

# Generate for specific architecture
make generate_kernels_gfx942
make generate_kernels_gfx90a
```

## Examples

| Example | Description | Complexity |
|---------|-------------|------------|
| [01_conv_forward.cpp](01_conv_forward.cpp) | 2D forward with tensor setup | ★★☆☆☆ |
| [02_conv_validation.cpp](02_conv_validation.cpp) | CPU reference validation | ★★☆☆☆ |
| [03_multi_size.cpp](03_multi_size.cpp) | Multiple problem sizes | ★★☆☆☆ |
| [04_benchmark.cpp](04_benchmark.cpp) | Advanced benchmark with full control | ★★★☆☆ |
| [05_heuristics.cpp](05_heuristics.cpp) | Heuristic kernel selection | ★★★☆☆ |
| [06_json_export.cpp](06_json_export.cpp) | Export registry to JSON | ★★☆☆☆ |
| [07_multi_registry.cpp](07_multi_registry.cpp) | Multiple registries | ★★★☆☆ |
| [08_conv3d_forward.cpp](08_conv3d_forward.cpp) | 3D volumetric convolution | ★★★☆☆ |
| [09_bwd_data.cpp](09_bwd_data.cpp) | Backward data gradient | ★★★☆☆ |
| [10_bwd_weight.cpp](10_bwd_weight.cpp) | Backward weight gradient | ★★★☆☆ |

## Declarative Kernel Pattern

Convolution examples use the **Signature/Algorithm/Arch** declarative pattern:

```cpp
DECL_CONV_KERNEL_SET(my_kernels,
    .add(
        ConvSig()                        // WHAT: convolution signature
            .dtype("fp16")               // Data type (fp16, bf16, fp32, fp8, int8)
            .layout("nhwgc")             // Tensor layout
            .conv_type("forward")        // Direction: forward, bwd_data, bwd_weight
            .dims(2),                    // Spatial dims: 1, 2, or 3
        ConvAlgo()                       // HOW: algorithm details
            .tile(1, 128, 128)           // Block tile (M, N, K)
            .wave(2, 2, 1)               // Wave distribution (M, N, K warps)
            .warp(32, 32, 16)            // Warp tile sizes (M, N, K per warp)
            .pipeline("compv4")          // Pipeline: mem, compv3, compv4, compv5
            .scheduler("intrawave")      // Scheduler: intrawave, interwave
            .vector_sizes(4, 8, 8)       // Vector sizes (A, B, C)
            .block_per_cu(1),            // Blocks per CU hint
        "gfx942"                         // WHERE: target architecture
    )
);
```

## Complete Configuration Parameters

### ConvSignature (WHAT operation)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dtype_in_` | string | "fp16" | Input data type |
| `dtype_wei_` | string | "fp16" | Weight data type |
| `dtype_out_` | string | "fp16" | Output data type |
| `dtype_acc_` | string | "fp32" | Accumulator type |
| `dtype_workspace_` | string | "fp32" | Workspace type (two-stage) |
| `dtype_bias_` | string | "fp16" | Bias type (bias epilogue) |
| `layout_` | string | "nhwc" | Data layout |
| `conv_op_` | string | "forward" | Direction |
| `num_dims_` | int | 2 | Spatial dimensions (1, 2, 3) |
| `groups_` | int | 1 | Group convolution count |
| `specialization_` | string | "default" | Filter specialization |

### ConvAlgorithm (HOW computed)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tile_m_`, `tile_n_`, `tile_k_` | int | 1, 128, 128 | Block tile dimensions |
| `wave_m_`, `wave_n_`, `wave_k_` | int | 2, 2, 1 | Wave/warp distribution |
| `warp_m_`, `warp_n_`, `warp_k_` | int | 32, 32, 16 | Warp tile sizes |
| `vector_a_`, `vector_b_`, `vector_c_` | int | 4, 8, 8 | Vector sizes |
| `pipeline_` | string | "compv4" | Pipeline: mem, compv3, compv4, compv5 |
| `scheduler_` | string | "intrawave" | Scheduler: intrawave, interwave |
| `epilogue_` | string | "cshuffle" | Epilogue: cshuffle, default |
| `memory_op_` | string | "set" | Memory op: set, atomic_add |
| `block_per_cu_` | int | 1 | Blocks per CU hint |
| `num_wave_groups_` | int | 1 | Wave groups (V5 pipeline) |
| `num_groups_to_merge_` | int | 1 | Groups to merge |
| `double_smem_buffer_` | bool | false | Double buffering |
| `pad_m_`, `pad_n_`, `pad_k_` | bool | true | Dimension padding |

### Supported Data Types

| Type | Description | Accumulator |
|------|-------------|-------------|
| fp32 | 32-bit float | fp32 |
| fp16 | 16-bit float (half) | fp32 |
| bf16 | 16-bit bfloat | fp32 |
| fp8 | 8-bit E4M3 float | fp32 |
| bf8 | 8-bit E5M2 float | fp32 |
| int8 | 8-bit signed integer | int32 |

## Convolution Problem Definition

```cpp
#include "ck_tile/dispatcher/conv_utils.hpp"

// Create 2D problem
auto problem = create_conv2d_problem(
    N,           // Batch size
    C,           // Input channels
    K,           // Output channels
    Hi, Wi,      // Input spatial size
    Y, X,        // Filter size
    stride,      // Stride
    pad,         // Padding
    ConvOp::Forward  // Direction
);

// Create 3D problem
auto problem = create_conv3d_problem(
    N, C, K,
    Di, Hi, Wi,  // 3D input
    Z, Y, X,     // 3D filter
    stride, pad,
    ConvOp::Forward
);
```

## Benchmark Parameters (stream_config)

Example 04 demonstrates all benchmark parameters matching CK Tile's `stream_config`:

```cpp
// Create stream_config with all parameters
ck_tile::stream_config cfg{
    nullptr,    // stream_id       - HIP stream (nullptr = default)
    true,       // time_kernel     - Enable timing
    1,          // log_level       - Verbosity (0=quiet, 1=normal, 2=verbose)
    5,          // cold_niters     - Warmup iterations (discarded)
    20,         // nrepeat         - Benchmark iterations (averaged)
    true,       // is_gpu_timer    - Use GPU events (true) or CPU chrono (false)
    false,      // flush_cache     - Flush L2 cache between iterations
    1           // rotating_count  - Rotating buffers for cache simulation
};

// Launch kernel with config
float avg_time_ms = SelectedConvKernelLauncher::launch(args, cfg);
```

### Command Line Options

```bash
./conv_04_benchmark --warmup 10 --repeat 100
./conv_04_benchmark --flush-cache --rotating-count 4
./conv_04_benchmark --cpu-timer
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--warmup N` | 5 | Warmup iterations (discarded from timing) |
| `--repeat N` | 20 | Benchmark iterations (averaged) |
| `--flush-cache` | off | Flush GPU L2 cache between iterations |
| `--rotating-count N` | 1 | Rotating buffers (for cache simulation) |
| `--cpu-timer` | off | Use CPU timer instead of GPU events |

### Use Cases

| Scenario | Recommended Settings |
|----------|---------------------|
| Quick test | `--warmup 1 --repeat 3` |
| Stable benchmark | `--warmup 10 --repeat 100` |
| Memory-bound analysis | `--flush-cache --rotating-count 4` |
| Debug timing | `--cpu-timer` |

## Example Details

### 01_conv_forward.cpp - Forward Pass
Shows complete forward convolution:
- Input/Weight/Output tensor creation
- GPU memory allocation and transfer
- Kernel execution and timing

### 02_conv_validation.cpp - Validation
Demonstrates correctness verification:
- CPU reference implementation
- GPU execution
- Numerical comparison with tolerance

### 09_bwd_data.cpp - Backward Data
Backward data gradient:
- dL/dInput computation
- Gradient propagation for backprop
- CPU reference validation with `--verify` flag

### 10_bwd_weight.cpp - Backward Weight
Backward weight gradient:
- dL/dWeight computation
- Filter gradient for training
- CPU reference validation with `--verify` flag
- Supports `NumGroupsToMerge` optimization

## Related Documentation

- [Python Conv Examples](../python/README.md)
- [C++ GEMM Examples](../../gemm/cpp/README.md)
- [Main Dispatcher README](../../../README.md)
