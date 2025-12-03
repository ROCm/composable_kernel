# GEMM C++ Examples

CK Tile Dispatcher C++ examples for GEMM (General Matrix Multiplication) operations.

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

# Build (kernels generated automatically by CMake)
make -j$(nproc)

# Run examples
cd examples
./gemm_01_basic
./gemm_03_benchmark
./gemm_04_validation
```

## Examples

| Example | Description | Complexity |
|---------|-------------|------------|
| [01_basic_gemm.cpp](01_basic_gemm.cpp) | Basic GEMM with declarative API | ★☆☆☆☆ |
| [02_multi_size.cpp](02_multi_size.cpp) | Multiple problem sizes | ★★☆☆☆ |
| [03_benchmark.cpp](03_benchmark.cpp) | Performance benchmarking | ★★☆☆☆ |
| [04_validation.cpp](04_validation.cpp) | CPU reference validation | ★★☆☆☆ |
| [05_heuristics.cpp](05_heuristics.cpp) | Heuristic kernel selection | ★★★☆☆ |
| [06_json_export.cpp](06_json_export.cpp) | Registry JSON export | ★★☆☆☆ |
| [07_preshuffle.cpp](07_preshuffle.cpp) | Layout optimization | ★★★☆☆ |
| [08_multi_d.cpp](08_multi_d.cpp) | Multi-D tensor ops | ★★★☆☆ |
| [09_multi_registry.cpp](09_multi_registry.cpp) | Multiple registries | ★★★★☆ |

## Example Details

### 01_basic_gemm.cpp - Basic GEMM
The simplest example demonstrating:
- Declarative kernel specification using `DECL_KERNEL_SET`
- Signature/Algorithm/Arch pattern
- Registry creation and kernel dispatch

```cpp
DECL_KERNEL_SET(basic_kernels,
    .add(
        Signature().dtype("fp16").layout("rcr"),
        Algorithm().tile(256, 256, 32).wave(2, 2, 1).warp(32, 32, 16)
                   .pipeline("compv4").scheduler("intrawave"),
        "gfx942"
    )
);
```

### 02_multi_size.cpp - Multiple Sizes
- Run the same kernel on different matrix sizes
- Track performance across problem sizes
- Dynamic workload handling

### 03_benchmark.cpp - Advanced Benchmarking
Demonstrates benchmark parameters (matching CK Tile `stream_config`):
- Warmup iterations (discarded)
- Benchmark iterations (averaged)
- Statistics: min/max/mean/median

```bash
./gemm_03_benchmark --warmup 10 --iterations 100
```

### 04_validation.cpp - CPU Validation
- CPU reference implementation
- Numerical comparison with tolerance
- Correctness verification workflow

### 05_heuristics.cpp - Heuristic Selection
- Problem size analysis
- Automatic kernel selection
- Compute-bound vs memory-bound heuristics

### 06_json_export.cpp - JSON Export
- Exporting registry to JSON format
- Kernel metadata serialization
- External tool integration

### 07_preshuffle.cpp - Preshuffle Optimization
- Preshuffled matrix layouts
- Memory access optimization
- Performance tuning techniques

### 08_multi_d.cpp - Multi-D Tensors
- Tensor operations beyond 2D matrices
- Bias and element-wise operations
- Fused kernel patterns

### 09_multi_registry.cpp - Multiple Registries
- Separate registries for different workloads
- Compute-optimized vs latency-optimized kernels
- Registry selection strategies

## Benchmark Parameters (stream_config)

CK Tile uses `stream_config` for benchmark control:

```cpp
ck_tile::stream_config cfg{
    nullptr,    // stream_id       - HIP stream (nullptr = default)
    true,       // time_kernel     - Enable timing
    1,          // log_level       - Verbosity (0=quiet, 1=normal)
    5,          // cold_niters     - Warmup iterations
    20,         // nrepeat         - Benchmark iterations
    true,       // is_gpu_timer    - Use GPU events vs CPU chrono
    false,      // flush_cache     - Flush L2 cache between iterations
    1           // rotating_count  - Rotating buffers for cache simulation
};
```

| Parameter | CLI Option | Default | Description |
|-----------|------------|---------|-------------|
| `cold_niters_` | `--warmup` | 5 | Warmup iterations |
| `nrepeat_` | `--iterations` | 100 | Benchmark iterations |
| `flush_cache_` | - | false | Flush L2 cache |
| `rotating_count_` | - | 1 | Rotating buffers |
| `is_gpu_timer_` | - | true | GPU timer vs CPU |

## Declarative Kernel Pattern

All examples use the declarative kernel pattern:

```cpp
DECL_KERNEL_SET(my_kernels,
    .add(
        Signature()               // WHAT: operation signature
            .dtype("fp16")        // Data type
            .layout("rcr"),       // Matrix layouts (A=row, B=col, C=row)
        Algorithm()               // HOW: implementation details  
            .tile(256, 256, 32)   // Tile sizes (M, N, K)
            .wave(2, 2, 1)        // Wave configuration
            .warp(32, 32, 16)     // Warp tile sizes
            .pipeline("compv4")   // Pipeline type
            .scheduler("intrawave"), // Scheduler type
        "gfx942"                  // WHERE: target architecture
    )
);
```

## Related Documentation

- [Python GEMM Examples](../python/README.md)
- [Convolution Examples](../../conv/cpp/README.md)
- [Main Dispatcher README](../../../README.md)
