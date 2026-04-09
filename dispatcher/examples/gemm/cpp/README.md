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
./gemm_03_benchmark_validation
./gemm_04_heuristics
```

## Examples

| Example | Description |
|---------|-------------|
| [01_basic_gemm.cpp](01_basic_gemm.cpp) | Basic GEMM with declarative API, autofill, autocorrect |
| [02_multi_size.cpp](02_multi_size.cpp) | Wildcard expansion for multiple configurations |
| [03_benchmark_validation.cpp](03_benchmark_validation.cpp) | Performance benchmarking with CPU reference validation |
| [04_heuristics.cpp](04_heuristics.cpp) | Heuristic-based kernel selection |
| [05_json_export.cpp](05_json_export.cpp) | Registry JSON export for external tools |
| [06_multi_registry.cpp](06_multi_registry.cpp) | Multiple registries with named kernel sets |

## Example Details

### 01_basic_gemm.cpp - Basic GEMM
Demonstrates the declarative kernel API with three patterns:

1. **Autofill Pattern** - Minimal specification, defaults filled automatically
2. **Autocorrect Pattern** - Invalid parameters corrected at build time
3. **Full Specification Pattern** - Complete kernel configuration

```cpp
DECL_KERNEL_SET(basic_kernels,
    // Pattern 1: Autofill - minimal specification
    .add(
        Signature().dtype("fp16").layout("rcr"),
        Algorithm(),  // Defaults filled by autofill
        "gfx942"
    )
    // Pattern 2: Full specification
    .add(
        Signature().dtype("fp16").layout("rcr"),
        Algorithm().tile(256, 256, 32).wave(2, 2, 1).warp(32, 32, 16)
                   .pipeline("compv4").scheduler("intrawave"),
        "gfx942"
    )
);
```

**Features:**
- Uses generic `REGISTER_GENERATED_KERNELS` macro
- `print_registered_kernels()` utility for debugging
- Demonstrates autofill messages during build

### 02_multi_size.cpp - Wildcard Expansion
Demonstrates automatic generation of multiple kernel configurations:

```cpp
DECL_KERNEL_SET(multi_kernels,
    .add(
        Signature().dtype("fp16").layout("rcr"),
        Algorithm().tile(*, *, 32)     // Wildcard tile M and N
                   .wave(2, 2, 1)
                   .warp(32, 32, 16)
                   .pipeline("compv4")
                   .scheduler("intrawave"),
        "gfx942"
    )
);
```

**Wildcard Values:**
- `*`, `-1`, or `ANY_INT` expand to all valid configurations
- Architecture filter prunes invalid combinations automatically
- Example generates 5 valid kernels after arch filtering (from 7 expansions)

### 03_benchmark_validation.cpp - Benchmark + Validation
Consolidated example combining performance benchmarking with correctness validation:

```bash
# Benchmark only
./gemm_03_benchmark_validation --warmup 10 --iterations 100

# With CPU validation
./gemm_03_benchmark_validation --verify 1 --rtol 1e-3 --atol 1e-3

# With GPU reference validation (faster for large matrices)
./gemm_03_benchmark_validation --verify 2
```

**Features:**
- Warmup iterations (discarded from timing)
- Benchmark iterations with statistics (min/max/mean/median)
- CPU reference validation using `ck_tile::reference_gemm`
- GPU reference validation using `ck_tile::reference_gemm_gpu`
- Configurable tolerances

### 04_heuristics.cpp - Heuristic Selection
Demonstrates custom kernel selection based on problem characteristics:

```cpp
// Problem size analysis
auto heuristic = [](const Problem& p) -> std::optional<KernelKey> {
    if (p.M() * p.N() < 256 * 256) {
        return small_kernel_key;   // Memory-bound heuristic
    } else {
        return large_kernel_key;   // Compute-bound heuristic
    }
};

dispatcher.set_heuristic(heuristic);
```

**Features:**
- Problem size analysis (small vs large matrices)
- Compute-bound vs memory-bound selection
- Custom heuristic function registration

### 05_json_export.cpp - JSON Export
Exports registry information to JSON for external tool integration:

```cpp
auto json = registry.to_json();
std::ofstream file("kernels.json");
file << json;
```

**Use Cases:**
- Kernel metadata serialization
- External analysis tools
- Configuration management

### 06_multi_registry.cpp - Multiple Registries
Demonstrates using multiple registries with named kernel sets:

```cpp
// Define separate kernel sets
DECL_KERNEL_SET(compute_optimized, ...);
DECL_KERNEL_SET(latency_optimized, ...);

// Register to specific registries
Registry compute_registry, latency_registry;
REGISTER_KERNEL_SET(compute_optimized, compute_registry);
REGISTER_KERNEL_SET(latency_optimized, latency_registry);

// Use appropriate registry based on workload
Dispatcher compute_dispatcher(compute_registry);
Dispatcher latency_dispatcher(latency_registry);
```

**Features:**
- Named kernel set registration with `REGISTER_KERNEL_SET` macro
- Separate registries for different optimization goals
- Dynamic kernel set selection by name

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

All examples use the declarative `DECL_KERNEL_SET` macro:

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

**Key Macros:**
- `DECL_KERNEL_SET(name, ...)` - Declare a kernel set
- `REGISTER_GENERATED_KERNELS` - Register all kernels from this example
- `REGISTER_KERNEL_SET(name, registry)` - Register specific kernel set to a registry

## Related Documentation

- [Python GEMM Examples](../python/README.md)
- [C++ Headers (GEMM + Grouped Conv)](../../../include/ck_tile/dispatcher/README.md)
- [Main Dispatcher README](../../../README.md)
