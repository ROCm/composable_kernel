# CK Tile Dispatcher - C++ Headers

C++ API for the CK Tile dispatcher (GEMM and Grouped Convolution).

> **See also:** [Main Dispatcher README](../../../../README.md) for installation and core concepts.

## File Organization

```
dispatcher/
|---- dispatcher.hpp          # Main include (includes all below)
|
|---- # GEMM Headers
|---- registry.hpp            # Kernel registry (storage & lookup)
|---- problem.hpp             # GEMM problem specification
|---- kernel_key.hpp          # Kernel configuration key
|---- kernel_instance.hpp     # Kernel instance interface
|---- utils.hpp               # Utilities (timers, GPU buffers)
|
|---- # Grouped Convolution Headers
|---- grouped_conv_config.hpp       # GroupedConvDirection, GroupedConvConfig
|---- grouped_conv_problem.hpp      # GroupedConvProblem + ProblemBuilder
|---- grouped_conv_kernel_decl.hpp  # GroupedConvKernelDecl, DECL_GROUPED_CONV_KERNEL_SET
|---- grouped_conv_registry.hpp     # Thread-safe registry with JSON export & filtering
|---- grouped_conv_utils.hpp        # Config creators, validation, benchmark utilities
|
+---- backends/               # Backend implementations
    |---- generated_tile_backend.hpp  # CK Tile kernels (production)
    +---- tile_backend.hpp            # Tile backend base
```

## Quick Start

```cpp
#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;

int main() {
    // 1. Build kernel key
    KernelKeyBuilder builder = KernelKeyBuilder::fp16_rcr();
    builder.tile_m = 128;
    builder.tile_n = 128;
    builder.tile_k = 32;
    KernelKey key = builder.build();

    // 2. Register kernel
    auto kernel = create_generated_tile_kernel<...>(key, "my_kernel");
    Registry::instance().register_kernel(kernel, Priority::High);

    // 3. Run GEMM
    Dispatcher dispatcher;
    Problem problem(1024, 1024, 1024);
    float time_ms = dispatcher.run(a_ptr, b_ptr, c_ptr, problem, nullptr);
}
```

## Core Classes

### KernelKey (`kernel_key.hpp`)

Uniquely identifies a kernel configuration:

```cpp
KernelKeyBuilder builder;
builder.dtype_a = DataType::FP16;
builder.layout_a = LayoutTag::Row;
builder.tile_m = 256;
builder.pipeline = Pipeline::CompV4;
KernelKey key = builder.build();
```

### Registry (`registry.hpp`)

Thread-safe kernel storage:

```cpp
auto& registry = Registry::instance();
registry.register_kernel(kernel, Priority::High);
registry.get_kernel_count();
registry.export_json();
```

### Dispatcher (`dispatcher.hpp`)

Kernel selection and execution:

```cpp
Dispatcher dispatcher;

// Strategies
dispatcher.set_strategy(SelectionStrategy::FirstFit);
dispatcher.set_strategy(SelectionStrategy::Heuristic);

// Run
float time = dispatcher.run(a, b, c, problem, stream);
```

### Problem (`problem.hpp`)

GEMM problem specification:

```cpp
Problem problem(M, N, K);
problem.batch_size = 4;
problem.alpha = 1.0f;
problem.beta = 0.0f;

// Auto-inference
auto p = Problem::from_ab(a_rows, a_cols, b_rows, b_cols, trans_a, trans_b);
```

## Utilities (`utils.hpp`)

### GPU Memory

```cpp
GpuBuffer<half_t> buffer(size);
buffer.copy_from_host(host_ptr);
buffer.copy_to_host(host_ptr);
buffer.zero();
```

### Timing

```cpp
GpuTimer timer;
timer.start();
// kernel...
timer.stop();
float ms = timer.elapsed_ms();
```

### Quick Helpers

```cpp
// Create FP16 RCR key
auto key = create_fp16_rcr_key(tile_m, tile_n, tile_k, ...);

// Performance
double tflops = calculate_tflops(M, N, K, time_ms);

// Validation
auto result = validate_result(gpu_ptr, cpu_ptr, size);
```

## Backend

### Generated Tile Backend

```cpp
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"

auto kernel = create_generated_tile_kernel<
    SelectedKernel, ADataType, BDataType, CDataType, AccDataType
>(key, name);
```

## Grouped Convolution API

### GroupedConvProblem (`grouped_conv_problem.hpp`)

Problem specification with builder pattern:

```cpp
#include "ck_tile/dispatcher/grouped_conv_problem.hpp"

using namespace ck_tile::dispatcher;

auto problem = GroupedConvProblemBuilder()
    .n(2).g(1).c(128).k(256)
    .input_spatial({28, 28})
    .filter_spatial({3, 3})
    .strides({1, 1})
    .dilations({1, 1})
    .left_pads({1, 1})
    .right_pads({1, 1})
    .build();

bool ok = problem.is_valid();
```

### GroupedConvRegistry (`grouped_conv_registry.hpp`)

Thread-safe registry with JSON export and filtering:

```cpp
#include "ck_tile/dispatcher/grouped_conv_registry.hpp"

auto& registry = GroupedConvRegistry::instance();

// Thread-safe registration
registry.register_kernel(kernel);

// JSON export
std::string json = registry.export_json();
registry.export_json_to_file("kernels.json");

// Filtering
auto gfx942_kernels = registry.filter_by_arch("gfx942");
auto matched = registry.filter([](const auto& k) { return k.is_fwd(); });
```

### DECL_GROUPED_CONV_KERNEL_SET (`grouped_conv_kernel_decl.hpp`)

Declarative kernel definition:

```cpp
DECL_GROUPED_CONV_KERNEL_SET(my_conv_kernels,
    .add(
        GroupedConvSignature().dtype("fp16").layout("nhwgc"),
        GroupedConvAlgorithm().tile(128, 128, 32).wave(2, 2, 1)
                             .warp(32, 32, 16).pipeline("compv4"),
        "gfx942"
    )
);

// Register all matching current arch
DECL_GROUPED_CONV_KERNEL_ALL(all_conv_kernels, "gfx942");
```

## Best Practices

1. Use `Release` build for performance
2. Register kernels at startup
3. Use `Priority::High` for hand-tuned kernels
4. Reuse dispatcher instances
5. Clear registry between test runs
6. Use `GroupedConvProblemBuilder` for validated problem construction
7. Leverage `export_json()` for kernel inventory and debugging

---

> **More info:** See [../../../../README.md](../../../../README.md) for full documentation.
