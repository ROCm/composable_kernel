# CK Tile Dispatcher - C++ Headers

C++ API for the CK Tile dispatcher.

> **See also:** [Main Dispatcher README](../../../../README.md) for installation and core concepts.

## File Organization

```
dispatcher/
├── dispatcher.hpp          # Main dispatcher (kernel selection)
├── registry.hpp            # Kernel registry (storage & lookup)
├── problem.hpp             # Problem specification
├── kernel_key.hpp          # Kernel configuration key
├── kernel_instance.hpp     # Kernel instance interface
├── utils.hpp               # Utilities (timers, GPU buffers)
│
└── backends/               # Backend implementations
    ├── generated_tile_backend.hpp  # CK Tile kernels (production)
    └── tile_backend.hpp            # Tile backend base
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

## Best Practices

1. Use `Release` build for performance
2. Register kernels at startup
3. Use `Priority::High` for hand-tuned kernels
4. Reuse dispatcher instances
5. Clear registry between test runs

---

> **More info:** See [../../../../README.md](../../../../README.md) for full documentation.
