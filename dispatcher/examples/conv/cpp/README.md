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

# Build all conv examples (kernels are generated automatically by CMake)
make -j$(nproc)

# Run examples
cd examples
./conv_01_basic
./conv_03_validation
./conv_10_bwd_data --verify
./conv_11_bwd_weight --verify
```

## Examples

| Example | Description | Complexity |
|---------|-------------|------------|
| [01_basic_conv.cpp](01_basic_conv.cpp) | Basic 2D conv with declarative API | ★☆☆☆☆ |
| [02_conv_forward.cpp](02_conv_forward.cpp) | 2D forward with tensor setup | ★★☆☆☆ |
| [03_conv_validation.cpp](03_conv_validation.cpp) | CPU reference validation | ★★☆☆☆ |
| [04_multi_size.cpp](04_multi_size.cpp) | Multiple problem sizes | ★★☆☆☆ |
| [05_benchmark.cpp](05_benchmark.cpp) | ResNet/VGG layer benchmarks | ★★☆☆☆ |
| [06_heuristics.cpp](06_heuristics.cpp) | Heuristic kernel selection | ★★★☆☆ |
| [07_json_export.cpp](07_json_export.cpp) | Export registry to JSON | ★★☆☆☆ |
| [08_multi_registry.cpp](08_multi_registry.cpp) | Multiple registries | ★★★☆☆ |
| [09_conv3d_forward.cpp](09_conv3d_forward.cpp) | 3D volumetric convolution | ★★★☆☆ |
| [10_bwd_data.cpp](10_bwd_data.cpp) | Backward data gradient | ★★★☆☆ |
| [11_bwd_weight.cpp](11_bwd_weight.cpp) | Backward weight gradient | ★★★☆☆ |

## Example Details

### 01_basic_conv.cpp - Basic Convolution
The simplest example demonstrating:
- Declarative kernel specification using `DECL_CONV_KERNEL_SET`
- ConvSignature/ConvAlgorithm/Arch pattern
- Registry creation and convolution dispatch

```cpp
DECL_CONV_KERNEL_SET(basic_conv_kernels,
    .add(
        ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
        ConvAlgo().tile(1, 128, 128).wave(2, 2, 1).warp(32, 32, 16)
                  .pipeline("compv3").scheduler("intrawave"),
        "gfx942"
    )
);
```

### 02_conv_forward.cpp - Forward Pass
Shows complete forward convolution:
- Input/Weight/Output tensor creation
- GPU memory allocation and transfer
- Kernel execution and timing

### 03_conv_validation.cpp - Validation
Demonstrates correctness verification:
- CPU reference implementation
- GPU execution
- Numerical comparison with tolerance

### 04_multi_size.cpp - Multiple Sizes
Shows running on various input sizes:
- Small (14x14), Medium (28x28), Large (56x56)
- Performance comparison across sizes

### 05_benchmark.cpp - Benchmarking
Professional benchmarking with:
- ResNet layer configurations
- VGG-16 layer configurations
- TFLOPS measurement and reporting

### 06_heuristics.cpp - Heuristic Selection
Intelligent kernel selection:
- Problem analysis (pointwise, depthwise, etc.)
- Workload classification
- Automatic kernel matching

### 07_json_export.cpp - JSON Export
Registry serialization:
- Export kernel metadata
- Configuration documentation
- Tool integration

### 08_multi_registry.cpp - Multiple Registries
Advanced registry patterns:
- Compute-optimized registry
- Memory-optimized registry
- Workload-based selection

### 09_conv3d_forward.cpp - 3D Convolution
Volumetric convolution for:
- Video processing
- Medical imaging (CT, MRI)
- Point cloud processing

### 10_bwd_data.cpp - Backward Data
Backward data gradient:
- dL/dInput computation
- Gradient propagation for backprop
- CPU reference validation with `--verify` flag

### 11_bwd_weight.cpp - Backward Weight
Backward weight gradient:
- dL/dWeight computation
- Filter gradient for training
- CPU reference validation with `--verify` flag

## Declarative Kernel Pattern

Convolution examples use the declarative pattern:

```cpp
DECL_CONV_KERNEL_SET(my_kernels,
    .add(
        ConvSig()                    // WHAT: convolution signature
            .dtype("fp16")           // Data type
            .layout("nhwgc")         // Tensor layout
            .conv_type("forward")    // Operation direction
            .dims(2),                // 2D or 3D
        ConvAlgo()                   // HOW: algorithm details
            .tile(1, 128, 128)       // Tile sizes (G, M, N)
            .wave(2, 2, 1)           // Wave configuration
            .warp(32, 32, 16)        // Warp tile sizes
            .pipeline("compv3")      // Pipeline type
            .scheduler("intrawave"), // Scheduler type
        "gfx942"                     // WHERE: target architecture
    )
);
```

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

## Related Documentation

- [Python Conv Examples](../python/README.md)
- [C++ GEMM Examples](../../gemm/cpp/README.md)
- [Main Dispatcher README](../../../README.md)
