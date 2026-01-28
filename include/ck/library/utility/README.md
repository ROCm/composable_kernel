# CK Library Utility

This directory contains utility headers for testing, benchmarking, and validating Composable Kernel (CK) operations. The utilities support both modern GPU-first validation for high-performance testing and legacy CPU-based approaches for backward compatibility.

## Quick Start

1. **Use GPU validation** for all new tests (10-100x faster than CPU validation)
2. **Let the system compute tolerances** automatically based on data types
3. **Only transfer error statistics**, not full tensors

## File-to-Purpose Quick Reference

| Need to...                          | Use this file                     | Key function/class        |
|-------------------------------------|-----------------------------------|---------------------------|
| Validate on GPU (recommended)       | `gpu_verification.hpp`            | `gpu_verify()`            |
| Validate on CPU (legacy/debugging)  | `check_err.hpp`                   | `check_err()`             |
| Compute tolerances automatically    | `check_err.hpp`                   | `get_relative_threshold<>()` |
| Allocate GPU memory                 | `device_memory.hpp`               | `DeviceMem`               |
| Create CPU tensors                  | `host_tensor.hpp`                 | `Tensor<T>`               |
| Generate test data on GPU           | `device_tensor_generator.hpp`     | `FillUniformRandFp()`     |
| Generate test data on CPU (legacy)  | `host_tensor_generator.hpp`       | `GeneratorTensor_*`       |
| Set up convolution parameters       | `convolution_parameter.hpp`       | `ConvParam`               |
| Create tensor descriptors           | `host_tensor.hpp`                 | `HostTensorDescriptor`    |

## Core Validation Tools

### GPU Validation (Recommended)

**`gpu_verification.hpp`** - Complete on-device verification

- `gpu_verify()`: Compares device tensors entirely on GPU
  - Automatic tolerance computation based on data types
  - Only transfers error statistics (~12 bytes), not tensors
  - Detailed error reporting (count, max error, percentage)
  - Supports all CK data types (fp32, fp16, bf16, fp8, int8, etc.)
- `gpu_reduce_max()`: Computes max(abs(tensor)) on GPU for tolerance scaling
- Grid-stride kernels with LDS reduction for optimal performance

**Performance**: 10-100x faster than CPU validation for large tensors.

**Example usage:**

```cpp
// Explicit tolerance
bool pass = gpu_verify<float>(output_dev, reference_dev, 1e-5f, 1e-6f, size);

// Automatic tolerance for mixed precision
bool pass = gpu_verify<float, half_t, float>(output_dev, reference_dev, K_dim, size);
```

**See:** `test/gpu_verification/test_gpu_verification.cpp`

### Tolerance Computation

**`check_err.hpp`** - Automatic tolerance calculation

- `get_relative_threshold<ComputeType, OutType, AccType>()`: Computes relative tolerance from mantissa bits
- `get_absolute_threshold<ComputeType, OutType, AccType>()`: Computes absolute tolerance scaled by magnitude
- Type-specific overloads for all CK data types
- Accumulation-aware error bounds

**Theory**: Based on IEEE 754 floating-point arithmetic and error propagation analysis.

### Legacy CPU Validation

**`check_err.hpp`** - CPU-based error checking (legacy)

- Overloaded `check_err()` functions for different data types
- Type-aware default tolerances
- Detailed error reporting (first 5 mismatches, statistics)

**Note**: Requires full tensor transfer to CPU - slow for large tensors. Use `gpu_verification.hpp` for new tests.

**See:** `test/convnd_fwd/convnd_fwd_naive.cpp` for legacy CPU validation patterns

## Numerical Validation Strategy

**TL;DR:** CK computes tolerances from IEEE 754 precision limits, not arbitrary values. FP32 gets ~1e-5 relative tolerance, FP16 gets ~1e-3, etc. The system accounts for accumulation effects in matrix operations.

CK implements a **theoretically-grounded approach to numerical validation** that goes beyond simple fixed tolerances. The validation system is designed around three core principles:

### 1. Type-Aware Tolerance Computation

Rather than using arbitrary threshold values, CK computes tolerances based on the datatypes:

- **Relative tolerance**: Derived from mantissa bits as `2^(-mantissa_bits) * 0.5`
- **Absolute tolerance**: Scaled by value magnitude as `2^(exponent - mantissa_bits) * 0.5`
- **Multi-type analysis**: Considers compute type, output type, and accumulator type separately
- **Conservative bounds**: Takes maximum error across all data paths

### 2. Algorithm-Aware Validation

Different algorithms have different error characteristics:

- **Accumulation effects**: Matrix operations (GEMM, convolution) accumulate errors proportional to the number of operations
- **Precision cascades**: Mixed-precision operations require careful tolerance selection based on the weakest link
- **Operation-specific bounds**: Tolerances scale with problem size (e.g., K dimension in GEMM)

The validation system accepts `number_of_accumulations` to adjust tolerances for algorithmic context.

### 3. Data Type Characteristics

Each data type has inherent precision limits that inform validation:

| Data Type | Mantissa Bits | Typical rtol | Typical atol |
|-----------|---------------|--------------|--------------|
| FP32      | 23            | 1e-5         | 3e-6         |
| TF32      | 10            | 5e-4         | 5e-4         |
| FP16      | 10            | 1e-3         | 1e-3         |
| BF16      | 7             | 1e-1         | 1e-3         |
| FP8       | 3-4           | 1e-3         | 1e-3         |
| BF8       | 2-3           | 1e-3         | 1e-3         |
| FP4       | 2             | 0.5          | 0.5          |
| INT8/INT32| N/A           | 0            | 0            |

## GPU-First Validation Philosophy

Modern CK testing emphasizes **pure GPU validation** to eliminate performance bottlenecks:

### Traditional CPU-Based Approach (Legacy)

```text
GPU Kernel → Transfer to CPU → CPU Verification
            ↑ BOTTLENECK: PCIe transfer of entire tensor
```

- **Problem**: Transferring multi-GB tensors over PCIe is 10-100x slower than computation
- **Impact**: Test suites become I/O bound rather than compute bound
- **Limitation**: Cannot efficiently test large-scale problems

### Modern GPU-First Approach (Recommended)

```text
GPU Kernel → GPU Reference → GPU Verification → Transfer scalars only
                                               ↑ Only ~12 bytes transferred
```

- **Advantage**: All data stays on GPU, only error statistics transfer to CPU
- **Performance**: 10-100x faster for large tensors
- **Scalability**: Enables testing of multi-GB tensors efficiently
- **Completeness**: Detailed error reporting (count, max error, percentage) without full transfer

### When to Use Each Approach

**Use GPU-First Validation When:**

- Testing production kernels (performance matters)
- Working with large tensors (>1MB)
- Running extensive test suites
- Validating at scale

**Use CPU-Based Validation When:**

- Debugging specific values (need to inspect individual elements)
- Working with tiny tensors (<1KB)
- Maintaining backward compatibility
- Implementing CPU reference algorithms

## Testing Workflow Comparison

### Modern GPU-First Workflow (Recommended)

```cpp
// 1. Allocate device memory only
DeviceMem input_dev(size), output_dev(size), reference_dev(size);

// 2. Initialize on GPU (no CPU involvement)
input_dev.FillUniformRandFp<float>(-1.0f, 1.0f);

// 3. Run kernel under test
run_kernel(input_dev, output_dev, params);

// 4. Run reference on GPU
run_reference_kernel(input_dev, reference_dev, params);

// 5. Verify on GPU (only transfers ~12 bytes of error stats)
bool pass = gpu_verify<float>(output_dev, reference_dev, rtol, atol, size);
if (!pass) {
    std::cout << "Validation failed!" << std::endl;
    return false;
}
```

**Key advantage**: Zero tensor transfers - all data stays on GPU.

### Legacy CPU-Based Workflow

```cpp
// 1. Create host tensors (allocates CPU memory)
Tensor<float> input_host(dims), output_host(dims), reference_host(dims);

// 2. Generate on CPU
input_host.GenerateTensorValue(GeneratorTensor_3<float>{-1.0f, 1.0f});

// 3. Allocate device memory
DeviceMem input_dev(size), output_dev(size);

// 4. Transfer to device (slow for large tensors)
input_dev.ToDevice(input_host.data());

// 5. Run kernel
run_kernel(input_dev, output_dev, params);

// 6. Transfer back to CPU (slow for large tensors)
output_dev.FromDevice(output_host.data());

// 7. Compute reference on CPU
compute_reference(input_host, reference_host, params);

// 8. Verify on CPU
bool pass = check_err(output_host, reference_host, "Test failed");
```

**Bottleneck**: Steps 4 and 6 transfer entire tensors over PCIe.

## Supporting Utilities

### Tensor Management

- **`host_tensor.hpp`**: CPU-side tensor container with multi-dimensional support
  - `HostTensorDescriptor`: Dimension, stride, and layout management
  - `Tensor<T>`: Host tensor with generation and conversion utilities
- **`device_memory.hpp`**: GPU memory management with RAII semantics
  - `DeviceMem`: Device allocation, transfer, and initialization
  - Device-side random value generation
  - `SetZero()`: Zero-initialize device memory (required for backward passes)

### Data Generation

- **`device_tensor_generator.hpp`**: GPU-side tensor initialization (recommended)
  - `FillUniformRandFp<T>()`: Fill with uniform random floating-point values
  - `FillUniformRandInt<T>()`: Fill with uniform random integer values
- **`host_tensor_generator.hpp`**: CPU-side functor-based generators (legacy)
  - Various patterns: zero, constant, random, sequential, diagonal, checkerboard
- **`fill.hpp`**: STL-style fill functors for containers

### Convolution Utilities

- **`convolution_parameter.hpp`**: Convolution parameter management
  - `ConvParam`: Encapsulates dimensions, strides, padding, dilations
  - Output dimension calculation and FLOP estimation
- **`convolution_host_tensor_descriptor_helper.hpp`**: Tensor descriptor creation helpers
- **`conv_common.hpp`**: Common convolution utilities

**See:** `test/convnd_fwd/convnd_fwd_naive.cpp` for convolution parameter usage

### Workspace Management

Some operations require temporary GPU memory for intermediate computations:

```cpp
// Check if workspace is needed
const std::size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());

// Allocate and set workspace if needed
if (workspace_sz > 0) {
    DeviceMem workspace_dev(workspace_sz);
    op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());
}
```

### Algorithmic Utilities

- **`algorithm.hpp`**: Generic algorithms
- **`ranges.hpp`**: Range-based utilities and concepts
- **`iterator.hpp`**: Custom iterator implementations
- **`numeric.hpp`**: Numeric operations

### Miscellaneous

- **`host_common_util.hpp`**: Common host-side utilities
- **`host_gemm.hpp`**: CPU reference GEMM implementation
- **`literals.hpp`**: User-defined literals
- **`thread.hpp`**: Threading utilities

## Best Practices

### Choosing Tolerances

1. **Prefer automatic computation**: Use `gpu_verify()` with automatic tolerance calculation
2. **Consider accumulation**: Pass `number_of_accumulations` for matrix operations
3. **Respect data type limits**: Don't expect FP16 to match FP32 precision
4. **Account for algorithm**: Different operations have different error characteristics

### Performance Optimization

1. **Use GPU-first validation** for all new tests
2. **Avoid CPU transfers** unless debugging specific values
3. **Generate data on GPU** when possible
4. **Batch verification** to amortize kernel launch overhead
