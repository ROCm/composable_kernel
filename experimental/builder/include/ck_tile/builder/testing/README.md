# CK-Builder Testing Utilities

This directory contains testing utilities designed to simplify the process of writing unit tests for GPU kernels built with `ck_tile::builder`. These utilities enable a clean, expressive **Given-When-Then** (Given-When-Then) testing pattern that separates test setup, execution, and validation.

## Overview

Testing GPU kernels typically involves significant boilerplate: allocating device memory, initializing test data, launching kernels, and validating results. The utilities in this directory abstract away these repetitive tasks, allowing you to focus on defining test cases and verifying correctness.

The core components are:

- **`Args`**: A struct template that holds runtime parameters for a specific test case.
- **`Input`** and **`Output`**: Helper classes that groups operation inputs and outputs.
- **`Validator`**: A utility that performs on-GPU validation and integrates with GoogleTest/GoogleMock.

Together, these components enable a structured approach to kernel testing that mirrors the Given-When-Then pattern commonly used in behavior-driven development.

## The Given-When-Then Testing Pattern

The Given-When-Then pattern organizes tests into three distinct phases:

1. **Given**: Set up the preconditions and test data
2. **When**: Execute the action being tested
3. **Then**: Verify the expected outcome

This structure makes tests easier to read, write, and maintain. Each phase has a clear purpose, and the testing utilities are designed to support this workflow.

### Given: Defining the Test Case

The "Given" phase establishes the context for your test. This includes both the compile-time characteristics of the kernel and the runtime parameters for the specific test case.

#### `ConvSignature`

The `ConvSignature` defines the **mathematical contract** that the kernel must satisfy. It specifies compile-time properties such as:

- Spatial dimensionality (1D, 2D, or 3D)
- Convolution direction (Forward, Backward Data, Backward Weight)
- Tensor memory layout (e.g., NHWC, NCHW)
- Data types (FP32, FP16, BF16, etc.)
- Fused element-wise operations (e.g., Bias, ReLU)

The signature is enforced at compile time using C++20 concepts, ensuring type safety and enabling compile-time optimizations.

```cpp
struct ConvSignature {
    int spatial_dim = 2;
    ck_tile::builder::ConvDirection direction =
        ck_tile::builder::ConvDirection::FORWARD;
    ck_tile::builder::GroupConvLayout2D layout =
        ck_tile::builder::GroupConvLayout2D::NHWGC_GKYXC_NHWGK;
    ck_tile::builder::DataType data_type =
        ck_tile::builder::DataType::FP16;
    ck_tile::builder::ElementwiseOperation elementwise_operation =
        ck_tile::builder::ElementwiseOperation::NONE;
};
static_assert(ck_tile::builder::ConvSignatureDescriptor<ConvSignature>);
constexpr auto SIGNATURE = ConvSignature{
    .spatial_dim = 2,
    .direction = ck_tile::builder::ConvDirection::FORWARD,
    .layout = ck_tile::builder::GroupConvLayout2D::NHWGC_GKYXC_NHWGK,
    .data_type = ck_tile::builder::DataType::FP16,
    .elementwise_operation = ck_tile::builder::ElementwiseOperation::NONE,
};
```

#### Run-time Arguments

The `Args` struct template provides the **runtime parameters** for your test case. It is parameterized by the `SIGNATURE` and contains fields for tensor dimensions, strides, dilations, and other dynamic properties. Note that the exact parameters required for each `Args` depends on the `SIGNATURE`: For example, a `SIGNATURE` that represents a forward convolution requires specifying the number of batches, groups, input- and output-channels, filter dimensions, filter strides, and so on. A `SIGNATURE` that represents a simple GEMM operation may instead require only the dimensions of the A-, B- and C-matrices.

```cpp
ck_tile::testing::Args<SIGNATURE> args = {
    .lengths = {
        .batch_size      = 128,
        .groups          = 1,
        .input_channels  = 64,
        .output_channels = 128,
        .image           = {.height = 56, .width = 56},
        .filter          = {.height = 3,  .width = 3},
    },
    .filter_strides  = {.height = 1, .width  = 1},
    .filter_dilation = {.height = 1, .width  = 1},
    .input_left_pad  = {.width  = 1, .height = 1},
    .input_right_pad = {.width  = 1, .height = 1},
};
```

#### Tensor Memory Management

Tensor memory is passed around using the `Inputs<SIGNATURE>` and `Outputs<SIGNATURE>` structures. These group all inputs and outputs for an operation. Note that these structures do not "own" the memory inside: They only logically group the inputs so that they can be passed around as a common type. The amount of inputs and outputs may differ depending on the `SIGNATURE`, and this avoids having to pass around additional values and accept additional parameters in those situations.

The `Inputs` and `Outputs` structures can be constructed manually from external data, however, the `UniqueInputs<SIGNATURE>` and `UniqueOutputs<SIGNATURE>` structures can be used to manage memory using RAII. The `alloc_inputs` and `alloc_outputs` functions are used to initialize these types: They take an `Args` structure and allocate the appropriate amounts of memory.

```cpp
auto inputs = allocate_inputs(args);
auto outputs = allocate_outputs(args);
```

Note that these functions merely _allocate_ memory: After allocation, the memory is yet uninitialized.

#### Tensor Initialization



### When: Executing the Kernel

The "When" phase is where the kernel to be tested is actually executed. This involves selecting an algorithm and using the `Builder` to generate the kernel.

#### `ConvAlgorithm`

The `ConvAlgorithm` defines the **implementation strategy** for the kernel. It specifies low-level details such as:

- Thread block dimensions and tile sizes
- GEMM implementation (XDL or WMMA)
- Data transfer vectorization
- Pipeline scheduling

```cpp
struct ConvAlgorithm {
    // Thread block configuration
    ThreadBlock thread_block = /* ... */;

    // Gridwise GEMM configuration
    GridwiseXdlGemm gridwise_gemm = /* ... */;

    // Block transfer configuration
    Transfer transfer = /* ... */;

    // Additional tuning parameters
    // ...
};
static_assert(ck_tile::builder::ConvAlgorithmDescriptor<ConvAlgorithm>);
constexpr auto ALGORITHM = ConvAlgorithm{};
```

#### Building and Running the Kernel

The `Builder` combines the `ConvSignature` (what to compute) with the `ConvAlgorithm` (how to compute it) to generate a runnable kernel operation.

```cpp
using ConvOp = ck_tile::builder::Builder<ConvSignature, ConvAlgorithm>::op;

// Launch the kernel with tensor pointers from TensorMemoryManager
ConvOp::Run(
    dev_mem.input_ptr(),
    dev_mem.weight_ptr(),
    dev_mem.output_ptr(),
    args
);
```

### Then: Verifying the Results

The "Then" phase validates that the kernel produced the expected output.

#### `Validator<ConvSignature>`

The `Validator` class encapsulates the validation logic. It performs on-GPU correctness checks by comparing the kernel's output against a reference implementation or expected properties.

```cpp
ck_tile::testing::Validator<ConvSignature> validator(args, dev_mem);
```

The `Validator` provides methods that return GoogleMock matchers, enabling clean integration with GoogleTest:

```cpp
EXPECT_THAT(validator.result(), validator.is_ok());
```

The `is_ok()` matcher checks that the output is numerically correct within acceptable tolerances. The `Validator` can also provide more detailed diagnostics, such as:

- Maximum absolute error
- Maximum relative error
- Number of mismatched elements
- Specific locations of errors

## Complete Example

Here's a complete test that demonstrates the Given-When-Then pattern:

```cpp
#include <gtest/gtest.h>
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/conv_builder.hpp"
#include "ck_tile/testing/tensor_memory_manager.hpp"
#include "ck_tile/testing/validator.hpp"

// Define the convolution signature
struct ConvSignature {
    static constexpr int spatial_dim = 2;
    static constexpr ck_tile::builder::ConvDirection direction =
        ck_tile::builder::ConvDirection::FORWARD;
    static constexpr ck_tile::builder::GroupConvLayout2D layout =
        ck_tile::builder::GroupConvLayout2D::NHWGC_GKYXC_NHWGK;
    static constexpr ck_tile::builder::DataType data_type =
        ck_tile::builder::DataType::FP16;
    static constexpr ck_tile::builder::ElementwiseOperation elementwise_operation =
        ck_tile::builder::ElementwiseOperation::NONE;
    static constexpr ck_tile::builder::GroupConvDeviceOp device_operation =
        ck_tile::builder::GroupConvDeviceOp::IMPLICIT_GEMM;
};
static_assert(ck_tile::builder::ConvSignatureDescriptor<ConvSignature>);

// Define the convolution algorithm
struct ConvAlgorithm {
    // Algorithm configuration details...
    // (Omitted for brevity)
};
static_assert(ck_tile::builder::ConvAlgorithmDescriptor<ConvAlgorithm>);

TEST(ConvolutionTest, Forward2D_FP16) {
    // ===== GIVEN: Set up the test case =====

    // Define runtime parameters
    ck_tile::testing::Args<ConvSignature> args = {
        .batch_size = 128,
        .num_groups = 1,
        .input_channels = 64,
        .output_channels = 128,
        .input_height = 56,
        .input_width = 56,
        .filter_height = 3,
        .filter_width = 3,
        .stride_height = 1,
        .stride_width = 1,
        .dilation_height = 1,
        .dilation_width = 1,
        .pad_height = 1,
        .pad_width = 1,
    };

    // Allocate and initialize GPU memory
    ck_tile::testing::TensorMemoryManager<ConvSignature> dev_mem(args);
    dev_mem.initialize();

    // ===== WHEN: Execute the kernel =====

    using ConvOp = ck_tile::builder::Builder<ConvSignature, ConvAlgorithm>::op;

    ConvOp::Run(
        dev_mem.input_ptr(),
        dev_mem.weight_ptr(),
        dev_mem.output_ptr(),
        args
    );

    // ===== THEN: Verify the results =====

    ck_tile::testing::Validator<ConvSignature> validator(args, dev_mem);
    EXPECT_THAT(validator.result(), validator.is_ok());
}
```

## Benefits of This Approach

1. **Clarity**: The Given-When-Then structure makes tests self-documenting. Each phase has a clear purpose.

2. **Reduced Boilerplate**: The utilities handle memory management, initialization, and validation, eliminating repetitive code.

3. **Type Safety**: The use of C++20 concepts ensures that signatures and algorithms are well-formed at compile time.

4. **Flexibility**: The `Args` struct can be easily extended to support different test scenarios, and the `TensorMemoryManager` supports various initialization patterns.

5. **Integration**: The `Validator` integrates seamlessly with GoogleTest/GoogleMock, providing familiar assertion syntax.

6. **Maintainability**: Changes to the testing infrastructure are localized to the utility classes, not scattered across individual tests.

## Future Enhancements

Potential improvements to the testing utilities include:

- Support for custom reference implementations in the `Validator`
- Performance benchmarking utilities
- Automatic test case generation from parameter ranges
- Enhanced error reporting with visual diffs
- Support for multi-GPU testing scenarios
