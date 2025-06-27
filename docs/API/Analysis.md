# Composable Kernel API Analysis

## Architecture Overview

The Composable Kernel library implements a hierarchical GPU kernel design with three main abstraction levels:

### 1. Device Level (`include/ck/tensor_operation/gpu/device/`)
High-level operation interfaces that provide the main entry points for tensor operations.

### 2. Grid Level (`include/ck/tensor_operation/gpu/grid/`)
Kernel implementation and execution logic that manages thread block coordination.

### 3. Block/Thread/Warp Level (`include/ck/tensor_operation/gpu/{block,thread,warp}/`)
Low-level compute primitives that implement the actual computation patterns.

## Device Interface Architecture

### Base Classes

```cpp
namespace ck::tensor_operation::device {
    struct BaseOperator {
        virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(...) = 0;
        virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
        virtual bool IsSupportedArgument(const BaseArgument* arg) = 0;
        virtual std::string GetTypeString() const = 0;
    };
}
```

All device operations inherit from `BaseOperator` and implement:
- **MakeArgumentPointer()**: Creates operation-specific argument objects containing tensor pointers, dimensions, and operation parameters
- **MakeInvokerPointer()**: Creates execution invokers that launch the actual GPU kernels
- **IsSupportedArgument()**: Validates whether the operation supports the given problem configuration
- **GetTypeString()**: Returns a string identifier for the operation type

### Template Parameter System

#### Core Template Structure
```cpp
template <typename ALayout,                    // Memory layout (Row/Column major)
          typename BLayout, 
          typename CLayout,
          typename ADataType,                  // Data types (half_t, float, int8_t, etc.)
          typename BDataType,
          typename CDataType,
          typename AccDataType,                // Accumulation type (usually float)
          typename AElementwiseOperation,      // Element-wise operations (PassThrough, etc.)
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec>         // Specialization flags
```

#### Tuning Parameters
The device implementations expose extensive tuning parameters:
```cpp
// Block and thread configuration
static constexpr auto BlockSize = 256;          // Thread block size
static constexpr auto MPerBlock = 256;          // M dimension per block
static constexpr auto NPerBlock = 128;          // N dimension per block
static constexpr auto K0PerBlock = 4;           // K0 dimension per block
static constexpr auto K1 = 8;                   // K1 vectorization

// XDL (Matrix Core) configuration  
static constexpr auto MPerXDL = 32;             // M per matrix instruction
static constexpr auto NPerXDL = 32;             // N per matrix instruction
static constexpr auto MXdlPerWave = 4;          // Matrix instructions per wave in M
static constexpr auto NXdlPerWave = 2;          // Matrix instructions per wave in N

// Memory transfer configuration
using ABlockTransferThreadClusterLengths = S<4, 64, 1>;  // Thread cluster for A block transfer
using ABlockTransferSrcAccessOrder = S<1, 0, 2>;         // Memory access order
static constexpr auto ABlockTransferSrcVectorDim = 2;     // Vectorization dimension
static constexpr auto ABlockTransferDstScalarPerVector = 8; // Vector width
```

## Operation Categories

### GEMM Operations

#### Basic GEMM
- **DeviceGemm**: Standard matrix multiplication C = A × B
- **DeviceGemmMultipleD**: GEMM with auxiliary tensors for element-wise operations
- **DeviceGemmSplitK**: K-dimension splitting for increased parallelism

#### Specialized GEMM
- **DeviceBatchedGemm**: Batched matrix operations for multiple independent problems
- **DeviceGroupedGemm**: Grouped operations for different problem sizes
- **DeviceGemmReduce**: GEMM combined with reduction operations

#### Advanced GEMM Fusions
- **DeviceGemmBiasEPermute**: GEMM + Bias + Element-wise + Permutation
- **DeviceGemmLayernorm**: GEMM + Layer Normalization
- **DeviceGemmSoftmax**: GEMM + Softmax (for attention mechanisms)

### Convolution Operations

#### Forward Convolution
- **DeviceConvNdFwd**: N-dimensional forward convolution using implicit GEMM
- **DeviceGroupedConvFwd**: Grouped convolutions for efficient channel-wise operations
- **DeviceConvFwdBiasActivation**: Convolution + Bias + Activation fusion

#### Backward Convolution
- **DeviceConvNdBwdData**: Backward data pass (gradient w.r.t. input)
- **DeviceConvNdBwdWeight**: Backward weight pass (gradient w.r.t. weights)
- **DeviceGroupedConvBwdWeight**: Grouped backward weight computation

### Normalization Operations

#### Layer Normalization
- **DeviceNormalizationFwd**: Forward layer normalization with Welford's algorithm
- **DeviceNormalizationBwd**: Backward layer normalization with gradient computation

#### Batch Normalization
- **DeviceBatchnormFwd**: Forward batch normalization
- **DeviceBatchnormBwd**: Backward batch normalization

#### Group Normalization
- **DeviceGroupnormFwd**: Forward group normalization
- **DeviceGroupnormBwd**: Backward group normalization

### Reduction Operations
- **DeviceReduce**: Parallel reductions with various operators (Sum, Max, Min, etc.)
- **DeviceMultipleReduce**: Multiple simultaneous reductions
- **DeviceSoftmax**: Numerically stable softmax implementation

### Memory Operations
- **DevicePermute**: Tensor transposition and dimension reordering
- **DeviceElementwise**: Element-wise tensor operations
- **DevicePutElement**: Scatter operations for sparse updates

## Implementation Patterns

### XDL (Matrix Core) Integration
Most high-performance operations leverage AMD's matrix core instructions:

```cpp
// Matrix instruction configuration
using ABlockTransferThreadClusterLengths_AK0_M_AK1 = S<4, 64, 1>;
using BBlockTransferThreadClusterLengths_BK0_N_BK1 = S<4, 32, 1>;

// XDL mapping to matrix cores
static constexpr auto MPerXDL = 32;    // Maps to MFMA instruction dimensions
static constexpr auto NPerXDL = 32;
```

### Memory Access Optimization
- **Coalesced Access**: Thread clusters designed for optimal memory coalescing
- **Vectorized Loads**: Configurable vector widths for memory transfers
- **LDS Management**: Shared memory usage optimization with double buffering

### Fusion Strategies
- **Epilogue Fusion**: Operations fused into GEMM epilogue to avoid intermediate memory
- **Producer-Consumer**: Direct data flow between operations in shared memory
- **Multi-stage Pipelines**: Complex operation chains with intermediate buffering

## Performance Characteristics

### Compute Intensity
- **GEMM Operations**: Compute-bound, high FLOP/byte ratio
- **Convolutions**: Compute-bound when using implicit GEMM
- **Reductions**: Memory-bound, limited by reduction tree depth
- **Element-wise**: Memory-bound, bandwidth limited

### Memory Patterns
- **Sequential Access**: Optimized for coalesced memory transactions
- **Strided Access**: Handled efficiently through vector loads
- **Random Access**: Minimized through careful data layout

### Fusion Benefits
- **Bandwidth Reduction**: Eliminates intermediate tensor storage
- **Cache Locality**: Data reuse within GPU cache hierarchy
- **Kernel Launch Overhead**: Reduces GPU kernel launch costs
- **Instruction Scheduling**: Better compute/memory overlap

## Integration with Deep Learning Frameworks

### PyTorch Integration
```cpp
// Example integration pattern for PyTorch
auto gemm_op = DeviceGemmInstance{};
auto argument = gemm_op.MakeArgument(
    tensor_a.data_ptr<half>(),
    tensor_b.data_ptr<half>(), 
    tensor_c.data_ptr<half>(),
    M, N, K,
    stride_a, stride_b, stride_c,
    PassThrough{}, PassThrough{}, PassThrough{});

auto invoker = gemm_op.MakeInvoker();
invoker.Run(argument, StreamConfig{stream});
```

### Optimizations for AI/ML Workloads
- **Mixed Precision**: FP16 input with FP32 accumulation
- **Quantization**: INT8/INT4 support with on-the-fly dequantization
- **Attention Mechanisms**: Specialized fused attention kernels
- **Transformer Optimizations**: Layer norm, GELU, and residual fusions
