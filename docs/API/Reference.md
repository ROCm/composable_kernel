# Composable Kernel API Reference

## Core Device Operations

### GEMM Operations

#### DeviceGemm
```cpp
template <typename ALayout, typename BLayout, typename CLayout,
          typename ADataType, typename BDataType, typename CDataType,
          typename AElementwiseOperation, typename BElementwiseOperation, 
          typename CElementwiseOperation>
struct DeviceGemm : public BaseOperator
```

**Purpose**: Standard matrix multiplication C = A × B with configurable element-wise operations.

**Key Methods**:
- `MakeArgumentPointer(p_a, p_b, p_c, M, N, K, StrideA, StrideB, StrideC, a_op, b_op, c_op)`
- `MakeInvokerPointer()`

**Specializations**:
- `DeviceGemmXdl`: XDL/MFMA-based implementation
- `DeviceGemm_Xdl_CShuffle`: XDL with C-Shuffle for larger blocks

#### DeviceGemmMultipleD
```cpp
template <typename ALayout, typename BLayout, typename DsLayout, typename ELayout,
          typename ADataType, typename BDataType, typename DsDataType, 
          typename EDataType, typename AElementwiseOperation, 
          typename BElementwiseOperation, typename CDEElementwiseOperation>
struct DeviceGemmMultipleD
```

**Purpose**: GEMM with multiple auxiliary tensors for complex epilogue operations.

**Common Patterns**:
- Bias addition: `E = (A × B) + D`
- Element-wise operations: `E = f(A × B, D0, D1, ...)`

#### DeviceBatchedGemm
```cpp
template <typename ALayout, typename BLayout, typename CLayout,
          typename ADataType, typename BDataType, typename CDataType,
          typename AElementwiseOperation, typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceBatchedGemm
```

**Purpose**: Batched matrix operations for multiple independent GEMM problems.

**Usage**: Deep learning inference with multiple samples or attention heads.

### Convolution Operations

#### DeviceConvNdFwd
```cpp
template <ck::index_t NDimSpatial, typename InLayout, typename WeiLayout, 
          typename OutLayout, typename InDataType, typename WeiDataType, 
          typename OutDataType, typename InElementwiseOperation,
          typename WeiElementwiseOperation, typename OutElementwiseOperation,
          ConvolutionForwardSpecialization ConvForwardSpecialization>
struct DeviceConvNdFwd
```

**Purpose**: N-dimensional forward convolution using implicit GEMM transformation.

**Specializations**:
- `ConvolutionForwardSpecialization::Default`: Standard convolution
- `ConvolutionForwardSpecialization::Filter1x1Stride1Pad0`: Optimized 1x1 convolution
- `ConvolutionForwardSpecialization::Filter1x1Pad0`: 1x1 with stride optimization

#### DeviceGroupedConvFwd
```cpp
template <ck::index_t NDimSpatial, typename InLayout, typename WeiLayout,
          typename OutLayout, typename InDataType, typename WeiDataType,
          typename OutDataType, typename InElementwiseOperation,
          typename WeiElementwiseOperation, typename OutElementwiseOperation,
          ConvolutionForwardSpecialization ConvForwardSpecialization>
struct DeviceGroupedConvFwd
```

**Purpose**: Grouped convolutions for efficient depthwise and channel-wise operations.

**Applications**: MobileNet, EfficientNet, and other efficient CNN architectures.

### Normalization Operations

#### DeviceNormalizationFwd
```cpp
template <typename XDataType, typename GammaDataType, typename BetaDataType,
          typename YDataType, typename SaveMeanInvStdDataType,
          typename XElementwiseOperation, typename YElementwiseOperation>
struct DeviceNormalizationFwd
```

**Purpose**: Layer normalization forward pass with Welford's algorithm for numerical stability.

**Features**:
- Welford's online algorithm for mean/variance computation
- Configurable epsilon for numerical stability
- Optional statistics saving for backward pass

#### DeviceBatchnormFwd
```cpp
template <typename XDataType, typename ScaleBiasDataType, typename YDataType,
          typename SaveMeanInvStdDataType, typename XElementwiseOperation,
          typename YElementwiseOperation>
struct DeviceBatchnormFwd
```

**Purpose**: Batch normalization forward pass for CNN training and inference.

**Modes**:
- Training mode: Compute and save running statistics
- Inference mode: Use pre-computed statistics

### Attention and Transformer Operations

#### DeviceGemmSoftmaxGemm
```cpp
template <typename ALayout, typename B0Layout, typename B1Layout, 
          typename CLayout, typename ADataType, typename B0DataType,
          typename B1DataType, typename CDataType, typename Acc0DataType,
          typename Acc1DataType, typename AElementwiseOperation,
          typename B0ElementwiseOperation, typename Acc0ElementwiseOperation,
          typename B1ElementwiseOperation, typename CElementwiseOperation>
struct DeviceGemmSoftmaxGemm
```

**Purpose**: Fused attention mechanism: Q×K^T → Softmax → ×V.

**Optimizations**:
- Online softmax algorithm for memory efficiency
- Tiled computation to fit in shared memory
- Numerical stability through max subtraction

#### DeviceGemmSoftmaxGemmPermute
```cpp
// Extended version with output permutation for multi-head attention
```

**Purpose**: Attention with output permutation for head merging in transformers.

### Reduction Operations

#### DeviceReduce
```cpp
template <typename InDataType, typename AccDataType, typename OutDataType,
          ck::index_t Rank, ck::index_t NumReduceDim,
          typename ReduceOperation, typename InElementwiseOperation,
          typename AccElementwiseOperation, bool PropagateNan,
          bool UseIndex>
struct DeviceReduce
```

**Purpose**: Parallel tensor reductions with configurable operators.

**Supported Operations**:
- `ck::reduce::Add`: Summation
- `ck::reduce::Max`: Maximum value
- `ck::reduce::Min`: Minimum value  
- `ck::reduce::Amax`: Absolute maximum

#### DeviceSoftmax
```cpp
template <typename InDataType, typename AccDataType, typename OutDataType,
          ck::index_t Rank, ck::index_t NumReduceDim,
          typename InElementwiseOperation, typename AccElementwiseOperation>
struct DeviceSoftmax
```

**Purpose**: Numerically stable softmax implementation.

**Algorithm**: Two-pass algorithm with max subtraction for stability.

### Memory and Utility Operations

#### DevicePermute
```cpp
template <typename InDataType, typename OutDataType, ck::index_t NDimSpatial,
          typename InElementwiseOperation, typename OutElementwiseOperation>
struct DevicePermute
```

**Purpose**: Tensor dimension reordering and transposition.

**Applications**:
- NCHW ↔ NHWC layout conversion
- Multi-head attention tensor reshaping
- Preparing tensors for optimized operations

#### DeviceElementwise
```cpp
template <typename InDataTypes, typename OutDataTypes,
          typename ElementwiseOperation, ck::index_t NDimSpatial>
struct DeviceElementwise
```

**Purpose**: Multi-input element-wise tensor operations.

**Supported Operations**:
- Arithmetic: Add, Subtract, Multiply, Divide
- Activation functions: ReLU, GELU, Sigmoid, Tanh
- Comparison: Equal, Greater, Less

## Performance Tuning Parameters

### Block Configuration
```cpp
static constexpr auto BlockSize = 256;          // Total threads per block
static constexpr auto MPerBlock = 256;          // M-dimension tile size
static constexpr auto NPerBlock = 128;          // N-dimension tile size
static constexpr auto KPerBlock = 16;           // K-dimension tile size
```

### XDL/MFMA Configuration
```cpp
static constexpr auto MPerXDL = 32;             // Matrix instruction M-dimension
static constexpr auto NPerXDL = 32;             // Matrix instruction N-dimension
static constexpr auto MXdlPerWave = 4;          // XDL instructions per wave (M)
static constexpr auto NXdlPerWave = 2;          // XDL instructions per wave (N)
```

### Memory Transfer Optimization
```cpp
using ABlockTransferThreadClusterLengths = S<4, 64, 1>;
static constexpr auto ABlockTransferSrcVectorDim = 2;
static constexpr auto ABlockTransferDstScalarPerVector = 8;
```

## Common Usage Patterns

### Basic GEMM Usage
```cpp
// Setup data types and layouts
using ADataType = ck::half_t;
using BDataType = ck::half_t; 
using CDataType = ck::half_t;
using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

// Create device operation
auto gemm_op = DeviceGemmInstance{};

// Create argument
auto argument = gemm_op.MakeArgument(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a, stride_b, stride_c,
    PassThrough{}, PassThrough{}, PassThrough{});

// Execute
auto invoker = gemm_op.MakeInvoker();
float time = invoker.Run(argument, StreamConfig{stream, true});
```

### Fused Operations Usage
```cpp
// GEMM + Bias + ReLU
auto gemm_bias_relu = DeviceGemmBiasReLUInstance{};
auto argument = gemm_bias_relu.MakeArgument(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_a, stride_b, stride_bias, stride_c,
    PassThrough{}, PassThrough{}, 
    UnaryOp<ReLU>{});  // Fused ReLU activation
```

### Attention Mechanism Usage
```cpp
// Fused attention: Q×K^T → Softmax → ×V
auto attention_op = DeviceGemmSoftmaxGemmInstance{};
auto argument = attention_op.MakeArgument(
    q_ptr, k_ptr, v_ptr, out_ptr,
    batch_size, seq_len, head_dim,
    num_heads, 
    PassThrough{}, PassThrough{}, PassThrough{});
```

## Integration Examples

### PyTorch Custom Operator
```cpp
#include <torch/extension.h>
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"

torch::Tensor composable_kernel_gemm(
    torch::Tensor a, torch::Tensor b) {
    
    auto c = torch::empty({a.size(0), b.size(1)}, a.options());
    
    auto gemm_op = DeviceGemmInstance{};
    auto argument = gemm_op.MakeArgument(
        a.data_ptr<at::Half>(),
        b.data_ptr<at::Half>(),
        c.data_ptr<at::Half>(),
        a.size(0), b.size(1), a.size(1),
        a.stride(0), b.stride(0), c.stride(0),
        PassThrough{}, PassThrough{}, PassThrough{});
    
    auto invoker = gemm_op.MakeInvoker();
    invoker.Run(argument, StreamConfig{
        at::cuda::getCurrentCUDAStream()});
    
    return c;
}
