# GEMM with Microscaling

This example demonstrates a **GEMM operation with microscaling**, an advanced quantization technique that applies fine-grained scaling to small blocks of data. Microscaling enables more precise quantization than traditional methods by using different scale factors for small groups of elements, leading to better accuracy preservation in quantized neural network inference.

## Mathematical Formulation

Microscaling applies block-wise scaling to quantized data, where each small block (typically 2-8 elements) has its own scale factor.

Given:
- Matrix `A` with microscaling: blocks of quantized values with per-block scale factors
- Matrix `B` with microscaling: blocks of quantized values with per-block scale factors
- Block size `BS` (e.g., 2, 4, or 8 elements per block)

For each block `k` in the matrices:
1.  **Dequantization**: Convert quantized values to higher precision using block-specific scales.
    $A_{block\_k} = \text{scale}_A[k] \times A_{quantized\_k}$
    $B_{block\_k} = \text{scale}_B[k] \times B_{quantized\_k}$

2.  **Matrix Multiplication**: Perform GEMM on the dequantized values.
    $C = A_{dequantized} \times B_{dequantized}$

3.  **Optional Output Quantization**: The result can be quantized with its own microscaling.
    $C_{quantized\_block\_j} = \text{round}(C_{block\_j} / \text{scale}_C[j])$

The key optimization is that dequantization happens on-the-fly during the GEMM computation, avoiding the need to store full-precision intermediate tensors.

## Algorithmic Strategy: Block-wise Dequantization with Tiled GEMM

The implementation integrates microscaling dequantization into the tiled GEMM algorithm.

1.  **Tiled GEMM with Microscaling**: Standard tiling approach enhanced with block-wise dequantization:
    -   **Load Quantized Tiles**: Read quantized data and scale factors from global memory
    -   **Block-wise Dequantization**: Apply appropriate scale factors to each block within the tile
    -   **Compute in Higher Precision**: Perform matrix multiplication using the dequantized values
    -   **Accumulate in Full Precision**: Maintain high precision for intermediate accumulations

2.  **Memory Access Optimization**:
    -   **Coalesced Quantized Reads**: Efficiently load quantized data and scales
    -   **Scale Factor Caching**: Cache scale factors in shared memory when possible
    -   **Vectorized Dequantization**: Use vectorized operations for block-wise scaling

3.  **Precision Management**:
    -   **On-the-fly Conversion**: Perform dequantization in registers without intermediate storage
    -   **Accumulator Precision**: Use appropriate precision (FP32) for accumulation to maintain accuracy
    -   **Output Precision Control**: Handle output quantization if required

## Source Code Organization

-   [`gemm_microscaling_xdl.cpp`](./gemm_microscaling_xdl.cpp): The main example file. It sets up microscaled matrices with quantized data and scale factors, and instantiates the `DeviceGemmMicroscaling` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_gemm_microscaling.hpp`](../../include/ck/tensor_operation/gpu/device/device_gemm_microscaling.hpp): The device interface for GEMM with microscaling support.
-   The underlying kernel implements sophisticated block-wise dequantization integrated into the GEMM computation pipeline.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/67_gemm_microscaling
mkdir build && cd build

cmake \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_PREFIX_PATH="/opt/rocm;${CK_INSTALL_PATH}" \
  ..

make -j
```

### Run the Example

Custom verification parameters:
```bash
# arg1: verification (0=no, 1=CPU)
# arg2: initialization (0=constant values, 1=integer values, 2=decimal values)
# arg3: time kernel (0=no, 1=yes)
# arg4: verbosity (0=no info, 1=verbose info)
# arg5 to 10: M(128x), N(128x), K(64x), StrideA, StrideB, StrideC
# arg11: KBatch
./bin/example_gemm_mx_fp8 1 1 0 1
```

Custom tensor shapes:
```bash
./bin/example_gemm_mx_fp8 1 2 1 0 128  128  256 -1 -1 -1 1
```

Default invocation:
```bash
# Implies: ./bin/example_gemm_mx_fp8 1 2 0 0
./bin/example_gemm_mx_fp8
```

## Applications in Advanced Quantization

Microscaling represents a cutting-edge approach to neural network quantization:

-   **High-Precision Quantization**: Achieve better accuracy than traditional uniform quantization by adapting to local data statistics
-   **Large Language Models**: Enable efficient deployment of very large models with minimal accuracy loss
-   **Scientific Computing**: Maintain precision for scientific applications that require quantization for memory constraints
-   **Edge AI**: Optimize models for deployment on resource-constrained devices while preserving quality
-   **Training Acceleration**: Use microscaling in training to reduce memory usage while maintaining convergence

## Advantages of Microscaling

Microscaling provides several benefits over traditional quantization approaches:

-   **Adaptive Precision**: Different blocks can have different scales based on their dynamic range
-   **Better Accuracy**: Reduced quantization error compared to global or channel-wise quantization
-   **Hardware Efficiency**: Can be implemented efficiently in hardware with modest overhead
-   **Gradient Preservation**: Better gradient flow in training scenarios

## Performance Considerations

Microscaling introduces additional complexity but can be implemented efficiently:

-   **Scale Factor Overhead**: Additional memory for storing per-block scale factors
-   **Computation Overhead**: Block-wise scaling operations during dequantization
-   **Memory Bandwidth**: Scale factors can be cached effectively due to their smaller size
-   **Precision Trade-offs**: Balance between accuracy improvement and computational overhead

## Comparison with Other Quantization Methods

| Method | Scale Granularity | Accuracy | Implementation Complexity | Memory Overhead |
|--------|------------------|----------|---------------------------|-----------------|
| **Global** | Entire tensor | Lowest | Simple | Minimal |
| **Channel-wise** | Per channel | Medium | Moderate | Low |
| **Block-wise** | Small blocks | High | High | Medium |
| **Microscaling** | Micro-blocks (2-8 elements) | Highest | Very High | Medium-High |

Microscaling represents the state-of-the-art in quantization techniques, providing the best accuracy preservation at the cost of increased implementation complexity. This example demonstrates how advanced quantization methods can be efficiently implemented in high-performance GPU kernels.
