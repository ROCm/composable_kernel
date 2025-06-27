# GEMM with Bilinear Operations

## Theory

This example demonstrates **GEMM with bilinear operations on auxiliary tensors**. Bilinear fusion patterns are used in neural networks for gating, attention, and multimodal feature fusion, where the output of a matrix multiplication is combined elementwise with one or more additional tensors.

**Mathematical Formulation:**
$$
F = \text{BilinearOp}(A \times B, D, E)
$$
- $A$: [M, K] input matrix
- $B$: [K, N] weight matrix
- $D$, $E$: [M, N] auxiliary tensors (or broadcastable)
- $F$: [M, N] output

**Examples:**
- Elementwise: $F = (A \times B) \odot D \odot E$
- Gated: $F = (A \times B) \odot \sigma(D) + E$
- Weighted: $F = \alpha (A \times B) + \beta (D \odot E)$

**Algorithmic Background:**
- The GEMM result is kept in registers and combined with auxiliary tensors in the epilogue.
- No intermediate results are written to global memory.
- This pattern is common in attention, gating, and feature interaction layers.

## How to Run

### Prerequisites
```bash
cd composable_kernel/build
make -j install
```

### Build and Execute
```bash
cd composable_kernel/example/02_gemm_bilinear
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./gemm_bilinear_xdl -M 1024 -N 1024 -K 512 --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/02_gemm_bilinear/
├── gemm_bilinear_xdl.cpp         # Main example: sets up, runs, and verifies GEMM with bilinear fusion
include/ck/tensor_operation/gpu/device/
│   └── device_gemm_multiple_d.hpp       # Device-level API for multi-tensor GEMM
include/ck/tensor_operation/gpu/device/impl/
│   └── device_gemm_bilinear_impl.hpp    # Bilinear operation implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_gemm_multiple_d.hpp     # Grid-level multi-tensor GEMM kernel
include/ck/tensor_operation/gpu/element/
    └── element_wise_operation.hpp       # Elementwise operation definitions
```

### Key Classes and Functions

- **DeviceGemmMultipleD** (in `device_gemm_multiple_d.hpp`):  
  Device API for GEMM with multiple auxiliary tensors and fused epilogues.
  ```cpp
  template <typename ALayout, typename BLayout, typename DsLayout, typename ELayout,
            typename ADataType, typename BDataType, typename DsDataType,
            typename EDataType, typename AElementwiseOperation,
            typename BElementwiseOperation, typename CDEElementwiseOperation>
  struct DeviceGemmMultipleD : public BaseOperator
  ```
- **gridwise_gemm_multiple_d** (in `gridwise_gemm_multiple_d.hpp`):  
  Implements the tiled/blocking GEMM kernel with multi-tensor epilogue.
- **element_wise_operation** (in `element_wise_operation.hpp`):  
  Defines bilinear and other elementwise operations.

This example demonstrates how Composable Kernel supports complex multi-tensor fusion patterns for advanced neural network architectures.
