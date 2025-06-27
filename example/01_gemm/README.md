# Basic GEMM (General Matrix Multiplication)

## Theory

GEMM (General Matrix Multiplication) is a fundamental operation in linear algebra and deep learning. It computes the product of two matrices, optionally adds a bias or residual, and is the core of many neural network layers (MLPs, attention, convolutions via im2col).

**Mathematical Formulation:**
$$
C = \alpha (A \times B) + \beta D
$$
- $A$: [M, K] input matrix
- $B$: [K, N] weight matrix
- $D$: [M, N] optional bias/residual
- $C$: [M, N] output
- $\alpha, \beta$: scalars (often 1.0, 0.0)

**Algorithmic Background:**
- GEMM is implemented using a tiled/blocking strategy to maximize data reuse and memory bandwidth.
- Modern GPU implementations use matrix core/XDL/MFMA instructions for high throughput.
- The operation is the computational backbone for transformer attention, MLPs, CNNs (via lowering), and more.

## How to Run

### Prerequisites
```bash
cd composable_kernel/build
make -j install
```

### Build and Execute
```bash
cd composable_kernel/example/01_gemm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (FP16)
./gemm_xdl_fp16 -M 4096 -N 4096 -K 4096 -v 1 -t 1
```

## Source Code Structure

### Directory Layout
```
example/01_gemm/
├── gemm_xdl_fp16.cpp         # Main example: sets up, runs, and verifies GEMM (FP16)
├── gemm_xdl_fp32.cpp         # Main example: FP32 variant
include/ck/tensor_operation/gpu/device/
│   └── device_gemm.hpp       # Device-level GEMM API (templated)
include/ck/tensor_operation/gpu/device/impl/
│   └── device_gemm_xdl.hpp   # XDL-based GEMM implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_gemm_xdl.hpp # Grid-level tiled GEMM kernel
include/ck/tensor_operation/gpu/block/
│   └── blockwise_gemm_xdl.hpp # Block-level tiled GEMM
library/reference_tensor_operation/cpu/
    └── reference_gemm.hpp    # CPU reference GEMM for correctness checking
```

### Key Classes and Functions

- **DeviceGemmXdl** (in `device_gemm.hpp`):  
  Main device API for launching GEMM kernels.  
  ```cpp
  template <typename ALayout, typename BLayout, typename CLayout,
            typename ADataType, typename BDataType, typename CDataType,
            typename AElementwiseOperation, typename BElementwiseOperation,
            typename CElementwiseOperation, typename GemmSpecialization>
  struct DeviceGemmXdl : public BaseOperator
  ```
- **GridwiseGemmXdl** (in `gridwise_gemm_xdl.hpp`):  
  Implements the tiled/blocking GEMM kernel for the GPU grid.
- **BlockwiseGemmXdl** (in `blockwise_gemm_xdl.hpp`):  
  Handles block-level computation and shared memory tiling.
- **reference_gemm** (in `reference_gemm.hpp`):  
  CPU implementation for result verification.

This example is the foundation for all matrix operations in Composable Kernel and is the basis for more advanced fused and batched operations.
