# N-Dimensional Convolution Forward

## Theory

This example demonstrates the **N-dimensional convolution forward pass** using Composable Kernel. Convolution is a fundamental operation in deep learning, especially in convolutional neural networks (CNNs) for images, audio, and volumetric data.

**Mathematical Formulation:**
Given:
- Input tensor: $X[N, C_{in}, D_1, D_2, ..., D_n]$
- Weight tensor: $W[C_{out}, C_{in}, K_1, K_2, ..., K_n]$
- Output tensor: $Y[N, C_{out}, O_1, O_2, ..., O_n]$

The convolution computes:
$$
Y[n, c_{out}, o_1, ..., o_n] = \sum_{c_{in}} \sum_{k_1} ... \sum_{k_n} X[n, c_{in}, o_1 + k_1, ..., o_n + k_n] \cdot W[c_{out}, c_{in}, k_1, ..., k_n]
$$

Stride, padding, and dilation parameters control the mapping between input and output indices.

**Algorithmic Background:**
- Composable Kernel implements convolution as an implicit GEMM (matrix multiplication) for efficiency.
- The input and weight tensors are transformed into matrices, and the convolution is performed as a GEMM.

## How to Run

### Prerequisites
```bash
cd composable_kernel/build
make -j install
```

### Build and Execute
```bash
cd composable_kernel/example/09_convnd_fwd
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./convnd_fwd_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/09_convnd_fwd/
├── convnd_fwd_xdl.cpp         # Main example: sets up, runs, and verifies N-D convolution
include/ck/tensor_operation/gpu/device/
│   └── device_convnd_fwd.hpp       # Device-level convolution API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_convnd_fwd_xdl.hpp   # XDL-based convolution implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_convnd_fwd_xdl.hpp # Grid-level convolution kernel
include/ck/tensor_operation/gpu/block/
    └── blockwise_convnd_fwd_xdl.hpp # Block-level convolution
```

### Key Classes and Functions

- **DeviceConvNdFwd** (in `device_convnd_fwd.hpp`):  
  Device API for N-dimensional convolution.
  ```cpp
  template <typename InLayout, typename WeiLayout, typename OutLayout,
            typename InDataType, typename WeiDataType, typename OutDataType,
            typename InElementwiseOperation, typename WeiElementwiseOperation,
            typename OutElementwiseOperation, typename ConvSpecialization>
  struct DeviceConvNdFwd : public BaseOperator
  ```
- **gridwise_convnd_fwd_xdl** (in `gridwise_convnd_fwd_xdl.hpp`):  
  Implements the tiled/blocking convolution kernel.
- **blockwise_convnd_fwd_xdl** (in `blockwise_convnd_fwd_xdl.hpp`):  
  Handles block-level computation and shared memory tiling.

This example demonstrates how Composable Kernel implements efficient N-dimensional convolution using implicit GEMM, supporting a wide range of deep learning applications.
