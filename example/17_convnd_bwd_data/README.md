# N-Dimensional Convolution Backward Pass for Data

This example demonstrates the backward data pass of an N-dimensional convolution, often denoted as `conv_bwd_data`. This operation is a crucial part of the backpropagation algorithm for training Convolutional Neural Networks (CNNs). Its purpose is to compute the gradient of the loss function with respect to the convolution's *input data*, which is then passed back to the preceding layer in the network.

## Mathematical Formulation

The backward data pass computes the gradient $\frac{\partial L}{\partial \text{In}}$, given the gradient from the subsequent layer, $\frac{\partial L}{\partial \text{Out}}$, and the filter weights `W` used in the forward pass.

Let the forward convolution be defined as:
$\text{Out} = \text{In} \star W$

The backward data pass is mathematically equivalent to a "full" convolution between the output gradient tensor `dL/dOut` and the 180-degree rotated (or transposed and flipped) weight tensor `W`.

$\frac{\partial L}{\partial \text{In}} = \frac{\partial L}{\partial \text{Out}} \star \text{rot180}(W)$

This operation propagates the error signal from the output back to the input, weighted by the same filters that were used in the forward pass.

## Algorithmic Strategy: Implicit GEMM

As with the forward pass, the most efficient way to implement the backward data pass on a GPU is to transform the convolution into a General Matrix-Matrix Multiplication (GEMM) problem.

1.  **Output Gradient Reshaping**: The output gradient tensor `dL/dOut` is logically reshaped into a matrix `dL/dOut'` of shape `[K, (N*Ho*Wo)]`. This becomes the "A" matrix in the GEMM.

2.  **Weight Reshaping**: The weight tensor `W` is logically reshaped into a matrix `W'` of shape `[K, (C*Y*X)]`. This becomes the "B" matrix in the GEMM.

3.  **Implicit GEMM**: The core computation is then formulated as a GEMM operation. However, the output of this GEMM is not a simple matrix; it's the `dL/dIn` tensor.
    $(\text{dL/dIn})' = (W')^T \times (\text{dL/dOut})'$

    The key insight is that this operation can be performed without explicitly forming the matrices. The GEMM kernel is designed to read from `dL/dOut` and `W` and write its results directly to the appropriate locations in the `dL/dIn` tensor. This process is sometimes referred to as an "implicit `col2im`" (column-to-image), as it is the inverse of the `im2col` transformation used in the forward pass.

This "implicit GEMM" approach is highly efficient. It avoids the massive memory and bandwidth overhead of materializing intermediate matrices, which is critical for performance.

## Source Code Organization

-   [`conv_bwd_data_xdl.cpp`](./conv_bwd_data_xdl.cpp): The main example file that defines the parameters for a 2D convolution and instantiates the generic `DeviceConvNdBwdData` kernel to compute the input gradients.
-   [`../../include/ck/tensor_operation/gpu/device/device_conv_bwd_data.hpp`](../../include/ck/tensor_operation/gpu/device/device_conv_bwd_data.hpp): The high-level device interface for the backward data convolution. It is templated on the dimensionality, layouts, and data types of the problem.
-   [`../../include/ck/tensor_operation/gpu/grid/gridwise_gemm_implicit_gemm_v1r2_xdlops_nchw_kcyx_nkhw.hpp`](../../include/ck/tensor_operation/gpu/grid/gridwise_gemm_implicit_gemm_v1r2_xdlops_nchw_kcyx_nkhw.hpp): An example of a specific grid-wise kernel that implements the implicit GEMM algorithm for the backward data pass. The library contains multiple such kernels optimized for different layouts and problem types, and the `DeviceConvNdBwdData` interface selects the most appropriate one.
-   [`../../library/include/ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp`](../../library/include/ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp): A CPU reference implementation used to verify the correctness of the GPU kernel's output.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/17_convnd_bwd_data
mkdir build && cd build

cmake \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_PREFIX_PATH="/opt/rocm;${CK_INSTALL_PATH}" \
  ..

make -j
```

### Run the Example
```bash
# Run the example with default settings
./conv_bwd_data_xdl

# Run with verification, data initialization, and timing
./conv_bwd_data_xdl 1 2 1
```

## Relationship to Other Passes

The training of a single convolutional layer requires three distinct steps:

1.  **Forward Pass (`conv_fwd`)**: Computes the output feature maps.
    -   `Out = In * W`
2.  **Backward Data Pass (`conv_bwd_data`)**: Computes the gradient with respect to the input, propagating the error to the previous layer. This is the focus of the current example.
    -   `dL/dIn = dL/dOut * rot180(W)`
3.  **Backward Weight Pass (`conv_bwd_weight`)**: Computes the gradient with respect to the weights, which is needed for the weight update.
    -   `dL/dW = In * dL/dOut`

All three passes are critical for training a CNN, and all are typically implemented as high-performance implicit GEMM operations.
