# Binary Element-wise Operations with Broadcasting

This example demonstrates a generic binary element-wise operation, a fundamental building block in numerical computing. It covers two important cases:
1.  **Simple Element-wise**: Applying a binary function to two input tensors of the *same* shape.
2.  **Element-wise with Broadcasting**: Applying a binary function to two input tensors of *different but compatible* shapes.

Broadcasting defines a set of rules for applying element-wise operations on tensors of different sizes, and it is a cornerstone of libraries like NumPy and TensorFlow.

## Mathematical Formulation

### Simple Element-wise
Given two input tensors, A and B, of the same rank and dimensions, and a binary operator $\odot$, the operation computes an output tensor C where each element is:

$C_{i,j,k,\dots} = A_{i,j,k,\dots} \odot B_{i,j,k,\dots}$

### Element-wise with Broadcasting
Broadcasting allows element-wise operations on tensors with different shapes, provided they are compatible. Two dimensions are compatible if they are equal, or if one of them is 1. The operation implicitly "stretches" or "duplicates" the tensor with the dimension of size 1 to match the other tensor's shape.

For example, adding a bias vector `B` of shape `(1, N)` to a matrix `A` of shape `(M, N)`:
$C_{i,j} = A_{i,j} + B_{0,j}$

Here, the single row of `B` is broadcast across all `M` rows of `A`. The output tensor `C` has the shape `(M, N)`.

Common binary element-wise operations include addition, subtraction, multiplication (Hadamard product), division, max, and min.

## Algorithmic Strategy: Grid-Stride Loop with Broadcasting

The implementation for both cases relies on the efficient **grid-stride loop**, which is adapted to handle broadcasting.

1.  **Grid Partitioning**: The problem is mapped to a 1D grid of threads based on the number of elements in the **output** tensor.

2.  **Grid-Stride Loop**: Each thread iterates through a subset of the output elements. For each output index, it must calculate the corresponding indices into the input tensors A and B.
    ```
    for (int out_idx = ...; out_idx < N_out; out_idx += ... )
    {
        // Map 1D output index to multi-dimensional coordinates (i, j, k...)
        Coord out_coord = get_coord(out_idx);

        // Calculate input indices based on broadcasting rules
        Coord a_coord = get_broadcast_coord(out_coord, shape_A);
        Coord b_coord = get_broadcast_coord(out_coord, shape_B);

        // Map multi-dimensional input coords back to 1D indices
        int a_idx = get_idx(a_coord, strides_A);
        int b_idx = get_idx(b_coord, strides_B);

        C[out_idx] = A[a_idx] op B[b_idx];
    }
    ```
-   The core of the broadcasting logic lies in the `get_broadcast_coord` function. If an input tensor's dimension is 1, the coordinate for that dimension is always set to 0, effectively reusing the same element across the broadcast dimension. If the dimension matches the output, the coordinate is passed through.
-   This strategy ensures that memory accesses to the larger tensor remain coalesced, while accesses to the smaller, broadcasted tensor will naturally involve re-reading the same values, which is efficiently handled by the GPU's cache hierarchy.

Like the simple case, broadcasted element-wise operations are almost always memory-bandwidth-bound.

## Source Code Organization

This example contains multiple files to demonstrate different scenarios:

-   [`binary_elementwise_xdl.cpp`](./binary_elementwise_xdl.cpp): Demonstrates the simple case where both input tensors have the same shape.
-   [`broadcast_add_2d_amn_bn.cpp`](./broadcast_add_2d_amn_bn.cpp): A specific example of broadcasting, adding a tensor of shape `(B, N)` to a tensor of shape `(A, M, N)`.
-   [`../../include/ck/tensor_operation/gpu/device/device_elementwise.hpp`](../../include/ck/tensor_operation/gpu/device/device_elementwise.hpp): The high-level device interface. It is generic enough to handle both simple and broadcasted operations by correctly interpreting the tensor descriptors, which contain shape and stride information.
-   [`../../include/ck/tensor_operation/gpu/grid/gridwise_elementwise.hpp`](../../include/ck/tensor_operation/gpu/grid/gridwise_elementwise.hpp): The grid-wise kernel that implements the grid-stride loop. The tensor coordinate logic within this kernel correctly handles broadcasting based on the provided tensor descriptors.
-   [`../../include/ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp`](../../include/ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp): Defines the various binary operator functors (like `Add`, `Multiply`, etc.).

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/19_binary_elementwise
mkdir build && cd build

cmake \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_PREFIX_PATH="/opt/rocm;${CK_INSTALL_PATH}" \
  ..

make -j
```

### Run the Example
```bash
# Run the simple element-wise example
./binary_elementwise_xdl 1 2 1

# Run the broadcasting example
./broadcast_add_2d_amn_bn 1 2 1
```

## Applications

Broadcasting is a powerful feature that makes code more concise and memory-efficient.
-   **Adding Bias**: The most common use case in deep learning is adding a bias vector (shape `[N]`) to a matrix of activations (shape `[Batch, N]`).
-   **Feature Scaling**: Multiplying a feature map (shape `[N, C, H, W]`) by a per-channel scaling factor (shape `[1, C, 1, 1]`).
-   **Standardization**: In data preprocessing, subtracting the mean (a vector) and dividing by the standard deviation (another vector) from a data matrix.
-   **Coordinate Grids**: Creating coordinate grids by adding a row vector `[0, 1, 2...]` to a column vector `[0, 1, 2...]^T`.
