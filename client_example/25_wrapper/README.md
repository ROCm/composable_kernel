[Back to the main page](../../README.md)
# Composable Kernel Wrapper GEMM Tutorial

This tutorial demonstrates how to implement matrix multiplication using the Composable Kernel (CK) wrapper. The examples show both basic and optimized GEMM implementations, as well as how to use the wrapper for tensor transformations such as im2col.

---

## Theory

The CK wrapper provides a high-level interface for launching GEMM and tensor operations, abstracting away many of the low-level details. This enables rapid prototyping and experimentation with different kernel traits, layouts, and memory strategies.

**Mathematical Formulation:**
- GEMM: $C = A \times B$
- im2col: Rearranges image blocks into columns for GEMM-based convolution.

**Algorithmic Background:**
- The wrapper allows you to specify layouts, memory types (global/LDS), and tiling strategies.
- Optimized GEMM kernels use advanced tiling, vectorization, and memory coalescing for performance.

---

## How to Run

### Prerequisites
```bash
cd composable_kernel/build
make -j install
```

### Build and Execute
```bash
cd composable_kernel/client_example/25_wrapper
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (basic GEMM)
./wrapper_basic_gemm

# Example run (optimized GEMM)
./wrapper_optimized_gemm

# Example run (im2col transformation)
./wrapper_img2col

# Example run (tensor transform using wrapper)
./tensor_transform_using_wrapper
```

---

## Source Code Structure

### Directory Layout
```
client_example/25_wrapper/
├── wrapper_basic_gemm.cpp         # Basic GEMM using CK wrapper
├── wrapper_optimized_gemm.cpp     # Optimized GEMM using CK wrapper
├── wrapper_img2col.cpp            # im2col transformation using CK wrapper
├── tensor_transform_using_wrapper.cpp # General tensor transform example
├── CMakeLists.txt                 # Build configuration for the example
├── README.md                      # This tutorial and reference
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input tensors, configures wrapper parameters, launches the kernel, and verifies the result.
- **CK wrapper API usage**:  
  Demonstrates how to create layouts, tensors, and launch GEMM or tensor transforms using the wrapper.

---

## Additional Details

- The wrapper supports padding, tiling, and flexible memory layouts.
- Optimized GEMM uses `gridwise_gemm_xdlops_v2r3` for high performance.
- See comments in each example for step-by-step usage.

---

## Related Examples

- [01_gemm](../01_gemm/README.md): Basic GEMM client example
- [22_im2col_col2im](../22_im2col_col2im/README.md): im2col/col2im transformations
- [25_gemm_bias_e_permute](../../example/25_gemm_bias_e_permute/README.md): GEMM with bias and permutation in the main example directory

---
[Back to Client Examples](../README.md)
