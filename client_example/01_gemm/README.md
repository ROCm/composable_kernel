# Client Example: Basic GEMM

## Theory

This client example demonstrates a basic **GEMM (General Matrix Multiplication)** operation using the Composable Kernel library. GEMM is a core operation in linear algebra and deep learning, computing the product of two matrices and optionally adding a bias or scaling.

**Mathematical Formulation:**
$$
C = \alpha (A \times B) + \beta D
$$
- $A$: [M, K] input matrix
- $B$: [K, N] weight matrix
- $D$: [M, N] optional bias or residual
- $C$: [M, N] output
- $\alpha, \beta$: scalars (often 1.0, 0.0)

**Algorithmic Background:**
- The operation is implemented using a tiled/blocking strategy for memory efficiency.
- GEMM is the computational backbone for transformer attention, MLPs, and CNNs (via im2col).

## How to Run

### Prerequisites
```bash
cd composable_kernel/build
make -j install
```

### Build and Execute
```bash
cd composable_kernel/client_example/01_gemm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./gemm
```

## Source Code Structure

### Directory Layout
```
client_example/01_gemm/
├── gemm.cpp         # Main client example: sets up, runs, and verifies GEMM
├── CMakeLists.txt   # Build configuration for the example
```

### Key Functions

- **main()** (in `gemm.cpp`):  
  Sets up input matrices, configures GEMM parameters, launches the GEMM kernel, and verifies the result.
- **GEMM kernel invocation**:  
  Uses the Composable Kernel device API to launch the GEMM operation.

This client example provides a minimal, end-to-end demonstration of using Composable Kernel for matrix multiplication in a user application.
