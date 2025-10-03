# Client Example: Basic GEMM

## Theory

This client example demonstrates a basic **GEMM (General Matrix Multiplication)** operation using the Composable Kernel (CK) library. GEMM is a core operation in linear algebra and deep learning, computing the product of two matrices with optional scaling and bias.

**Mathematical Formulation:**
$$
C = \alpha (A \times B) + \beta D
$$
- $A$: [M, K] input matrix
- $B$: [K, N] weight matrix
- $D$: [M, N] optional bias or residual
- $C$: [M, N] output
- $\alpha, \beta$: scalars (often 1.0, 0.0)

## Run the example

Install CK using the [instructions in the official documentation](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/install/Composable-Kernel-install.html). [Docker images](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/install/Composable-Kernel-Docker.html) with all required prerequisites are also available.

Build and run the example:

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
