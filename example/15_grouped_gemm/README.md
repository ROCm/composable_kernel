# Grouped GEMM

## Theory

This example demonstrates **grouped GEMM**: performing multiple independent GEMM operations (with potentially different shapes) in a single kernel launch. Grouped GEMM is used in transformer models (e.g., multi-head attention), mixture-of-experts, and other architectures requiring heterogeneous batched matrix multiplications.

**Mathematical Formulation:**
For $G$ groups, each with its own $A_g$, $B_g$, $C_g$:
$$
C_g = A_g \times B_g \quad \text{for} \quad g = 1, 2, ..., G
$$
- $A_g$: [M_g, K_g] input matrix for group $g$
- $B_g$: [K_g, N_g] weight matrix for group $g$
- $C_g$: [M_g, N_g] output matrix for group $g$

**Algorithmic Background:**
- Each group can have different matrix sizes and strides.
- The kernel launches a grid covering all groups, with each block assigned to a group.
- Useful for variable-length sequences, multi-head attention, and expert routing.

## How to Run

### Prerequisites
```bash
cd composable_kernel/build
make -j install
```

### Build and Execute
```bash
cd composable_kernel/example/15_grouped_gemm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./grouped_gemm_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/15_grouped_gemm/
├── grouped_gemm_xdl.cpp         # Main example: sets up, runs, and verifies grouped GEMM
include/ck/tensor_operation/gpu/device/
│   └── device_grouped_gemm_xdl.hpp       # Device-level grouped GEMM API
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_grouped_gemm_xdl.hpp     # Grid-level grouped GEMM kernel
```

### Key Classes and Functions

- **DeviceGroupedGemmXdl** (in `device_grouped_gemm_xdl.hpp`):  
  Device API for grouped GEMM.
  ```cpp
  template <typename ALayout, typename BLayout, typename CLayout,
            typename ADataType, typename BDataType, typename CDataType,
            typename AElementwiseOperation, typename BElementwiseOperation,
            typename CElementwiseOperation, typename GemmSpecialization>
  struct DeviceGroupedGemmXdl : public BaseOperator
  ```
- **gridwise_grouped_gemm_xdl** (in `gridwise_grouped_gemm_xdl.hpp`):  
  Implements the tiled/blocking grouped GEMM kernel.

This example demonstrates how Composable Kernel supports efficient heterogeneous batched matrix multiplication for advanced AI/ML workloads.
