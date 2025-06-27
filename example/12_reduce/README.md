# Parallel Reduction Operations

## Theory

This example demonstrates **parallel reduction operations** (e.g., sum, max, min, mean) over tensors. Reduction is a fundamental operation in deep learning for computing statistics (such as batch mean/variance), loss aggregation, and normalization.

**Mathematical Formulation:**
Given a tensor $X$ and a reduction axis $a$:
$$
Y = \text{reduce}_{a}(X)
$$
- For sum: $Y = \sum_{i \in a} X_i$
- For max: $Y = \max_{i \in a} X_i$
- For mean: $Y = \frac{1}{|a|} \sum_{i \in a} X_i$

**Algorithmic Background:**
- Reductions are implemented using parallel tree reduction or segmented reduction algorithms.
- Efficient reductions require careful memory access, synchronization, and sometimes numerically stable algorithms (e.g., Welford's for variance).

## How to Run

### Prerequisites
```bash
cd composable_kernel/build
make -j install
```

### Build and Execute
```bash
cd composable_kernel/example/12_reduce
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (sum reduction)
./reduce_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/12_reduce/
├── reduce_xdl.cpp         # Main example: sets up, runs, and verifies reduction
include/ck/tensor_operation/gpu/device/
│   └── device_reduce.hpp       # Device-level reduction API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_reduce_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_reduce.hpp     # Grid-level reduction kernel
include/ck/tensor_operation/gpu/block/
    └── blockwise_reduce.hpp    # Block-level reduction
```

### Key Classes and Functions

- **DeviceReduce** (in `device_reduce.hpp`):  
  Device API for reductions.
  ```cpp
  template <typename InDataType, typename OutDataType, typename AccDataType,
            typename ReduceOperation, typename InElementwiseOperation,
            typename AccElementwiseOperation, typename OutElementwiseOperation>
  struct DeviceReduce : public BaseOperator
  ```
- **gridwise_reduce** (in `gridwise_reduce.hpp`):  
  Implements the tiled/blocking reduction kernel.
- **blockwise_reduce** (in `blockwise_reduce.hpp`):  
  Handles block-level reduction and shared memory.

This example demonstrates how Composable Kernel implements efficient parallel reductions for deep learning and scientific computing.
