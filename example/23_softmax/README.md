# Parallel Softmax

## Theory

This example demonstrates **parallel softmax computation** over tensors. Softmax is a key operation in deep learning, especially in attention mechanisms and classification, converting logits into normalized probabilities.

**Mathematical Formulation:**
Given input $X$ and axis $a$:
$$
\text{softmax}(X)_i = \frac{\exp(X_i)}{\sum_j \exp(X_j)}
$$

**Algorithmic Background:**
- Softmax is implemented using a numerically stable algorithm:
  1. Subtract the maximum value for numerical stability.
  2. Exponentiate and sum.
  3. Normalize by the sum.
- Efficient parallel softmax requires careful reduction and memory access patterns.

## How to Run

### Prerequisites
```bash
cd composable_kernel/build
make -j install
```

### Build and Execute
```bash
cd composable_kernel/example/23_softmax
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./softmax_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/23_softmax/
├── softmax_xdl.cpp         # Main example: sets up, runs, and verifies softmax
include/ck/tensor_operation/gpu/device/
│   └── device_softmax.hpp       # Device-level softmax API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_softmax_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_softmax.hpp     # Grid-level softmax kernel
include/ck/tensor_operation/gpu/block/
    └── blockwise_softmax.hpp    # Block-level softmax
```

### Key Classes and Functions

- **DeviceSoftmax** (in `device_softmax.hpp`):  
  Device API for softmax.
  ```cpp
  template <typename InDataType, typename OutDataType, typename AccDataType,
            typename ReduceOp, typename InElementwiseOperation,
            typename AccElementwiseOperation, typename OutElementwiseOperation>
  struct DeviceSoftmax : public BaseOperator
  ```
- **gridwise_softmax** (in `gridwise_softmax.hpp`):  
  Implements the tiled/blocking softmax kernel.
- **blockwise_softmax** (in `blockwise_softmax.hpp`):  
  Handles block-level softmax and shared memory.

This example demonstrates how Composable Kernel implements efficient, numerically stable softmax for deep learning models.
