# 2D Pooling Forward

## Theory

This example demonstrates the **2D pooling forward pass**, a key operation in convolutional neural networks (CNNs) for spatial downsampling. Pooling reduces the spatial dimensions of feature maps, providing translation invariance and reducing computation.

**Mathematical Formulation:**
Given input $X[N, C, H_{in}, W_{in}]$, pooling window $(k_H, k_W)$, stride $(s_H, s_W)$, and padding $(p_H, p_W)$:
- Output $Y[N, C, H_{out}, W_{out}]$
- $H_{out} = \left\lfloor \frac{H_{in} + 2p_H - k_H}{s_H} \right\rfloor + 1$
- $W_{out} = \left\lfloor \frac{W_{in} + 2p_W - k_W}{s_W} \right\rfloor + 1$

For each output position:
- **Max Pooling:** $Y_{n,c,h,w} = \max_{i,j} X_{n,c,h \cdot s_H + i, w \cdot s_W + j}$
- **Average Pooling:** $Y_{n,c,h,w} = \frac{1}{k_H k_W} \sum_{i,j} X_{n,c,h \cdot s_H + i, w \cdot s_W + j}$

**Algorithmic Background:**
- Each thread computes one or more output elements.
- Handles padding and boundary conditions.
- Optimizes memory access for bandwidth.

## How to Run

### Prerequisites
```bash
cd composable_kernel/build
make -j install
```

### Build and Execute
```bash
cd composable_kernel/example/13_pool2d_fwd
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./pool2d_fwd_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/13_pool2d_fwd/
├── pool2d_fwd_xdl.cpp         # Main example: sets up, runs, and verifies 2D pooling
include/ck/tensor_operation/gpu/device/
│   └── device_pool_fwd.hpp       # Device-level pooling API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_pool2d_fwd_nhwc.hpp # NHWC layout optimization
│   └── device_pool2d_fwd_nchw.hpp # NCHW layout optimization
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_pool_fwd.hpp     # Grid-level pooling kernel
include/ck/tensor_operation/gpu/block/
    └── blockwise_pool.hpp        # Block-level pooling
```

### Key Classes and Functions

- **DevicePoolFwd** (in `device_pool_fwd.hpp`):  
  Device API for pooling.
  ```cpp
  template <typename InDataType, typename OutDataType, typename IndexDataType,
            ck::index_t NDimSpatial, typename ReduceOperation,
            typename InElementwiseOperation, typename OutElementwiseOperation>
  struct DevicePoolFwd : public BaseOperator
  ```
- **gridwise_pool_fwd** (in `gridwise_pool_fwd.hpp`):  
  Implements the tiled/blocking pooling kernel.
- **blockwise_pool** (in `blockwise_pool.hpp`):  
  Handles block-level pooling and shared memory.

This example demonstrates how Composable Kernel implements efficient 2D pooling for CNNs and vision models.
