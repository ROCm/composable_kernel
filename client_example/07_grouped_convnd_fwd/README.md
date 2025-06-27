# Client Example: Grouped N-Dimensional Convolution Forward

## Theory

This client example demonstrates **grouped N-dimensional convolution forward** for 1D, 2D, and 3D inputs, supporting multiple data types (including BF8 and FP8). Grouped convolution is used in modern CNNs and vision transformers to reduce computation and enable channel-wise or expert-wise processing.

**Mathematical Formulation:**
Given input $X$ and weights $W$ for $G$ groups:
- For each group $g$:
  $$
  Y^g[n, c_{out}, ...] = \sum_{c_{in}} \sum_{k_1} ... \sum_{k_n} X^g[n, c_{in}, ...] \cdot W^g[c_{out}, c_{in}, ...]
  $$
- Each group operates on a subset of input/output channels.

**Algorithmic Background:**
- Grouped convolution splits the input and weights into $G$ groups and applies convolution independently to each group.
- Supports 1D, 2D, and 3D convolutions, and multiple data layouts (e.g., NCHW, NGCHW).
- Used for efficient CNNs, depthwise separable convolutions, and expert models.

## How to Run

### Prerequisites
```bash
cd composable_kernel/build
make -j install
```

### Build and Execute
```bash
cd composable_kernel/client_example/07_grouped_convnd_fwd
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (2D grouped convolution)
./grouped_conv2d_fwd

# Example run (3D grouped convolution, BF8)
./grouped_conv3d_fwd_bf8

# Example run (3D grouped convolution, FP8)
./grouped_conv3d_fwd_fp8
```

## Source Code Structure

### Directory Layout
```
client_example/07_grouped_convnd_fwd/
├── grouped_conv1d_fwd.cpp         # 1D grouped convolution
├── grouped_conv2d_fwd.cpp         # 2D grouped convolution (NCHW)
├── grouped_conv2d_fwd_ngchw.cpp   # 2D grouped convolution (NGCHW)
├── grouped_conv3d_fwd_bf8.cpp     # 3D grouped convolution (BF8)
├── grouped_conv3d_fwd_fp8.cpp     # 3D grouped convolution (FP8)
├── grouped_conv3d_fwd_bf8_fp8.cpp # 3D grouped convolution (BF8/FP8 mixed)
├── grouped_conv3d_fwd_fp8_bf8.cpp # 3D grouped convolution (FP8/BF8 mixed)
├── common.hpp                     # Common utilities for grouped convolution
├── CMakeLists.txt                 # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input tensors, configures grouped convolution parameters, launches the kernel, and verifies the result.
- **Grouped convolution kernel invocation**:  
  Uses the Composable Kernel device API to launch grouped convolution for different dimensions and data types.

This client example provides a comprehensive demonstration of grouped convolution for efficient CNN and vision transformer models.
