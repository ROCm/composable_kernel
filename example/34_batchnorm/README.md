# Batch Normalization Forward

## Theory

This example demonstrates **batch normalization forward pass**. Batch normalization is used in deep neural networks to normalize activations across the batch dimension, improving training stability and convergence.

**Mathematical Formulation:**
Given input $X[N, C, ...]$:
- Mean: $\mu_c = \frac{1}{N \cdot ...} \sum_{n,...} X_{n,c,...}$
- Variance: $\sigma^2_c = \frac{1}{N \cdot ...} \sum_{n,...} (X_{n,c,...} - \mu_c)^2$
- Normalized: $\hat{X}_{n,c,...} = \frac{X_{n,c,...} - \mu_c}{\sqrt{\sigma^2_c + \epsilon}}$
- Output: $Y_{n,c,...} = \gamma_c \hat{X}_{n,c,...} + \beta_c$

$\gamma_c$, $\beta_c$ are learnable scale and shift parameters per channel.

**Algorithmic Background:**
- Computes mean and variance per channel (across batch and spatial dimensions).
- Applies normalization and affine transformation.
- Used in CNNs, MLPs, and other deep learning models.

## How to Run

### Prerequisites
```bash
cd composable_kernel/build
make -j install
```

### Build and Execute
```bash
cd composable_kernel/example/34_batchnorm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./batchnorm_fwd_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/34_batchnorm/
├── batchnorm_fwd_xdl.cpp         # Main example: sets up, runs, and verifies batchnorm
include/ck/tensor_operation/gpu/device/
│   └── device_batchnorm_fwd.hpp       # Device-level batchnorm API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_batchnorm_fwd_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
    └── gridwise_batchnorm_fwd.hpp     # Grid-level kernel
```

### Key Classes and Functions

- **DeviceBatchnormFwd** (in `device_batchnorm_fwd.hpp`):  
  Device API for batch normalization.
  ```cpp
  template <typename XDataType, typename GammaBetaDataType, typename YDataType,
            typename MeanVarDataType, typename XElementwiseOperation,
            typename YElementwiseOperation>
  struct DeviceBatchnormFwd : public BaseOperator
  ```
- **gridwise_batchnorm_fwd** (in `gridwise_batchnorm_fwd.hpp`):  
  Implements the tiled/blocking batchnorm kernel.

This example demonstrates how Composable Kernel implements efficient batch normalization for deep learning models.
