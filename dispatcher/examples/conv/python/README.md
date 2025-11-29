# Convolution Python Examples

CK Tile Dispatcher Python examples for Convolution operations.

> **Main Documentation**: [Dispatcher README](../../../README.md) | [Examples Overview](../../README.md)

## Quick Start

### Build Library

```bash
cd /path/to/composable_kernel/dispatcher
mkdir -p build && cd build

cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DBUILD_DISPATCHER_EXAMPLES=ON

# Build Python libraries (kernels generated automatically)
make python_libs -j$(nproc)
```

### Run Examples

```bash
cd /path/to/composable_kernel/dispatcher

# Basic forward convolution
python3 examples/conv/python/01_basic_conv.py

# With validation
python3 examples/conv/python/04_conv2d_bwd_data.py --verify
python3 examples/conv/python/05_conv2d_bwd_weight.py --verify
```

## Examples

| Example | Description |
|---------|-------------|
| [01_basic_conv.py](01_basic_conv.py) | Basic 2D forward convolution |
| [02_conv2d_fwd.py](02_conv2d_fwd.py) | 2D forward patterns |
| [03_conv3d_fwd.py](03_conv3d_fwd.py) | 3D forward patterns |
| [04_conv2d_bwd_data.py](04_conv2d_bwd_data.py) | Backward data with validation |
| [05_conv2d_bwd_weight.py](05_conv2d_bwd_weight.py) | Backward weight with validation |
| [06_benchmark.py](06_benchmark.py) | Performance benchmarking |
| [07_validation.py](07_validation.py) | CPU vs GPU validation |
| [08_json_export.py](08_json_export.py) | Registry JSON export |
| [09_multi_registry.py](09_multi_registry.py) | Multiple registries |
| [10_conv3d_forward.py](10_conv3d_forward.py) | 3D conv with GPU |
| [11_bwd_data.py](11_bwd_data.py) | Backward data API |
| [12_bwd_weight.py](12_bwd_weight.py) | Backward weight API |

## Example Details

### 01_basic_conv.py - Basic Convolution
Complete example with GPU execution:

```python
from conv_utils import (
    ConvSignature, ConvAlgorithm, ArchInfo,
    ConvKernelSet, ConvProblem, GpuConvRunner
)

# Define kernel
sig = ConvSignature()
sig.dtype("fp16")
sig.layout = "nhwc"
sig.direction = "forward"
sig.num_dims = 2

algo = ConvAlgorithm()
algo.tile(1, 128, 128)
algo.pipeline = "compv3"

kernel_set = ConvKernelSet("basic_conv")
kernel_set.add(sig, algo, ArchInfo(name="gfx942"))

# Run on GPU
runner = GpuConvRunner()
result = runner.run(input_data, weight_data, problem)
print(f"Time: {result['time_ms']:.2f} ms, TFLOPS: {result['tflops']:.2f}")
```

### 02_conv2d_fwd.py - 2D Forward Patterns
Various 2D convolution configurations:
- Standard convolution
- Strided convolution
- Dilated convolution
- Depthwise convolution

### 03_conv3d_fwd.py - 3D Forward Patterns
3D convolution patterns for:
- Video processing
- Volumetric data
- Point clouds

### 04_conv2d_bwd_data.py - Backward Data
Backward data gradient with CPU validation:
- dL/dInput computation
- Use `--verify` flag to compare with CPU reference

### 05_conv2d_bwd_weight.py - Backward Weight
Backward weight gradient with CPU validation:
- dL/dWeight computation
- Use `--verify` flag to compare with CPU reference

### 06_benchmark.py - Benchmarking
Performance measurement:
- Multiple layer configurations
- TFLOPS reporting

### 07_validation.py - Validation
Correctness verification:
- NumPy reference implementation
- Tolerance checking

### 08_json_export.py - JSON Export
Registry serialization for tool integration.

### 09_multi_registry.py - Multiple Registries
Specialized registries for different workloads.

### 10_conv3d_forward.py - 3D Convolution
Full 3D convolution with GPU execution.

### 11_bwd_data.py & 12_bwd_weight.py - Backward APIs
API demonstrations for backward operations.

## Utility Module: conv_utils.py

```python
from conv_utils import (
    # Kernel specification
    ConvSignature,      # Operation signature
    ConvAlgorithm,      # Algorithm details
    ArchInfo,           # Target GPU
    
    # Kernel management
    ConvKernelConfig,   # Single kernel config
    ConvKernelSet,      # Collection of kernels
    
    # Problem specification
    ConvProblem,        # Convolution problem sizes
    
    # GPU execution
    GpuConvRunner,           # Forward/BwdData runner
    GpuConvBwdWeightRunner,  # BwdWeight runner (separate lib)
)
```

### ConvProblem Class

```python
problem = ConvProblem(
    N=1,             # Batch size
    C=64,            # Input channels
    K=128,           # Output channels
    Hi=28, Wi=28,    # Input spatial size
    Y=3, X=3,        # Filter size
    stride_h=1, stride_w=1,
    pad_h=1, pad_w=1,
    direction="forward"
)

# Properties
print(problem.Ho, problem.Wo)  # Output sizes
print(problem.flops)           # FLOPs
print(problem.is_pointwise())  # 1x1 check
```

## Convolution Types

| Type | Description | Use Case |
|------|-------------|----------|
| Forward | Input × Weight → Output | Inference, forward pass |
| Backward Data | dOutput × Weight → dInput | Backpropagation |
| Backward Weight | Input × dOutput → dWeight | Training, weight update |

## Tensor Layouts

| Layout | Description | Example Shape |
|--------|-------------|---------------|
| NHWC | Batch, Height, Width, Channel | (1, 28, 28, 64) |
| NHWGC | With groups | (1, 28, 28, 1, 64) |
| NDHWC | 3D with depth | (1, 8, 28, 28, 64) |

## Related Documentation

- [C++ Conv Examples](../cpp/README.md)
- [Python GEMM Examples](../../gemm/python/README.md)
- [Main Dispatcher README](../../../README.md)
