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

## Complete Configuration API

### ConvSignature (WHAT operation)

```python
from conv_utils import ConvSignature

sig = ConvSignature()

# Data types (all types can be set independently)
sig.dtype_in = "fp16"      # Input: fp16, bf16, fp32, fp8, int8
sig.dtype_wei = "fp16"     # Weight
sig.dtype_out = "fp16"     # Output
sig.dtype_acc = "fp32"     # Accumulator
sig.dtype_workspace = "fp32"  # Workspace (two-stage algorithms)
sig.dtype_bias = "fp16"    # Bias type (bias epilogue)

# Or set all at once
sig.dtype("fp16")          # Sets in/wei/out to fp16, acc to fp32

# Tensor layout
sig.layout = "nhwc"        # nhwc, nchw, nhwgc (with groups)

# Operation direction
sig.direction = "forward"  # forward, bwd_data, bwd_weight

# Spatial dimensions
sig.num_dims = 2           # 1, 2, or 3

# Groups
sig.groups = 1             # Group convolution

# Filter specialization (for optimized paths)
sig.specialization = "default"  # default, filter_1x1_pad0, filter_3x3
```

### ConvAlgorithm (HOW computed)

```python
from conv_utils import ConvAlgorithm

algo = ConvAlgorithm()

# Block tile dimensions (N=batch, K=output channels, C=input channels)
algo.tile(1, 128, 128)     # tile_n, tile_k, tile_c
# Or:
algo.tile_n = 1            # Batch tile (usually 1)
algo.tile_k = 128          # Output channel tile (K)
algo.tile_c = 128          # Input channel tile (C * filter)

# MNK convention (for unified API, maps to above):
algo.tile_m = 1            # Maps to tile_n

# Wave/warp distribution (number of warps per dimension)
algo.wave(2, 2, 1)         # wave_m, wave_n, wave_k
# Or:
algo.wave_m = 2
algo.wave_n = 2
algo.wave_k = 1

# Warp tile sizes (work per warp)
algo.warp(32, 32, 16)      # warp_m, warp_n, warp_k
# Or:
algo.warp_m = 32
algo.warp_n = 32
algo.warp_k = 16

# Vector sizes for memory access optimization
algo.vector_sizes(4, 8, 8)  # vector_size_a, b, c
# Or:
algo.vector_size_a = 4     # Input tensor
algo.vector_size_b = 8     # Weight tensor
algo.vector_size_c = 8     # Output tensor

# Pipeline and scheduler
algo.pipeline = "compv4"    # mem, compv3, compv4, compv5, compv6
algo.scheduler = "intrawave"  # default, intrawave, interwave
algo.epilogue = "cshuffle"  # cshuffle, default_2d

# Memory operation (for split-k reduction)
algo.memory_op = "set"      # set, atomic_add, atomic_max

# Occupancy hints
algo.block_per_cu = 1       # Blocks per CU
algo.num_wave_groups = 1    # Wave groups (V5 pipeline)
algo.num_groups_to_merge = 1  # Groups to merge optimization

# Double buffering
algo.double_buffer = False  # DoubleSmemBuffer

# Padding flags
algo.pad_m = True
algo.pad_n = True
algo.pad_k = True

# Helper methods
algo.occupancy(block_per_cu=2, num_wave_groups=1)
```

### Supported Data Types

| Type | Description | Accumulator | Architectures |
|------|-------------|-------------|---------------|
| FP32 | 32-bit float | fp32 | All |
| FP16 | 16-bit float (half) | fp32 | All |
| BF16 | 16-bit bfloat | fp32 | gfx90a+ |
| FP8_E4M3 | 8-bit E4M3 float | fp32 | gfx942+ |
| FP8_E5M2 | 8-bit E5M2 float (BF8) | fp32 | gfx942+ |
| INT8 | 8-bit signed integer | int32 | gfx942+ |
| FP4 | 4-bit float (MXFP4) | fp32 | gfx950+ |
| INT4 | 4-bit integer | int32 | gfx950+ |

### Pipeline Versions

| Pipeline | Description | Best For |
|----------|-------------|----------|
| mem | Memory-bound pipeline | Bandwidth-limited workloads |
| compv3 | Compute V3 (intrawave only) | Balanced workloads |
| compv4 | Compute V4 (double buffer, ping-pong LDS) | Large tiles |
| compv5 | Compute V5 (wave groups) | Maximum throughput |

## Full Example

```python
from conv_utils import (
    ConvSignature, ConvAlgorithm, ArchInfo,
    ConvKernelSet, ConvProblem, GpuConvRunner
)

# Define kernel signature (WHAT)
sig = ConvSignature()
sig.dtype("fp16")
sig.layout = "nhwc"
sig.direction = "forward"
sig.num_dims = 2

# Define algorithm (HOW)
algo = ConvAlgorithm()
algo.tile(1, 128, 128)
algo.wave(2, 2, 1)
algo.warp(32, 32, 16)
algo.pipeline = "compv4"
algo.scheduler = "intrawave"
algo.vector_sizes(4, 8, 8)
algo.block_per_cu = 1

# Target architecture (WHERE)
arch = ArchInfo(name="gfx942")

# Create kernel set
kernel_set = ConvKernelSet("my_conv")
kernel_set.add(sig, algo, arch)

# Run on GPU
runner = GpuConvRunner()
result = runner.run(input_data, weight_data, problem)
print(f"Time: {result['time_ms']:.2f} ms, TFLOPS: {result['tflops']:.2f}")
```

## ConvProblem Class

```python
from conv_utils import ConvProblem

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
print(problem.is_depthwise())  # Depthwise check
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

## Advanced Benchmarking

Example 13 demonstrates all benchmark parameters:

```bash
python3 13_advanced_benchmark.py --help

# Benchmark parameters
python3 13_advanced_benchmark.py \
    --warmup 10 \
    --repeat 100 \
    --flush-cache \
    --timer gpu

# Memory-bound analysis
python3 13_advanced_benchmark.py \
    --flush-cache \
    --rotating-count 4 \
    --init constant
```

### GpuConvRunner with Benchmark Settings

```python
from conv_utils import GpuConvRunner

runner = GpuConvRunner(
    warmup=10,           # Warmup iterations
    repeat=100,          # Benchmark iterations
    flush_cache=True,    # Flush L2 cache between iterations
    rotating_count=4,    # Rotating buffers for cache simulation
    timer="gpu",         # Timer type: "gpu" or "cpu"
)

result = runner.run(input_data, weight_data, problem)
print(f"Time: {result['time_ms']:.4f} ms")
```

## Related Documentation

- [C++ Conv Examples](../cpp/README.md)
- [Python GEMM Examples](../../gemm/python/README.md)
- [Main Dispatcher README](../../../README.md)
