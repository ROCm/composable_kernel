# CK Tile Engine Pool Operations

## Overview

The CK Tile Engine Pool module provides a comprehensive system for generating, building, and benchmarking pooling kernels (2D and 3D) with various configurations. It supports multiple data types, reduce operations (max, min, average), and optimization strategies. The system follows the same architecture as the GEMM module with individual kernel compilation for better build parallelism and targeted testing capabilities.

## Table of Contents

1. [Build System Architecture](#build-system-architecture)
2. [Build Instructions](#build-instructions)
3. [Running Benchmarks](#running-benchmarks)
4. [Configuration System](#configuration-system)
5. [Scripts and Tools](#scripts-and-tools)
6. [Command Line Options](#command-line-options)
7. [Understanding Kernel Names](#understanding-kernel-names)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tips](#performance-tips)

## Build System Architecture

### Individual Kernel Compilation

The tile engine benchmark system compiles each kernel configuration into a separate executable. This provides:
- Better build parallelism
- Faster incremental builds
- More targeted testing
- Easier debugging of specific configurations

Each benchmark executable follows the naming pattern:
```
benchmark_pool<dim>d_<dtype>_<reduce_op>_<output_index>_<propagate_nan>_<block_config>
```

## Build Instructions

### Prerequisites
- ROCm installation
- CMake 3.16 or higher
- C++17 compatible compiler
- Python 3.6 or higher

### Basic Build

```bash
# In the root of composable kernel, create build directory
mkdir build && cd build

# Configure with specific datatypes and reduce operations
# Replace [Arch] with your GPU architecture (e.g., gfx90a, gfx942)
../script/cmake-ck-dev.sh ../ [Arch] -DPOOL_DATATYPE="fp16;fp32" -DPOOL_REDUCE_OP="max;avg"

# Build specific benchmarks
make benchmark_pool_fp16_max -j
```

### Configuration Options

The build system supports several configuration options:

#### Using Custom Config Files
```bash
# Method 1: CMake variable (config file must be in configs/ directory)
cmake -DPOOL_CONFIG_FILE=my_custom_config.json ...

# Method 2: Environment variable (takes precedence over CMake variable)
export POOL_CONFIG_FILE=my_custom_config.json
cmake ...
```

#### Config File Priority Order
1. **Environment variable** `POOL_CONFIG_FILE` (highest priority)
2. **CMake variable** `POOL_CONFIG_FILE`
3. **Default config** (default_config.json)

**Note**: All custom config files must be placed in the `tile_engine/ops/pooling/configs/` directory.

### Example Build Commands

```bash
# Build for gfx942 with fp16 datatype, max reduce operation
mkdir build && cd build
../script/cmake-ck-dev.sh ../ gfx942 -DPOOL_DATATYPE="fp16;fp32" -DPOOL_REDUCE_OP="max;avg"
make benchmark_pool_fp16_max -j
make benchmark_pool_fp32_avg -j
```

### Building Individual Kernels

```bash
# Build a specific kernel configuration
make benchmark_pool3d_fp16_max_True_False_128x1_1x1_2x1

# Build all fp16 max pooling benchmarks
make benchmark_pool_fp16_max -j$(nproc)

# Build all 3D pooling benchmarks
make benchmark_pool3d -j$(nproc)
```

### Rebuilding After Configuration Changes

If you modify the configuration file, you must rebuild:
```bash
rm -rf tile_engine/ && make benchmark_pool_[Datatype]_[ReduceOp] -j
```

## Running Benchmarks

### Individual Kernel Execution

```bash
cd /path/to/build/directory
./bin/benchmark_pool3d_fp16_max_True_False_128x1_1x1_2x1 \
    -N=2 -D=30 -H=30 -W=30 -C=32 \
    -Z=2 -Y=2 -X=2 \
    -Sz=2 -Sy=2 -Sx=2 \
    -verify=1
```

### Using the Benchmark Python Script

```bash
# Run benchmark sweep
python pool_benchmark.py /path/to/build \
    --problem-sizes "2,30,30,30,32" "4,64,64,64,64" \
    --window-sizes "2,2,2" "3,3,3" \
    --stride-sizes "2,2,2" \
    --pool-dim 3 \
    --verify \
    --json results.json
```

## Configuration System

### Configuration Files

The system uses JSON configuration files to specify kernel parameters:

- `configs/default_config.json` - Default configurations

### Configuration Structure

```json
{
    "block_config": {
        "block_m": {"values": [64, 128, 256]},
        "block_n": {"values": [1]},
        "warp_m": {"values": [1, 2]},
        "warp_n": {"values": [1]},
        "thread_tile_m": {"values": [1, 2, 4]},
        "thread_tile_n": {"values": [1]}
    },
    "trait_config": {
        "output_index": {"values": [true, false]},
        "propagate_nan": {"values": [false]},
        "pool_dim": {"values": [2, 3]}
    },
    "k_block_per_cu": 1
}
```

### Configuration Parameters

- **block_m/block_n**: Block tile dimensions for output
- **warp_m/warp_n**: Number of warps per block
- **thread_tile_m/thread_tile_n**: Thread tile sizes
- **output_index**: Whether to output indices (for max/min pooling)
- **propagate_nan**: Whether to propagate NaN values
- **pool_dim**: Pooling dimension (2 for 2D, 3 for 3D)

## Scripts and Tools

### Python Scripts

#### pool_instance_builder.py
**Purpose**: Main kernel instance generation script that creates C++ kernel implementations based on configuration files.

**Key Features**:
- Generates individual kernel header files for separate compilation
- Supports multiple data types (fp16, fp32, bf16)
- Validates block configurations for correctness
- Creates CMake integration files

**Usage**:
```bash
python pool_instance_builder.py \
    --working_path ./generated \
    --datatype fp16 \
    --reduce_op max \
    --config_json configs/default_config.json \
    --gen_all_individual \
    --gpu_target gfx942
```

#### pool_benchmark.py
**Purpose**: Python script for running and analyzing pool benchmarks.

**Features**:
- Automated benchmark execution
- Performance data collection
- Result analysis and reporting
- CSV and JSON export

**Usage**:
```bash
python pool_benchmark.py /path/to/build \
    --problem-sizes "2,30,30,30,32" \
    --window-sizes "2,2,2" \
    --verbose \
    --json results.json
```

## Command Line Options

All benchmark executables support the following options:

### Tensor Dimensions
- `-N=<value>` - Batch size (default: 2)
- `-D=<value>` - Depth dimension for 3D pooling (default: 30)
- `-H=<value>` - Height dimension (default: 30)
- `-W=<value>` - Width dimension (default: 30)
- `-C=<value>` - Channel dimension (default: 32)

### Window Parameters
- `-Z=<value>` - Window depth (default: 2)
- `-Y=<value>` - Window height (default: 2)
- `-X=<value>` - Window width (default: 2)

### Stride Parameters
- `-Sz=<value>` - Stride depth (default: 2)
- `-Sy=<value>` - Stride height (default: 2)
- `-Sx=<value>` - Stride width (default: 2)

### Dilation Parameters
- `-Dz=<value>` - Dilation depth (default: 1)
- `-Dy=<value>` - Dilation height (default: 1)
- `-Dx=<value>` - Dilation width (default: 1)

### Padding Parameters
- `-LeftPz=<value>` - Left padding depth (default: 0)
- `-LeftPy=<value>` - Left padding height (default: 0)
- `-LeftPx=<value>` - Left padding width (default: 0)
- `-RightPz=<value>` - Right padding depth (default: 0)
- `-RightPy=<value>` - Right padding height (default: 0)
- `-RightPx=<value>` - Right padding width (default: 0)

### Pool Dimension
- `-pool_dim=<2|3>` - Pooling dimension (default: 3)

### Verification
- `-verify=<0|1>` - Verification mode
  - 0: No verification
  - 1: CPU verification (default)

### Performance Testing
- `-warmup=<value>` - Warmup iterations (default: 20)
- `-repeat=<value>` - Benchmark iterations (default: 100)
- `-timer=<true|false>` - Use GPU timer (default: true)
- `-flush_cache=<true|false>` - Flush cache between runs (default: true)
- `-rotating_count=<value>` - Cache rotation count (default: 1000)

### Initialization
- `-init=<0|1|2>` - Tensor initialization method
  - 0: Random values [-5, 5] (default)
  - 1: Linear sequence
  - 2: Constant value (1.0)

### Output Options
- `-log=<true|false>` - Enable verbose logging (default: false)
- `-metric=<0|1|2>` - Performance metric
  - 0: Latency in ms
  - 1: TFLOPS
  - 2: Bandwidth in GB/s (default)
- `-json_output=<true|false>` - JSON format output (default: false)
- `-csv_filename=<filename>` - Save results to CSV

## Understanding Kernel Names

The kernel naming convention encodes the configuration:

```
benchmark_pool3d_fp16_max_True_False_128x1_1x1_2x1
              ^^^^ ^^^^ ^^^ ^^^^ ^^^^^ ^^^^^ ^^^ ^^^
              |    |    |   |    |     |     |   |
              |    |    |   |    |     |     |   Thread tile (MxN)
              |    |    |   |    |     |     Warp config (MxN)
              |    |    |   |    |     Block tile (MxN)
              |    |    |   |    Propagate NaN
              |    |    |   Output Index
              |    |    Reduce operation
              |    Data type
              Pool dimension (2D or 3D)
```

### Components:
- **Pool dimension**: 2d, 3d
- **Data type**: fp16, fp32, bf16
- **Reduce op**: max, min, avg
- **Output Index**: True/False (whether to output argmax/argmin)
- **Propagate NaN**: True/False
- **Block config**: Block_MxBlock_N_Warp_MxWarp_N_ThreadTile_MxThreadTile_N

## Troubleshooting

### Common Issues

1. **Kernel not found**
   - Ensure the specific benchmark executable is built
   - Check the build directory bin/ folder

2. **Verification failures**
   - Check tensor dimensions are valid for the window/stride configuration
   - Verify padding values are reasonable

3. **Build failures**
   - Check GPU architecture compatibility
   - Ensure ROCm is properly installed
   - Verify configuration file syntax

4. **Performance variations**
   - Increase warmup iterations
   - Disable CPU frequency scaling
   - Use GPU timer for accurate measurements

### Debug Options

Enable verbose logging:
```bash
./bin/benchmark_pool... -log=true -verify=1
```

## Performance Tips

1. **Optimal Problem Sizes**: Use sizes that are multiples of block dimensions
2. **Warmup**: Use at least 20-50 warmup iterations
3. **GPU Timer**: Always use `-timer=true` for accurate measurements
4. **Cache Management**: Enable cache flushing for consistent results
5. **Output Index**: Disable output index if not needed (reduces memory bandwidth)

## Integration Examples

### Python Integration

```python
import subprocess
import json

# Run benchmark with JSON output
result = subprocess.run([
    './bin/benchmark_pool3d_fp16_max_...', 
    '-N=2', '-D=30', '-H=30', '-W=30', '-C=32',
    '-json_output=true'
], capture_output=True, text=True)

# Parse results
data = json.loads(result.stdout)
print(f"Bandwidth: {data['bandwidth_gb_s']} GB/s")
```

### Batch Testing Script

```bash
#!/bin/bash
SIZES="32 64 128 256"
for size in $SIZES; do
    echo "Testing HxW=${size}x${size}"
    ./bin/benchmark_pool... -H=$size -W=$size \
        -verify=1 -csv_filename=results.csv
done
```

## Contributing

When adding new features or configurations:
1. Update the instance builder (`pool_instance_builder.py`)
2. Update configuration examples in `configs/`
3. Document new command-line options in this README
4. Add appropriate tests

For more information about the Composable Kernel project, visit the main repository documentation.

