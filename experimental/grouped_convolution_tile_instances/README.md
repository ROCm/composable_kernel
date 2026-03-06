# CK Tile Instance Generation and Integration

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Instance Generation Workflow](#instance-generation-workflow)
4. [Configuration Files](#configuration-files)
5. [Python Generation Script](#python-generation-script)
6. [Generated Artifacts](#generated-artifacts)
7. [Integration with CK Profiler](#integration-with-ck-profiler)
8. [Directory Structure](#directory-structure)
9. [Usage](#usage)

---

## Overview

The CK Tile instance generation system provides an automated way to create optimized convolution kernel instances using the **CK Builder** pattern. These instances are:

- **Generated** from configuration files containing instance parameter strings
- **Integrated** with the CK Profiler for benchmarking and validation

### Key Components

1. **CK Builder** (`/projects/composablekernel/experimental/builder`)
   - High-level C++20 interface for constructing composable kernel operations
   - Provides compile-time dispatch from builder descriptors to specialized kernel implementations

2. **Instance Generator** (`/projects/composablekernel/experimental/grouped_convolution_tile_instances`)
   - Python-based code generation system
   - Parses configuration files with instance strings
   - Generates C++ wrapper files using templates

3. **CK Profiler Integration** (`projects/composablekernel/profiler`)
   - Benchmarks generated instances
   - Validates correctness against reference implementations
   - Selects best-performing kernels

---

## Architecture

### CK Builder Design

The CK Builder uses a **builder pattern** that separates:

1. **Signature** - Defines the operation (data type, layout, direction)
2. **Algorithm** - Specifies tile parameters and optimizations
3. **Instance** - The compiled kernel from Builder + Algorithm

```cpp
// Example: Building a convolution instance
using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

auto conv = Instance{};
ckt::RunResult result = ckt::run(conv, args, inputs, outputs, stream_config);
```

### Convolution Signatures

Signatures are compile-time constants that define the operation:

```cpp
constexpr auto SIGNATURE_NHWGC_FP16_FWD = ckt::ConvSignature{
    .spatial_dim            = 2,              // 2D convolution
    .direction              = ckb::ConvDirection::FORWARD,
    .data_type              = ckb::DataType::FP16,
    .accumulation_data_type = ckb::DataType::FP32,
    .input                  = {.config = {.layout = ckb::TensorLayout::NHWGC}},
    .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
    .output                 = {.config = {.layout = ckb::TensorLayout::NHWGK}}
};
```

### Tile Algorithm Configuration

Algorithms specify tile sizes, GEMM parameters, and optimizations:

```cpp
constexpr auto ALGORITHM = cku::ConvAlgorithm_Tile_GroupedConvolutionKernel{}
    .with_tile_specializations(ckb::TileConvSpecialization::DEFAULT)
    .with_tile_thread_block(ckt::TileThreadBlock{
        .tile_size = {.m = 128, .n = 128, .k = 32}
    })
    .with_tile_block_gemm(ckt::TileBlockGemm{
        .warps              = {.m = 2, .n = 2, .k = 1},
        .warp_tile          = {.m = 32, .n = 32, .k = 16},
        .double_smem_buffer = false,
        .num_wave_groups    = 1,
        .pipeline_version   = ckb::PipelineVersion::V1,
        .scheduler          = ckb::PipelineScheduler::INTRAWAVE
    })
    .with_tile_transfer(ckt::TileTransfer{
        .a_scalar_per_vector = 8,
        .b_scalar_per_vector = 8,
        .c_scalar_per_vector = 8
    })
    .with_tile_optimizations(ckt::TileOptimizations{
        .num_groups_to_merge = 1,
        .split_image         = false,
        .explicit_gemm       = false
    });
```

---

## Instance Generation Workflow

### Step 1: Configuration Files

Instance strings are defined in configuration files organized by:
- **Direction**: `forward`, `backward_weight`, `backward_data`
- **Purpose**: `profiler` (all instances), `tests` (limited set), `compilation` (empty)
- **Layout & Data Type**: e.g., `nhwgc_fp16.conf`, `ndhwgc_bf16.conf`

**Location**: `configs/{direction}/{purpose}/{layout_dtype}.conf`

### Step 2: Python Generation

Run `generate_instances.py` to parse configs and generate C++ files:

```bash
python generate_instances.py \
    --mode profiler \
    --direction all \
    --filter_pattern convolution
```

### Step 3: Generated Files

For each instance, the script generates:

1. **Individual C++ files** (one per instance)
   - Location: `instances/{direction}/{config}/{instance_name}.cpp`
   - Contains instance-specific kernel wrapper

2. **Include files** (`.inc` headers)
   - `{problem_name}.inc` - Function declarations
   - `{problem_name}_calls.inc` - Function call invocations

3. **CMake integration** (via `CMakeLists.txt`)
   - Compiles all generated instances
   - Links with profiler

### Step 4: Compilation

CMake compiles the generated instances with:
- GPU-specific optimizations
- Target architecture (e.g., `gfx942`)
- C++20 standard required

### Step 5: Profiler Integration

Generated instances are integrated via include files in profiler headers.

---

## Configuration Files

### Instance String Format

Configuration files contain instance strings that define kernel parameters. The format varies by device operation type.

#### Forward Convolution Example

```
DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 1, 1, 1, 1, 1, 1>
```

**Parameters** (parsed by `parse_fwd_instances`):
1. `block_size` = 64 (total threads per block)
2. `m_per_block` = 64 (M dimension of tile)
3. `n_per_block` = 64 (N dimension of tile)
4. `k_per_block` = 32 (K dimension of tile)
5. `spec` = Default (specialization: Default, Filter1x1Pad0, Filter1x1Stride1Pad0, OddC, Filter3x3)
6. `m_per_xdl` = 32 (M dimension per XDL instruction)
7. `n_per_xdl` = 32 (N dimension per XDL instruction)
8. `m_xdl_per_wave` = 2 (XDL tiles in M per wave)
9. `n_xdl_per_wave` = 2 (XDL tiles in N per wave)
10. `a_scalar_per_vector` = 1 (vectorization for input)
11. `b_scalar_per_vector` = 1 (vectorization for weight)
12. `c_scalar_per_vector` = 1 (vectorization for output)
13-14. Optional pipeline parameters
15. Optional `num_groups_to_merge`

#### Backward Weight Convolution Example (V3 Instance)

```
DeviceGroupedConvBwdWeight_Xdl_CShuffleV3<256, 128, 128, 64, Default, 32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>
```

**Additional Parameters** (V3 instances):
- `BlkGemmPipelineScheduler` - Intrawave or Interwave
- `BlkGemmPipelineVersion` - v1, v2, v3, v4, or v5

### Specializations

- **DEFAULT** - General purpose convolution
- **FILTER_1X1_PAD0** - Optimized for 1x1 filters with no padding
- **FILTER_1X1_STRIDE1_PAD0** - Optimized for 1x1 filters, stride 1, no padding
- **FILTER_3x3** - Optimized for 3x3 filters
- **OddC** - Optimized for odd channel counts

### Pipeline Versions

- **v1** - Basic pipeline
- **v2** - Enhanced pipeline with better scheduling
- **v3** - Advanced pipeline optimizations
- **v4** - Double shared memory buffering
- **v5** - Two wave groups (2x parallelism)

---

## Python Generation Script

### Script: `generate_instances.py`

#### Key Functions

1. **`parse_fwd_instances(instances, problem_name)`**
   - Parses forward convolution instance strings
   - Extracts tile sizes, GEMM parameters, specializations
   - Returns list of `ConvInstanceTemplateParams` objects

2. **`parse_bwd_weight_instances(instances, problem_name)`**
   - Parses backward weight convolution instance strings
   - Handles V1, V3, and TwoStage variants
   - Extracts pipeline scheduler and version parameters

3. **`parse_bwd_data_instances(instances, problem_name)`**
   - Placeholder for backward data parsing (not yet implemented)

4. **`generate_conv_cpp(instances, problem_name, config, direction, signature_name, filter_pattern)`**
   - Generates individual C++ wrapper files from template
   - One file per instance

5. **`generate_defs_inc(instances, problem_name, signature, direction, filter_pattern)`**
   - Generates function declarations (`.inc` file)
   - Used by profiler to call instances

6. **`generate_calls_inc(instances, problem_name, direction, filter_pattern)`**
   - Generates function call statements (`.inc` file)
   - Invokes each instance in profiler benchmark loop

#### Template System

**Template**: `instances/grouped_convolution_tile.cpp.in`

**Placeholders**:
- `gen_signature` → Signature constant name
- `gen_instance_name` → Unique instance function name
- `gen_specialization` → Tile specialization enum
- `gen_thread_block` → Thread block configuration
- `gen_block_gemm_desc` → Block GEMM descriptor
- `gen_block_transfer` → Transfer parameters
- `gen_optimizations` → Optimization settings

**Generated Output**: `instances/{direction}/{config}/{instance_name}.cpp`

#### Command-Line Arguments

```bash
python generate_instances.py \
    --mode {compilation|tests|profiler} \
    --direction {forward|backward_weight|backward_data|all} \
    --filter_pattern {pattern}
```

**Modes**:
- `compilation` - Empty instance list (compile-time check only)
- `tests` - Limited instances for testing
- `profiler` - All instances for benchmarking

---

## Generated Artifacts

### Directory Structure

```
instances/
├── forward/
│   ├── nhwgc_fp16/
│   │   ├── grouped_convolution_forward_tile_nhwgc_fp16_0.cpp
│   │   ├── grouped_convolution_forward_tile_nhwgc_fp16_1.cpp
│   │   └── ...
│   ├── grouped_convolution_forward_tile_nhwgc_fp16.inc
│   └── grouped_convolution_forward_tile_nhwgc_fp16_calls.inc
├── backward_weight/
│   ├── nhwgc_bf16/
│   │   └── ...
│   └── ...
├── instance_includes.inc     # Shared headers and signatures
└── instance_run.inc          # Shared instance execution logic
```

### File Types

1. **Instance Implementation** (`.cpp`)
   ```cpp
   // grouped_convolution_forward_tile_nhwgc_fp16_0.cpp
   #include "../../instance_includes.inc"
   namespace ck_tile::builder::profiling {
       constexpr auto SIGNATURE = SIGNATURE_NHWGC_FP16_FWD;
       std::tuple<bool, float, std::string> run_grouped_convolution_forward_tile_nhwgc_fp16_0(
           const ckt::Args<SIGNATURE>& args,
           const ckt::Inputs<SIGNATURE>& inputs,
           const ckt::Outputs<SIGNATURE>& outputs,
           const ck_tile::stream_config& s_conf)
       {
           constexpr auto ALGORITHM = /* ... */;
           #include "../../instance_run.inc"
       }
   }
   ```

2. **Function Declarations** (`.inc`)
   ```cpp
   // grouped_convolution_forward_tile_nhwgc_fp16.inc
   std::tuple<bool, float, std::string> run_grouped_convolution_forward_tile_nhwgc_fp16_0(...);
   std::tuple<bool, float, std::string> run_grouped_convolution_forward_tile_nhwgc_fp16_1(...);
   // ...
   ```

3. **Function Calls** (`_calls.inc`)
   ```cpp
   // grouped_convolution_forward_tile_nhwgc_fp16_calls.inc
   run_alg(run_grouped_convolution_forward_tile_nhwgc_fp16_0);
   run_alg(run_grouped_convolution_forward_tile_nhwgc_fp16_1);
   // ...
   ```

---

## Integration with CK Profiler

### Profiler Header: `grouped_convolution_forward_tile_algs.hpp`

This file orchestrates the benchmarking of all CK Tile instances.

#### Key Components

1. **Include Generated Instances**
   ```cpp
   #include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_nhwgc_fp32.inc"
   #include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_nhwgc_bf16.inc"
   #include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_nhwgc_fp16.inc"
   // ... more includes
   ```

2. **Benchmark Loop** (`run_grouped_conv_forward_tile_algs`)
   ```cpp
   template <auto SIGNATURE>
   std::tuple<bool, float, std::string> run_grouped_conv_forward_tile_algs(
       const ckt::Args<SIGNATURE>& args,
       const ckt::Inputs<SIGNATURE>& inputs,
       const ckt::Outputs<SIGNATURE>& outputs,
       const ck_tile::stream_config& s_conf)
   {
       float best_avg_time = std::numeric_limits<float>::max();
       std::string best_op_name;
       bool valid = true;

       // Generate reference output
       auto reference = ckt::alloc_outputs(args);
       using ReferenceInstance = /* ... */;
       auto ref_conv = ReferenceInstance{};
       auto ref_result = ckt::run(ref_conv, args, inputs, reference.get());

       // Lambda to run and validate each instance
       auto run_alg = [&](auto&& run_alg_func) {
           auto [is_supported, avg_time, op_name] = run_alg_func(args, inputs, outputs, s_conf);
           if(is_supported) {
               best_avg_time = std::min(best_avg_time, avg_time);
               best_op_name = (best_avg_time < avg_time) ? best_op_name : op_name;
               
               // Validate correctness
               valid = ck_tile::check_err(outputs, reference, rtol, atol);
               
               std::cout << "Perf: " << avg_time << " ms, " << op_name << std::endl;
           }
       };

       // Run all instances based on signature
       if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP16_FWD) {
           #include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_nhwgc_fp16_calls.inc"
       }
       // ... more signature branches

       return std::make_tuple(valid, best_avg_time, best_op_name);
   }
   ```

---

## Directory Structure

```
projects/composablekernel/
├── experimental/
│   ├── builder/                           # CK Builder framework
│   │   ├── include/ck_tile/builder/       # Builder API
│   │   │   ├── conv_builder.hpp           # Main builder interface
│   │   │   ├── factory/                   # Dispatch to kernel implementations
│   │   │   └── reflect/                   # Instance traits and reflection
│   │   ├── test/                          # Builder tests and utilities
│   │   └── README.md                      # Builder documentation
│   │
│   └── grouped_convolution_tile_instances/ # Instance generation system
│       ├── generate_instances.py           # Main generation script
│       ├── CMakeLists.txt                  # Build configuration
│       ├── README.md                       # Brief overview
│       │
│       ├── configs/                        # Configuration files
│       │   ├── forward/
│       │   │   ├── profiler/               # All instances for profiling
│       │   │   │   ├── nhwgc_fp16.conf
│       │   │   │   ├── nhwgc_fp32.conf
│       │   │   │   ├── nhwgc_bf16.conf
│       │   │   │   ├── ndhwgc_fp16.conf
│       │   │   │   ├── ndhwgc_fp32.conf
│       │   │   │   └── ndhwgc_bf16.conf
│       │   │   └── tests/                  # Limited instances for testing
│       │   ├── backward_weight/
│       │   │   └── profiler/
│       │   └── backward_data/
│       │       └── profiler/
│       │
│       └── instances/                      # Generated C++ files
│           ├── instance_includes.inc       # Shared headers and signatures
│           ├── instance_run.inc            # Shared execution logic
│           ├── grouped_convolution_tile.cpp.in  # Template file
│           │
│           ├── forward/                    # Forward instances
│           │   ├── nhwgc_fp16/
│           │   │   ├── grouped_convolution_forward_tile_nhwgc_fp16_0.cpp
│           │   │   ├── grouped_convolution_forward_tile_nhwgc_fp16_1.cpp
│           │   │   └── ...
│           │   ├── grouped_convolution_forward_tile_nhwgc_fp16.inc
│           │   ├── grouped_convolution_forward_tile_nhwgc_fp16_calls.inc
│           │   └── ...
│           │
│           └── backward_weight/            # Backward weight instances
│               └── ...
│
└── profiler/
    ├── include/profiler/
    │   ├── grouped_convolution_forward_tile_algs.hpp  # Profiler integration
    │   └── ...
    └── src/
        └── profile_grouped_conv_fwd.cpp    # Main profiler entry point
```

---

## Usage

### 1: Generate All Instances for Profiling

```bash
cd projects/composablekernel/experimental/grouped_convolution_tile_instances

# Generate all forward, backward_weight, and backward_data instances
python generate_instances.py --mode profiler --direction all
```

**Output**:
- Generates `.cpp` files for all instances
- Creates `.inc` declaration and call files
- Ready to compile with CMake

#### 1.1: Generate Only Forward Instances for Testing

```bash
# Generate limited forward instances from test configs
python generate_instances.py --mode tests --direction forward
```

#### 1.2: Filter Specific Instances

```bash
# Only generate instances matching "fp16"
python generate_instances.py \
    --mode profiler \
    --direction forward \
    --filter_pattern fp16
```

### 2: Compile the Generated Instances

```bash
cd build
cmake -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -D CMAKE_BUILD_TYPE=Release \
      -D GPU_TARGETS="gfx942" \
      -D CK_EXPERIMENTAL_BUILDER=ON \
      -D CMAKE_CXX_STANDARD=20 \
      -G Ninja \
      ..
      
ninja device_grouped_conv_fwd_tile_instances
ninja device_grouped_conv_bwd_weight_tile_instances
```

### 3: Run the Profiler

```bash
# Profile 2D convolution (NHWGC layout, FP16 data type)
./bin/ckProfiler conv fwd \
    1 0 2 \              # data_type=FP16, layout=NHWGC, spatial_dim=2
    1 128 128 64 \       # G=1, N=128, K=128, C=64
    3 3 \                # filter: 3x3
    28 28 \              # input: 28x28
    1 1 \                # stride: 1x1
    1 1 \                # dilation: 1x1
    1 1 1 1 \            # padding: 1,1,1,1
    1 0 1                # verification, initialization, profiling

# Output shows performance of each instance and selects best kernel
```

## Adding a New Instance Configuration

1. **Edit config file**: `configs/forward/profiler/nhwgc_fp16.conf`

   ```
   DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 64, Default, 32, 32, 4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v5>
   ```

2. **Regenerate instances**:
   ```bash
   python generate_instances.py --mode profiler --direction forward --filter_pattern fp16
   ```

3. **Rebuild**:
   ```bash
   ninja grouped_convolution_tile_instances
   ```

4. **Profile**:
   ```bash
   ./bin/ckProfiler conv fwd ...
   ```

---

## References

- [CK Builder README](../builder/README.md)
- [CK Builder Design](../builder/include/ck_tile/builder/README.md)
- [CK Builder Factory](../builder/include/ck_tile/builder/factory/README.md)
- [CK Builder Testing](../builder/include/ck_tile/builder/testing/README.md)

---
