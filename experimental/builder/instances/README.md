# ConvBuilder: Automated Convolution Kernel Instantiation

This directory contains the experimental ConvBuilder system for generating convolution kernel instances from structured JSON descriptions.

## Overview

The ConvBuilder provides a declarative way to instantiate all 753 forward convolution kernels from the legacy device operation templates by:
1. Reading structured JSON descriptions of device op instantiations
2. Generating compile-time C++ ConvBuilder instantiations
3. Building kernels through the ConvBuilder factory system

## Directory Structure

```
experimental/builder/
├── include/ck_tile/builder/                # ConvBuilder headers
│   ├── conv_builder.hpp                    # Main builder interface
│   ├── conv_factory.hpp                    # Factory for creating kernel instances
│   ├── conv_signature_concepts.hpp         # Signature concepts and types
│   └── conv_algorithm_concepts.hpp         # Algorithm concepts and types
├── test/impl/                              # Concrete types implementing concepts
│   ├── conv_signature_types.hpp            # ConvSignature struct
│   └── conv_algorithm_types.hpp            # Algorithm structs (V3, Standard XDL, WMMA)
├── instances/                              # JSON instantiation database
│   └── forward_conv_structured_instantiations.json                           
│   └── generate_conv_builder_instances.py  # Code generation scripts
└── codegen/                                # Generated C++ files (created by script)
    ├── CMakeLists.txt
    └── conv_instances_batch_*.cpp
```

## Strategy

### JSON-to-C++ Mapping

The JSON file contains convolution instantiations, each with:
- **Signature**: Mathematical interface (spatial dim, direction, layouts, data types, elementwise op)
- **Algorithm**: Implementation parameters (block sizes, tiling, pipeline configuration)

### Device Operation Types

Three distinct device operation types are supported:

1. **DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3** 
   - V3 pipeline with `block_gemm` configuration

2. **DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle**
   - Standard XDL pipeline
   - Uses: `num_gemm_k_prefetch_stages`, `loop_scheduler`, `num_groups_to_merge`

3. **DeviceGroupedConvFwdMultipleD_Wmma_CShuffle**
   - WMMA-based implementation
   - Uses `gridwise_wmma_gemm` instead of `gridwise_xdl_gemm`
   - Uses: `num_gemm_k_prefetch_stages`, `loop_scheduler`

### Code Generation Approach

The Python script (`generate_conv_builder_instances.py`):
1. Parses the JSON file
2. Batches instantiations into multiple C++ files (configurable batch size)
3. Generates constexpr `ConvSignature` and algorithm structs
4. Creates type aliases for `ConvBuilder<sig, algo>` instantiations
5. Generates CMakeLists.txt for building

## Usage

### Generate C++ Files

```bash
# Generate with default settings (batch size=50)
python3 experimental/builder/scripts/generate_conv_builder_instances.py --cmake

# Custom batch size and output directory
python3 experimental/builder/scripts/generate_conv_builder_instances.py \
    --output my_generated_dir \
    --batch-size 100 \
    --cmake

# Show help
python3 experimental/builder/scripts/generate_conv_builder_instances.py --help
```

### Build Generated Instances

```bash
# Build the generated instances library
make ckb_instances
```

### Use in Your Code

```cpp
#include "experimental/builder/codegen/conv_instances_batch_00.cpp"

using namespace ck_tile::builder::generated::batch_0;

// Access a specific kernel instance
using MyKernel = Instance_0;

// The Instance type provides:
// - RunKernel() method
// - MakeArgument() method  
// - GetDeviceKernelInfo() method
```

## Architecture Details

### C++20 Concepts

The system uses C++20 concepts for compile-time validation:

**Signature Concepts** (`conv_signature_concepts.hpp`):
- `ConvSignatureDescriptor<T>`: Validates signature structure
- `ValidConvSignature<auto Sig>`: Validates signature values
- `ConvDirectionIsForward<auto Sig>`: Direction predicates

**Algorithm Concepts** (`conv_algorithm_concepts.hpp`):
- `ThreadBlockDescriptor<T>`: Thread block configuration
- `GridwiseXdlGemmDescriptor<T>`: XDL GEMM parameters
- `GridwiseWmmaGemmDescriptor<T>`: WMMA GEMM parameters  
- `BlockTransferDescriptor<T>`: Data transfer configuration
- `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<T>`: V3 algorithm concept
- `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<T>`: Standard XDL concept
- `DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<T>`: WMMA concept


## Future Enhancements

- Generate backward data/weight convolution instances
- Add filtering/selection mechanisms to build only needed kernels
- Runtime kernel selection based on signature matching
- Integration with profiler for automatic tuning
- Support for dynamic instantiation patterns

## Development

To add a new instantiation:
1. Add entry to `instances/forward_conv_structured_instantiations.json`
2. Run the generation script
3. Rebuild

To modify the generator:
1. Edit `instances/generate_conv_builder_instances.py`
2. Test with `--batch-size 5` for quick iteration
3. Regenerate and verify build

## License

Copyright (C) Advanced Micro Devices, Inc., or its affiliates.  
SPDX-License-Identifier: MIT
