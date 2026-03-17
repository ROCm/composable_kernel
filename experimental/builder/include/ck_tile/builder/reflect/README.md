# Convolution Reflection Directory

This directory contains tools for "reflecting" on convolution kernel instances. It allows developers to inspect the compile-time configuration of a kernel and generate detailed, human-readable descriptions.

See the [main builder documentation](../README.md) for an overview.

## Design Overview

The reflection system works by extracting properties from a convolution kernel *type* and formatting them into a string. This is useful for debugging, performance tuning, and generating documentation.

1. **Trait Extraction**: The `ConvTraits` template (in `conv_traits.hpp`) is specialized for each kernel instance. It extracts low-level details like tile sizes, data layouts, and pipeline versions from the kernel's type definition.
This template is common for XDL and WMMA, forward and backward weight kernels. `std::optional` is used for parameters that are only used by some kernels.

2. **Description Generation**: The `describe<Instance>()` function (in `conv_description.hpp`) uses `ConvTraits` to populate a `ConvDescription` (`Description`) object.

3. **Formatting**: The `ConvDescription` class (which implements `Description`) contains methods like `brief()` and `detailed()` that format the extracted properties into well-structured strings for display.

## Key Files

- **`description.hpp`**: The generalized Description base class with no implementation.

- **`conv_description.hpp`**: The main entry point. Contains the `ConvDescription` struct and the `describe()` factory function.
- **`conv_traits.hpp`**: Home of the `ConvTraits` template, which is the core of the property extraction mechanism.
- **`tree_formatter.hpp`**: A tree-building utility that generates indented, tree-like output for the `detailed()` description.

## Usage

To get a description of a convolution kernel instance, use the `describe` function and call one of its formatting methods:

```cpp
#include "ck_tile/builder/reflect/conv_description.hpp"

// Assume MyConvFwdInstance is a type alias for a specific kernel instance
using MyConvFwdInstance = /* ... some kernel type ... */;

// Describe the instance
const auto description = ck_tile::reflect::conv::Describe<MyConvFwdInstance>();

// Print the detailed description
std::cout << description.detailed() << std::endl;
```

## Appendix: Current Limitations

### Supported Instance Types

The reflection system (`ckr::describe`) currently supports the following convolution instance types:

- **Standard XDL Forward Convolution** (`DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle`)
- **Large Tensor XDL Forward Convolution** (`DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor`)
- **V3 XDL Forward Convolution** (`DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3`)
- **WMMA Forward Convolution** (`DeviceGroupedConvFwdMultipleD_Wmma_CShuffle`)
- **XDL Backward Weight Convolution** (`DeviceGroupedConvBwdWeight_Xdl_CShuffle`)
- **V3 XDL Backward Weight Convolution** (`DeviceGroupedConvBwdWeight_Xdl_CShuffleV3`)
- **XDL Multiple D Backward Weight Convolution** (`DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle`)
- **Two Stage XDL Backward Weight Convolution** (`DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle`)
- **V3 Two Stage XDL Backward Weight Convolution** (`DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3`)
- **Wmma Backward Weight Convolution** (`DeviceGroupedConvBwdWeight_Wmma_CShuffle`) 
- **V3 Wmma Backward Weight Convolution** (`DeviceGroupedConvBwdWeight_Wmma_CShuffleV3`)
- **V3 Wmma Multiple D Backward Weight Convolution** (`DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3`)

These variants all share similar template parameter structures and are compatible with the current `ConvTraits` implementation.

#### CK Tile Instance Types

The reflection system also provides `InstanceTraits` specializations for CK Tile kernel instances:

- **Tile Forward Convolution** (`GroupedConvolutionForwardKernel`)
- **Tile Backward Weight Convolution** (`GroupedConvolutionBackwardWeightKernel`)
- **Tile Backward Data Convolution** (`GroupedConvolutionBackwardDataKernel`)
- **Reference Convolution** (reference implementation)

#### Unsupported Instance Types

- **DL (non-XDLops) Forward** (`DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK`) has `InstanceTraits` but uses a different internal parameter structure (`K0PerBlock`, `K1`, `M1PerThread` instead of standard block/warp parameters). It can use `GetInstanceString()` through the base class pointer but cannot use `describe()`.

### Reflection Coverage: ConvTraits Bridge

The reflection system operates at two levels:

1. **`InstanceTraits`** (compile-time): Extracts raw template parameters from a kernel type. Specializations exist for both old CK and CK Tile instances.

2. **`ConvTraits`** (runtime): A unified, type-erased data structure representing kernel configuration in convolution-specific terms. Populated by `instance_to_conv_traits<Instance>()` specializations.

`ConvTraits` captures the common ground shared by both backends: spatial dimensions, tensor layouts, data types, elementwise operations, tile dimensions, pipeline version/scheduler, and memory access patterns. Within old CK, `ConvTraits` already unifies across the MFMA/WMMA instruction set boundary — XDL and WMMA forward instances both produce the same `ConvTraits` representation, demonstrating that instruction-set differences can be abstracted at this level.

Currently, `instance_to_conv_traits()` specializations exist only for old CK instances (forward XDL, XDL V3, WMMA, large tensor, and 8 backward weight variants). CK Tile instances have `InstanceTraits` but lack `instance_to_conv_traits()` specializations — there is no bridge from CK Tile's `InstanceTraits` to the unified `ConvTraits` representation.

This is the critical gap in the reflection system. Today the builder has 16+ per-variant factories, each with its own algorithm descriptor shape. `ConvTraits` is the mechanism for discovering which parameters are genuinely variant-specific versus which can be expressed in a single unified algorithm descriptor. Closing the CK Tile bridge means writing `instance_to_conv_traits()` specializations for the CK Tile kernel types that map their `InstanceTraits` fields to the `ConvTraits` struct. Once this bridge exists, both backends produce the same `ConvTraits` output — making it possible to define a single algorithm descriptor format that the dispatcher decomposes into variant-specific parameters internally.

### Future Work

The priorities for the reflection system are:

1. **CK Tile ConvTraits bridge.** Write `instance_to_conv_traits()` specializations for `GroupedConvolutionForwardKernel`, `GroupedConvolutionBackwardWeightKernel`, and `GroupedConvolutionBackwardDataKernel`. This is the prerequisite for unified algorithm descriptors.

2. **DL variant support.** The DL forward kernel needs a specialized `ConvTraits` mapping due to its different internal parameter structure.

3. **Generalization beyond convolution.** `ConvTraits` is designed to evolve toward a more general `KernelTraits` covering GEMM, flash attention, and other operations.
