# Convolution Reflection Directory

This directory contains tools for "reflecting" on convolution kernel instances. It allows developers to inspect the compile-time configuration of a kernel and generate detailed, human-readable descriptions.

See the [main builder documentation](../README.md) for an overview.

## Design Overview

The reflection system works by extracting properties from a convolution kernel *type* and formatting them into a string. This is useful for debugging, performance tuning, and generating documentation.

1. **Trait Extraction**: The `ConvTraits` template (in `conv_traits.hpp`) is specialized for each kernel instance. It extracts low-level details like tile sizes, data layouts, and pipeline versions from the kernel's type definition.

2. **Description Generation**: The `Describe<Instance>()` function (in `conv_description.hpp`) uses `ConvTraits` to populate a `ConvDescription` struct.

3. **Formatting**: The `ConvDescription` struct contains methods like `brief()` and `detailed()` that format the extracted properties into well-structured strings for display.

## Key Files

- **`conv_description.hpp`**: The main entry point. Contains the `ConvDescription` struct and the `Describe()` factory function.
- **`conv_traits.hpp`**: Home of the `ConvTraits` template, which is the core of the property extraction mechanism.
- **`tree_formatter.hpp`**: A simple utility for generating the indented, tree-like format used in the `detailed()` description.

## Usage

To get a description of a convolution kernel instance, use the `Describe` function and call one of its formatting methods:

```cpp
#include "ck_tile/builder/reflect/conv_description.hpp"

// Assume MyConvFwdInstance is a type alias for a specific kernel instance
using MyConvFwdInstance = /* ... some kernel type ... */;

// Describe the instance
const auto description = ck_tile::reflect::conv::Describe<MyConvFwdInstance>();

// Print the detailed description
std::cout << description.detailed() << std::endl;
```
