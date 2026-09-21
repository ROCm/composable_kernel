# Builder

This directory contains the experimental builder feature for composable_kernel.

* Status: In development (October 2025 - March 2026)

## Overview

The builder provides a high-level, semantically-clear interface for constructing composable kernel operations, with an initial focus on convolution kernels for MIOpen. It leverages modern C++20 features (such as POD structs as non-type template parameters, concepts, and designated initializers) to simplify kernel instantiation and improve developer experience.

This project is a prototype for a more general builder pattern for all of composable_kernel (CK) and CK Tile, but is currently limited to formalizing the interface between MIOpen and CK.

## Design Direction

The builder's primary goal is transparent dispatch across two backend implementations: old CK (template-heavy device operations) and CK Tile (modern tile-based API). MIOpen, the consumer library, should construct kernels through the builder without needing to know which backend provides the implementation.

**Current state:** The builder dispatches correctly, but each kernel variant (forward XDL, forward WMMA, backward weight XDL V3, etc.) has its own factory and its own algorithm descriptor shape. The result is 16+ per-variant facades rather than one unified facade. Unification across three axes — CK vs CK Tile backend, MFMA vs WMMA instruction set, and forward vs backward direction — is the central design challenge.

Three principles guide the design toward that unification:

1. **Unified vocabulary through reflection.** The reflection system (`reflect/`) extracts kernel traits from both backends into a common `ConvTraits` representation. This shared vocabulary is the mechanism for discovering what algorithm parameters are truly variant-specific versus what can be expressed once and mapped to multiple backends.

2. **Expert overrides.** Power users can pin to a specific backend or device operation when needed, bypassing automatic dispatch.

3. **Versioned API evolution.** The builder uses semantic version strings (`"0.0.0"`, `"0.1.0"`) to manage API changes predictably. The `ConvBuilder` template defaults to the latest version but accepts explicit version pinning.

## Design descriptions

- [CK Builder design description](include/ck_tile/builder/README.md)
- [CK Builder factory design](include/ck_tile/builder/factory/README.md)
- [CK Builder testing design](include/ck_tile/builder/testing/README.md)
- [CK Builder reflection design](include/ck_tile/builder/reflect/README.md)

## Directory Structure

- `include/ck_tile/builder/`
  Core builder headers and public API.
- `include/ck_tile/builder/reflect`
  Reflection mechanism.
- `include/ck_tile/builder/factory`
  Compile-time dispatch from builder descriptors to our existing specialized convolution kernel implementations.
- `test/`
  Unit tests and example usage of the builder pattern.
- `CMakeLists.txt`
  CMake configuration for building the experimental builder and its tests.

## CMake Configuration

To enable the experimental builder, configure your build with:

```bash
cmake                                                                                             \
  -D CMAKE_PREFIX_PATH=/opt/rocm                                                                  \
  -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                       \
  -D CMAKE_BUILD_TYPE=Release                                                                     \
  -D GPU_TARGETS="gfx942"                                                                         \
  -D CK_EXPERIMENTAL_BUILDER=ON                                                                   \
  -D CMAKE_CXX_STANDARD=20                                                                        \
  -G Ninja                                                                                        \
  ..
```

Note: The tests for WMMA builders are only built when `CK_USE_WMMA` is enabled. Add e.g. 
`gfx1121` or any of the other `gfx11`/`gfx12` architectures to the GPU targets. Alternatively, 
one can add flag `-D CK_USE_WMMA=ON` to build the tests. For the end-to-end tests that use 
the instances from builder, one needs an actual Navi card.

## Building and Testing

The builder test suite is organized into two main categories:

### Smoke Tests (Fast Unit Tests)
Quick unit tests that verify the builder's internal logic without compiling GPU kernels. These complete in under 1 second total and are suitable for frequent execution during development.

```sh
ninja smoke-builder
```

### Regression Tests (Integration Tests)
Integration tests that compile actual GPU kernels to verify that the builder generates valid, compilable code. These are more expensive than smoke tests (can take minutes to compile) but cover more functionality.

```sh
ninja regression-builder
```

### Running All Tests
To build and run the complete test suite:

```sh
ninja check-builder
```

### Building Individual Tests
To build and run a specific test:

```sh
ninja test_ckb_conv_builder && bin/test_ckb_conv_builder
```

### Test Organization
- **Smoke tests**: Fast feedback during active development
- **Regression tests**: Thorough validation before submitting changes
- **Factory tests**: Expensive tests that build all MIOpen kernels (included in regression tests)

When adding new tests, please follow the convention where the CMake build target starts with a prefix `test_ckb`. This allows filtering of CK Builder tests from the full CK repository test suite.
