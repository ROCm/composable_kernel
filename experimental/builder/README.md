# Builder

This directory contains the experimental builder feature for composable_kernel.

* Status: In development (October - November 2025)

## Overview

The builder provides a high-level, semantically-clear interface for constructing composable kernel operations, with an initial focus on convolution kernels for MIOpen. It leverages modern C++20 features (such as POD structs as non-type template parameters, concepts, and designated initializers) to simplify kernel instantiation and improve developer experience.

This project is a prototype for a more general builder pattern for all of composable_kernel (CK) and CKTile, but is currently limited to formalizing the interface between MIOpen and CK.

## Directory Structure

- `include/ck_tile/builder/`  
  Core builder headers and public API.
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
  -D GPU_TARGETS="gfx942;gfx950"                                                                  \
  -D CK_EXPERIMENTAL_BUILDER=ON                                                                   \
  -D CMAKE_CXX_STANDARD=20                                                                        \
  -G Ninja                                                                                        \
  ..
```

## Building and testing

During development, all CK Builder tests can be built with command

```sh
ninja test_ckb_all
```

To execute all tests, run

```sh
ls bin/test_ckb_* | xargs -n1 sh -c
```

Some tests involve building old CK convolution factories, which will take a long time.
Hence, one might want to build only single test targets. For example

```sh
ninja test_ckb_conv_builder && bin/test_ckb_conv_builder
```

When adding new tests, please follow the convention where the CMake build target starts with a prefix `test_ckb`.
This allows us to filter out the CK Builder tests from the set full CK repository tests.
Also, the `test_ckb_all` target that builds all CK Builder tests relies on having the `test_ckb` prefix on the CMake build targets.
