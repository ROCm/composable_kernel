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

```sh
cmake -DCK_EXPERIMENTAL_BUILDER=ON -DCMAKE_CXX_STANDARD=20 ...
```
## Building and testing

During development, build and test from the CK build directory with
```sh
ninja test_conv_builder && bin/test_conv_builder
```
