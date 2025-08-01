# Experimental GEMM Example

This directory contains an experimental project to build a C++20 GEMM builder over CK Tile.

* **Current state**: Early stages of development.
* **Future work**
  * Fake implemenation as a fascade over CK Tile universal GEMM.
  * Add a `Describe(gemm)` function.
  * Build out flexibilty.
  * Experiment with some more complex build functionality.
  * Explore design to describe more GEMM details.

## File Overview

* `gemm_example.cpp` — Main example code, including device memory helpers and GEMM builder usage
* `builder.h` — C++20 concepts and builder pattern for GEMM configuration
* `CMakeLists.txt` — Build configuration for CMake
