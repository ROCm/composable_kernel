# FLATMM Matrix Multiplication with CK Tile

This example demonstrates FLATMM (flattened matrix multiplication) using the CK Tile programming model. FLATMM is a variant of GEMM optimized for certain memory layouts and batch processing patterns. Currently, it only supports the basic feature of the CK Tile FLATMM, but creates the placeholders for the future support on different FLATMM pipeline and different FLATMM modules. In the near future, we will gradually migrate all the FLATMM features from old CK to CK Tile.

---

## Algorithm and Math

Given:
- $A$: $[\text{batch}, M, K]$
- $B$: $[\text{batch}, K, N]$
- $C$: $[\text{batch}, M, N]$

For each batch $b$:
$$
C^{(b)} = A^{(b)} \times B^{(b)}
$$

- **Tilewise FLATMM**: Each thread block processes a tile of $C$ for a specific batch, loading corresponding tiles from $A$ and $B$, performing blockwise matrix multiply-accumulate, and writing results. FLATMM may use flattened or packed memory layouts for improved memory access.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of $C$ for a given batch.
- **Tile Engine**: Handles loading tiles, performing GEMM in registers, and storing results.
- **Pipeline**: Modular, supports different memory/computation pipelines and flat/padded layouts.

---

## Features

- **Flexible Layouts**: Supports row/column-major and custom strides for $A$, $B$, $C$.
- **Batching**: Efficiently computes multiple GEMMs in parallel.
- **Precision**: Supports fp16, bf16, fp8, bf8.
- **Validation**: CPU/GPU validation and error tolerance options.

---

## Build & Run

```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
# The basic pipeline method on the flatmm calculation
make tile_example_flatmm_basic -j
```
This will result in an executable `build/bin/tile_example_flatmm_basic`

### Arguments
```
args:
          -b    batch size (default:1)
          -m    m dimension (default:1024)
          -n    n dimension (default:2048)
          -k    k dimension (default:64)
   -a_layout    Tensor A data layout (default: R)
   -b_layout    Tensor B data layout (default: R)
   -c_layout    Tensor C data layout (default: R)
   -stride_a    Tensor A stride (default:0)
   -stride_b    Tensor B stride (default:0)
   -stride_c    Tensor C stride (default:0)
          -v    0. No validation, 1. Validation on CPU, 2. Validation on GPU (default:2)
          -e    Absolute error tolerance (default:1e-5)
       -prec    data type. fp16/bf16/fp8/bf8 (default:fp16)
     -warmup    number of iterations before benchmark the kernel (default:10)
     -repeat    number of iterations to benchmark the kernel (default:100)
      -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
```

---

## Source Structure

- **Kernel**: [`flatmm_basic.hpp`](flatmm_basic.hpp) (tile-programming kernel template)
- **Executable**: [`flatmm_basic.cpp`](flatmm_basic.cpp)
- **Build**: `CMakeLists.txt`, `run_flatmm_example.inc`, `script/`

---

## Related CK Tile Examples

- [16_batched_gemm](../16_batched_gemm/README.md): Batched GEMM with tiles
- [03_gemm](../03_gemm/README.md): Single GEMM with tiles
- [17_grouped_gemm](../17_grouped_gemm/README.md): Grouped GEMM with tiles

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
