# GEMM with CK Tile

This example demonstrates matrix multiplication (GEMM) using the CK Tile programming model, focusing on tile-based parallelism and modular kernel design.

---

## Algorithm and Math

GEMM computes:
$$
C = A \times B
$$
where $A$ is $[M, K]$, $B$ is $[K, N]$, and $C$ is $[M, N]$.

- **Tilewise GEMM**: Each thread block computes a tile of $C$ by loading tiles of $A$ and $B$, performing blockwise matrix multiply-accumulate, and writing results back.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of $C$.
- **Tile Engine**: Handles loading tiles from global memory, performing GEMM in registers, and storing results.
- **Pipeline**: Modular design allows swapping different memory/computation pipelines (e.g., basic, memory-bound).

---

## Features

- **Flexible Layouts**: Supports row/column-major and custom strides for $A$, $B$, $C$.
- **Batching**: Batched GEMM supported.
- **Precision**: Supports fp16, bf16, fp8, bf8.
- **Validation**: CPU/GPU validation and error tolerance options.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_gemm_basic -j
make tile_example_gemm_universal -j
```
This will result in an executable `build/bin/tile_example_gemm_basic` & `build/bin/tile_example_gemm_universal`

## example
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
       -prec    data type. fp16/bf16/fp8/bf8/int8 (default:fp16)
     -warmup    number of iterations before benchmark the kernel (default:10)
     -repeat    number of iterations to benchmark the kernel (default:100)
      -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
```


## Source Structure

- **Kernels**: `gemm_basic.cpp`, `universal_gemm.cpp` (different pipelines)
- **Utils**: `gemm_utils.hpp` (helper functions)
- **Build**: `CMakeLists.txt`, `run_gemm_example.inc`
- **Scripts**: `script/` (build and run helpers)

---

## Related CK Tile Examples

- [01_fmha](../01_fmha/README.md): Fused multi-head attention (FMHA)
- [02_layernorm2d](../02_layernorm2d/README.md): Tile-programming LayerNorm
- [16_batched_gemm](../16_batched_gemm/README.md): Batched GEMM with tiles

For tile engine and distribution, see `include/ck_tile/tile_engine/` and `include/ck_tile/tile_program/tile_distribution/`.

---
[Back to CK Tile Examples](../README.md)
