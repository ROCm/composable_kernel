# FLATMM Matrix Multiplication with CK Tile

This example demonstrates FLATMM (flattened matrix multiplication) using the CK Tile programming model. FLATMM is a variant of GEMM optimized for certain memory layouts and batch processing patterns.

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

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_flatmm_basic -j
./bin/tile_example_flatmm_basic -?
```

Example:
```bash
./bin/tile_example_flatmm_basic -b=1 -m=1024 -n=2048 -k=64
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
