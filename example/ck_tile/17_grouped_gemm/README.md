# Grouped GEMM with CK Tile

This example demonstrates grouped GEMM (multiple independent GEMMs with different shapes) using the CK Tile programming model. Grouped GEMM is essential for workloads like MoE, variable-length sequences, and multi-head attention.

---

## Algorithm and Math

Given $G$ groups, each with its own $A_g$, $B_g$, $C_g$:
$$
C_g = A_g \times B_g
$$

- **Tilewise Grouped GEMM**: Each thread block processes a tile of $C_g$ for a specific group, loading corresponding tiles from $A_g$ and $B_g$, performing blockwise matrix multiply-accumulate, and writing results.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of $C_g$ for a given group.
- **Tile Engine**: Handles loading tiles, performing GEMM in registers, and storing results.
- **Pipeline**: Modular, supports different memory/computation pipelines and group-specific parameters.

---

## Features

- **Flexible Layouts**: Supports row/column-major and custom strides for $A_g$, $B_g$, $C_g$.
- **Grouped Execution**: Efficiently computes multiple GEMMs with different shapes in parallel.
- **Validation**: CPU validation and benchmarking options.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_grouped_gemm -j
./bin/tile_example_grouped_gemm -?
```

---

## Source Structure

- **Kernel**: [`grouped_gemm.hpp`](grouped_gemm.hpp) (tile-programming kernel template)
- **Executables**: [`grouped_gemm.cpp`](grouped_gemm.cpp), [`grouped_gemm_tileloop.cpp`](grouped_gemm_tileloop.cpp)
- **Build**: `CMakeLists.txt`, `run_grouped_gemm_example.inc`

---

## Related CK Tile Examples

- [16_batched_gemm](../16_batched_gemm/README.md): Batched GEMM with tiles
- [15_fused_moe](../15_fused_moe/README.md): Fused MoE block (uses grouped GEMM)
- [03_gemm](../03_gemm/README.md): Single GEMM with tiles

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
