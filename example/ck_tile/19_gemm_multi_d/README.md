# Multiple D GEMM with CK Tile

This example demonstrates GEMM with multiple D tensors (multi-output GEMM) using the CK Tile programming model. This is useful for fused operations where the GEMM output is combined with multiple side inputs (e.g., bias, residual, or other elementwise sources).

---

## Algorithm and Math

Given:
- $A$: $[M, K]$
- $B$: $[K, N]$
- $D_0, D_1, ..., D_n$: $[M, N]$ (multiple side inputs)
- $E$: $[M, N]$ (output)

The operation:
$$
E = f(A \times B, D_0, D_1, ..., D_n)
$$
where $f$ is a fused elementwise function (e.g., add, multiply, activation).

- **Tilewise Multi-D GEMM**: Each thread block processes a tile of $E$, loading corresponding tiles from $A$, $B$, and all $D_i$, performing blockwise GEMM and fused elementwise operations.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of $E$.
- **Tile Engine**: Handles loading tiles, performing GEMM, and applying fused elementwise ops.
- **Pipeline**: Modular, supports different memory/computation pipelines and multi-D fusion.

---

## Features

- **Multiple D Inputs**: Supports arbitrary number of side inputs for fusion.
- **Flexible Layouts**: Supports row/column-major and custom strides for all tensors.
- **SplitK**: Supports K-batching for large K dimensions.
- **Validation**: GPU validation and benchmarking options.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_gemm_multi_d_fp16 -j
./bin/tile_example_gemm_multi_d_fp16 -?
```

Example:
```bash
./bin/tile_example_gemm_multi_d_fp16 -m=3840 -n=4096 -k=4096
```

---

## Source Structure

- **Kernel**: [`gemm_multi_d_fp16.hpp`](gemm_multi_d_fp16.hpp) (tile-programming kernel template)
- **Executable**: [`gemm_multi_d_fp16.cpp`](gemm_multi_d_fp16.cpp)
- **Utils**: [`utils.hpp`](utils.hpp)
- **Build**: `CMakeLists.txt`, `run_gemm_multi_d_fp16_example.inc`

---

## Related CK Tile Examples

- [03_gemm](../03_gemm/README.md): Single GEMM with tiles
- [16_batched_gemm](../16_batched_gemm/README.md): Batched GEMM with tiles
- [17_grouped_gemm](../17_grouped_gemm/README.md): Grouped GEMM with tiles

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
