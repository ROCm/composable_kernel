# Batched GEMM with CK Tile

This example demonstrates batched matrix multiplication (Batched GEMM) using the CK Tile programming model, enabling efficient parallel computation of multiple independent GEMMs in a single kernel launch.

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

- **Tilewise Batched GEMM**: Each thread block processes a tile of $C$ for a specific batch, loading corresponding tiles from $A$ and $B$, performing blockwise matrix multiply-accumulate, and writing results.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of $C$ for a given batch.
- **Tile Engine**: Handles loading tiles from global memory, performing GEMM in registers, and storing results.
- **Pipeline**: Modular, supports different memory/computation pipelines.

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
make tile_example_batched_gemm -j
./bin/tile_example_batched_gemm -?
```

Example:
```bash
./bin/tile_example_batched_gemm -batch_count=16 -m=256 -n=128 -k=128
```

---

## Source Structure

- **Kernel**: [`batched_gemm.hpp`](batched_gemm.hpp) (tile-programming kernel template)
- **Executable**: [`batched_gemm.cpp`](batched_gemm.cpp)
- **Build**: `CMakeLists.txt`, `run_batched_gemm_example.inc`

---

## Related CK Tile Examples

- [03_gemm](../03_gemm/README.md): Single GEMM with tiles
- [15_fused_moe](../15_fused_moe/README.md): Fused MoE block (uses group/batched GEMM)
- [13_moe_sorting](../13_moe_sorting/README.md): MoE sorting for expert dispatch

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
