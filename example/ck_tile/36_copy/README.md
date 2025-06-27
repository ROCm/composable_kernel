# Copy Kernel with CK Tile

This example demonstrates a basic copy kernel using the CK Tile programming model. It is designed as a minimal platform for new CK Tile kernel developers to test and understand tile-based data movement and memory hierarchy.

---

## Algorithm and Math

Given an input matrix $X$ of shape $[M, N]$, the copy kernel performs:
$$
Y_{i, j} = X_{i, j}
$$

- **Tilewise Copy**: Each thread block processes a tile (block) of the input, moving data from global memory (DRAM) to registers, registers to LDS (shared memory), LDS to registers, and finally to output DRAM.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of the input matrix.
- **Tile Engine**: Demonstrates all stages of data movement: DRAM $\leftrightarrow$ registers $\leftrightarrow$ LDS $\leftrightarrow$ registers $\leftrightarrow$ DRAM.
- **Pipeline**: Simple, but can be extended for more complex memory patterns or fused operations.

---

## Features

- **Memory Hierarchy**: Illustrates DRAM, LDS, and register usage in CK Tile.
- **Minimal Example**: Ideal for learning and debugging tile-programming concepts.
- **Validation**: CPU validation and benchmarking options.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make test_copy -j
./bin/test_copy_kernel -?
```

Example:
```bash
./bin/test_copy_kernel -m=64 -n=8
```

---

## Source Structure

- **Kernel**: [`test_copy.hpp`](test_copy.hpp) (tile-programming kernel template)
- **Executable**: [`test_copy.cpp`](test_copy.cpp)
- **Build**: `CMakeLists.txt`

---

## Related CK Tile Examples

- [03_gemm](../03_gemm/README.md): GEMM with tiles
- [35_batched_transpose](../35_batched_transpose/README.md): Batched transpose with tiles
- [06_permute](../06_permute/README.md): Generic permutation with tiles

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
