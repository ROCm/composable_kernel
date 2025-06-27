# Batched Transpose with CK Tile

This example demonstrates batched tensor transpose using the CK Tile programming model. It supports common layout conversions such as NCHW <-> NHWC, which are essential for deep learning frameworks and hardware accelerators.

---

## Algorithm and Math

Given a batch of tensors $X$ of shape $[N, C, H, W]$, the transpose operation rearranges axes to produce $Y$ of shape $[N, H, W, C]$ (NCHW to NHWC) or other permutations.

For each element:
$$
Y_{n, h, w, c} = X_{n, c, h, w}
$$

- **Tilewise Batched Transpose**: Each thread block processes a tile (block) of the input, computes the permuted indices, and writes to the output.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of the input tensor for a given batch.
- **Tile Engine**: Loads tiles, computes permuted indices, and writes results.
- **Pipeline**: Modular, can be extended for vectorized or fused operations.

---

## Features

- **Flexible Layouts**: Supports NCHW <-> NHWC and other axis permutations.
- **Batching**: Efficiently transposes multiple tensors in parallel.
- **Validation**: CPU validation and benchmarking options.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_batched_transpose -j
./bin/tile_example_batched_transpose -?
```

Example:
```bash
./bin/tile_example_batched_transpose -N=2 -C=16 -H=1 -W=16 -layout_in=NCHW -layout_out=NHWC
```

---

## Source Structure

- **Kernel**: [`batched_transpose_example.hpp`](batched_transpose_example.hpp) (tile-programming kernel template)
- **Executables**: [`batched_transpose_example.cpp`](batched_transpose_example.cpp), [`batched_transpose_api.cpp`](batched_transpose_api.cpp)
- **Build**: `CMakeLists.txt`, `script/`

---

## Related CK Tile Examples

- [06_permute](../06_permute/README.md): Generic permutation with tiles
- [03_gemm](../03_gemm/README.md): GEMM with tiles
- [16_batched_gemm](../16_batched_gemm/README.md): Batched GEMM with tiles

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
