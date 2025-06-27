# Batched Transpose (Block Transpose) with CK Tile

This example demonstrates a high-performance batched block transpose kernel using the CK Tile programming model, with a focus on architectures like gfx950. The kernel is optimized for tiled memory access and is suitable for layout conversions such as NCHW <-> NHWC in deep learning.

---

## Algorithm and Math

Given a batch of tensors $X$ of shape $[N, C, H, W]$, the block transpose operation rearranges axes to produce $Y$ of shape $[N, H, W, C]$ (NCHW to NHWC) or other permutations.

For each element:
$$
Y_{n, h, w, c} = X_{n, c, h, w}
$$

- **Blockwise Transpose**: Each thread block processes a tile (block) of the input, reading and writing in a coalesced, tiled fashion for optimal memory throughput.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of the input tensor for a given batch.
- **Tile Engine**: Loads tiles, computes permuted indices, and writes results using blockwise memory access.
- **Policy**: [`transpose_policy.hpp`](transpose_policy.hpp) defines tile/block size and memory access patterns for optimal performance.

---

## Features

- **Optimized for Architecture**: Designed for architectures like gfx950 with constraints on input tile distribution.
- **Flexible Layouts**: Supports NCHW <-> NHWC and other axis permutations.
- **Batching**: Efficiently transposes multiple tensors in parallel.
- **Validation**: CPU validation and benchmarking options.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_transpose -j
./bin/tile_example_transpose -?
```

Example:
```bash
./bin/tile_example_transpose -N=2 -C=64 -H=1 -W=64 -layout_in=NCHW -layout_out=NHWC
```

---

## Source Structure

- **Kernel**: [`block_transpose.hpp`](block_transpose.hpp), [`batched_transpose_kernel.hpp`](batched_transpose_kernel.hpp), [`transpose_policy.hpp`](transpose_policy.hpp)
- **Executables**: [`transpose_example.cpp`](transpose_example.cpp), [`transpose_api.cpp`](transpose_api.cpp)
- **Build**: `CMakeLists.txt`

---

## Related CK Tile Examples

- [35_batched_transpose](../35_batched_transpose/README.md): Batched transpose with tiles
- [06_permute](../06_permute/README.md): Generic permutation with tiles
- [03_gemm](../03_gemm/README.md): GEMM with tiles

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
