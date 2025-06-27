# Permute with CK Tile

This example demonstrates generic tensor permutation using the CK Tile programming model, similar to `torch.permute` (with `contiguous`). It supports up to rank-8 tensors and arbitrary axis permutations in a single kernel.

---

## Algorithm and Math

Given a tensor $X$ of shape $[d_0, d_1, ..., d_{n-1}]$ and a permutation $\pi$, compute:
$$
Y_{i_0, i_1, ..., i_{n-1}} = X_{i_{\pi(0)}, i_{\pi(1)}, ..., i_{\pi(n-1)}}
$$

- **Tilewise Permute**: Each thread block processes a tile (block) of the input, computes the permuted indices, and writes to the output.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of the input tensor.
- **Tile Engine**: Loads tiles, computes permuted indices, and writes results.
- **Alternative Implementation**: For rank-7 tensors, a swizzled layout is supported for matrix core-friendly data loading.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_permute -j
./bin/tile_example_permute -?
```

Example:
```bash
./bin/tile_example_permute -shape=2,3,4,6 -perm=0,3,2,1
```

---

## Source Structure

- **Kernel**: `permute.hpp` (tile-programming kernel template)
- **Executable**: `permute.cpp` (argument parsing, kernel launch)
- **Alternative**: `alternative_impl/` (swizzled layout for rank-7 tensors)
- **Build**: `CMakeLists.txt`, `script/`

---

## Related CK Tile Examples

- [03_gemm](../03_gemm/README.md): GEMM with tiles
- [05_reduce](../05_reduce/README.md): Reductions with tiles
- [35_batched_transpose](../35_batched_transpose/README.md): Batched transpose with tiles

For tile engine and distribution, see `include/ck_tile/tile_engine/` and `include/ck_tile/tile_program/tile_distribution/`.

---
[Back to CK Tile Examples](../README.md)
