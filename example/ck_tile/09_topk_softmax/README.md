# TopK-Softmax with CK Tile

This example demonstrates a tile-programming implementation of TopK-Softmax, commonly used in Mixture-of-Experts (MoE) models to select top-k experts per token after softmax.

---

## Algorithm and Math

Given a matrix $X$ of shape $[\text{tokens}, \text{experts}]$:
1. **Softmax per row**: $S_{i,j} = \frac{\exp(X_{i,j})}{\sum_k \exp(X_{i,k})}$
2. **TopK selection**: For each row $i$, select the $k$ largest $S_{i,j}$ and their indices.

**Output**:  
- $[\text{tokens}, k]$ weights (fp32)
- $[\text{tokens}, k]$ indices (int32)

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile (block of rows).
- **Tile Engine**: Loads tiles, computes softmax, finds top-k, and writes results.
- **Pipeline**: Modular, can be extended for fused operations.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_topk_softmax -j
./bin/tile_example_topk_softmax -?
```

Example:
```bash
./bin/tile_example_topk_softmax -t=32 -e=8 -k=2
```

---

## Source Structure

- **Kernel**: [`topk_softmax_api.hpp`](topk_softmax_api.hpp) (tile-programming kernel template)
- **Executable**: [`topk_softmax.cpp`](topk_softmax.cpp) (argument parsing, kernel launch)
- **Build**: `CMakeLists.txt`, `script/`

---

## Related CK Tile Examples

- [15_fused_moe](../15_fused_moe/README.md): Fused MoE block using TopK-Softmax
- [05_reduce](../05_reduce/README.md): Reductions with tiles
- [03_gemm](../03_gemm/README.md): GEMM with tiles

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
