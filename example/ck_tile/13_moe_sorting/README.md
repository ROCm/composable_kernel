# MoE Sorting with CK Tile

This example demonstrates MoE (Mixture-of-Experts) sorting using the CK Tile programming model. MoE sorting rearranges token-to-expert assignments for efficient dispatch to expert GEMMs, a key step in large language models with MoE layers.

---

## Algorithm and Math

Given:
- **Input**: $[\text{tokens}, \text{topk}]$ indices and weights (from TopK-Softmax)
- **Goal**: Rearrange tokens so each expert receives its assigned tokens in contiguous blocks

**Steps:**
1. For each token, for each of its top-k experts, assign the token to the expert's input buffer.
2. Output:
   - Expert-wise token lists (indices)
   - Corresponding weights

This enables efficient batched GEMM per expert.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile (block of tokens or experts).
- **Tile Engine**: Loads token assignments, performs sorting, and writes expert-wise outputs.
- **Pipeline**: Modular, can be extended for further fusion or dispatch.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_moe_sorting -j
```

Example:
```bash
./bin/tile_example_moe_sorting -t=128 -e=8 -k=4
```

---

## Source Structure

- **Kernel**: [`moe_sorting_api.hpp`](moe_sorting_api.hpp) (tile-programming kernel template)
- **Executable**: [`moe_sorting.cpp`](moe_sorting.cpp), [`moe_sorting_api.cpp`](moe_sorting_api.cpp)
- **Build**: `CMakeLists.txt`, `script/`

---

## Related CK Tile Examples

- [09_topk_softmax](../09_topk_softmax/README.md): TopK-Softmax for MoE gating
- [15_fused_moe](../15_fused_moe/README.md): Fused MoE block
- [03_gemm](../03_gemm/README.md): GEMM with tiles

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
