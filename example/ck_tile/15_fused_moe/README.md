# Fused-MoE with CK Tile

This example implements a highly optimized fused Mixture-of-Experts (MoE) block using the CK Tile programming model. The design fuses MoE sorting, group-GEMM, activation, and top-k weighting into a single kernel, minimizing memory traffic and maximizing throughput for large language models.

---

## Algorithm and Math

### MoE Block Structure

Given:
- **Input**: $X$ of shape $[\text{tokens}, \text{hidden}]$
- **TopK indices/weights**: $I, W$ from gating (shape $[\text{tokens}, \text{topk}]$)
- **Expert weights**: $[\text{experts}, \text{hidden}, \text{hidden}]$

**Steps:**
1. **MoE Sorting**: Rearrange tokens so each expert receives its assigned tokens in contiguous blocks (see [13_moe_sorting](../13_moe_sorting/README.md)).
2. **Group-GEMM**: For each expert, perform GEMM on its assigned tokens:
   $$
   Y^{(e)} = X^{(e)} W^{(e)}
   $$
3. **Activation + TopK Weighting**: Apply activation (e.g., GELU) and multiply by top-k weights.
4. **Scatter/Gather**: Write results back to the original token order.

### Technical Details

- **Scatter/Gather Group-GEMM**: Uses indirect indexing to map tokens to experts and back.
- **Block Partitioning**: Tokens are partitioned into slices per expert, with padding for alignment.
- **Atomic Accumulation**: Second GEMM uses atomics for accumulation to support overlapping tokens.
- **Buffer Zeroing**: Output buffer is zeroed in the sorting step, eliminating extra kernels.
- **Pre-shuffled Weights**: Expert weights are pre-shuffled for coalesced memory access.
- **Micro-kernel Pipeline**: Uses block-inline-asm micro-kernels for peak performance, while retaining composability.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile (block of tokens or experts).
- **Tile Engine**: Orchestrates sorting, GEMM, activation, weighting, and scatter/gather.
- **Pipeline**: Fuses all MoE steps into a single, composable kernel.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_fused_moe -j
./bin/tile_example_fused_moe -?
```

---

## Source Structure

- **Kernel**: [`fused_moe.hpp`](fused_moe.hpp), [`fused_moegemm.hpp`](fused_moegemm.hpp), [`fused_moesorting.hpp`](fused_moesorting.hpp)
- **Executable**: [`main.cpp`](main.cpp)
- **Build**: `CMakeLists.txt`, `instances/`, `misc/`

---

## Technical Notes

- **Indexing**: See the detailed block partitioning and indexing scheme in the source and README comments for efficient expert-token mapping.
- **Numerical Stability**: Use BF16 for gate+up cases to avoid fp16 accumulator overflow.
- **No Workspace**: The design eliminates extra workspace, reducing memory footprint.

---

## Related CK Tile Examples

- [13_moe_sorting](../13_moe_sorting/README.md): MoE sorting for expert dispatch
- [09_topk_softmax](../09_topk_softmax/README.md): TopK-Softmax for MoE gating
- [03_gemm](../03_gemm/README.md): GEMM with tiles

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
