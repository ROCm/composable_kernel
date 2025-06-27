# MoE-SmoothQuant with CK Tile

This example demonstrates MoE-SmoothQuant, a fused quantization operation for Mixture-of-Experts (MoE) models, using the CK Tile programming model. Unlike standard SmoothQuant, the input scale is expert-dependent, and the operation is fused with top-k expert selection.

---

## Algorithm and Math

Given:
- **Input**: $X$ of shape $[\text{tokens}, \text{topk}, \text{hidden}]$
- **Expert scales**: $S$ of shape $[\text{experts}, \text{hidden}]$
- **TopK indices**: $I$ of shape $[\text{tokens}, \text{topk}]$

**Steps:**
1. For each token $t$ and its $k$ selected experts:
   - Select scale $S_{I_{t,k}, :}$ for the $k$-th expert.
   - Scale: $Y_{t,k,j} = X_{t,k,j} \cdot S_{I_{t,k}, j}$
2. **Rowwise Dynamic Quantization** (per token-expert pair):
   - $s_{t,k} = \max_j |Y_{t,k,j}| / 127$
   - $Q_{t,k,j} = \text{round}(Y_{t,k,j} / s_{t,k})$, $Q_{t,k,j} \in \text{int8}$

**Output**:  
- Quantized tensor $Q$ (int8)
- Per-token-expert scale $s$ (fp32)

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile (block of tokens, experts, or hidden units).
- **Tile Engine**: Loads input, selects expert scales via top-k indices, applies scaling and quantization, and writes results.
- **Pipeline**: Modular, can be extended for further fusion.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_moe_smoothquant -j
./bin/tile_example_moe_smoothquant -?
```

---

## Source Structure

- **Kernel**: [`moe_smoothquant.hpp`](moe_smoothquant.hpp) (tile-programming kernel template)
- **Executable**: [`moe_smoothquant.cpp`](moe_smoothquant.cpp)
- **Build**: `CMakeLists.txt`, `instances/`, `misc/`, `script/`

---

## Technical Notes

- **Expert-dependent scaling**: Each token's top-k experts use their own per-hidden-unit scale, requiring indirect indexing and efficient memory access.
- **Fused with top-k**: The kernel uses top-k indices from gating to select the correct expert scale for each token.
- **Rowwise quantization**: Each token-expert pair is quantized independently for maximum accuracy.

---

## Related CK Tile Examples

- [09_topk_softmax](../09_topk_softmax/README.md): TopK-Softmax for MoE gating
- [13_moe_sorting](../13_moe_sorting/README.md): MoE sorting for expert dispatch
- [12_smoothquant](../12_smoothquant/README.md): Standard SmoothQuant

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
