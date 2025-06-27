# SmoothQuant with CK Tile

This example demonstrates SmoothQuant, a quantization technique for transformer models, using the CK Tile programming model. SmoothQuant enables efficient int8 inference by scaling activations and weights to balance quantization error.

---

## Algorithm and Math

Given input $X$ and per-channel scale $S$:
1. **Scale**: $Y_{i,j} = X_{i,j} \cdot S_j$
2. **Rowwise Dynamic Quantization**:
   - For each row, $s = \max(|Y|) / 127$
   - $Q_{i,j} = \text{round}(Y_{i,j} / s)$, $Q_{i,j} \in \text{int8}$

**Output**:  
- Quantized tensor $Q$ (int8)
- Per-row scale $s$ (fp32)

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile (row or block).
- **Tile Engine**: Loads tiles, applies scaling, performs quantization, and writes results.
- **Pipeline**: Modular, can be extended for further fusion.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_smoothquant -j
./bin/tile_smoothquant -?
```

Example:
```bash
./bin/tile_smoothquant -m=3328 -n=4096
```

---

## Source Structure

- **Kernel**: [`smoothquant.hpp`](smoothquant.hpp) (tile-programming kernel template)
- **Executable**: [`smoothquant.cpp`](smoothquant.cpp), [`example_smoothquant.cpp`](example_smoothquant.cpp)
- **Build**: `CMakeLists.txt`, `instances/`, `script/`

---

## Related CK Tile Examples

- [11_add_rmsnorm2d_rdquant](../11_add_rmsnorm2d_rdquant/README.md): Add + RMSNorm2D + RDQuant
- [10_rmsnorm2d](../10_rmsnorm2d/README.md): RMSNorm2D with tiles
- [02_layernorm2d](../02_layernorm2d/README.md): LayerNorm2D with tiles

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
