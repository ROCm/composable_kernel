# RMSNorm2D Forward with CK Tile

This example demonstrates 2D Root Mean Square Layer Normalization (RMSNorm) using the CK Tile programming model, a normalization technique widely used in LLMs and transformers.

---

## Algorithm and Math

For each row $x$:
$$
\text{rms}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2 + \epsilon}
$$
$$
y_i = \frac{x_i}{\text{rms}(x)} \cdot \gamma_i
$$
where $\gamma$ is a learnable scale parameter.

- **Tilewise RMSNorm**: Each thread block processes a tile (row or block), computes the mean square, normalizes, and applies scale.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of the input matrix.
- **Tile Engine**: Loads tiles, computes mean square, normalizes, and writes results.
- **Pipeline**: Modular, can be extended for fused operations.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_rmsnorm2d_fwd -j
./bin/tile_rmsnorm2d_fwd -?
```

Example:
```bash
./bin/tile_rmsnorm2d_fwd -m=3328 -n=4096
```

---

## Source Structure

- **Kernel**: [`rmsnorm2d_fwd.hpp`](rmsnorm2d_fwd.hpp) (tile-programming kernel template)
- **Executable**: [`rmsnorm2d_fwd.cpp`](rmsnorm2d_fwd.cpp) (argument parsing, kernel launch)
- **Build**: `CMakeLists.txt`, `generate.py`, `script/`

---

## Related CK Tile Examples

- [02_layernorm2d](../02_layernorm2d/README.md): LayerNorm2D with tiles
- [12_smoothquant](../12_smoothquant/README.md): SmoothQuant with tiles
- [05_reduce](../05_reduce/README.md): Reductions with tiles

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
