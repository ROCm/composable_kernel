# LayerNorm2D Forward with CK Tile

This example demonstrates efficient 2D layer normalization using the CK Tile programming model, leveraging tile-based parallelism and advanced fusion for transformer and LLM workloads.

---

## Algorithm and Math

LayerNorm computes, for each row $x$:
$$
\mu = \frac{1}{N} \sum_{i=1}^N x_i,\quad \sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2
$$
$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}},\quad y_i = \gamma \hat{x}_i + \beta
$$

- **Welford's Algorithm**: Used for numerically stable, blockwise mean/variance computation. For $N \leq 4096$, a one-pass algorithm is used; for large $N$, a two-pass approach is adopted.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile (row or block of the input matrix).
- **Tile Engine**: Handles data movement, reduction, and normalization in a pipelined fashion.
- **Fusion**: Supports pre-add (prenorm), post-add (postnorm), and quantization (smooth/dynamic quant) in the same kernel.

---

## Features

- **Prenorm/Postnorm Fusion**: Fused residual addition before/after normalization for transformer blocks.
- **Smooth/Dynamic Quantization**: Rowwise int8 quantization with per-token scale, supporting smoothquant for LLMs.
- **Flexible Precision**: Supports fp16, bf16, int8 output.
- **Efficient for Large N**: Two-pass pipeline for $N > 4096$.
- **Highly Modular**: Easily extendable for new fusion or quantization strategies.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_layernorm2d_fwd -j
./bin/tile_example_layernorm2d_fwd -?
```

Example:
```bash
./bin/tile_example_layernorm2d_fwd -m=10 -n=1024
./bin/tile_example_layernorm2d_fwd -m=10 -n=1024 -prec_o=int8 -fquant=1
```

---

## Source Structure

- **Kernel**: `layernorm2d_fwd.hpp` (tile-programming kernel template)
- **Executable**: `layernorm2d_fwd.cpp` (argument parsing, kernel launch)
- **Codegen**: `generate.py` (instantiates kernels for different configs)
- **Misc**: `misc/` (algorithm diagrams, e.g., prenorm/postnorm, quantization)

---

## Related CK Tile Examples

- [01_fmha](../01_fmha/README.md): Fused multi-head attention (FMHA)
- [03_gemm](../03_gemm/README.md): Tile-programming GEMM
- [12_smoothquant](../12_smoothquant/README.md): Standalone smoothquant kernel

For tile engine and distribution, see `include/ck_tile/tile_engine/` and `include/ck_tile/tile_program/tile_distribution/`.

---
[Back to CK Tile Examples](../README.md)
