# CK Tile Distribution Encoding Tutorial

## Overview

Every `load_tile` and `store_tile` in CK needs to know **which thread reads which data element**.
This mapping is defined by a `tile_distribution_encoding` — a compile-time struct with 6 template
parameters:

```cpp
tile_distribution_encoding<Rs, Hs, Ps_major, Ps_minor, Ys_major, Ys_minor>
```

Every level of **Hs** (hierarchical dimensions) is assigned to exactly one role:

| Role | Meaning |
|------|---------|
| **P** (parallel) | Thread ID selects which slice — different threads get different data |
| **Y** (yield) | Each thread owns the entire range in its buffer |
| **R** (replicate) | Identical data broadcast to multiple thread groups |

## Tutorials

These tutorials use the exact tile sizes from the naive GEMM tutorial
(`01_naive_gemm/`): MPerBlock=256, NPerBlock=128, KPerBlock=32, BlockSize=256, fp16.

| # | File | Matrix | Tile | Key Concept |
|---|------|--------|------|-------------|
| 1 | `tile_distribution_1.cpp` | A (DRAM load) | 256×32 | NDimP=2, warp\_id→M1, lane\_id→M2×K0 (coalesced) |
| 2 | `tile_distribution_2.cpp` | B (DRAM load) | 128×32 | Same pattern as A, but N0=2 iterations (vs A's M0=4) due to smaller N |
| 3 | `tile_distribution_3.cpp` | C (registers) | 256×128 | Warp-level MFMA output + block-level composition, standard vs transposed |

Tutorial 3 responds to `CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION` — rebuild with `=0` or `=1`
to see both C register layouts.

**Architecture note:** All comments and concrete values assume **CDNA (warp_size=64)**.
On RDNA (warp_size=32), the thread-to-data mapping will differ.

## Building

```bash
cd <repo-root>/projects/composablekernel/build

# Build all tutorials:
make tutorials -j
# or: ninja tutorials

# Or build individually:
make tile_tutorial_tile_distribution_1 -j
make tile_tutorial_tile_distribution_2 -j
make tile_tutorial_tile_distribution_3 -j

# Tutorial 3 with standard (non-transposed) C:
cmake -DCMAKE_CXX_FLAGS="-DCK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION=0" ..
make tile_tutorial_tile_distribution_3 -j
```

## Reference

- Encoding definition: `include/ck_tile/core/tensor/tile_distribution_encoding.hpp`
- Thread identity (NDimP): `include/ck_tile/core/tensor/tile_distribution.hpp`
- MFMA warp output layout: `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp`
- Production A/B distributions: `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp`
- Naive GEMM tutorial: `tutorial/ck_tile/gemm/01_naive_gemm/`
