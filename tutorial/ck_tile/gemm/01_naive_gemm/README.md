# CK Tile Naive GEMM Tutorial

A minimal GEMM (`C = A × B`) using the CK Tile API. No optimizations — just the
core data flow through the three-level hierarchy: host → block → warp.

## Key Terms

| Term | What it is |
|------|-----------|
| **Problem** | Shape, data types, and layout of the GEMM matrices |
| **Policy** | How data and computation are mapped to threads (tile distributions, warp configs) |
| **Pipeline** | The loop that moves data through DRAM → VGPRs → LDS → MFMA and accumulates C |
| **Epilogue** | Post-GEMM work (e.g. activation, scaling). Not used in this tutorial |

## Execution Hierarchy

```
practice_gemm.cpp          ← host: parse args, allocate, launch, verify
  └─ grid_gemm.hpp         ← host-level: block-to-tile mapping, create tile windows
      └─ block_gemm_pipeline_agmem_bgmem_creg.hpp
         │                  ← block-level: loop over K, DRAM→VGPR→LDS, call warp GEMM
         └─ block_gemm_asmem_bsmem_creg.hpp
                            ← warp-level: LDS→VGPR, MFMA m32n32k8, accumulate C
```

**Data flow per K-iteration:**
```
A,B in DRAM ──load_tile──► VGPRs ──store_tile──► LDS ──sync──► warp GEMM (MFMA) ──► C in VGPRs
```

After all K-iterations, C is stored back to DRAM.

## File Guide

| File | Role |
|------|------|
| `practice_gemm.cpp` | Entry point: sizes, host tensors, kernel launch, verification |
| `practice_gemm.hpp` | Composes `GridGemmProblem`, `BlockGemmPipelineProblem`, and `Gemm` struct |
| `reference_gemm.hpp` | CPU reference for correctness checking |
| `grid_gemm.hpp` | Host-level pipeline: maps `blockIdx` to tile coordinates, creates A/B/C tile windows |
| `block_gemm_pipeline_agmem_bgmem_creg.hpp` | Block-level pipeline: K-loop, DRAM→LDS data movement |
| `block_gemm_pipeline_agmem_bgmem_creg_policy.hpp` | Policy: A/B DRAM tile distributions, LDS descriptors |
| `block_gemm_asmem_bsmem_creg.hpp` | Warp-level: reads A/B from LDS, runs MFMA, accumulates C |
| `block_gemm_asmem_bsmem_creg_policy.hpp` | Policy: WarpGemm type selection (standard vs transposed C) |

## Tile Sizes

From `practice_gemm.cpp` (fp16, `BlockSize=256`):

| Matrix | Block tile | Description |
|--------|-----------|-------------|
| A | 256 × 32 | M × K per block |
| B | 128 × 32 | N × K per block |
| C | 256 × 128 | M × N per block (accumulated in registers) |

Each block tile is further split across 4 warps (MWarp=4, NWarp=1).
The warp-level MFMA instruction is `m32n32k8`.

## Tile Distributions

The policy files define how threads map to tile elements:

**A and B DRAM loads** (`block_gemm_pipeline_agmem_bgmem_creg_policy.hpp`):
- Factor M (or N) into `M0 × M1 × M2`, K into `K0 × K1`
- `P0 = warp_id → M1`, `P1 = lane_id → M2 × K0` (merged for coalescing)
- `Y0 = M0` (iterations), `Y1 = K1` (vector load width = 8 for fp16)
- See the `tile_distribution/` tutorials for worked examples with these exact shapes

**C register layout** (`block_gemm_asmem_bsmem_creg_policy.hpp`):
- Determined by the WarpGemm type (MFMA instruction output mapping)
- Standard: M-dimension in Hs[0], N-dimension in Hs[1]
- Transposed: swaps M/N dimensions, changes which lanes hold which C elements

## Transposed C Distribution Switch

The macro `CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION` (default: 1) selects between
two WarpGemm variants:

| Value | WarpGemm | C layout |
|-------|----------|----------|
| 1 (default) | `WarpGemmMfma*TransposedCDistribution` | Swapped A/B in MFMA, transposed C register layout |
| 0 | `WarpGemmMfma*` | Standard MFMA, standard C register layout |

To build with the standard (non-transposed) variant, pass the define via compiler flags:
```bash
cmake -DCMAKE_CXX_FLAGS="-DCK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION=0" ..
```

Both variants produce correct results — they differ only in how C elements are
distributed across thread registers, which affects downstream store coalescing.

## Building and Running

```bash
cd <repo-root>/projects/composablekernel/build

# Configure (first time)
../script/cmake-ck-dev.sh ../ <arch>

# Build
make tile_tutorial_naive_gemm -j
# or: ninja tile_tutorial_naive_gemm

# Run (default: M=3328, N=4096, K=4096)
./bin/tile_tutorial_naive_gemm

# Custom sizes (positional args: verification M N K)
./bin/tile_tutorial_naive_gemm 0 1024 512 256
```

## Reference

- Tile distribution encoding: `include/ck_tile/core/tensor/tile_distribution_encoding.hpp`
- MFMA warp gemm: `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp`
- Production GEMM pipeline: `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp`
