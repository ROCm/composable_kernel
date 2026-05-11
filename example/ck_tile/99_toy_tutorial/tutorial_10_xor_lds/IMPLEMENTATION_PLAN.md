# Plan: Tutorial 09 - Optimized LDS Staging

## Objective
Create **tutorial_09_optimized_lds** as an advanced version that demonstrates LDS optimizations like separate copy distributions, following patterns from `02_gemm`.

## Differences from Tutorial 08

| Aspect | Tutorial 08 (Simple) | Tutorial 09 (Optimized) |
|--------|---------------------|------------------------|
| Distributions | Same for all operations | Separate copy vs GEMM distributions |
| Global→LDS | Uses GEMM distribution | Uses optimized copy distribution |
| LDS→Registers | Uses GEMM distribution | Uses GEMM distribution |
| Goal | Understanding LDS concept | Production-ready patterns |
| Complexity | Minimal | Realistic |

## Key Optimizations in Tutorial 09

### 1. Separate Copy Distribution
Optimized for coalesced global memory access (all 256 threads participate efficiently).

### 2. Bank Conflict Avoidance
Optional: Add padding or XOR-based layout transformations.

### 3. Double Buffering (Optional)
Ping-pong buffers for overlapping compute and memory operations.

## Implementation Strategy

Build on tutorial_08, add:
1. `MakeACopyDistribution()` - optimized for global memory coalescing
2. `MakeBCopyDistribution()` - optimized for global memory coalescing
3. Separate windows: `a_copy_dram_window`, `a_copy_lds_window`, `a_lds_gemm_window`

This follows the pattern from `02_gemm/block_gemm_pipeline_agmem_bgmem_creg.hpp`.
