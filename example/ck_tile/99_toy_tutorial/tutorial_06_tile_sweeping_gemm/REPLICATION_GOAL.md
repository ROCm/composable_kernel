# What We Want: Tile Distribution with Replication

## Goal for Tutorial 06

We have 256 threads organized as 4 warps in a 2×2 configuration:

```
Warp Layout (2×2):
┌─────────┬─────────┐
│ Warp 0  │ Warp 1  │  ← N-warp 0 and 1 (same M-row)
│ (M0,N0) │ (M0,N1) │
├─────────┼─────────┤
│ Warp 2  │ Warp 3  │  ← N-warp 0 and 1 (same M-row)
│ (M1,N0) │ (M1,N1) │
└─────────┴─────────┘
    ↑         ↑
  M-warp    M-warp
    0         1
```

## CORRECTED Understanding

**We DON'T want all warps to load identical data!**

Each warp computes a different 64×64 output region and needs DIFFERENT input data:

### A Matrix Access Pattern (for one K-iteration)

```
A Matrix (128×16):  ← Block-level tile
┌───────────────────┐
│ M[0-63]   K[0-15] │  ← Warp 0 & Warp 1 need this (M-warp 0)
├───────────────────┤
│ M[64-127] K[0-15] │  ← Warp 2 & Warp 3 need this (M-warp 1)
└───────────────────┘

Warp 0 (M0,N0): Needs A[0-63, 0-15]    ┐ Same M-rows
Warp 1 (M0,N1): Needs A[0-63, 0-15]    ┘ (NWarp replication)

Warp 2 (M1,N0): Needs A[64-127, 0-15]  ┐ Same M-rows  
Warp 3 (M1,N1): Needs A[64-127, 0-15]  ┘ (NWarp replication)
```

### B Matrix Access Pattern (for one K-iteration)

```
B Matrix (16×128):  ← Block-level tile
┌────────────────────────────┐
│ K[0-15]                    │
│ N[0-63]  │  N[64-127]      │
│    ↓           ↓           │
│  N-warp 0   N-warp 1       │
└────────────────────────────┘

Warp 0 (M0,N0): Needs B[0-15, 0-63]    ┐ Same N-cols
Warp 2 (M1,N0): Needs B[0-15, 0-63]    ┘ (MWarp replication)

Warp 1 (M0,N1): Needs B[0-15, 64-127]  ┐ Same N-cols
Warp 3 (M1,N1): Needs B[0-15, 64-127]  ┘ (MWarp replication)
```

## The Real Goal:

For tutorial_06, we're testing with a SINGLE 16×16 tile (not 128×128), so:
- **Without tile sweeping**: Each warp would load its own 16×16 portion
- **With replication for testing**: We're artificially making all warps load the same 16×16 tile to verify the replication mechanism works

This is a TEST scenario, not the actual GEMM pattern!

## Current Test Results

### test_b: ✓ WORKS
- All 4 warps load B[0-3, 0-3] (identical)
- Replication verified

### test_a: ✗ NOT WORKING
- Warp 0: A[0-3, 0-3]
- Warp 1: A[0-3, 4-7]  ← Different K! Should be same
- Warp 2: A[0-3, 8-11]
- Warp 3: A[0-3, 12-15]

**Problem**: Warps are loading different K slices instead of being replicated

## What Needs to Happen

For a single 16×16 tile loaded by 256 threads with replication:
- **All 256 threads** should collectively load the 16×16 tile
- **Replication** means the same 64-thread pattern is repeated across the replicated dimension
- For A with NWarp=2 replication: The 128-thread pattern (2 M-warps × 64) is replicated twice
- For B with MWarp=2 replication: The 128-thread pattern (2 N-warps × 64) is replicated twice

The distribution encoding must ensure that the R dimension causes true replication, not just partitioning of the data.
