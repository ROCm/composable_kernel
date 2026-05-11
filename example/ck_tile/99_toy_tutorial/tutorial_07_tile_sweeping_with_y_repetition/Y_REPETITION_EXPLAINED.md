# Y-Dimension Repetition for Tile Sweeping

This document explains how Y-dimension repetition enables true tile sweeping in CK Tile distributions.

## Overview

**Y-dimension repetition** is a mechanism in tile distribution encodings that allows each thread/warp to process multiple tiles of data. This is the key to implementing efficient tile sweeping patterns in GEMM kernels.

## Comparison: Tutorial 06 vs Tutorial 07

### Tutorial 06: No Y-Repetition (Single Tile per Warp)

```cpp
// Each warp processes exactly ONE 16×16 tile
constexpr auto a_block_outer_dstr_encode = tile_distribution_encoding<
    sequence<NWarp>,                      // Replication
    tuple<sequence<MWarp>, sequence<>>,   // H: Just warp organization
    tuple<sequence<0, 1>>,                // Ps_to_Hs
    tuple<sequence<0, 0>>,                // Ps_in_Hs
    sequence<>,                           // Ys_to_Hs: NO Y-dimension
    sequence<>>{};                        // Ys_in_Hs: NO Y-dimension
```

**Result:** Each warp computes ONE 16×16 output tile
- Block output: 32×32 (2 warps × 16 in each dimension)
- No tile sweeping

### Tutorial 07: With Y-Repetition (Multiple Tiles per Warp)

```cpp
// Each warp processes MULTIPLE tiles via Y-repetition
constexpr index_t MIterPerWarp = 2;  // 2 iterations in M dimension

constexpr auto a_block_outer_dstr_encode = tile_distribution_encoding<
    sequence<NWarp>,                                    // Replication
    tuple<sequence<MIterPerWarp, MWarp>,                // H: Iterations × Warps
          sequence<KIterPerWarp>>,                      // H: K iterations
    tuple<sequence<1, 0>>,                              // Ps_to_Hs
    tuple<sequence<1, 0>>,                              // Ps_in_Hs
    sequence<1, 2>,                                     // Ys_to_Hs: Y maps to BOTH dims
    sequence<0, 0>>{};                                  // Ys_in_Hs: Y position
```

**Result:** Each warp sweeps over 2×2 = 4 tiles of 16×16
- Warp output: 32×32 (2 iters × 16 in each dimension)
- Block output: 64×64 (2 warps × 32 in each dimension)
- TRUE tile sweeping!

## Key Parameters

### MIterPerWarp and NIterPerWarp

These control how many tiles each warp processes:

```cpp
static constexpr index_t MIterPerWarp = 2;  // Each warp: 2 tiles in M
static constexpr index_t NIterPerWarp = 2;  // Each warp: 2 tiles in N
```

Total tiles per warp = `MIterPerWarp × NIterPerWarp = 2 × 2 = 4 tiles`

### Ys_to_Hs and Ys_in_Hs

These parameters define the Y-dimension mapping:

- `Ys_to_Hs`: Which H-dimensions does Y map to?
  - `sequence<1, 2>` means Y maps to BOTH dimension 1 (M or N) and dimension 2 (K)
  
- `Ys_in_Hs`: Position of Y within each H-dimension
  - `sequence<0, 0>` means Y is at position 0 in both dimensions

## The H-Space Structure

With Y-repetition, the H-space becomes multi-dimensional:

### For A Matrix (M×K):
```
H0 (M dimension): sequence<MIterPerWarp, MWarp>
                  = sequence<2, 2>
                  = [iter0, iter1] × [warp0, warp1]
                  
H1 (K dimension): sequence<KIterPerWarp>
                  = sequence<1>
```

### For B Matrix (N×K):
```
H0 (N dimension): sequence<NIterPerWarp, NWarp>
                  = sequence<2, 2>
                  = [iter0, iter1] × [warp0, warp1]
                  
H1 (K dimension): sequence<KIterPerWarp>
                  = sequence<1>
```

## Extracting Tiles with get_y_sliced_thread_data

The Y-repetition creates a block tensor that contains ALL tiles for ALL iterations. We extract specific tiles using Y-slicing:

```cpp
static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
        // Extract the tile for iteration [mIter, nIter]
        auto c_warp_tensor = make_static_distributed_tensor<AccDataType>(
            make_static_tile_distribution(c_warp_dstr_encode));
        
        // Y-slice: Get data for this specific iteration
        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
        
        // Process this tile...
        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
        
        // Write back
        c_block_tile.set_y_sliced_thread_data(
            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
            c_warp_tensor.get_thread_buffer());
    });
});
```

### Understanding the Y-Slice Parameters

1. **Y-index**: `merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros)`
   - Specifies WHICH tile to extract
   - `sequence<mIter, nIter>` selects the iteration indices
   - `c_warp_y_index_zeros` fills in zeros for other Y-dimensions

2. **Y-length**: `merge_sequences(sequence<1, 1>{}, c_warp_y_lengths)`
   - Specifies HOW MANY tiles to extract
   - `sequence<1, 1>` means extract 1 tile in each iteration dimension
   - `c_warp_y_lengths` provides lengths for other Y-dimensions

## Memory Layout

### Without Y-Repetition (Tutorial 06):
```
Block Tensor Layout:
[Warp0_Tile] [Warp1_Tile]
[Warp2_Tile] [Warp3_Tile]

Each warp has 1 tile worth of data
```

### With Y-Repetition (Tutorial 07):
```
Block Tensor Layout (conceptual):
Warp 0: [Iter0,0] [Iter0,1]    Warp 1: [Iter0,0] [Iter0,1]
        [Iter1,0] [Iter1,1]            [Iter1,0] [Iter1,1]

Warp 2: [Iter0,0] [Iter0,1]    Warp 3: [Iter0,0] [Iter0,1]
        [Iter1,0] [Iter1,1]            [Iter1,0] [Iter1,1]

Each warp has 4 tiles worth of data (2×2 iterations)
```

## Replication Still Works!

Y-repetition is orthogonal to replication:

```cpp
// A matrix: Replicate across N-warps, sweep in M dimension
constexpr auto a_block_outer_dstr_encode = tile_distribution_encoding<
    sequence<NWarp>,                      // ← Replication
    tuple<sequence<MIterPerWarp, MWarp>,  // ← Y-repetition in M
          sequence<KIterPerWarp>>,        // ← Y-repetition in K
    ...
    sequence<1, 2>,                       // ← Y maps to both M and K
    sequence<0, 0>>{};
```

**Result:**
- Warps 0 and 2 (both N-warp 0) load identical A data
- Warps 1 and 3 (both N-warp 1) load identical A data
- But each warp sweeps over 2 M-iterations

## Scaling to Production

This pattern scales directly to production kernels:

### Example: 256×256 Block with 4×4 Warps

```cpp
static constexpr index_t MWarp = 4;
static constexpr index_t NWarp = 4;
static constexpr index_t MIterPerWarp = 4;  // Each warp: 4 M-iterations
static constexpr index_t NIterPerWarp = 4;  // Each warp: 4 N-iterations

// Each warp: 4×4 iters × 16×16 per tile = 64×64 output
// Each block: 4×4 warps × 64×64 per warp = 256×256 output
```

## Benefits of Y-Repetition

1. **Compile-time tile iteration**: `static_for` loops unroll at compile time
2. **Efficient register usage**: All tiles for a warp are in registers
3. **Flexible tile counts**: Easy to adjust `MIterPerWarp` and `NIterPerWarp`
4. **Production-ready pattern**: Used in 02_gemm and real kernels
5. **Works with replication**: Orthogonal concepts that compose well

## Summary

| Aspect | Tutorial 06 | Tutorial 07 |
|--------|-------------|-------------|
| Tiles per warp | 1 (16×16) | 4 (2×2 iters of 16×16) |
| Warp output | 16×16 | 32×32 |
| Block output | 32×32 | 64×64 |
| Y-repetition | No | Yes (MIterPerWarp=2, NIterPerWarp=2) |
| Tile extraction | Direct load | get_y_sliced_thread_data |
| Iteration | None | static_for over iterations |
| Pattern | Basic multi-warp | Production-ready sweeping |

Y-dimension repetition is the key mechanism that enables efficient, scalable tile sweeping in CK Tile GEMM kernels!
