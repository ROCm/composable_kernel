# Tutorial 09: Optimized LDS with Separate Copy/GEMM Distributions

## Overview

This tutorial demonstrates **the fundamental optimization pattern** used in ALL production GPU kernels: **separate copy and GEMM distributions**. This is the critical bridge between educational code and production-ready implementations.

## Key Concepts

### Two Distribution Types

1. **Copy Distribution** (for Global ↔ LDS transfers)
   - Optimized for **memory bandwidth**
   - No replication (`sequence<1>`)
   - All 256 threads cooperatively load
   - Vector loads (8 elements = 16 bytes)
   - Perfect memory coalescing

2. **GEMM Distribution** (for LDS → Registers and compute)
   - Optimized for **compute efficiency**
   - With replication (`sequence<NWarp>` or `sequence<MWarp>`)
   - Warp-based partitioning
   - Enables efficient LDS broadcast
   - Matches MFMA instruction requirements

### Six Windows Instead of Four

Tutorial 08 used **4 windows** (same distribution):
- 2 global memory windows (A and B)
- 2 LDS windows (A and B)

Tutorial 09 uses **6 windows** (separate distributions):
- 2 copy DRAM windows (A and B) - with copy distribution
- 2 copy LDS windows (A and B) - with copy distribution
- 2 GEMM LDS windows (A and B) - with GEMM distribution

**Key insight:** Same LDS buffer, different access patterns! The distribution determines HOW threads access the buffer, not the buffer itself.

## Data Flow Comparison

### Tutorial 08 (Simple)
```
Global → [GEMM dist] → Regs → [GEMM dist] → LDS → [GEMM dist] → MFMA
         (Same distribution everywhere - suboptimal)
```

### Tutorial 09 (Optimized)
```
Global → [COPY dist] → Regs → [COPY dist] → LDS → [GEMM dist] → MFMA
         ↑______ bandwidth ______↑              ↑___ compute ___↑
```

## Copy Distribution Details

For A matrix (M×K):
```cpp
constexpr index_t K1 = 16 / sizeof(DataType);  // 8 for half_t
constexpr index_t K0 = kKPerBlock / K1;         // 32 / 8 = 4
constexpr index_t M2 = kWaveSize / K0;          // 64 / 4 = 16
constexpr index_t M1 = kBlockSize / kWaveSize;  // 256 / 64 = 4
constexpr index_t M0 = kMPerBlock / (M2 * M1);  // 64 / (16 * 4) = 1
```

**Key properties:**
- `sequence<1>`: NO replication
- `K1 = 8`: Vector load of 8 half_t elements = 16 bytes
- All 256 threads: (64×32) / 256 = 8 elements per thread
- Perfect coalescing: consecutive threads access consecutive addresses

## GEMM Distribution Details

For A matrix (M×K):
```cpp
// Block-level with REPLICATION
sequence<NWarp>  // Data REPLICATED across N-dimension warps
tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>
```

**Key properties:**
- `sequence<NWarp>`: Data replicated across N-warps (all N-warps read same A data)
- Warp-based partitioning matches MFMA requirements
- Enables efficient LDS broadcast (one read serves multiple warps)

## K-Loop Phases

The K-loop demonstrates the separate distributions:

```cpp
for(index_t k_iter = 0; k_iter < num_k_loops; ++k_iter)
{
    // PHASE 1: Global → Registers (COPY distribution)
    const auto a_block_tile_copy = load_tile(a_copy_dram_window);
    const auto b_block_tile_copy = load_tile(b_copy_dram_window);

    // PHASE 2: Registers → LDS (COPY distribution)
    store_tile(a_copy_lds_window, a_block_tile_copy);
    store_tile(b_copy_lds_window, b_block_tile_copy);

    // PHASE 3: Synchronization
    block_sync_lds();

    // PHASE 4: LDS → Registers (GEMM distribution)
    // NOTE: Same LDS buffer, different distribution!
    const auto a_block_tile_gemm = load_tile(a_lds_gemm_window);
    const auto b_block_tile_gemm = load_tile(b_lds_gemm_window);

    // PHASE 5: Compute (using GEMM tiles)
    // ... MFMA operations ...

    // PHASE 6: Move windows
    move_tile_window(a_copy_dram_window, {0, kKPerBlock});
    move_tile_window(b_copy_dram_window, {kKPerBlock, 0});
    // GEMM windows stay at {0,0} - they always read from LDS
}
```

## Why This is Faster

1. **Memory Bandwidth Optimization**
   - Copy distribution: All 256 threads cooperatively load
   - Vector loads: 8 elements = 16 bytes (optimal for global memory)
   - Perfect coalescing: consecutive threads → consecutive addresses

2. **Compute Efficiency Optimization**
   - GEMM distribution: Warp-based partitioning
   - Data replication via LDS broadcast
   - Matches MFMA instruction requirements

3. **Best of Both Worlds**
   - Memory transfer: bandwidth-optimized
   - Computation: compute-optimized
   - LDS acts as the redistribution point

## Performance Expectations

For small problems (K=64):
- Should match Tutorial 08 numerically (same computation)
- Performance may be similar (only 2 K-iterations)

For larger problems (K >> 64):
- Better memory coalescing visible
- More efficient LDS utilization
- Scalable to production sizes

## Code Structure

```cpp
// 1. Copy distribution functions
MakeACopyDistribution<DataType>()  // A: M×K
MakeBCopyDistribution<DataType>()  // B: K×N

// 2. GEMM distribution functions
MakeAGemmDistribution()  // A: M×K with NWarp replication
MakeBGemmDistribution()  // B: K×N with MWarp replication

// 3. Six windows creation
a_copy_dram_window   // Global A with copy dist
b_copy_dram_window   // Global B with copy dist
a_copy_lds_window    // LDS A with copy dist
b_copy_lds_window    // LDS B with copy dist
a_lds_gemm_window    // LDS A with GEMM dist
b_lds_gemm_window    // LDS B with GEMM dist

// 4. K-loop with appropriate windows
load_tile(a_copy_dram_window)     // Use copy for transfer
store_tile(a_copy_lds_window, ...)
load_tile(a_lds_gemm_window)      // Use GEMM for compute
```

## Comparison Table

| Aspect | Tutorial 08 | Tutorial 09 |
|--------|-------------|-------------|
| **Distributions** | 1 type (GEMM) | 2 types (copy + GEMM) |
| **Windows** | 4 windows | 6 windows |
| **Global→LDS** | GEMM dist | Copy dist ✓ |
| **LDS→Compute** | GEMM dist | GEMM dist ✓ |
| **Memory coalescing** | Suboptimal | Optimal ✓ |
| **Compute efficiency** | Good | Good ✓ |
| **Production-ready** | No | Yes ✓ |

## Educational Value

This tutorial teaches:

1. **Why separate distributions matter**
   - Different operations have different optimization requirements
   - Memory bandwidth ≠ compute efficiency

2. **The production pattern**
   - ALL optimized GPU kernels use this pattern
   - GEMM, Convolution, Attention - all use copy + GEMM distributions

3. **How redistribution works**
   - Same LDS buffer, different access patterns
   - LDS acts as the redistribution point

4. **Foundation for advanced optimizations**
   - Double buffering (overlap copy and compute)
   - Bank conflict avoidance (XOR swizzle)
   - Prefetching (hide latency)

## Building and Running

```bash
cd build
cmake ..
make aa_tutorial_09_optimized_lds
./bin/aa_tutorial_09_optimized_lds
```

Expected output:
```
Tutorial 09: Optimized LDS with Copy/GEMM Distributions
...
Results:
  Correctness: ✓ PASSED
  Max error: ~5.7e-6
...
```

## Next Steps

After understanding Tutorial 09, you're ready for:

- **Tutorial 10**: Double buffering (overlap copy and compute)
- **Advanced optimizations**: Bank conflict avoidance with XOR swizzle
- **Production kernels**: Study `02_gemm` implementation
- **Other kernels**: Apply same pattern to Convolution, Attention

## References

### Production Examples
- `example/ck_tile/99_toy_example/02_gemm/block_gemm_pipeline_agmem_bgmem_creg_default_policy.hpp` (lines 213-262)
  - Copy distribution pattern
  - Vector width calculation

- `example/ck_tile/99_toy_example/02_gemm/block_gemm_asmem_bsmem_creg.hpp` (lines 51-88)
  - GEMM distribution pattern
  - Embedded warp distributions

- `example/ck_tile/99_toy_example/02_gemm/block_gemm_pipeline_agmem_bgmem_creg.hpp` (lines 236-402)
  - Six-window setup
  - K-loop with separate distributions

### Learning Path
1. Tutorial 08: Understand LDS staging concept (simple)
2. **Tutorial 09: Understand distribution optimization (realistic)** ← You are here
3. Tutorial 10+: Advanced optimizations (double buffering, etc.)

## Key Takeaways

- **THE fundamental production pattern:** Separate copy and GEMM distributions
- **Memory hierarchy optimization:** Different distributions for different operations
- **Bandwidth vs compute tradeoff:** Copy optimizes memory, GEMM optimizes compute
- **Same buffer, different access:** LDS enables redistribution without data movement
- **Universal pattern:** Applies to ALL GPU compute kernels

This is not just an optimization - it's **the standard approach** in production code!
