# Plan: Tutorial 08 - Simple LDS Staging

## Objective
Create **tutorial_08_lds_staging** as a **simple, direct continuation** of tutorial_07 that adds LDS (Local Data Share / shared memory) staging to demonstrate: **Global Memory → LDS → Registers → Compute**.

**Note**: This is the SIMPLE version for learning. Tutorial 09 will add optimizations like separate copy distributions.

## Simplified Tutorial Approach

**KEY PRINCIPLE**: Keep it simple for educational purposes
- ✅ Use the **SAME tile distributions** as tutorial_07 (no new distributions!)
- ✅ Just add the LDS staging layer between global memory and compute
- ✅ Only change: increase `kKPerBlock` from 16 to 32 for `KIterPerWarp = 2`
- ✅ Minimal code changes from tutorial_07

**What we DON'T do** (to keep it simple):
- ❌ No separate copy distributions (like 02_gemm's optimized version)
- ❌ No ENABLE_PREFETCH complexity
- ❌ No XOR-based bank conflict avoidance
- ❌ No complex optimization strategies

**Data flow**:
```
Tutorial 07: Global → Registers → MFMA
Tutorial 08: Global → Registers → LDS → Registers → MFMA
                     (same distribution everywhere)
```

## Understanding Data Reuse in 02_gemm

### How 02_gemm Implements LDS Reuse

Looking at `02_gemm/block_gemm_asmem_bsmem_creg.hpp`, the key parameters are:

```cpp
constexpr index_t KPerBlock    = BlockGemmShape::kK;      // e.g., 32 or 64
constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK; // e.g., 32/16 = 2
```

**The reuse pattern**:
1. **One load from global to LDS**: The entire `kKPerBlock` K-chunk (e.g., 32 elements in K) is loaded to LDS once
2. **Multiple iterations within LDS**: The inner `static_for<0, KIterPerWarp, 1>` loop iterates `KIterPerWarp` times over K-slices **within LDS**
3. **Reuse via replication**: A is replicated across `NWarp` warps, B across `MWarp` warps

**Example with KPerBlock=32, WarpGemm::kK=16**:
- Load A[M×32] and B[32×N] to LDS once
- Inner loop iterates 2 times: kIter=0 uses K[0:16], kIter=1 uses K[16:32]
- Each K-slice in LDS is read by all MWarp (for B) or NWarp (for A) warps

### Tutorial 07's Problem

Tutorial 07 has `KIterPerWarp = 1` and `kWarpK = 16`, so each K-chunk loaded is used only once - **no temporal reuse in K-dimension**. The only reuse is:
- A replicated across 2 NWarps (each A element used 2 times)
- B replicated across 2 MWarps (each B element used 2 times)

This spatial reuse doesn't benefit from LDS staging since global memory coalescing is already good.

## Solution: Increase kKPerBlock for Tutorial 08

For meaningful LDS benefit, we need `KIterPerWarp > 1`:

**Tutorial 08 Configuration**:
```cpp
static constexpr index_t kKPerBlock = 32;    // Load 32 K-elements to LDS
static constexpr index_t kWarpK = 16;        // Each MFMA uses 16
static constexpr index_t KIterPerWarp = 2;   // 2 iterations within each LDS load
```

**Data reuse calculation**:
- A tile: 64×32 elements loaded once, used by 2 NWarps × 2 KIters = 4× reuse
- B tile: 32×64 elements loaded once, used by 2 MWarps × 2 KIters = 4× reuse

## Current State (Tutorial 07)
- **File**: `example/ck_tile/99_toy_example/tutorial_07_tile_sweeping_with_y_repetition/tile_sweeping_with_y_repetition.cpp`
- **Current flow**: Global Memory → Registers → MFMA (no LDS staging)
- **Block config**: 2×2 warps (256 threads), 64×64 output per block
- **K-loop**: 4 iterations with kWarpK=16

---

## Implementation Steps

### Step 0: Create New Tutorial Directory

```bash
mkdir -p example/ck_tile/99_toy_example/tutorial_08_lds_staging
```

Copy `tutorial_07` as a starting point and modify.

### Step 1: Update Kernel Constants

Change the K-dimension parameters to enable temporal reuse:

```cpp
// Tutorial 07 values (no temporal reuse):
// static constexpr index_t kWarpK = 16;
// static constexpr index_t KIterPerWarp = 1;

// Tutorial 08 values (with temporal reuse):
static constexpr index_t kWarpK = 16;         // MFMA K dimension (unchanged)
static constexpr index_t kKPerBlock = 32;     // NEW: K-tile loaded to LDS
static constexpr index_t KIterPerWarp = kKPerBlock / kWarpK;  // = 2
```

### Step 2: Add LDS Size Calculation and Descriptor Functions

Add static member functions to the kernel struct:

```cpp
// LDS descriptor for A: [M=64][K=32] with kKPack=8
CK_TILE_HOST_DEVICE static constexpr auto MakeALdsDescriptor()
{
    constexpr index_t kMPerBlock = MWarp * MIterPerWarp * kWarpM;  // 64
    constexpr index_t kKPack = 8;

    constexpr auto a_lds_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kMPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
        make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
        number<kKPack>{},
        number<1>{});

    constexpr auto a_lds_desc = transform_tensor_descriptor(
        a_lds_desc_0,
        make_tuple(make_pass_through_transform(kMPerBlock),
                   make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
        make_tuple(sequence<0>{}, sequence<1, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return a_lds_desc;
}

// LDS descriptor for B: [N=64][K=32]
CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsDescriptor()
{
    constexpr index_t kNPerBlock = NWarp * NIterPerWarp * kWarpN;  // 64
    constexpr index_t kKPack = 8;

    constexpr auto b_lds_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kNPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
        make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
        number<kKPack>{},
        number<1>{});

    constexpr auto b_lds_desc = transform_tensor_descriptor(
        b_lds_desc_0,
        make_tuple(make_pass_through_transform(kNPerBlock),
                   make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
        make_tuple(sequence<0>{}, sequence<1, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return b_lds_desc;
}

// LDS size calculation
CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
{
    return integer_divide_ceil(
               sizeof(ADataType) * MakeALdsDescriptor().get_element_space_size(), 16) * 16 +
           sizeof(BDataType) * MakeBLdsDescriptor().get_element_space_size();
}
```

### Step 3: NO NEED for Separate Copy Distributions!

**IMPORTANT FOR TUTORIAL SIMPLICITY**: We will use the **SAME** distributions from tutorial_07 for all operations:
- Load from global memory
- Store to LDS
- Load from LDS

This keeps the tutorial simple and focused on the LDS staging concept, not on distribution optimization.

### Step 4: Add `void* p_smem` Parameter to Kernel

Modify the kernel operator signature:

```cpp
CK_TILE_DEVICE void operator()(const ADataType* a,
                               const BDataType* b,
                               const CDataType* c,
                               CDataType* d,
                               index_t M, index_t N, index_t K,
                               index_t lda, index_t ldb, index_t ldc, index_t ldd,
                               AccDataType alpha, AccDataType beta,
                               void* p_smem) const  // NEW: LDS pointer
```

### Step 5: Create LDS Tensor Views and Windows

Inside the kernel operator, add after creating the global tensor views:

```cpp
// ============================================================================
// LDS SETUP (Tutorial 08 Addition)
// ============================================================================

// A tile in LDS
ADataType* p_a_lds = static_cast<ADataType*>(p_smem);
constexpr auto a_lds_desc = MakeALdsDescriptor();
auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_desc);

constexpr index_t a_lds_size_aligned =
    integer_divide_ceil(sizeof(ADataType) * a_lds_desc.get_element_space_size(), 16) * 16;

// B tile in LDS
BDataType* p_b_lds = static_cast<BDataType*>(
    static_cast<void*>(static_cast<char*>(p_smem) + a_lds_size_aligned));
constexpr auto b_lds_desc = MakeBLdsDescriptor();
auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_desc);

// Create windows using the SAME distributions from tutorial_07
// Global memory windows
auto a_block_window = make_tile_window(
    a_tensor,
    make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
    {m_block_base, 0},
    a_block_distribution  // Same as tutorial_07
);

auto b_block_window = make_tile_window(
    b_tensor,
    make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}),
    {0, n_block_base},
    b_block_distribution  // Same as tutorial_07
);

// LDS windows (NEW - use SAME distributions!)
auto a_lds_window = make_tile_window(
    a_lds_block,
    make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
    {0, 0},
    a_block_distribution  // Reuse the same distribution!
);

auto b_lds_window = make_tile_window(
    b_lds_block,
    make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}),
    {0, 0},
    b_block_distribution  // Reuse the same distribution!
);
```

### Step 6: Update Block Distributions for KIterPerWarp=2

The existing block distributions need to account for `KIterPerWarp = 2`:

```cpp
// A Distribution with K-iteration Y-repetition
constexpr auto a_block_outer_dstr_encode = tile_distribution_encoding<
    sequence<NWarp>,                                        // Replicate across N-warps
    tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,  // 2×2 in M, 2 in K
    tuple<sequence<0, 1>>,
    tuple<sequence<0, 1>>,
    sequence<1, 2>,                                         // Y maps to BOTH M and K
    sequence<0, 0>>{};

// B Distribution with K-iteration Y-repetition
constexpr auto b_block_outer_dstr_encode = tile_distribution_encoding<
    sequence<MWarp>,                                        // Replicate across M-warps
    tuple<sequence<KIterPerWarp>, sequence<NIterPerWarp, NWarp>>,  // 2 in K, 2×2 in N
    tuple<sequence<2, 0>>,
    tuple<sequence<1, 0>>,
    sequence<1, 2>,                                         // Y maps to BOTH K and N
    sequence<0, 0>>{};
```

### Step 7: Modify K-Loop with LDS Staging

Replace the current K-loop with the LDS-staged version:

```cpp
// Main K-loop with LDS staging
const index_t num_k_loops = K / kKPerBlock;  // Now K/32 instead of K/16
for(index_t k_iter = 0; k_iter < num_k_loops; ++k_iter)
{
    // Phase 1: Global -> Registers
    const auto a_global_tile = load_tile(a_block_window);
    const auto b_global_tile = load_tile(b_block_window);

    // Phase 2: Registers -> LDS
    store_tile(a_lds_window, a_global_tile);
    store_tile(b_lds_window, b_global_tile);

    // Phase 3: Synchronize (wait for all threads to finish writing to LDS)
    block_sync_lds();

    // Phase 4: LDS -> Registers (same distribution, just different source!)
    const auto a_block_tile = load_tile(a_lds_window);
    const auto b_block_tile = load_tile(b_lds_window);

    // Phase 5: Compute (SAME as tutorial_07, just with KIterPerWarp=2 now)
    static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            // ... existing Y-slicing code from tutorial_07 ...
        });
    });

    // Phase 6: Move to next K chunk
    if(k_iter < num_k_loops - 1) {
        move_tile_window(a_block_window, {0, kKPerBlock});
        move_tile_window(b_block_window, {kKPerBlock, 0});
    }
}
```

### Step 8: Update Kernel Launch with LDS Size

In `main()`, update the `launch_kernel` call:

```cpp
constexpr index_t lds_size = LdsStagingHgemmKernel<
    InputType, InputType, AccumType, AccumType>::GetStaticLdsSize();

launch_kernel(stream,
             make_kernel<block_size>(
                 LdsStagingHgemmKernel<InputType, InputType, AccumType, AccumType>{},
                 dim3(grid_size),
                 dim3(block_size),
                 lds_size,  // Was 0, now actual LDS size (~8KB)
                 // ... rest of arguments ...
                 ));
```

---

## LDS Memory Layout

```
LDS Memory:
+------------------+------------------+
|   A tile (64×32) |   B tile (32×64) |
|   4096 bytes     |   4096 bytes     |
|   (aligned 16B)  |                  |
+------------------+------------------+
Total: ~8KB (well within 64KB limit)
```

## Data Flow Summary

**Before (Tutorial 07)**:
```
For each K-chunk (16 elements):
  Global Memory → Registers → MFMA Compute
  (No temporal reuse in K)
```

**After (Tutorial 08 with LDS)**:
```
For each K-chunk (32 elements):
  Global Memory → Registers → LDS → block_sync_lds()
  For kIter in [0, 1]:  # KIterPerWarp = 2
    LDS → Registers → MFMA Compute
  (Temporal reuse: K-chunk used 2 times)
```

---

## Verification

1. **Build**: Compile the new tutorial
2. **Run**: Execute `aa_tutorial_08_lds_staging`
3. **Verify correctness**: Should pass with same tolerance (~1e-2)
4. **Check LDS usage**: Can use `rocprof` to verify LDS allocation

## Educational Additions

Add comments explaining:
- Why LDS is beneficial (data reuse, bandwidth hierarchy)
- The relationship between KIterPerWarp > 1 and temporal reuse in K
- How the same distribution works for global and LDS operations
- Synchronization requirements (`block_sync_lds()`)
- Memory layout considerations
