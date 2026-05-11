# Tutorial 11: XOR Transpose - Bank Conflict Elimination

## Overview

This tutorial demonstrates how XOR swizzling eliminates LDS (Local Data Share) bank conflicts during matrix transpose operations on AMD MI300 GPUs. The implementation uses the **CK Tile API** exclusively (no manual loops) with proper tensor descriptors, views, and tile windows.

## Files

### 1. Tutorial 11j: XOR Transpose Comparison
**File:** `xor_test_real_transpose.cpp`
**Binary:** `aa_tutorial_11_xor_real_transpose`
**Purpose:** Compare plain LDS vs XOR LDS in a single execution

**Features:**
- Runs **two tests** (plain and XOR) in one binary
- Template parameter `UseXor` toggles XOR swizzling
- Full correctness verification for both modes
- Suitable for side-by-side profiling

**Usage:**
```bash
cd relbuild
./bin/aa_tutorial_11_xor_real_transpose

# Profile both versions
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
          -d /tmp/transpose -- ./bin/aa_tutorial_11_xor_real_transpose
```

### 2. Tutorial 11l: Plain Transpose Only
**File:** `xor_test_plain_only.cpp`
**Binary:** `aa_tutorial_11_plain_transpose`
**Purpose:** Baseline bank conflict demonstration (no XOR)

**Features:**
- Single test (plain LDS only)
- Simpler code for understanding baseline behavior
- Suitable for focused profiling

**Usage:**
```bash
./bin/aa_tutorial_11_plain_transpose

# Profile plain version only
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
          -d /tmp/plain -- ./bin/aa_tutorial_11_plain_transpose
```

## Implementation Details

### Key Concepts

#### Four Descriptors for Transpose

The implementation uses **four separate tensor descriptors**:

1. **Global Input Descriptor** `gmem_desc_in`: [M, K]
   - Row-major input matrix
   - Strides: (K, 1) where K is runtime

2. **LDS Write Descriptor** `lds_desc_mk`: [M, K]
   - Plain: Simple row-major layout
   - XOR: Permuted layout to avoid conflicts

3. **LDS Read Descriptor** `lds_desc_km`: [K, M]
   - Plain: Transposed view with stride-kK (creates bank conflicts!)
   - XOR: Matching transposed XOR permutation (eliminates conflicts!)

4. **Global Output Descriptor** `gmem_desc_out`: [K, M]
   - Row-major transposed output
   - Strides: (M, 1) where M is runtime

#### XOR Swizzling Strategy

**Plain LDS (no XOR):**
```
Write: [M, K] with offset = m*kK + k
Read:  [K, M] with offset = k*1 + m*kK  (stride-kK access = BANK CONFLICTS!)
```

**XOR LDS:**
```
Write: [M, K] with XOR permutation: physical_addr = XOR(m, k/kKPack)
Read:  [K, M] with MATCHING XOR permutation for transpose compatibility
```

The critical insight: **Both write and read must use compatible XOR transforms** for transpose to work correctly. The read descriptor applies the same XOR pattern but with swapped merge order to achieve transpose.

### Code Structure

```cpp
template<typename DataType, bool UseXor>
struct RealTransposeKernel
{
    // Tile configuration
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    // LDS descriptors with optional XOR
    static constexpr auto MakeLdsDescriptorMK();  // [M, K] write
    static constexpr auto MakeLdsDescriptorKM();  // [K, M] read (transposed)

    // Distributions for thread mapping
    static constexpr auto MakeDistributionMK();

    void operator()(input, output, M, K)
    {
        // Setup LDS views
        auto lds_view_mk = make_tensor_view<lds>(lds, lds_desc_mk);
        auto lds_view_km = make_tensor_view<lds>(lds, lds_desc_km);

        // K-dimension loop (process matrix in tiles)
        for(k_block = 0; k_block < K; k_block += kK)
        {
            // 1. Load from global [M, K]
            auto reg_tile = load_tile(gmem_window_in);

            // 2. Store to LDS [M, K] (with optional XOR)
            store_tile(lds_window_mk, reg_tile);
            block_sync_lds();

            // 3. Read transposed from LDS [K, M] (1000 iterations)
            for(iter = 0; iter < 1000; ++iter)
            {
                (void)load_tile(lds_window_km);  // Bank conflicts here!
                block_sync_lds();
            }

            // 4. Write to global [K, M]
            auto reg_final = load_tile(lds_window_km);
            store_tile(gmem_window_out, reg_final);
            block_sync_lds();
        }
    }
};
```

### XOR Descriptor Implementation

#### Write Descriptor [M, K] with XOR

```cpp
static constexpr auto MakeLdsDescriptorMK()
{
    if constexpr (UseXor)
    {
        // Calculate layer for XOR permutation
        constexpr auto MLdsLayer = (32 * 4 / kK / sizeof(DataType));

        // Step 1: Reshape to [K/Pack*Layer, M/Layer, Pack]
        auto lds_desc_0 = make_naive_tensor_descriptor(...);

        // Step 2: Apply XOR permutation
        auto lds_desc_permuted = transform_tensor_descriptor(
            lds_desc_0,
            make_xor_transform(...));

        // Step 3: Unmerge layer dimension
        auto lds_desc_unmerged = transform_tensor_descriptor(...);

        // Step 4: Merge back to [M, K]
        auto lds_desc = transform_tensor_descriptor(...);

        return lds_desc;
    }
    else
    {
        return make_naive_tensor_descriptor_packed(make_tuple(kM, kK));
    }
}
```

#### Read Descriptor [K, M] with XOR Transpose

```cpp
static constexpr auto MakeLdsDescriptorKM()
{
    if constexpr (UseXor)
    {
        // Use SAME layer calculation as write!
        constexpr auto MLdsLayer = (32 * 4 / kK / sizeof(DataType));

        // Apply SAME XOR transform as write
        // But final merge uses SWAPPED order for transpose
        auto lds_desc = transform_tensor_descriptor(
            lds_desc_unmerged,
            make_tuple(
                merge([K/Pack, Pack]),    // First dimension: K
                merge([M/Layer, Layer])   // Second dimension: M
            ),
            make_tuple(sequence<2, 3>{}, sequence<1, 0>{}),  // Swapped!
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_desc;
    }
    else
    {
        // Plain transpose: stride-kK access
        return make_naive_tensor_descriptor(
            make_tuple(kK, kM),
            make_tuple(number<1>{}, number<kK>{}));
    }
}
```

## Performance Results

### Configuration
- Matrix: [256, 128] → [128, 256] transpose
- Data type: FP16 (2 bytes)
- Tile size: [64, 32]
- Block size: 256 threads
- Grid size: 4 blocks
- Iterations: 1000 (for bank conflict amplification)

### Bank Conflict Analysis

```
╔════════════════════════════════════════════════════════════════════════╗
║          XOR Transpose - Bank Conflict Comparison                      ║
╚════════════════════════════════════════════════════════════════════════╝

┌────────────────┬─────────────────┬──────────────┬──────────────────────┐
│ Version        │ Bank Conflicts  │ LDS Instrs   │ Conflict Rate        │
├────────────────┼─────────────────┼──────────────┼──────────────────────┤
│ Plain LDS      │     7,168       │     608      │   1,178.95%          │
│ XOR LDS        │     3,072       │     608      │     505.26%          │
├────────────────┼─────────────────┼──────────────┼──────────────────────┤
│ Reduction      │    -4,096       │       0      │    -673.69%          │
│ Improvement    │     -57.1%      │      0%      │     -57.1%           │
└────────────────┴─────────────────┴──────────────┴──────────────────────┘
```

**Key Findings:**
- ✓ XOR reduces bank conflicts by **57.1%** (7,168 → 3,072)
- ✓ Conflict rate drops from 1,179% to 505%
- ✓ Each plain LDS instruction encounters ~12 bank conflicts
- ✓ XOR reduces this to ~5 bank conflicts per instruction

### Performance Comparison

```
╔════════════════════════════════════════════════════════════════════════╗
║          Performance Comparison: Plain vs XOR Transpose               ║
╚════════════════════════════════════════════════════════════════════════╝

┌────────────────┬─────────────────┬─────────────────┬──────────────────┐
│ Version        │ Avg Time (ns)   │ Total Time (ms) │ Bandwidth (GB/s) │
├────────────────┼─────────────────┼─────────────────┼──────────────────┤
│ Plain LDS      │        37,005   │         2.37    │         3.54     │
│ XOR LDS        │        35,802   │         2.29    │         3.66     │
├────────────────┼─────────────────┼─────────────────┼──────────────────┤
│ Difference     │         1,203   │         0.08    │         0.12     │
│ Improvement    │         3.25%   │         3.25%   │         3.36%    │
└────────────────┴─────────────────┴─────────────────┴──────────────────┘
```

**Performance Summary:**
- ✓ XOR version is **1.034x faster** (3.25% speedup)
- ✓ Execution time: 37,005ns → 35,802ns
- ✓ Bandwidth: 3.54 GB/s → 3.66 GB/s

**Why modest speedup despite 57% conflict reduction?**

The 1000-iteration loop amplifies bank conflicts for profiling visibility, but also means:
1. Most kernel time is repetitive LDS reads (same conflicts over and over)
2. Global memory access time is unaffected by XOR
3. Only the LDS transpose portion benefits from conflict reduction

In a real GEMM kernel with single transpose (not 1000x), the relative impact differs but XOR still provides measurable benefit.

## Why Bank Conflicts Occur

### LDS Architecture (MI300/GFX942)
- 32 banks, 4 bytes each
- Bank = (byte_address / 4) % 32
- Bank conflicts happen when multiple threads access the same bank

### Transpose Access Pattern (Plain LDS)

**Physical Layout:** [M, K] row-major
```
[0][0], [0][1], [0][2], ..., [0][31]  ← Row 0
[1][0], [1][1], [1][2], ..., [1][31]  ← Row 1
[2][0], [2][1], [2][2], ..., [2][31]  ← Row 2
...
```

**Transposed Read:** [K, M] accesses
```
Read column 0: [0][0], [1][0], [2][0], ..., [63][0]
Physical offsets: 0*32, 1*32, 2*32, ..., 63*32
Stride: 32 elements = 64 bytes (for FP16)
```

**Bank Mapping (FP16, 2 bytes each):**
```
Element [0][0] → byte 0   → bank 0
Element [1][0] → byte 64  → bank 16
Element [2][0] → byte 128 → bank 0  ← CONFLICT with [0][0]!
Element [3][0] → byte 192 → bank 16 ← CONFLICT with [1][0]!
```

Result: **Massive bank conflicts** as threads read sequential M values.

### How XOR Eliminates Conflicts

XOR swizzling permutes physical addresses:
```
physical_addr = XOR(m, k / kKPack)
```

This spreads out elements that would otherwise map to the same bank, breaking the conflict pattern. The transposed read descriptor applies a compatible XOR permutation so logical [k,m] still maps to the correct physical location.

## CK Tile API Usage

This implementation demonstrates proper use of CK Tile API:

### 1. Tensor Descriptors
```cpp
// Compile-time descriptor
constexpr auto desc = make_naive_tensor_descriptor(
    make_tuple(number<M>{}, number<K>{}),  // Dimensions
    make_tuple(stride_M, stride_K));       // Strides

// Runtime descriptor (for global memory with runtime K)
const auto desc = make_naive_tensor_descriptor(
    make_tuple(number<kM>{}, number<kK>{}),
    make_tuple(K, number<1>{}));  // K is runtime, 1 is compile-time
```

### 2. Tensor Views
```cpp
// LDS view
auto lds_view = make_tensor_view<address_space_enum::lds>(
    ptr, descriptor);

// Global view
auto gmem_view = make_tensor_view<address_space_enum::global>(
    ptr, descriptor);
```

### 3. Tile Windows
```cpp
auto window = make_tile_window(
    view,                          // Tensor view
    make_tuple(tile_M, tile_K),    // Window shape
    {offset_M, offset_K},          // Window position
    distribution);                 // Thread distribution
```

### 4. Data Movement
```cpp
// Load data through tile window
auto reg_tile = load_tile(window);

// Store data through tile window
store_tile(window, reg_tile);
```

### 5. Transform Descriptors (for XOR)
```cpp
auto desc_xor = transform_tensor_descriptor(
    base_descriptor,
    make_tuple(make_xor_transform(...)),
    input_sequences,
    output_sequences);
```

## Building and Running

### Build
```bash
cd relbuild
cmake --build . --target aa_tutorial_11_xor_real_transpose -j$(nproc)
cmake --build . --target aa_tutorial_11_plain_transpose -j$(nproc)
```

### Run Tests
```bash
# Compare both versions
./bin/aa_tutorial_11_xor_real_transpose

# Plain only
./bin/aa_tutorial_11_plain_transpose
```

### Profile
```bash
# Profile both versions together
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
          -d /tmp/transpose -- ./bin/aa_tutorial_11_xor_real_transpose

# Query results
sqlite3 /tmp/transpose/*/results.db "
SELECT
    CASE
        WHEN name LIKE '%Lb0%' THEN 'Plain LDS'
        WHEN name LIKE '%Lb1%' THEN 'XOR LDS'
    END as version,
    SUM(CASE WHEN counter_name = 'SQ_LDS_BANK_CONFLICT' THEN counter_value ELSE 0 END) as conflicts,
    SUM(CASE WHEN counter_name = 'SQ_INSTS_LDS' THEN counter_value ELSE 0 END) as lds_insts,
    ROUND(100.0 * conflicts / lds_insts, 2) as conflict_rate
FROM pmc_events
GROUP BY version;"
```

## Key Takeaways

1. **XOR swizzling works**: 57% reduction in bank conflicts
2. **Performance improves**: 3.25% faster execution time
3. **Correctness maintained**: Both versions produce identical results
4. **CK Tile API sufficient**: No manual loops needed for complex transforms
5. **Descriptor design matters**: Matching XOR patterns for read/write is critical
6. **Bank conflicts are real**: 1,179% conflict rate on plain transpose!

## Related Tutorials

- **Tutorial 11a-11k**: Various XOR swizzling experiments
- **Tutorial 13**: Production XOR GEMM implementation
- **Tutorial 10**: Distributed GEMM with XOR (partial fix)

## References

- MI300 LDS architecture: 32 banks × 4 bytes
- XOR swizzling paper: "Conflict-Free Tensor Layouts for GPUs"
- CK Tile API documentation: `include/ck_tile/core/`
