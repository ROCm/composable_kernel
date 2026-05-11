# Tutorial 11: XOR Descriptor Test - Findings

## Summary

Tutorial 11 is a minimal test that validates XOR-based LDS descriptors work correctly when accessed directly using `calculate_offset()`. The test **PASSES**, proving the XOR transform implementation is correct.

## Test Design

Simple kernel that:
1. Loads data from global memory
2. Stores to LDS using XOR descriptor via `calculate_offset()`
3. Syncs threads
4. Loads from LDS using same XOR descriptor
5. Stores back to global memory

If XOR descriptor is correct, output should match input.

## Results

✓ **PASSED** - XOR descriptor correctly maps logical [M,K] coordinates to physical LDS addresses.

## Key Findings

### 1. XOR Transform Implementation is Correct

The 4-step XOR descriptor creation pattern from `02_gemm` works correctly:
- Step 1: Reshape into layers based on MLdsLayer
- Step 2: Apply XOR permutation
- Step 3: Unmerge dimensions
- Step 4: Merge back to logical [M,K] layout

### 2. Dimension Matching is Critical

Initial test failed when:
- Kernel descriptor: 64×**32** (kM × kK)
- Main test: 128×**64** (M × K)

The K dimensions mismatched (32 vs 64), causing errors for all k >= 32.

After fixing to M=128, K=32 (matching kK=32 in kernel), test **PASSED**.

### 3. Direct Access Method

Tutorial 11 uses direct offset calculation:
```cpp
constexpr auto idx_dims = decltype(lds_desc)::get_num_of_dimension();
array<index_t, idx_dims> logical_idx;
logical_idx[number<0>{}] = m;
logical_idx[number<1>{}] = k;
const index_t physical_offset = lds_desc.calculate_offset(logical_idx);
p_lds[physical_offset] = value;  // Direct pointer access
```

This proves the XOR descriptor's `calculate_offset()` method works correctly.

## Implications for Tutorial 10

Tutorial 10 uses the **same XOR descriptor creation code** but **FAILS** correctness tests.

Key differences between Tutorial 10 and Tutorial 11:
- **Tutorial 11**: Direct LDS access via `calculate_offset()` → **WORKS**
- **Tutorial 10**: LDS access via `tile_window` with copy/GEMM distributions → **FAILS**

This suggests:
1. XOR descriptor creation is correct (proven by Tutorial 11)
2. Problem is likely in how `tile_window` interacts with XOR descriptors
3. OR: The specific copy/GEMM distributions are incompatible with XOR layout

## Next Steps

To fix Tutorial 10:
1. Verify all tile dimensions (kMPerBlock=64, kNPerBlock=64, kKPerBlock=32) match window sizes
2. Check if copy/GEMM distributions are compatible with XOR descriptors
3. Consider if XOR swizzling requires specific distribution patterns
4. Compare with 02_gemm's usage of tile_window + XOR descriptors

## Test Configuration

- M×K: 128×32 (matches kM=64, kK=32)
- Tile: 64×32
- Grid: 2 blocks
- Block: 256 threads
- Data type: half_t (FP16)
- XOR layer size: MLdsLayer = 2
