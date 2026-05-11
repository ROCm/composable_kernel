# Tutorial 10 XOR Debugging Notes

## Investigation Summary

### XOR Descriptor Comparison
✅ **Tutorial 10's XOR descriptor matches 02_gemm and production code EXACTLY**
- Same 4-step transform pattern
- Same MLdsLayer calculation: `(32 * 4) / (kKPerBlock * DataTypeSize) = 2`
- Same XOR transform parameters
- Same unmerge/merge logic

### Key Findings

1. **02_gemm has two paths:**
   - `ENABLE_PREFETCH` path: Uses distributions with GEMM windows
   - Non-prefetch path: Creates windows WITHOUT distributions
   - Both paths work with XOR descriptors

2. **Tutorial 10 vs 02_gemm difference:**
   - Tutorial 10: Manually loads tiles using `load_tile(lds_gemm_window)`
   - 02_gemm: Passes windows to `BlockGemm()` class which handles loading internally

3. **Distribution usage:**
   - Tutorial 10 creates GEMM windows WITH `MakeAGemmDistribution()`
   - This is similar to 02_gemm's prefetch path
   - Should work, but doesn't

### Window Creation Comparison

**Tutorial 10:**
```cpp
auto a_lds_gemm_window = make_tile_window(
    a_lds_view,                            // XOR-swizzled view
    make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
    {0, 0},
    MakeAGemmDistribution());              // Custom GEMM distribution
```

**02_gemm (prefetch path):**
```cpp
auto a_lds_gemm_window = make_tile_window(
    a_lds_block,                           // XOR-swizzled view
    make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
    {0, 0},
    make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode()));
```

The key difference: Tutorial 10 uses a custom `MakeAGemmDistribution()` function, while 02_gemm uses `BlockGemm::MakeABlockDistributionEncode()`.

### GEMM Distribution Comparison Needed

Need to compare:
1. Tutorial 10's `MakeAGemmDistribution()`
2. vs BlockGemm's `MakeABlockDistributionEncode()`
3. vs production pipeline's GEMM distributions

The distribution might not be compatible with XOR swizzling.

## Next Steps

1. Compare Tutorial 10's GEMM distribution with 02_gemm's BlockGemm distribution
2. Check if the distribution is accessing LDS in a pattern that conflicts with XOR
3. Possibly use a different distribution that's XOR-compatible
