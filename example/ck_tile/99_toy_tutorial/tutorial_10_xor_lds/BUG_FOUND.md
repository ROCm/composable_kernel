# BUG FOUND: B Matrix Dimension Mismatch

## Root Cause

Tutorial 10 has a **dimension order mismatch** between the B LDS descriptor and the B LDS window:

- **B LDS XOR descriptor** (copied from 02_gemm): produces **[N, K]** dimensions
- **B LDS window usage** (copied from Tutorial 9): expects **[K, N]** dimensions

## Evidence

### Tutorial 9 (WORKS ✓)
```cpp
// B LDS descriptor: [K, N]
constexpr auto b_lds_desc = make_naive_tensor_descriptor_packed(
    make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}));  // [K=32, N=64]

// B LDS window: [K, N] - MATCHES!
auto b_lds_gemm_window = make_tile_window(
    b_lds_view,
    make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}),  // [K=32, N=64]
    {0, 0},
    MakeBGemmDistribution());
```

### Tutorial 10 (FAILS ✗)
```cpp
// B LDS XOR descriptor: [N, K]
constexpr auto b_lds_desc = transform_tensor_descriptor(...);
// After all transforms, final dimensions are [N, K]

// B LDS window: [K, N] - MISMATCH!
auto b_lds_gemm_window = make_tile_window(
    b_lds_view,
    make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}),  // [K=32, N=64]
    {0, 0},
    MakeBGemmDistribution());
```

### 02_gemm (Production Code)
```cpp
// B LDS XOR descriptor: [N, K]
constexpr auto b_lds_block_desc = transform_tensor_descriptor(...);
// Final dimensions are [N, K]

// B LDS window: [N, K] - MATCHES!
auto b_copy_lds_window = make_tile_window(
    b_lds_block,
    make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),  // [N, K]
    {0, 0},
    b_copy_dram_window.get_tile_distribution());
```

## Why [N, K] vs [K, N]?

Both layouts work, but they must be **consistent**:
- Tutorial 9 uses [K, N] everywhere (simple packed layout)
- 02_gemm uses [N, K] everywhere (XOR swizzled layout)
- Tutorial 10 mixed them: [N, K] descriptor with [K, N] window usage!

The XOR swizzling pattern from 02_gemm produces [N, K] because it's optimized for how the B matrix is accessed in GEMM (column-wise reads).

## The Fix

Change Tutorial 10's B LDS window creation from **[K, N]** to **[N, K]** to match the XOR descriptor:

```cpp
// BEFORE (wrong):
auto b_lds_gemm_window = make_tile_window(
    b_lds_view,
    make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}),  // [K, N]
    {0, 0},
    MakeBGemmDistribution());

// AFTER (correct):
auto b_lds_gemm_window = make_tile_window(
    b_lds_view,
    make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),  // [N, K]
    {0, 0},
    MakeBGemmDistribution());
```

Same fix needed for `b_copy_lds_window` and `b_lds_copy_window`.

## Test Results

After this fix, the copy-only test should pass for B matrix!
