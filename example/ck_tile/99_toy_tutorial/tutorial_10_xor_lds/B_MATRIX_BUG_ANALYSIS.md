# B Matrix XOR Descriptor Bug Analysis

## Test Results

**Copy-only test** (xor_copy_only_test.cpp):
- A matrix: ✓ PASSED (0 errors)
- B matrix: ✗ FAILED (3202/4096 errors, ~78% failure rate)

## What This Tells Us

1. The distributions are correct (A works, user confirmed they work in Tutorial 9)
2. The A matrix XOR descriptor is correct
3. **The B matrix XOR descriptor has a bug**

## Descriptor Comparison

### A Matrix XOR Descriptor (WORKS)
```cpp
// Initial: [K/kKPack*MLdsLayer, M/MLdsLayer, kKPack] = [8, 64, 8]
constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
    make_tuple(number<kKPerBlock / kKPack * MLdsLayer>{},  // 32/8*2 = 8
               number<kMPerBlock / MLdsLayer>{},            // 128/2 = 64
               number<kKPack>{}),                           // 8
    make_tuple(number<kKPack>{},                            // stride: 8
               number<kKPerBlock * MLdsLayer>{},            // stride: 64
               number<1>{}),                                // stride: 1
    number<kKPack>{}, number<1>{});

// XOR permutation on dims [1, 0]
constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
    a_lds_block_desc_0,
    make_tuple(make_xor_transform(make_tuple(number<kMPerBlock / MLdsLayer>{},      // 64
                                             number<kKPerBlock / kKPack * MLdsLayer>{})),  // 8
               make_pass_through_transform(number<kKPack>{})),
    make_tuple(sequence<1, 0>{}, sequence<2>{}),
    make_tuple(sequence<1, 0>{}, sequence<2>{}));

// Unmerge dim 0 into [MLdsLayer, K/kKPack]
constexpr auto a_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
    a_lds_block_desc_permuted,
    make_tuple(make_unmerge_transform(make_tuple(number<MLdsLayer>{},              // 2
                                                 number<kKPerBlock / kKPack>{})),   // 4
               make_pass_through_transform(number<kMPerBlock / MLdsLayer>{}),       // 64
               make_pass_through_transform(number<kKPack>{})),                      // 8
    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
    make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));
// Output dims: [MLdsLayer=2, M/MLdsLayer=64, K/kKPack=4, kKPack=8]
//              [dim0,        dim1,           dim2,        dim3]

// Final merge to [M, K]
constexpr auto a_lds_desc = transform_tensor_descriptor(
    a_lds_block_desc_xk0_mnldslayer_mn_xk1,
    make_tuple(make_merge_transform(make_tuple(number<kMPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
               make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
    make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
    make_tuple(sequence<0>{}, sequence<1>{}));
// Merges [dim1, dim0] = [M/MLdsLayer, MLdsLayer] = M -> output dim 0
// Merges [dim2, dim3] = [K/kKPack, kKPack] = K -> output dim 1
// Final: [M=128, K=32] ✓
```

### B Matrix XOR Descriptor (FAILS)
```cpp
// Initial: [K/kKPack*NLdsLayer, N/NLdsLayer, kKPack] = [8, 64, 8]
constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
    make_tuple(number<kKPerBlock / kKPack * NLdsLayer>{},  // 32/8*2 = 8
               number<kNPerBlock / NLdsLayer>{},            // 128/2 = 64
               number<kKPack>{}),                           // 8
    make_tuple(number<kKPack>{},                            // stride: 8
               number<kKPerBlock * NLdsLayer>{},            // stride: 64
               number<1>{}),                                // stride: 1
    number<kKPack>{}, number<1>{});

// XOR permutation on dims [1, 0]
constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
    b_lds_block_desc_0,
    make_tuple(make_xor_transform(make_tuple(number<kNPerBlock / NLdsLayer>{},      // 64
                                             number<kKPerBlock / kKPack * NLdsLayer>{})),  // 8
               make_pass_through_transform(number<kKPack>{})),
    make_tuple(sequence<1, 0>{}, sequence<2>{}),
    make_tuple(sequence<1, 0>{}, sequence<2>{}));

// Unmerge dim 0 into [NLdsLayer, K/kKPack]
constexpr auto b_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
    b_lds_block_desc_permuted,
    make_tuple(make_unmerge_transform(make_tuple(number<NLdsLayer>{},              // 2
                                                 number<kKPerBlock / kKPack>{})),   // 4
               make_pass_through_transform(number<kNPerBlock / NLdsLayer>{}),       // 64
               make_pass_through_transform(number<kKPack>{})),                      // 8
    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
    make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));
// Output dims: [NLdsLayer=2, N/NLdsLayer=64, K/kKPack=4, kKPack=8]
//              [dim0,        dim1,           dim2,        dim3]

// Final merge - THIS IS WHERE THE BUG MIGHT BE
constexpr auto b_lds_desc = transform_tensor_descriptor(
    b_lds_block_desc_xk0_mnldslayer_mn_xk1,
    make_tuple(make_merge_transform(make_tuple(number<kNPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
               make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
    make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
    make_tuple(sequence<0>{}, sequence<1>{}));
// Merges [dim1, dim0] = [N/NLdsLayer, NLdsLayer] = N -> output dim 0
// Merges [dim2, dim3] = [K/kKPack, kKPack] = K -> output dim 1
// Final: [N=128, K=32] ← Should be [K=32, N=128]!  ✗✗✗
```

## THE BUG

**B matrix final dimensions are [N, K] but should be [K, N]!**

The B matrix in global memory is transposed (N×K layout), but in LDS it should be stored as K×N for efficient GEMM access.

The final merge creates [N, K] instead of [K, N]. This means all accesses are transposed!

## The Fix

The B matrix final merge should be:
```cpp
constexpr auto b_lds_desc = transform_tensor_descriptor(
    b_lds_block_desc_xk0_mnldslayer_mn_xk1,
    make_tuple(make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{})),
               make_merge_transform(make_tuple(number<kNPerBlock / NLdsLayer>{}, number<NLdsLayer>{}))),
    make_tuple(sequence<2, 3>{}, sequence<1, 0>{}),  // SWAPPED!
    make_tuple(sequence<0>{}, sequence<1>{}));
// Merges [dim2, dim3] = [K/kKPack, kKPack] = K -> output dim 0
// Merges [dim1, dim0] = [N/NLdsLayer, NLdsLayer] = N -> output dim 1
// Final: [K=32, N=128] ✓
```

## Wait... Check 02_gemm!

Need to verify if 02_gemm has the same bug or if they handle B differently!
