// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace tutorial_10 {

using namespace ck_tile;

// ============================================================================
// XOR DESCRIPTOR CREATION
// ============================================================================
// Creates XOR-swizzled LDS descriptors for bank conflict avoidance
//
// Pattern from 02_gemm production code:
// 1. Reshape into layers based on bank width (128 bytes)
// 2. Apply XOR permutation to redistribute addresses
// 3. Unmerge dimensions
// 4. Merge back to logical [M,K] or [K,N] layout
//
// XOR formula: idx_new = idx_old ^ (other_idx % length)
// This spreads consecutive accesses across different banks

template<typename DataType, index_t kMPerBlock, index_t kKPerBlock>
CK_TILE_HOST_DEVICE static constexpr auto MakeALdsXorDescriptor()
{
    constexpr index_t kKPack = 8;  // Vector width for half_t

    // Calculate layer size for XOR swizzling
    constexpr auto DataTypeSize = sizeof(DataType);
    constexpr auto MLdsLayer =
        (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

    // Step 1: Reshape into [K/kKPack * MLdsLayer, M/MLdsLayer, kKPack]
    constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kKPerBlock / kKPack * MLdsLayer>{},
                   number<kMPerBlock / MLdsLayer>{},
                   number<kKPack>{}),
        make_tuple(number<kKPack>{},
                   number<kKPerBlock * MLdsLayer>{},
                   number<1>{}),
        number<kKPack>{},
        number<1>{});

    // Step 2: Apply XOR permutation
    constexpr auto lds_desc_permuted = transform_tensor_descriptor(
        lds_desc_0,
        make_tuple(make_xor_transform(make_tuple(number<kMPerBlock / MLdsLayer>{},
                                                 number<kKPerBlock / kKPack * MLdsLayer>{})),
                   make_pass_through_transform(number<kKPack>{})),
        make_tuple(sequence<1, 0>{}, sequence<2>{}),
        make_tuple(sequence<1, 0>{}, sequence<2>{}));

    // Step 3: Unmerge
    constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
        lds_desc_permuted,
        make_tuple(make_unmerge_transform(
                       make_tuple(number<MLdsLayer>{}, number<kKPerBlock / kKPack>{})),
                   make_pass_through_transform(number<kMPerBlock / MLdsLayer>{}),
                   make_pass_through_transform(number<kKPack>{})),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
        make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

    // Step 4: Merge back to [M, K]
    constexpr auto lds_desc = transform_tensor_descriptor(
        lds_desc_unmerged,
        make_tuple(
            make_merge_transform(make_tuple(number<kMPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
            make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return lds_desc;
}

template<typename DataType, index_t kNPerBlock, index_t kKPerBlock>
CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsXorDescriptor()
{
    constexpr index_t kKPack = 8;  // Vector width for half_t

    // Calculate layer size for XOR swizzling
    constexpr auto DataTypeSize = sizeof(DataType);
    constexpr auto NLdsLayer =
        (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

    // Step 1: Reshape into [K/kKPack * NLdsLayer, N/NLdsLayer, kKPack]
    constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kKPerBlock / kKPack * NLdsLayer>{},
                   number<kNPerBlock / NLdsLayer>{},
                   number<kKPack>{}),
        make_tuple(number<kKPack>{},
                   number<kKPerBlock * NLdsLayer>{},
                   number<1>{}),
        number<kKPack>{},
        number<1>{});

    // Step 2: Apply XOR permutation
    constexpr auto lds_desc_permuted = transform_tensor_descriptor(
        lds_desc_0,
        make_tuple(make_xor_transform(make_tuple(number<kNPerBlock / NLdsLayer>{},
                                                 number<kKPerBlock / kKPack * NLdsLayer>{})),
                   make_pass_through_transform(number<kKPack>{})),
        make_tuple(sequence<1, 0>{}, sequence<2>{}),
        make_tuple(sequence<1, 0>{}, sequence<2>{}));

    // Step 3: Unmerge
    constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
        lds_desc_permuted,
        make_tuple(make_unmerge_transform(
                       make_tuple(number<NLdsLayer>{}, number<kKPerBlock / kKPack>{})),
                   make_pass_through_transform(number<kNPerBlock / NLdsLayer>{}),
                   make_pass_through_transform(number<kKPack>{})),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
        make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

    // Step 4: Merge back to [K, N]
    constexpr auto lds_desc = transform_tensor_descriptor(
        lds_desc_unmerged,
        make_tuple(
            make_merge_transform(make_tuple(number<kNPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
            make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return lds_desc;
}

} // namespace tutorial_10
