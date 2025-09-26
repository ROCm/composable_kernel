// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/fused_moe/pipeline/fused_moegemm_pipeline_flatmm_policy.hpp"

namespace ck_tile {

// Fixed policy that maintains consistent alignment regardless of atomic setting
struct FusedMoeGemmPipelineFlatmmPolicy_NoAtomic : public FusedMoeGemmPipelineFlatmmPolicy
{
    // Override GetAlignment_O to maintain consistent alignment
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_O()
    {
        // FIXED: Always use the same alignment as atomic version
        // This prevents memory access pattern changes when atomics are disabled
        if constexpr(sizeof(typename Problem::ODataType) == 2)  // BF16/FP16
        {
            return 2;  // Same as atomic version
        }
        else if constexpr(sizeof(typename Problem::ODataType) == 4)  // FP32
        {
            return 1;  // Same as atomic version
        }
        else
        {
            // Fallback for other data types
            return 16 / sizeof(typename Problem::ODataType);
        }
    }
    
    // Note: All other functions inherited from base policy
    // This ensures we only change the alignment behavior
};

} // namespace ck_tile
