// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

// Default policy for MHC kernel
// This policy provides warp gemm configuration for MHC operations
struct MHCDefaultPolicy
{
    // Provide warp gemm configuration for float data types
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        // For float x float -> float, provide a simple configuration
        if constexpr(std::is_same_v<typename Problem::ADataType, float> &&
                     std::is_same_v<typename Problem::BDataType, float> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
            // Use a simple warp gemm configuration for float
            // This is a basic configuration - can be optimized later
            using WG = WarpGemmDispatcher<float, float, float,
                                          16, 16, 16,  // M, N, K per warp
                                          true, false, false,
                                          WGAttrNumAccessEnum::Single>;
            return make_tuple(WG{}, 1, 1);  // 1 warp in M, 1 warp in N
        }
        else
        {
            // For other data types, delegate to default policy
            return BlockGemmASmemBSmemCRegV1DefaultPolicy::GetWarpGemmMWarpNWarp<Problem>();
        }
    }
    
    // Get shared memory size needed for the kernel
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // For MHC, we need shared memory for the BlockGemm operations
        // The size depends on the block shape and data types
        // This is a placeholder - actual size calculation would depend on
        // the specific BlockGemm implementation requirements
        return 0; // Will be calculated by BlockGemm internally
    }
};

} // namespace ck_tile
