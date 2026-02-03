// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

// Policy for BlockGemmASmemBSmemCReg with MFMA 16x16x(16x2) instruction
struct BlockGemmASmemBSmemCRegPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        // Kernel D uses no block tile shape adjustment, so kMWarp=4, kNWarp=1
        constexpr index_t kMWarp = 4;
        constexpr index_t kNWarp = 1;

        // MFMA 16x16x(16x2) - This is MFMA 16x16x32 instruction
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(
                WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution{}, kMWarp, kNWarp);
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(
                WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution{}, kMWarp, kNWarp);
        }
    }
};

} // namespace ck_tile
