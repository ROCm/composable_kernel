// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

// Policy for BlockGemmASmemBSmemCReg with MFMA_32x32x16 (8x2) instruction
struct BlockGemmASmemBSmemCRegPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        // KERNEL_A uses 4x1 warp configuration
        constexpr index_t kMWarp = 4;
        constexpr index_t kNWarp = 1;

        // KERNEL_A uses mfma m32 n32 k16 (8x2 variant)
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(
                WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution{}, kMWarp, kNWarp);
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(
                WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution{}, kMWarp, kNWarp);
        }
        else
        {
            static_assert(false, "Unsupported data type configuration for GEMM warp execution.");
        }
    }
};

} // namespace ck_tile
