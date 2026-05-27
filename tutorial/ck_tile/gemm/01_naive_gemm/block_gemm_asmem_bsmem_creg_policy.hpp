// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

// Controls whether to use the A/B-swapped MFMA variant with transposed C register layout.
// 0 = WarpGemmMfmaF16F16F32M32N32K8                        (standard, no swap, no transposed C)
// 1 = WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution  (swap A/B in MFMA + transposed C
// layout)
#ifndef CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION
#define CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION 1
#endif

namespace ck_tile {

// Default policy for BlockGemmASmemBSmemCReg
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmASmemBSmemCRegPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        constexpr index_t kMWarp = 4;
        constexpr index_t kNWarp = 1;

        // mfma m32 n32 k8
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
#if CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION
            return make_tuple(
                WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution{}, kMWarp, kNWarp);
#else
            return make_tuple(WarpGemmMfmaF16F16F32M32N32K8{}, kMWarp, kNWarp);
#endif
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
#if CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION
            return make_tuple(
                WarpGemmMfmaBf16Bf16F32M32N32K8TransposedCDistribution{}, kMWarp, kNWarp);
#else
            return make_tuple(WarpGemmMfmaBf16Bf16F32M32N32K8{}, kMWarp, kNWarp);
#endif
        }
        else
        {
            static_assert(false, "Unsupported data type configuration for GEMM warp execution.");
        }
    }
};

} // namespace ck_tile
