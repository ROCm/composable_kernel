// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

struct BlockFmhaPipelineQRKSVSAsyncDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                          /* AsyncCopy = */ true,
                                          /* NumPrefetchV = */ 2>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        constexpr index_t BlockGemmK = (KLoadOnce && Problem::BlockFmhaShape::kQKHeaddim ==
                                                         Problem::BlockFmhaShape::kSubQKHeaddim)
                                           ? Problem::BlockFmhaShape::kSubQKHeaddim
                                           : Problem::BlockFmhaShape::kK0;

        using GemmProblem = BlockGemmProblem<
            typename Problem::QDataType,
            typename Problem::KDataType,
            typename Problem::SaccDataType,
            Problem::kNumGemm0Warps * get_warp_size(),
            TileGemmShape<
                sequence<Problem::BlockFmhaShape::kM0, Problem::BlockFmhaShape::kN0, BlockGemmK>,
                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        constexpr auto warp_gemm = []() {
            constexpr index_t WarpGemmM = Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{});
            static_assert(WarpGemmM == 4 || WarpGemmM == 16 || WarpGemmM == 32);

            if constexpr(std::is_same_v<typename Problem::QDataType, half_t> &&
                         std::is_same_v<typename Problem::KDataType, half_t> &&
                         std::is_same_v<typename Problem::SaccDataType, float>)
            {
                if constexpr(WarpGemmM == 32)
                    return WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr(WarpGemmM == 16)
                    return WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{};
                else // WarpGemmM == 4
                    return WarpGemmMfmaF16F16F32M4N64K16{};
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, bf16_t> &&
                              std::is_same_v<typename Problem::KDataType, bf16_t> &&
                              std::is_same_v<typename Problem::SaccDataType, float>)
            {
                if constexpr(WarpGemmM == 32)
                    return WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr(WarpGemmM == 16)
                    return WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{};
                else // WarpGemmM == 4
                    return WarpGemmMfmaBf16Bf16F32M4N64K16{};
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, fp8_t> &&
                              std::is_same_v<typename Problem::KDataType, fp8_t> &&
                              std::is_same_v<typename Problem::SaccDataType, float>)
            {
                static_assert(WarpGemmM == 32);

                // TODO: hard coded here. Otherwise, it may incorrect result
                constexpr index_t swizzle_factor = 4;
                return WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution<
                    swizzle_factor>{};
            } // TODO - bf8_t
        }();

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::QDataType,
                                                 typename Problem::KDataType,
                                                 typename Problem::SaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                 decltype(warp_gemm)>;

        if constexpr(1 < Problem::kNumGemm0Warps)
            return BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
