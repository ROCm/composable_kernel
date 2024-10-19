// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
struct BlockFmhaFwdSplitKVPipelineQRKSVSDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                          /* AsyncCopyK = */ false,
                                          /* AsyncCopyV = */ false,
                                          /* NumPrefetchK = */ 1,
                                          /* NumPrefetchV = */ 1>
{
    // hide the one inherited from BlockFmhaPipelineQXCustomPolicy<true>
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN0,
                                                    Problem::BlockFmhaShape::kK0>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        constexpr auto warp_gemm = []() {
            if constexpr(std::is_same_v<typename Problem::QDataType, half_t> &&
                         std::is_same_v<typename Problem::KDataType, half_t> &&
                         std::is_same_v<typename Problem::SaccDataType, float>)
            {
                return WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{};
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, bf16_t> &&
                              std::is_same_v<typename Problem::KDataType, bf16_t> &&
                              std::is_same_v<typename Problem::SaccDataType, float>)
            {
                return WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{};
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, fp8_t> &&
                              std::is_same_v<typename Problem::KDataType, fp8_t> &&
                              std::is_same_v<typename Problem::SaccDataType, float>)
            {
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

        return BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
