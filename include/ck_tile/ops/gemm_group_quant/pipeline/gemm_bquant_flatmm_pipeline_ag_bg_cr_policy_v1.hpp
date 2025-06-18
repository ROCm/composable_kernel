// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"
#include "ck_tile/ops/gemm_group_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

struct GemmBQuantFlatmmPipelineAgBgCrDefaultPolicyV1 : public UniversalFlatmmPipelineAgBgCrPolicy
{

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeBQ()
    {
        using BQDataType              = remove_cvref_t<typename Problem::BQDataType>;
        constexpr index_t NPerBlock   = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockBQ = KPerBlock / Problem::kQuantGroupSize;

        return GetABQGlobalVectorLoadSize<Problem, BQDataType, NPerBlock, KPerBlockBQ>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBQDramTileDistribution()
    {
        return GemmBQuantPipelineAgBgCrDefaultPolicy::MakeBQDramTileDistribution<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockPrimitive()
    {
        using AccDataType = float;
        using BlockWarps  = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile    = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm    = WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                                   typename Problem::BDataType,
                                                   AccDataType,
                                                   WarpTile::at(I0),
                                                   WarpTile::at(I1),
                                                   WarpTile::at(I2),
                                                   Problem::TransposeC>;

        // TODO : Use a custom block policy for AsBrCr
        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                                      typename Problem::BDataType,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return BlockGemmBQuantASmemBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
