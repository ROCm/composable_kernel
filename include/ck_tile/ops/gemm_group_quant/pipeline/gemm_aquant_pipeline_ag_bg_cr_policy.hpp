// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "gemm_group_quant_utils.hpp"

namespace ck_tile {

struct GemmAQuantPipelineAgBgCrDefaultPolicy : public UniversalGemmPipelineAgBgCrPolicy
{
    using Base = UniversalGemmPipelineAgBgCrPolicy;
    using Base::I0;
    using Base::I1;
    using Base::I2;

    using Base::ATileAccessPattern;
    using Base::BTileAccessPattern;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeAQ()
    {
        using AQLayout                = remove_cvref_t<typename Problem::AQLayout>;
        using AQDataType              = remove_cvref_t<typename Problem::AQDataType>;
        constexpr index_t MPerBlock   = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockAQ = KPerBlock / Problem::kQuantGroupSize;

        static_assert(std::is_same_v<AQLayout, ck_tile::tensor_layout::gemm::RowMajor>);
        return GetAQGlobalVectorLoadSize<Problem, AQDataType, MPerBlock, KPerBlockAQ>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeAQDramTileDistribution()
    {
        using AQLayout       = remove_cvref_t<typename Problem::AQLayout>;
        using BlockGemmShape = typename Problem::BlockGemmShape;

        constexpr index_t BlockSize   = Problem::kBlockSize;
        constexpr index_t MPerBlock   = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockAQ = KPerBlock / Problem::kQuantGroupSize;
        constexpr index_t VecLoadSize = GetVectorSizeAQ<Problem>();
        using WarpTile                = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm                = WarpGemmMfmaDispatcher<typename Problem::ComputeDataType,
                                                               typename Problem::ComputeDataType,
                                                               typename Problem::CDataType,
                                                               WarpTile::at(I0),
                                                               WarpTile::at(I1),
                                                               WarpTile::at(I2),
                                                               false>;

        static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
        using TileEncodingPattern = TileDistributionEncodingPatternAQ<BlockGemmShape,
                                                                      WarpGemm,
                                                                      BlockSize,
                                                                      MPerBlock,
                                                                      KPerBlockAQ,
                                                                      VecLoadSize>;

        return TileEncodingPattern::Make2DStaticTileDistribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        static_assert(Problem::kQuantGroupSize % WarpTile::at(I2) == 0,
                      "KPerWarpGemm must be a multiple of kQuantGroupSize!");

        using WarpGemm = WarpGemmMfmaDispatcher<typename Problem::ComputeDataType,
                                                typename Problem::ComputeDataType,
                                                typename Problem::CDataType,
                                                WarpTile::at(I0),
                                                WarpTile::at(I1),
                                                WarpTile::at(I2),
                                                false>;
        static_assert(std::is_same_v<typename Problem::ComputeDataType, fp8_t> ||
                      std::is_same_v<typename Problem::ComputeDataType, bf8_t>);
        static_assert(std::is_same_v<typename Problem::CDataType, float>);
        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                                      typename Problem::BDataType,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return AQuantBlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
