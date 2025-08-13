// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "gemm_mx_utils.hpp"

namespace ck_tile {

struct GemmMXPipelineAgBgCrDefaultPolicy : public UniversalGemmPipelineAgBgCrPolicy
{
    using Base = UniversalGemmPipelineAgBgCrPolicy;
    using Base::I0;
    using Base::I1;
    using Base::I2;

    using Base::ATileAccessPattern;
    using Base::BTileAccessPattern;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeAScale()
    {
        using AScaleLayout               = remove_cvref_t<typename Problem::AScaleLayout>;
        using AScaleDataType             = remove_cvref_t<typename Problem::AScaleDataType>;
        constexpr index_t MPerBlock      = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock      = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockScale = KPerBlock / Problem::kBlockScaleSize;

        static_assert(std::is_same_v<AScaleLayout, ck_tile::tensor_layout::gemm::RowMajor>);
        return GetScaleGlobalVectorLoadSize<Problem, AScaleDataType, MPerBlock, KPerBlockScale>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeBScale()
    {
        using BScaleLayout               = remove_cvref_t<typename Problem::BScaleLayout>;
        using BScaleDataType             = remove_cvref_t<typename Problem::BScaleDataType>;
        constexpr index_t NPerBlock      = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock      = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockScale = KPerBlock / Problem::kBlockScaleSize;

        static_assert(std::is_same_v<BScaleLayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
        return GetScaleGlobalVectorLoadSize<Problem, BScaleDataType, NPerBlock, KPerBlockScale>();
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

    // A Scale DRAM tile distribution
    // This is used to load the A scale data from DRAM into shared memory.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeAScaleDramTileDistribution()
    {
        using AScaleLayout   = remove_cvref_t<typename Problem::AScaleLayout>;
        using BlockGemmShape = typename Problem::BlockGemmShape;

        constexpr index_t BlockSize      = Problem::kBlockSize;
        constexpr index_t MPerBlock      = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock      = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockScale = KPerBlock / Problem::kBlockScaleSize;
        constexpr index_t VecLoadSize    = GetVectorSizeAScale<Problem>();
        using WarpTile                   = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm                   = WarpGemmMfmaDispatcher<typename Problem::ComputeDataType,
                                                typename Problem::ComputeDataType,
                                                typename Problem::CDataType,
                                                WarpTile::at(I0),
                                                WarpTile::at(I1),
                                                WarpTile::at(I2),
                                                false>;
        static constexpr auto MXdlPack   = 2;
        static constexpr auto NXdlPack   = 2;
        static constexpr auto KXdlPack   = 2;

        static_assert(std::is_same_v<AScaleLayout, tensor_layout::gemm::RowMajor>);
        using TileEncodingPattern = TileDistributionEncodingPatternAScale<BlockGemmShape,
                                                                          WarpGemm,
                                                                          BlockSize,
                                                                          MPerBlock,
                                                                          KPerBlockScale,
                                                                          MXdlPack,
                                                                          KXdlPack>;
        return TileEncodingPattern::Make2DStaticTileDistribution();
    }

    // B Scale DRAM tile distribution
    // This is used to load the B scale data from DRAM into shared memory.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBScaleDramTileDistribution()
    {
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        static_assert(Problem::kBlockScaleSize % WarpTile::at(I2) == 0,
                      "KPerWarpGemm must be a multiple of kBlockScaleSize!");

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
        return BlockUniversalGemmMXAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
