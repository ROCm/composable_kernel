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

    static constexpr auto MXdlPack = 2;
    static constexpr auto NXdlPack = 2;
    static constexpr auto KXdlPack = 2;

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

    // A Scale DRAM tile distribution
    // This is used to load the A scale data from DRAM into shared memory.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeAScaleDramTileDistribution()
    {
        using AScaleLayout   = remove_cvref_t<typename Problem::AScaleLayout>;
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using AScaleDataType = remove_cvref_t<typename Problem::AScaleDataType>;
        static constexpr index_t APackedSize =
            ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;

        constexpr index_t BlockSize      = Problem::kBlockSize;
        constexpr index_t MPerBlockScale = Problem::BlockGemmShape::kM / MXdlPack;
        constexpr index_t KPerBlock      = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockScale =
            KPerBlock * APackedSize / (Problem::kBlockScaleSize * KXdlPack);
        using WarpTile = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm = WarpGemmMfmaDispatcher<typename Problem::ComputeDataType,
                                                typename Problem::ComputeDataType,
                                                typename Problem::CDataType,
                                                WarpTile::at(I0),
                                                WarpTile::at(I1),
                                                WarpTile::at(I2),
                                                false>;

        static_assert(std::is_same_v<AScaleLayout, tensor_layout::gemm::RowMajor>);
        using TileEncodingPattern = TileDistributionEncodingPatternAScale<BlockGemmShape,
                                                                          WarpGemm,
                                                                          BlockSize,
                                                                          MPerBlockScale,
                                                                          KPerBlockScale>;
        return TileEncodingPattern::Make2DStaticTileDistribution();
    }

    // B Scale DRAM tile distribution
    // This is used to load the B scale data from DRAM into shared memory.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBScaleDramTileDistribution()
    {
        using BScaleLayout   = remove_cvref_t<typename Problem::BScaleLayout>;
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BScaleDataType = remove_cvref_t<typename Problem::BScaleDataType>;
        static constexpr index_t BPackedSize =
            ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

        constexpr index_t BlockSize      = Problem::kBlockSize;
        constexpr index_t NPerBlockScale = Problem::BlockGemmShape::kN / NXdlPack;
        constexpr index_t KPerBlock      = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockScale =
            KPerBlock * BPackedSize / (roblem::kBlockScaleSize * KXdlPack);
        using WarpTile = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm = WarpGemmMfmaDispatcher<typename Problem::ComputeDataType,
                                                typename Problem::ComputeDataType,
                                                typename Problem::CDataType,
                                                WarpTile::at(I0),
                                                WarpTile::at(I1),
                                                WarpTile::at(I2),
                                                false>;

        static_assert(std::is_same_v<BScaleLayout, tensor_layout::gemm::ColumnMajor>);
        using TileEncodingPattern = TileDistributionEncodingPatternBScale<BlockGemmShape,
                                                                          WarpGemm,
                                                                          BlockSize,
                                                                          NPerBlockScale,
                                                                          KPerBlockScale>;
        return TileEncodingPattern::Make2DStaticTileDistribution();
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
