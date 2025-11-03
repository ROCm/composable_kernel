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

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeAQ()
    {
        using AQLayout                = remove_cvref_t<typename Problem::AQLayout>;
        using AQDataType              = remove_cvref_t<typename Problem::AQDataType>;
        constexpr index_t MPerBlock   = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockAQ = KPerBlock / Problem::QuantGroupSize::kK;

        static_assert(std::is_same_v<AQLayout, ck_tile::tensor_layout::gemm::RowMajor>);
        return GetABQGlobalVectorLoadSize<Problem, AQDataType, MPerBlock, KPerBlockAQ>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeAQDramTileDistribution()
    {
        using AQLayout       = remove_cvref_t<typename Problem::AQLayout>;
        using BlockGemmShape = typename Problem::BlockGemmShape;

        constexpr index_t BlockSize    = Problem::kBlockSize;
        constexpr index_t MPerBlock    = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock    = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockAQ  = KPerBlock / Problem::QuantGroupSize::kK;
        constexpr index_t VecLoadSize  = GetVectorSizeAQ<Problem>();
        constexpr bool PreshuffleQuant = Problem::Traits::PreshuffleQuant;
        using WarpTile                 = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm                 = WarpGemmDispatcher<typename Problem::ComputeDataType,
                                                            typename Problem::ComputeDataType,
                                                            typename Problem::CDataType,
                                                            WarpTile::at(I0),
                                                            WarpTile::at(I1),
                                                            WarpTile::at(I2),
                                                            Problem::TransposeC>;

        static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
        if constexpr(PreshuffleQuant)
        {
            using TileEncodingPattern = tile_distribution_encoding_pattern_aq<
                BlockGemmShape,
                WarpGemm,
                BlockSize,
                MPerBlock / WarpGemm::kM,
                ck_tile::integer_least_multiple(WarpGemm::kM * KPerBlockAQ, get_warp_size()),
                KPerBlockAQ,
                VecLoadSize,
                PreshuffleQuant>;

            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
        else
        {
            if constexpr(Problem::TransposeC)
            {
                using TileEncodingPatternTransposeC =
                    tile_distribution_encoding_pattern_aq_transposed_c<BlockGemmShape,
                                                                       WarpGemm,
                                                                       BlockSize,
                                                                       MPerBlock,
                                                                       KPerBlockAQ,
                                                                       VecLoadSize>;
                return TileEncodingPatternTransposeC::make_2d_static_tile_distribution();
            }
            else
            {
                using TileEncodingPattern = tile_distribution_encoding_pattern_aq<BlockGemmShape,
                                                                                  WarpGemm,
                                                                                  BlockSize,
                                                                                  MPerBlock,
                                                                                  KPerBlockAQ,
                                                                                  KPerBlockAQ,
                                                                                  VecLoadSize,
                                                                                  PreshuffleQuant>;

                return TileEncodingPattern::make_2d_static_tile_distribution();
            }
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        static_assert(Problem::QuantGroupSize::kK % WarpTile::at(I2) == 0,
                      "KPerWarpGemm must be a multiple of QuantGroupSize::kK!");

        using WarpGemm = WarpGemmDispatcher<typename Problem::ComputeDataType,
                                            typename Problem::ComputeDataType,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC>;
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
