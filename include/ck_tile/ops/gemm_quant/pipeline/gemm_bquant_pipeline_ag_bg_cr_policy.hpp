// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "gemm_group_quant_utils.hpp"

namespace ck_tile {

struct GemmBQuantPipelineAgBgCrDefaultPolicy : public UniversalGemmPipelineAgBgCrPolicy
{
    using Base = UniversalGemmPipelineAgBgCrPolicy;
    using Base::I0;
    using Base::I1;
    using Base::I2;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeBQ()
    {
        using BQLayout                = remove_cvref_t<typename Problem::BQLayout>;
        using BQDataType              = remove_cvref_t<typename Problem::BQDataType>;
        constexpr index_t NPerBlock   = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockBQ = KPerBlock / Problem::kQuantGroupSize;

        static_assert(std::is_same_v<BQLayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
        return GetABQGlobalVectorLoadSize<Problem, BQDataType, NPerBlock, KPerBlockBQ>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBQDramTileDistribution()
    {
        using BQLayout       = remove_cvref_t<typename Problem::BQLayout>;
        using BlockGemmShape = typename Problem::BlockGemmShape;

        constexpr index_t BlockSize    = Problem::kBlockSize;
        constexpr index_t NPerBlock    = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock    = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockBQ  = KPerBlock / Problem::kQuantGroupSize;
        constexpr index_t VecLoadSize  = GetVectorSizeBQ<Problem>();
        constexpr bool PreshuffleQuant = Problem::Traits::PreshuffleQuant;

        using WarpTile = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm = WarpGemmDispatcher<typename Problem::ComputeDataType,
                                            typename Problem::ComputeDataType,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC>;

        static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>);
        // if(get_block_id() == 0 && get_thread_id() == 0)
        // {
        //     printf("BlockSize: %d,  KPerBlockBQ(YPerTile): %d, NPerBlock(XPerTile): %d, "
        //            "VecLoadSize: %d\n",
        //            BlockSize,
        //            KPerBlockBQ,
        //            NPerBlock,
        //            VecLoadSize);
        // }
        if constexpr(PreshuffleQuant)
        {
            // if(get_block_id() == 0 && get_thread_id() == 0)
            // {
            //     printf("Inside PreshuffleQuant\n BlockSize: %d,  YPerTile: %d, XPerTile: %d, "
            //            "VecLoadSize: %d\n",
            //            BlockSize,
            //            NPerBlock / WarpGemm::kN,
            //            ck_tile::integer_least_multiple(WarpGemm::kN * KPerBlockBQ,
            //            get_warp_size()), VecLoadSize);
            // }
            using TileEncodingPattern =
                tile_distribution_encoding_pattern_bq<BlockGemmShape,
                                                      WarpGemm,
                                                      BlockSize,
                                                      NPerBlock / WarpGemm::kN, // 64/16 = 4
                                                      ck_tile::integer_least_multiple(
                                                          WarpGemm::kN * KPerBlockBQ,
                                                          get_warp_size()), //(32, 64) = 64
                                                      VecLoadSize,
                                                      PreshuffleQuant>;
            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
        else
        {
            using TileEncodingPattern = tile_distribution_encoding_pattern_bq<BlockGemmShape,
                                                                              WarpGemm,
                                                                              BlockSize,
                                                                              KPerBlockBQ, // 2
                                                                              NPerBlock,   // 64
                                                                              VecLoadSize>;

            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        static_assert(Problem::kQuantGroupSize % WarpTile::at(I2) == 0,
                      "KPerWarpGemm must be a multiple of kQuantGroupSize!");

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
        return BQuantBlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
