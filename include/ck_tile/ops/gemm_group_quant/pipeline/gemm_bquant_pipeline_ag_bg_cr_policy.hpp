// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_group_quant/pipeline/gemm_group_quant_utils.hpp"

namespace ck_tile {

template <typename BlockGemmShape,
          typename WarpGemm,
          index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t VecSize>
struct TileDistributionEncodingPatternBQ : public TileDistributionEncodingPattern
{
    // TODO: make pattern where below condition does not need to hold - GGemmMultiDSplitk!
    static_assert(XPerTile % VecSize == 0, "XPerTile must be a multiple of VecSize!");
    static constexpr index_t warp_size = get_warp_size();
    static constexpr index_t num_warps = BlockSize / get_warp_size();

    static constexpr index_t MWarps = BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr index_t NWarps = BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr index_t KWarps = BlockGemmShape::BlockWarps::at(number<2>{});

    static constexpr index_t NIterPerWarp = BlockGemmShape::kN / (NWarps * WarpGemm::kN);

    static_assert(num_warps == MWarps * NWarps * KWarps);

    // KWarps > 1 isn't supported
    static_assert(KWarps == 1);

    // # of elements per thread
    static constexpr index_t X  = XPerTile;
    static constexpr index_t XR = 2;

    // Number of iters per warp
    // MIters are indexed using (Y0, Y1)
    static constexpr index_t Y0 = NIterPerWarp;

    // # of warps in Y dim
    static constexpr index_t Y1 = NWarps;

    static constexpr index_t Y2 = WarpGemm::kN;

    static_assert(Y0 * Y1 * Y2 == YPerTile, "Y0, Y1, Y2 must cover the blocktile along Y.");

    CK_TILE_HOST_DEVICE static constexpr auto Make2DStaticTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarps, XR>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X>>,
                                       tuple<sequence<0, 1>, sequence<0, 1>>,
                                       tuple<sequence<0, 1>, sequence<1, 2>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{});
    }
};

struct GemmBQuantPipelineAgBgCrDefaultPolicy : public UniversalGemmPipelineAgBgCrPolicy
{
    using Base = UniversalGemmPipelineAgBgCrPolicy;
    using Base::I0;
    using Base::I1;
    using Base::I2;

    using Base::ATileAccessPattern;
    using Base::BTileAccessPattern;

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

        constexpr index_t BlockSize   = Problem::kBlockSize;
        constexpr index_t NPerBlock   = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockBQ = KPerBlock / Problem::kQuantGroupSize;
        constexpr index_t VecLoadSize = GetVectorSizeBQ<Problem>();
        using WarpTile                = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm                = WarpGemmMfmaDispatcher<typename Problem::ComputeDataType,
                                                               typename Problem::ComputeDataType,
                                                               typename Problem::CDataType,
                                                               WarpTile::at(I0),
                                                               WarpTile::at(I1),
                                                               WarpTile::at(I2),
                                                               Problem::TransposeC>;

        static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>);
        using TileEncodingPattern = TileDistributionEncodingPatternBQ<BlockGemmShape,
                                                                      WarpGemm,
                                                                      BlockSize,
                                                                      NPerBlock,
                                                                      KPerBlockBQ,
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
                                                Problem::TransposeC || Problem::TransposedWarpGemm>;
        static_assert(std::is_same_v<typename Problem::ComputeDataType, fp8_t> || std::is_same_v<typename Problem::ComputeDataType, bf8_t>);
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
