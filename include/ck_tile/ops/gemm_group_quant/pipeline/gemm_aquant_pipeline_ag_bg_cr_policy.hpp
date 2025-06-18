// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "gemm_group_quant_utils.hpp"

namespace ck_tile {

// AQ holds groupquant scale data for A. Data is loaded from DRAM and partitioned across
// threads. Post mfma scales are shuffled across threads in the warp and applied to
// accum registers.
template <typename BlockGemmShape,
          typename WarpGemm,
          index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t VecSize>
struct TileDistributionEncodingPatternAQ : public TileDistributionEncodingPattern
{
    // TODO: make pattern where below condition does not need to hold - GGemmMultiDSplitk!
    static_assert(XPerTile % VecSize == 0, "XPerTile must be a multiple of VecSize!");
    static constexpr index_t warp_size = get_warp_size();
    static constexpr index_t num_warps = BlockSize / get_warp_size();

    static constexpr index_t MWarps = BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr index_t NWarps = BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr index_t KWarps = BlockGemmShape::BlockWarps::at(number<2>{});

    static constexpr index_t MIterPerWarp = BlockGemmShape::kM / (MWarps * WarpGemm::kM);

    static_assert(num_warps == MWarps * NWarps * KWarps);

    // KWarps > 1 isn't supported
    static_assert(KWarps == 1);

    // # of elements per thread
    static constexpr index_t X = XPerTile;

    // Number of iters per warp
    // MIters are indexed using (Y0, Y1)
    static constexpr index_t Y1 = warp_size / WarpGemm::kM;
    static constexpr index_t Y0 = MIterPerWarp / Y1;

    // # of warps in Y dim
    static constexpr index_t Y2 = MWarps;

    // # of rows per iter per warp
    static constexpr index_t Y3 = YPerTile / (Y1 * Y0 * Y2);
    static_assert(Y3 >= WarpGemm::kM, "Scales for all rows must be available within the warp.");
    static_assert(Y0 * Y1 * Y2 * Y3 == YPerTile,
                  "Y0, Y1, Y2, Y3 must cover the blocktile along Y.");

    CK_TILE_HOST_DEVICE static constexpr auto Make2DStaticTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps>,
                                       tuple<sequence<Y0, Y1, Y2, Y3>, sequence<X>>,
                                       tuple<sequence<1, 0>, sequence<1, 1>>,
                                       tuple<sequence<2, 0>, sequence<1, 3>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{});
    }
};

template <typename BlockGemmShape,
          typename WarpGemm,
          index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t VecSize>
struct TileDistributionEncodingPatternAQTransposedC : public TileDistributionEncodingPattern
{
    // TODO: make pattern where below condition does not need to hold - GGemmMultiDSplitk!
    static_assert(XPerTile % VecSize == 0, "XPerTile must be a multiple of VecSize!");
    static constexpr index_t warp_size = get_warp_size();
    static constexpr index_t num_warps = BlockSize / get_warp_size();

    static constexpr index_t MWarps = BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr index_t NWarps = BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr index_t KWarps = BlockGemmShape::BlockWarps::at(number<2>{});

    static constexpr index_t MIterPerWarp = BlockGemmShape::kM / (MWarps * WarpGemm::kM);

    static_assert(num_warps == MWarps * NWarps * KWarps);

    // KWarps > 1 isn't supported
    static_assert(KWarps == 1);

    // # of elements per thread
    static constexpr index_t X  = XPerTile;
    static constexpr index_t XR = 2;

    // Number of iters per warp
    // MIters are indexed using (Y0, Y1)
    static constexpr index_t Y0 = MIterPerWarp;

    // # of warps in Y dim
    static constexpr index_t Y1 = MWarps;

    static constexpr index_t Y2 = WarpGemm::kM;

    static_assert(Y0 * Y1 * Y2 == YPerTile, "Y0, Y1, Y2 must cover the blocktile along Y.");

    CK_TILE_HOST_DEVICE static constexpr auto Make2DStaticTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps, XR>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X>>,
                                       tuple<sequence<1, 0>, sequence<0, 1>>,
                                       tuple<sequence<1, 0>, sequence<1, 2>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{});
    }
};

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
        return GetABQGlobalVectorLoadSize<Problem, AQDataType, MPerBlock, KPerBlockAQ>();
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
                                                               Problem::TransposeC>;

        static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
        using TileEncodingPattern = TileDistributionEncodingPatternAQ<BlockGemmShape,
                                                                      WarpGemm,
                                                                      BlockSize,
                                                                      MPerBlock,
                                                                      KPerBlockAQ,
                                                                      VecLoadSize>;

        using TileEncodingPatternTransposeC =
            TileDistributionEncodingPatternAQTransposedC<BlockGemmShape,
                                                         WarpGemm,
                                                         BlockSize,
                                                         MPerBlock,
                                                         KPerBlockAQ,
                                                         VecLoadSize>;

        if constexpr (Problem::TransposedWarpGemm == true) {
            return TileEncodingPatternTransposeC::Make2DStaticTileDistribution();
        } else {
            return TileEncodingPattern::Make2DStaticTileDistribution();
        }
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
        return AQuantBlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
