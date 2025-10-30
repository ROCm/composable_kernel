// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

template <typename Problem, typename DataType, index_t YPerTile, index_t XPerTile>
CK_TILE_HOST_DEVICE static constexpr auto GetAQGlobalVectorLoadSize()
{
    using I1                 = number<1>;
    constexpr index_t NWarps = Problem::BlockGemmShape::BlockWarps::at(I1{});

    constexpr index_t BlockSize = Problem::kBlockSize;

    // Data is replicated across warps along NWarps, so we divide BlockSize by NWarps
    constexpr index_t elements_per_thread = (YPerTile * XPerTile) / (BlockSize / NWarps);
    constexpr index_t PackedSize = ck_tile::numeric_traits<remove_cvref_t<DataType>>::PackedSize;

    // Define vector load candidates in descending order of priority
    constexpr std::array<index_t, 5> candidates{
        PackedSize * 32 / sizeof(DataType),
        PackedSize * 16 / sizeof(DataType),
        PackedSize * 8 / sizeof(DataType),
        PackedSize * 4 / sizeof(DataType),
        PackedSize * 2 / sizeof(DataType),
    };

    for(const auto vec_size : candidates)
    {
        if(vec_size <= 0 || XPerTile % vec_size != 0 || elements_per_thread % vec_size != 0)
            continue;
        bool is_valid = (vec_size > 0) && (XPerTile % vec_size == 0) &&
                        (elements_per_thread % vec_size == 0) && vec_size != candidates[4];
        if(is_valid)
        {
            return vec_size;
        }
    }
    return PackedSize; // Absolute fallback
}

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

    static constexpr index_t Y0 = 1;
    static constexpr index_t Y1 = MIterPerWarp ? MIterPerWarp : 1;
    static constexpr index_t Y2 = MWarps;
    static constexpr index_t Y3 = WarpGemm::kM;
    static_assert(Y3 >= WarpGemm::kM, "Scales for all rows must be available within the warp.");
    static_assert(Y0 * Y1 * Y2 * Y3 == YPerTile,
                  "Y0, Y1, Y2, Y3 must cover the blocktile along Y.");

    CK_TILE_HOST_DEVICE static constexpr auto Make2DStaticTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps>,
                                       tuple<sequence<Y0, Y1, Y2, Y3>, sequence<X>>,
                                       tuple<sequence<1, 0>, sequence<1, 1>>,
                                       tuple<sequence<2, 0>, sequence<0, 3>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{});
    }
};

} // namespace ck_tile
