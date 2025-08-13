// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

template <typename Problem, typename DataType, index_t YPerTile, index_t XPerTile>
CK_TILE_HOST_DEVICE static constexpr auto GetScaleGlobalVectorLoadSize()
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

// A Scale data for A data is preshuffled and loaded from DRAM
// using v_mfama_f32_scale_f32_16x16x128_F8F6F4 instruction for calculating
template <typename BlockGemmShape,
          typename WarpGemm,
          index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t VecSize = 1>
struct TileDistributionEncodingPatternAScale : public TileDistributionEncodingPattern
{
    static_assert(XPerTile % VecSize == 0, "XPerTile must be a multiple of VecSize!");
    static constexpr index_t warp_size = get_warp_size();
    static constexpr index_t num_warps = BlockSize / warp_size;

    static constexpr index_t MWarps = BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr index_t NWarps = BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr index_t KWarps = BlockGemmShape::BlockWarps::at(number<2>{});

    static constexpr index_t MThreadPerXdl = WarpGemm::kM;
    static constexpr index_t KThreadPerXdl = warp_size / MThreadPerXdl;

    static_assert(num_warps == MWarps * NWarps * KWarps, "Block warps do not match block size");
    static_assert(KWarps == 1, "KWarps > 1 is not supported");

    // Y dimension (M) decomposition
    static constexpr index_t Y1 = MWarps;
    static constexpr index_t Y2 = MThreadPerXdl;
    static constexpr index_t Y0 = YPerTile / (MWarps * MThreadPerXdl);

    // X dimension (K) decomposition
    static constexpr index_t X0 = KThreadPerXdl;
    static constexpr index_t X1 = VecSize;

    static_assert(Y0 * Y1 * Y2 == YPerTile, "Y dimensions must cover the YPerTile");
    static_assert(X0 * X1 == XPerTile, "X dimensions must cover the XPerTile");

    CK_TILE_HOST_DEVICE static constexpr auto Make2DStaticTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<1, 0>, sequence<2, 1>>,
                                       tuple<sequence<1, 0>, sequence<0, 2>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }
};

// B Scale data for B data is preshuffled and loaded from DRAM
// using v_mfama_f32_scale_f32_16x16x128_F8F6F4 instruction for calculating
template <typename BlockGemmShape,
          typename WarpGemm,
          index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t NXdlPack,
          index_t VecSize = 1>
struct TileDistributionEncodingPatternBScale : public TileDistributionEncodingPattern
{
    static_assert(NPerBlock % NXdlPack == 0, "XPerTile must be a multiple of VecSize!");
    static constexpr index_t warp_size = get_warp_size();
    static constexpr index_t num_warps = BlockSize / warp_size;

    static constexpr index_t MWarps = BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr index_t NWarps = BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr index_t KWarps = BlockGemmShape::BlockWarps::at(number<2>{});

    static constexpr index_t NThreadPerXdl = WarpGemm::kN;
    static constexpr index_t KThreadPerXdl = warp_size / NThreadPerXdl;

    static_assert(num_warps == MWarps * NWarps * KWarps, "Block warps do not match block size");
    static_assert(KWarps == 1, "KWarps > 1 is not supported");

    // Y dimension (N) decomposition
    static constexpr index_t Y1 = NWarps;
    static constexpr index_t Y2 = NThreadPerXdl;
    static constexpr index_t Y0 = YPerTile / (NWarps * NThreadPerXdl);

    // X dimension (K) decomposition
    static constexpr index_t X0 = KThreadPerXdl;
    static constexpr index_t X1 = VecSize;

    static_assert(Y0 * Y1 * Y2 == YPerTile, "Y dimensions must cover the YPerTile");
    static_assert(X0 * X1 == XPerTile, "X dimensions must cover the XPerTile");

    CK_TILE_HOST_DEVICE static constexpr auto Make2DStaticTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarps>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<0, 1>, sequence<2, 1>>,
                                       tuple<sequence<0, 1>, sequence<0, 2>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }
};

} // namespace ck_tile
