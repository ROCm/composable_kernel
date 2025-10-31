// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

template <typename Problem, typename DataType, index_t YPerTile, index_t XPerTile>
CK_TILE_HOST_DEVICE static constexpr auto GetABQGlobalVectorLoadSize()
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
          index_t KPerBlockAQ,
          index_t VecSize,
          bool PreshuffleQuant>
struct tile_distribution_encoding_pattern_aq : public tile_distribution_encoding_pattern
{
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

    CK_TILE_HOST_DEVICE static constexpr auto make_2d_static_tile_distribution()
    {
        if constexpr(PreshuffleQuant)
        {
            // # of elements per thread
            static_assert(XPerTile >= warp_size && XPerTile % warp_size == 0);
            constexpr index_t X1 = warp_size;
            constexpr index_t X0 = XPerTile / warp_size;

            constexpr index_t Y1 = MWarps;
            constexpr index_t Y0 = YPerTile / Y1;
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<NWarps>,
                                           tuple<sequence<Y0, Y1>, sequence<X0, X1>>,
                                           tuple<sequence<1, 0>, sequence<2>>,
                                           tuple<sequence<1, 0>, sequence<1>>,
                                           sequence<1, 2>,
                                           sequence<0, 0>>{});
        }
        else
        {
            // # of elements per thread
            constexpr index_t X = XPerTile;

            constexpr index_t Y0 = 1;
            constexpr index_t Y1 = MIterPerWarp ? MIterPerWarp : 1;
            constexpr index_t Y2 = MWarps;
            constexpr index_t Y3 = WarpGemm::kM;
            static_assert(Y3 >= WarpGemm::kM,
                          "Scales for all rows must be available within the warp.");
            static_assert(Y0 * Y1 * Y2 * Y3 == YPerTile,
                          "Y0, Y1, Y2, Y3 must cover the blocktile along Y.");
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<NWarps>,
                                           tuple<sequence<Y0, Y1, Y2, Y3>, sequence<X>>,
                                           tuple<sequence<1, 0>, sequence<1, 1>>,
                                           tuple<sequence<2, 0>, sequence<0, 3>>,
                                           sequence<1, 2>,
                                           sequence<1, 0>>{});
        }
    }
};

template <typename BlockGemmShape,
          typename WarpGemm,
          index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t VecSize>
struct tile_distribution_encoding_pattern_aq_transposed_c
    : public tile_distribution_encoding_pattern
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

    CK_TILE_HOST_DEVICE static constexpr auto make_2d_static_tile_distribution()
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

// TODO:: might need to update
template <typename BlockGemmShape,
          typename WarpGemm,
          index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t XPerQ>
struct tile_distribution_encoding_pattern_bq : public tile_distribution_encoding_pattern
{
    static constexpr index_t warp_size = get_warp_size();
    static constexpr index_t num_warps = BlockSize / get_warp_size();

    static constexpr index_t MWarps = BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr index_t NWarps = BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr index_t KWarps = BlockGemmShape::BlockWarps::at(number<2>{});

    static constexpr index_t NIterPerWarp = BlockGemmShape::kN / (NWarps * WarpGemm::kN);

    static_assert(num_warps == MWarps * NWarps * KWarps);
    static_assert(KWarps == 1);

    /// @brief Creates a 2D tile distribution for BQ (B-matrix quantization scales)
    ///
    /// This function determines the optimal thread distribution pattern for loading and applying
    /// quantization scales to the B matrix based on the quantization group size (XPerQ) relative
    /// to warp dimensions.
    ///
    /// Three distinct distribution patterns are handled:
    ///
    /// 1. Fine-grained quantization (XPerQ < WarpGemm::kN):
    ///    - Multiple quantization groups exist within a single warp's N-dimension
    ///    - Each warp processes multiple scales (WarpGemm::kN / XPerQ scales per warp)
    ///    - Distribution includes explicit replication factor (XR = XPerQ) for scale broadcast
    ///    - Example: XPerQ=8, WarpGemm::kN=16, NWarps=4 → 2 scales per warp
    ///
    /// 2. Medium-grained quantization (WarpGemm::kN <= XPerQ <= WarpGemm::kN * NWarps):
    ///    - Each warp handles exactly one quantization scale
    ///    - Scales are distributed across warps with replication factor XR = XPerQ / WarpGemm::kN
    ///    - Example: XPerQ=64, WarpGemm::kN=16, NWarps=4 → 1 scale per warp, XR=4
    ///
    /// 3. Coarse-grained quantization (XPerQ > WarpGemm::kN * NWarps):
    ///    - Quantization group spans multiple warps
    ///    - All warps share the same scale value
    ///    - Example: XPerQ=128, WarpGemm::kN=16, NWarps=4 → all warps use same scale
    ///
    /// @return A static tile distribution encoding for the BQ scale tensor
    CK_TILE_HOST_DEVICE static constexpr auto make_2d_static_tile_distribution()
    {
        if constexpr(XPerQ < WarpGemm::kN)
        {
            // Case 1: Fine-grained - multiple quantization scales within a single warp
            constexpr index_t Y  = YPerTile;             // Full Y dimension of tile
            constexpr index_t YR = 1;                    // No Y replication needed
            constexpr index_t X0 = NIterPerWarp;         // Iterations per warp in N-dim
            constexpr index_t X1 = NWarps;               // Number of warps in N-dim
            constexpr index_t X2 = WarpGemm::kN / XPerQ; // Number of scales per warp
            constexpr index_t XR = XPerQ;                // Elements per quantization group

            static_assert(X0 * X1 * X2 == XPerTile, "X0, X1, X2 must cover the blocktile along X.");

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<MWarps, YR, XR>,
                                           tuple<sequence<Y>, sequence<X0, X1, X2>>,
                                           tuple<sequence<0, 2>, sequence<0, 2, 0>>,
                                           tuple<sequence<0, 1>, sequence<1, 2, 2>>,
                                           sequence<2, 1>,
                                           sequence<0, 0>>{});
        }
        else if constexpr(XPerQ <= WarpGemm::kN * NWarps)
        {
            // Case 2: Medium-grained - one quantization scale per warp
            constexpr auto XR = XPerQ / WarpGemm::kN; // Scale replication factor
            constexpr auto X1 = NWarps / XR;          // Warps per unique scale
            constexpr auto X0 = XPerTile / X1;        // Iterations to cover X dimension
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<MWarps, XR, get_warp_size()>,
                                           tuple<sequence<YPerTile>, sequence<X0, X1>>,
                                           tuple<sequence<0, 2, 0>, sequence<0>>,
                                           tuple<sequence<0, 1, 1>, sequence<2>>,
                                           sequence<2, 1>,
                                           sequence<0, 0>>{});
        }
        else // XPerQ > WarpGemm::kN * NWarps
        {
            // Case 3: Coarse-grained - quantization group spans all warps
            // All warps in N-dimension share the same quantization scale
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<MWarps, NWarps, get_warp_size()>,
                                           tuple<sequence<YPerTile>, sequence<XPerTile>>,
                                           tuple<sequence<0, 0>, sequence<0>>,
                                           tuple<sequence<0, 1>, sequence<2>>,
                                           sequence<2, 1>,
                                           sequence<0, 0>>{});
        }
    }
};

template <typename GroupSizes>
struct QuantGroupShape
{
    static constexpr index_t kM = GroupSizes::at(number<0>{});
    static constexpr index_t kN = GroupSizes::at(number<1>{});
    static constexpr index_t kK = GroupSizes::at(number<2>{});

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat('_', "quant_group_shape", concat('x', kM, kN, kK));
    }
};

} // namespace ck_tile
