// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
// Use `CK_PRINT<T1, T2, ...>()` to inspect values of type T1, T2, ...
// Use `CK_PRINT<v1, v2, ...>()` to inspect constexpr values of val1, val2, ... of the same type
// In a non-evaluated context, you can use `using _dummy = decltype(CK_PRINT<...>());`
// Set BUILD_DEV to OFF to avoid enabling Werror
template <auto... val>
[[deprecated("Help function to print value")]] inline constexpr void CK_PRINT()
{
}
template <typename... type>
[[deprecated("Help function to print value")]] inline constexpr void CK_PRINT()
{
}
struct TransposePolicy
{
    static constexpr auto TileAccessPattern = tile_distribution_pattern::thread_raked;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSize()
    {
        return 16 / sizeof(typename Problem::DataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return integer_least_multiple(
            sizeof(typename Problem::DataType) *
                MakeLdsBlockDescriptor<Problem>().get_element_space_size(),
            16);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeInputDistribution()
    {
        constexpr index_t BlockSize         = Problem::kBlockSize;
        constexpr index_t LeadDimPerBlock   = Problem::kLeadSizePerBlock;
        constexpr index_t SecondDimPerBlock = Problem::kSecondSizePerBlock;
        constexpr index_t VecLoadSize       = 16 / sizeof(typename Problem::DataType);

        using TileEncodingPattern = TileDistributionEncodingPattern2D<BlockSize,
                                                                      SecondDimPerBlock,
                                                                      LeadDimPerBlock,
                                                                      VecLoadSize,
                                                                      TileAccessPattern>;

        return TileEncodingPattern::Make2DStaticTileDistribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        constexpr index_t kLeadDimPerBlock   = Problem::kLeadSizePerBlock;
        constexpr index_t kSecondDimPerBlock = Problem::kSecondSizePerBlock;
        constexpr index_t kVectorSize        = 16 / sizeof(typename Problem::DataType);

        constexpr auto lds_block_desc = make_naive_tensor_descriptor(
            make_tuple(number<kSecondDimPerBlock>{}, number<kLeadDimPerBlock>{}),
            make_tuple(number<kLeadDimPerBlock>{}, number<1>{}),
            number<kVectorSize>{},
            number<1>{});

        return lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsLoadTileDistribution()
    {
        using DataType = typename Problem::DataType;

        // Extract base dimensions from the traits
        constexpr index_t kBaseLeadDim   = LaneGroupTransposeTraits<DataType>::kleadDim;
        constexpr index_t kBaseSecondDim = LaneGroupTransposeTraits<DataType>::ksecondDim;

        // Calculate block-level dimensions
        constexpr index_t kLead              = Problem::kLeadSizePerXdl;
        constexpr index_t kSecond            = Problem::kSecondSizePerXdl;
        constexpr index_t kLeadIterPerWarp   = Problem::kLeadXdlNumPerWarp;
        constexpr index_t kSecondIterPerWarp = Problem::kSecondXdlNumPerWarp;
        constexpr index_t kLeadNumWarps      = Problem::kLeadNumWarps;
        constexpr index_t kSecondNumWarps    = Problem::kSecondNumWarps;

        // Calculate repetitions of base pattern
        constexpr index_t kLeadRepetitions     = kLead / kBaseLeadDim;
        constexpr index_t kSecondRepetitions   = kSecond / kBaseSecondDim;
        constexpr index_t kSecondDimIterations = Problem::kIterationsInSecondDim;
        constexpr index_t kSecondDimStrSub     = kSecondRepetitions / kSecondDimIterations;
        // CK_PRINT<number<kBaseLeadDim>,          // 16
        //          number<kBaseSecondDim>,        // 4
        //          number<kLead>,                 // 16
        //          number<kSecond>,               // 32
        //          number<kLeadIterPerWarp>,      // 1
        //          number<kSecondIterPerWarp>,    // 1
        //          number<kLeadNumWarps>,         // 1
        //          number<kSecondNumWarps>,       // 1
        //          number<kLeadRepetitions>,      // 1
        //          number<kSecondRepetitions>,    // 8
        //          number<kSecondDimIterations>,  // 2
        //          number<kSecondDimStrSub>>();   // 4

        constexpr auto xdllevel_dstr_encoding = make_transposed_distr_encode<DataType,
                                                                             kSecondDimStrSub,
                                                                             kSecondDimIterations,
                                                                             kLeadRepetitions,
                                                                             1>();

        constexpr auto input_tile_encode =
            InputTileDistributionEncoding<decltype(xdllevel_dstr_encoding),
                                          kLeadIterPerWarp,
                                          kSecondIterPerWarp,
                                          kLeadNumWarps,
                                          kSecondNumWarps>();
        CK_PRINT<decltype(input_tile_encode)>();
        const ck_tile::tile_distribution_encoding<
            ck_tile::sequence<>,
            ck_tile::tuple<ck_tile::sequence<1, 2, 4, 2, 4>, ck_tile::sequence<1, 2, 1, 1, 4, 4>>,
            ck_tile::tuple<ck_tile::sequence<2, 1>, ck_tile::sequence<1, 2, 1, 2>>,
            ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2, 4, 4>>,
            ck_tile::sequence<2, 1, 2, 1, 2>,
            ck_tile::sequence<0, 0, 3, 3, 5>>;
        constexpr auto block_dstr = make_static_tile_distribution(input_tile_encode);

        return block_dstr;
    }
};

} // namespace ck_tile
