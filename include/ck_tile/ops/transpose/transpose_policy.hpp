// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename T, typename = void>
struct QuartTransposeTraits;

template <typename T>
struct QuartTransposeTraits<T, std::enable_if_t<sizeof(T) == 2>>
{
    static constexpr index_t ksecondDim = 4;
    static constexpr index_t kleadDim   = 16;
    using TileDistribution              = tile_distribution_encoding<sequence<>,
                                                        tuple<sequence<4>, sequence<4, 4>>,
                                                        tuple<sequence<1, 2>>,
                                                        tuple<sequence<0, 0>>,
                                                        sequence<2>,
                                                        sequence<1>>;
};

template <typename T>
struct QuartTransposeTraits<T, std::enable_if_t<sizeof(T) == 1>>
{
    static constexpr index_t ksecondDim = 8;
    static constexpr index_t kleadDim   = 16;
    using TileDistribution              = tile_distribution_encoding<sequence<>,
                                                        tuple<sequence<8>, sequence<2, 8>>,
                                                        tuple<sequence<1, 2>>,
                                                        tuple<sequence<0, 0>>,
                                                        sequence<2>,
                                                        sequence<1>>;
};

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
                MakeLdsStoreBlockDescriptor<Problem>().get_element_space_size(),
            16);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeInputDistribution()
    {
        constexpr index_t BlockSize         = Problem::kBlockSize;
        constexpr index_t LeadDimPerBlock   = Problem::kLeadDimPerBlock;
        constexpr index_t SecondDimPerBlock = Problem::kSecondDimPerBlock;
        constexpr index_t VecLoadSize       = 16 / sizeof(typename Problem::DataType);

        using TileEncodingPattern = TileDistributionEncodingPattern2D<BlockSize,
                                                                      SecondDimPerBlock,
                                                                      LeadDimPerBlock,
                                                                      VecLoadSize,
                                                                      TileAccessPattern>;
        return TileEncodingPattern::Make2DStaticTileDistribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOutputDistribution()
    {
        constexpr index_t BlockSize         = Problem::kBlockSize;
        constexpr index_t LeadDimPerBlock   = Problem::kLeadDimPerBlock;
        constexpr index_t SecondDimPerBlock = Problem::kSecondDimPerBlock;
        constexpr index_t VecLoadSize       = 16 / sizeof(typename Problem::DataType);

        using TileEncodingPattern = TileDistributionEncodingPattern2D<BlockSize,
                                                                      LeadDimPerBlock,
                                                                      SecondDimPerBlock,
                                                                      VecLoadSize,
                                                                      TileAccessPattern>;
        return TileEncodingPattern::Make2DStaticTileDistribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsStoreBlockDescriptor()
    {
        //using Layout                         = remove_cvref_t<typename Problem::Layout>;
        constexpr index_t kLeadDimPerBlock   = Problem::kLeadDimPerBlock;
        constexpr index_t kSecondDimPerBlock = Problem::kSecondDimPerBlock;
        constexpr index_t kVectorSize        = 16 / sizeof(typename Problem::DataType);

        constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kSecondDimPerBlock>{},
                       number<kLeadDimPerBlock / kVectorSize>{},
                       number<kVectorSize>{}),
            make_tuple(
                number<(kLeadDimPerBlock + 1) * kVectorSize>{}, number<kVectorSize>{}, number<1>{}),
            number<kVectorSize>{},
            number<1>{});

        constexpr auto lds_block_desc = transform_tensor_descriptor(
            lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<kSecondDimPerBlock>{}),
                       make_merge_transform(make_tuple(number<kLeadDimPerBlock / kVectorSize>{},
                                                       number<kVectorSize>{}))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsLoadBlockDescriptor()
    {
    }

    template <typename Problem, typename WarpLevelOuterDistribution_>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsLoadTileDistribution()
    {
        //using Layout = remove_cvref_t<typename Problem::Layout>;
        using QuartTransposeTileDistribution =
            typename QuartTransposeTraits<typename Problem::DataType>::TileDistribution;
        using WarpTransposeTileDistribution =
            decltype(detail::make_embed_tile_distribution_encoding(
                WarpLevelOuterDistribution_{}, QuartTransposeTileDistribution{}));
        constexpr index_t LeadDimIterPerWarp =
            Problem::kLeadDimPerBlock / (Problem::kLeadDimPerWarp * Problem::kLeadDimWarps);
        constexpr index_t SecondDimIterPerWarp =
            Problem::kSecondDimPerBlock / (Problem::kSecondDimPerWarp * Problem::kSecondDimWarps);

        constexpr auto block_outer_dst_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<SecondDimIterPerWarp>, sequence<LeadDimIterPerWarp>>,
            tuple<>,
            tuple<>,
            sequence<1, 2>,
            sequence<0, 0>>{};
        return detail::make_embed_tile_distribution_encoding(block_outer_dst_encoding,
                                                             WarpTransposeTileDistribution{});
    }
};

} // namespace ck_tile
