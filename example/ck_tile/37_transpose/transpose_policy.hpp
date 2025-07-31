// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

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
    CK_TILE_HOST_DEVICE static constexpr auto MakeOutputDistribution()
    {
        constexpr auto input_dstr = MakeLdsLoadTileDistribution<Problem>();

        using OutTileDstrEncode =
            typename OutputTileDistributionTraits<remove_cvref_t<decltype(input_dstr)>,
                                                  typename Problem::DataType>::OutDstrEncode;
        constexpr auto block_dstr = make_static_tile_distribution(OutTileDstrEncode{});

        return block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsStoreBlockDescriptor()
    {
        constexpr index_t kLeadDimPerBlock   = Problem::kLeadSizePerBlock;
        constexpr index_t kSecondDimPerBlock = Problem::kSecondSizePerBlock;
        constexpr index_t kVectorSize        = 16 / sizeof(typename Problem::DataType);

        constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kSecondDimPerBlock>{},
                       number<kLeadDimPerBlock / kVectorSize>{},
                       number<kVectorSize>{}),
            make_tuple(number<kLeadDimPerBlock>{}, number<kVectorSize>{}, number<1>{}),
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
        constexpr index_t kLeadDimPerBlock   = Problem::kLeadSizePerBlock;
        constexpr index_t kSecondDimPerBlock = Problem::kSecondSizePerBlock;

        constexpr index_t kVectorSize = 8 / sizeof(typename Problem::DataType);

        constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kSecondDimPerBlock>{},
                       number<kLeadDimPerBlock / kVectorSize>{},
                       number<kVectorSize>{}),
            make_tuple(number<kLeadDimPerBlock>{}, number<kVectorSize>{}, number<1>{}),
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
        constexpr auto block_dstr = make_static_tile_distribution(input_tile_encode);
        return block_dstr;
    }
};

} // namespace ck_tile
