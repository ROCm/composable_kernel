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
    // before transpose, 4x16
    static constexpr index_t ksecondDim = 4;
    static constexpr index_t kleadDim   = 16;
    // after transpose, 16x4
    static constexpr index_t ksecondDimT = 16;
    static constexpr index_t kleadDimT   = 4;
};

template <typename T>
struct QuartTransposeTraits<T, std::enable_if_t<sizeof(T) == 1>>
{
    static constexpr index_t ksecondDim = 8;
    static constexpr index_t kleadDim   = 16;

    static constexpr index_t ksecondDimT = 16;
    static constexpr index_t kleadDimT   = 8;
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
        // one xdl implement kSecond x kLead
        constexpr index_t kLead   = Problem::kLeadSizePerXdl;
        constexpr index_t kSecond = Problem::kSecondSizePerXdl;
        constexpr index_t kLeadDimstr =
            kLead / QuartTransposeTraits<typename Problem::DataType>::kleadDim;
        constexpr index_t kSecondDimstr =
            kSecond / QuartTransposeTraits<typename Problem::DataType>::ksecondDim;
        constexpr index_t kSecondDimIterations = Problem::kIterationsInSecondDim;
        constexpr index_t kSecondDimStrSub     = kSecondDimstr / kSecondDimIterations;
        constexpr auto xdllevel_dstr_encoding =
            make_transposed_distr_encode<typename Problem::DataType,
                                         kSecondDimStrSub,
                                         kSecondDimIterations,
                                         kLeadDimstr,
                                         1>();
        constexpr index_t kLeadIterPerWarp   = Problem::kLeadXdlNumPerWarp;
        constexpr index_t kSecondIterPerWarp = Problem::kSecondXdlNumPerWarp;
        constexpr index_t kLeadNumWarps      = Problem::kLeadNumWarps;
        constexpr index_t kSecondNumWarps    = Problem::kSecondNumWarps;
        constexpr auto block_outer_dst_encoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<kSecondIterPerWarp, kSecondNumWarps>,
                                             sequence<kLeadIterPerWarp, kLeadNumWarps>>,
                                       tuple<sequence<2, 1>>,
                                       tuple<sequence<1, 1>>,
                                       sequence<2, 1>,
                                       sequence<0, 0>>{};
        constexpr auto blk_distr_encode = detail::make_embed_tile_distribution_encoding(
            block_outer_dst_encoding, xdllevel_dstr_encoding);
        constexpr auto block_dstr = make_static_tile_distribution(blk_distr_encode);
        return block_dstr;
    }
};

} // namespace ck_tile
