// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

namespace ck_tile {

template <index_t AKDim_>
struct BlockGemmPipelineAGmemBGmemCRegSkipALdsPersistentQRegCachePolicy
{
    static constexpr index_t AKDim = AKDim_;

    template <typename Problem>
    __host__ __device__ static constexpr auto GetBlockGemm()
    {
        using BlockGemmPolicy = BlockGemmARegBSmemCRegV1K8Policy;

        return BlockGemmARegBSmemCRegV1<Problem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeARegBlockDescriptor()
    {
        constexpr auto blockgemm = GetBlockGemm<Problem>();
        using BlockGemm          = remove_cvref_t<decltype(blockgemm)>;

        static_assert((Problem::BlockGemmShape::kM == Problem::BlockGemmShape::kN), "wrong!");

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = AKDim;

        constexpr auto config =
            BlockGemm::BlockGemmPolicy::template GetWarpGemmMWarpNWarp<Problem, kMPerBlock>();

        using WG = remove_cvref_t<decltype(config.template get<0>())>;

        constexpr index_t MWarp = config.template get<1>();
        constexpr index_t NWarp = config.template get<2>();

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);

        return a_block_dstr;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeADramTileDistribution()
    {
        return MakeARegBlockDescriptor<Problem>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t kKPack     = 8;

        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr auto DataTypeSize = sizeof(BDataType);
        constexpr auto NLdsLayer =
            (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack * NLdsLayer>{},
                       number<kNPerBlock / NLdsLayer>{},
                       number<kKPack>{}),
            make_tuple(number<kKPack>{}, number<kKPerBlock * NLdsLayer>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<kNPerBlock / NLdsLayer>{},
                                                     number<kKPerBlock / kKPack * NLdsLayer>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        constexpr auto b_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
            b_lds_block_desc_permuted,
            make_tuple(make_unmerge_transform(
                           make_tuple(number<NLdsLayer>{}, number<kKPerBlock / kKPack>{})),
                       make_pass_through_transform(number<kNPerBlock / NLdsLayer>{}),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_xk0_mnldslayer_mn_xk1,
            make_tuple(
                make_merge_transform(
                    make_tuple(number<kNPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        return b_lds_block_desc;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBDramTileDistribution()
    {
        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = 16 / sizeof(BDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;

        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }
};

} // namespace ck_tile
