// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename AccDataType_,
          typename Shape_>
struct PracticeGemmBlockPipelineProblem
{
    using ADataType   = ADataType_;
    using BDataType   = BDataType_;
    using CDataType   = CDataType_;
    using AccDataType = AccDataType_;
    using Shape       = Shape_;
};

struct PracticeGemmBlockPipelinePolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPracticeWaveGemmPipeline()
    {
        // return PracticeWaveGemm<Problem>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockTile::at(number<0>{});
        constexpr index_t kKPerBlock = Problem::BlockTile::at(number<2>{});
        constexpr index_t kKPack     = 8;

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
            make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        return a_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockTile::at(number<1>{});
        constexpr index_t kKPerBlock = Problem::BlockTile::at(number<2>{});
        constexpr index_t kKPack     = 8;

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
            make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return b_lds_block_desc;
    }
};

template <typename Problem, typename Policy = PracticeGemmBlockPipelinePolicy>
struct PracticeGemmBlockPipelineAGmemBGmemCreg
{
    using ADataType   = typename Problem::ADataType;
    using BDataType   = typename Problem::BDataType;
    using CDataType   = typename Problem::CDataType;
    using AccDataType = typename Problem::AccDataType;

    using BlockTile = typename Problem::Shape::BlockTile;
    using WaveTile  = typename Problem::Shape::WaveTile;

    static constexpr index_t MPerBlock = BlockTile::at(number<0>{});
    static constexpr index_t NPerBlock = BlockTile::at(number<1>{});
    static constexpr index_t KPerBlock = BlockTile::at(number<2>{});

    static constexpr index_t MPerWave = WaveTile::at(number<0>{});
    static constexpr index_t NPerWave = WaveTile::at(number<1>{});
    static constexpr index_t KPerWave = WaveTile::at(number<2>{});

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetPracticeWaveGemm<Problem>())>;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetStaticLDSSize()
    {
        return integer_divide_ceil(
                   sizeof(ADataType) *
                       Policy::template MakeALdsBlockDescriptor<Problem>().get_element_space_size(),
                   16) *
                   16 +
               sizeof(BDataType) *
                   Policy::template MakeBLdsBlockDescriptor<Problem>().get_element_space_size();
    }
};

} // namespace ck_tile
