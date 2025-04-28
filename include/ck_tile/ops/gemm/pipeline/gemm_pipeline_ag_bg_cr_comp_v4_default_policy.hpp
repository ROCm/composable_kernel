// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {
// Default policy for GemmPipelineAGmemBGmemCregComputeV4, except the block gemm method, it shares
// the same vector size implementation, SmemSize, Global memory tile distiribution as the
// UniversalGemm Pipeline Policy.
// Default policy class should not be templated, put template on
// member functions instead.
struct GemmPipelineAgBgCrCompV4DefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompV4DefaultPolicy>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t KPack     = GetSmemPackA<Problem>();

        constexpr auto DataTypeSize = sizeof(ADataType);
        constexpr auto MLdsLayer =
            (32 * 4 / KPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / KPerBlock / DataTypeSize);

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<KPerBlock / KPack * MLdsLayer>{},
                       number<MPerBlock / MLdsLayer>{},
                       number<KPack>{}),
            make_tuple(number<KPack>{}, number<KPerBlock * MLdsLayer>{}, number<1>{}),
            number<KPack>{},
            number<1>{});

        constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<MPerBlock / MLdsLayer>{},
                                                     number<KPerBlock / KPack * MLdsLayer>{})),
                       make_pass_through_transform(number<KPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        constexpr auto a_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
            a_lds_block_desc_permuted,
            make_tuple(make_unmerge_transform(
                           make_tuple(number<MLdsLayer>{}, number<KPerBlock / KPack>{})),
                       make_pass_through_transform(number<MPerBlock / MLdsLayer>{}),
                       make_pass_through_transform(number<KPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_xk0_mnldslayer_mn_xk1,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<MPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        // using BLayout   = remove_cvref_t<typename Problem::BLayout>;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        // works for both row-major and column-major
        // TODO: Improve the performance for column-major
        {
            constexpr index_t KPack     = GetSmemPackB<Problem>();
            constexpr auto BK0          = number<KPerBlock / KPack>{};
            constexpr auto DataTypeSize = sizeof(BDataType);
            constexpr auto NLdsLayer =
                (32 * 4 / KPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / KPerBlock / DataTypeSize);

            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(
                    BK0 * number<NLdsLayer>{}, number<NPerBlock / NLdsLayer>{}, number<KPack>{}),
                make_tuple(number<KPack>{}, number<KPerBlock * NLdsLayer>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<NPerBlock / NLdsLayer>{},
                                                         BK0 * number<NLdsLayer>{})),
                           make_pass_through_transform(number<KPack>{})),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto b_lds_block_desc_bk0_nldslayer_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(number<NLdsLayer>{}, BK0)),
                           make_pass_through_transform(number<NPerBlock / NLdsLayer>{}),
                           make_pass_through_transform(number<KPack>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            constexpr auto b_lds_block_desc = transform_tensor_descriptor(
                b_lds_block_desc_bk0_nldslayer_n_bk1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<NPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                           make_merge_transform_v3_division_mod(make_tuple(BK0, number<KPack>{}))),
                make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
            return b_lds_block_desc;
        }
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using AccDataType     = float;
        using BlockWarps      = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile        = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm        = WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                                typename Problem::BDataType,
                                                AccDataType,
                                                WarpTile::at(I0),
                                                WarpTile::at(I1),
                                                WarpTile::at(I2),
                                                Problem::TransposeC>;
        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};
} // namespace ck_tile
