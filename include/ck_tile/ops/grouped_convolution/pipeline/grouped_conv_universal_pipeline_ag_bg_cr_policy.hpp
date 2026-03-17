// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

// UniversalGemm Policy
struct GroupedConvUniversalPipelineAgBgCrPolicy
    : public UniversalGemmBasePolicy<GroupedConvUniversalPipelineAgBgCrPolicy>
{

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        using ADataType             = OverrideADataType;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t KPack     = GetSmemPackA<Problem>();

        if constexpr(is_a_load_tr<Problem>)
        {
            // TODO: better lds descriptor for performance
            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor( //
                make_tuple(number<KPerBlock>{}, number<MPerBlock>{}),
                make_tuple(number<MPerBlock>{}, number<1>{}),
                number<MPerBlock>{},
                number<1>{});
            return a_lds_block_desc_0;
        }
        else
        {
            constexpr auto DataTypeSize    = sizeof(ADataType);
            constexpr uint64_t MinLdsLayer = 1ULL;
            constexpr auto MLdsLayer =
                max(MinLdsLayer,
                    get_n_lds_banks() * get_n_dwords_per_128b() / KPerBlock / DataTypeSize);

            constexpr index_t NBanks = get_n_lds_banks();
            static_assert(NBanks == 32 || NBanks == 64, "Unexpected LDS bank count");
            constexpr index_t RowMul = (NBanks == 64) ? 2 : 1;

            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<KPerBlock / KPack * MLdsLayer>{},
                           number<MPerBlock / MLdsLayer>{},
                           number<KPack>{}),
                make_tuple(number<KPack>{}, number<KPerBlock * MLdsLayer>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<MPerBlock / MLdsLayer * RowMul>{},
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
    }

    /**
     * @brief Create LDS block descriptor for B tensor.
     *
     * @tparam Problem  Gemm pipeline problem.
     * @return B tensor LDS block descriptor.
     */
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr bool IsBCastPolicyBeforeLDSWrite = IsBCastPolicyBeforeLDSWrite_v<Problem>;
        using BDataType                            = std::conditional_t<IsBCastPolicyBeforeLDSWrite,
                                                                        typename Problem::ADataType,
                                                                        typename Problem::BDataType>;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(is_b_load_tr<Problem>)
        {
            // TODO: better lds descriptor for performance
            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor( //
                make_tuple(number<KPerBlock>{}, number<NPerBlock>{}),
                make_tuple(number<NPerBlock>{}, number<1>{}),
                number<NPerBlock>{},
                number<1>{});
            return b_lds_block_desc_0;
        }
        else
        {
            constexpr index_t KPack        = GetSmemPackB<Problem>();
            constexpr auto BK0             = number<KPerBlock / KPack>{};
            constexpr auto DataTypeSize    = sizeof(BDataType);
            constexpr uint64_t MinLdsLayer = 1ULL;
            constexpr auto NLdsLayer =
                max(MinLdsLayer,
                    get_n_lds_banks() * get_n_dwords_per_128b() / KPerBlock / DataTypeSize);

            constexpr index_t NBanks = get_n_lds_banks();
            static_assert(NBanks == 32 || NBanks == 64, "Unexpected LDS bank count");
            constexpr index_t RowMul = (NBanks == 64) ? 2 : 1;

            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(
                    BK0 * number<NLdsLayer>{}, number<NPerBlock / NLdsLayer>{}, number<KPack>{}),
                make_tuple(number<KPack>{}, number<KPerBlock * NLdsLayer>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<NPerBlock / NLdsLayer * RowMul>{},
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
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        constexpr index_t vector_size =
            DS_READ_TR_SIZE() / sizeof(typename Problem::ComputeDataType);
        constexpr index_t thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();
        constexpr auto wg_attr_num_access =
            !(is_a_load_tr<Problem> || is_b_load_tr<Problem>) ? WGAttrNumAccessEnum::Single
            : vector_size == thread_elements                  ? WGAttrNumAccessEnum::Single
            : vector_size * 2 == thread_elements              ? WGAttrNumAccessEnum::Double
            : vector_size * 4 == thread_elements              ? WGAttrNumAccessEnum::Quad
                                                              : WGAttrNumAccessEnum::Invalid;

        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;
        using ATypeToUse =
            std::conditional_t<std::is_same_v<ADataType, pk_int4_t>, BDataType, ADataType>;
        using BTypeToUse = std::conditional_t<std::is_same_v<BDataType, pk_int4_t> ||
                                                  std::is_same_v<BDataType, pk_fp4_t> ||
                                                  sizeof(BDataType) < sizeof(ADataType),
                                              ADataType,
                                              BDataType>;

        using WarpGemm = WarpGemmDispatcher<ATypeToUse,
                                            BTypeToUse,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            Problem::UseStructuredSparsity,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<ATypeToUse,
                                                                      BTypeToUse,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return BlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
