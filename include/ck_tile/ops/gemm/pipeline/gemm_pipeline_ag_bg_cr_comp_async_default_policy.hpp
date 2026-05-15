// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {
// Default policy for GemmPipelineAgBgCrCompAsync
// Customized methods: MakeALdsBlockDescriptor, MakeBLdsBlockDescriptor
// GetBlockGemm implementation is copied from GemmPipelineAgBgCrCompV4DefaultPolicy
template <bool EnableSubTile = false>
struct GemmPipelineAgBgCrCompAsyncDefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncDefaultPolicy<EnableSubTile>>
{
    static constexpr auto ATileAccessPattern = tile_distribution_pattern::warp_raked;
    static constexpr auto BTileAccessPattern = tile_distribution_pattern::warp_raked;
    using Base = UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncDefaultPolicy<EnableSubTile>>;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::is_a_load_tr;
    using Base::is_b_load_tr;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
#if defined(__gfx125__)
        return Base::template MakeALdsBlockDescriptor<Problem>();
#else
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(Base::template is_a_load_tr<Problem>)
        {
            // TODO: better LDS descriptor for performance
            // This branch is reusing the logic from
            // UniversalGemmBasePolicy::MakeALdsBlockDescriptor
            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor( //
                make_tuple(number<KPerBlock>{}, number<MPerBlock>{}),
                make_tuple(number<MPerBlock>{}, number<1>{}),
                number<MPerBlock>{},
                number<1>{});
            return a_lds_block_desc_0;
        }
        else
        {
            constexpr index_t KPack = Base::template GetSmemPackA<Problem>();

            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<KPerBlock / KPack>{}, number<MPerBlock>{}, number<KPack>{}),
                make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            return transform_tensor_descriptor(
                a_lds_block_desc_0,
                make_tuple(
                    make_pass_through_transform(number<MPerBlock>{}),
                    make_merge_transform(make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                make_tuple(sequence<1>{}, sequence<0, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
#endif
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
#if defined(__gfx125__)
        return Base::template MakeBLdsBlockDescriptor<Problem>();
#else
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(Base::template is_b_load_tr<Problem>)
        {
            // TODO: better LDS descriptor for performance
            // This branch is reusing the logic from
            // UniversalGemmBasePolicy::MakeBLdsBlockDescriptor
            constexpr auto b_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<KPerBlock>{}, number<NPerBlock>{}),
                                             make_tuple(number<NPerBlock>{}, number<1>{}),
                                             number<NPerBlock>{},
                                             number<1>{});
            return b_lds_block_desc_0;
        }
        else
        {
            constexpr index_t KPack = Base::template GetSmemPackB<Problem>();

            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<KPerBlock / KPack>{}, number<NPerBlock>{}, number<KPack>{}),
                make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            return transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(
                    make_pass_through_transform(number<NPerBlock>{}),
                    make_merge_transform(make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                make_tuple(sequence<1>{}, sequence<0, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
#endif
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetEstimatedVgprCount()
    {
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;
        using CDataType = remove_cvref_t<typename Problem::CDataType>;

        constexpr index_t MWarps       = Problem::BlockGemmShape::BlockWarps::at(Base::I0);
        constexpr index_t NWarps       = Problem::BlockGemmShape::BlockWarps::at(Base::I1);
        constexpr index_t warpSize     = get_warp_size();
        constexpr index_t BlockSize    = Problem::kBlockSize;
        constexpr index_t BytesPerVGPR = 4;
        constexpr index_t AccVGPRNum =
            sizeof(CDataType) * MPerBlock * NPerBlock / BlockSize / BytesPerVGPR;

        constexpr index_t DoubleBufferFactor = 3;

        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;

        constexpr index_t ALoadVGPRNum = sizeof(ADataType) / APackedSize * MPerBlock * KPerBlock /
                                         MWarps / warpSize / BytesPerVGPR * DoubleBufferFactor;

        constexpr index_t BLoadVGPRNum = sizeof(BDataType) / BPackedSize * NPerBlock * KPerBlock /
                                         NWarps / warpSize / BytesPerVGPR * DoubleBufferFactor;

        constexpr index_t TotalInputVGPRNum = ALoadVGPRNum + BLoadVGPRNum;

        return make_tuple(number<AccVGPRNum>{}, number<TotalInputVGPRNum>{});
    }

    // this function is used to get SubTile Number
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetPipelineSubTileNum()
    {
        constexpr auto estimated_vgpr = GetEstimatedVgprCount<Problem>();

        constexpr auto acc_vgpr_num   = estimated_vgpr.at(number<0>{});
        constexpr auto input_vgpr_num = estimated_vgpr.at(number<1>{});

        constexpr index_t vgpr_capacity = get_max_vgpr_count();
        // sub tile number; have 1, 2, 4 choices
        constexpr index_t sub_tile_num = ((input_vgpr_num + acc_vgpr_num) <= vgpr_capacity) ? 1
                                         : ((input_vgpr_num / 2 + acc_vgpr_num) <= vgpr_capacity)
                                             ? 2
                                             : 4;

        return number<sub_tile_num>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

#if defined(__gfx950__)
        constexpr index_t vector_size =
            DS_READ_TR_SIZE() / sizeof(typename Problem::AComputeDataType);
        constexpr index_t thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();
        constexpr auto wg_attr_num_access =
            !(Base::template is_a_load_tr<Problem> || Base::template is_b_load_tr<Problem>)
                ? WGAttrNumAccessEnum::Single
            : vector_size == thread_elements     ? WGAttrNumAccessEnum::Single
            : vector_size * 2 == thread_elements ? WGAttrNumAccessEnum::Double
            : vector_size * 4 == thread_elements ? WGAttrNumAccessEnum::Quad
                                                 : WGAttrNumAccessEnum::Invalid;
#else
        constexpr auto wg_attr_num_access = WGAttrNumAccessEnum::Default;
#endif

        constexpr auto pipeline_tune_params = GetPipelineSubTileNum<Problem>();
        constexpr index_t sub_tile_num      = EnableSubTile ? pipeline_tune_params.value : 1;

        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType, // AccDataType
                                            WarpTile::at(Base::I0),
                                            WarpTile::at(Base::I1),
                                            WarpTile::at(Base::I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm,
                                                                    sub_tile_num>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};
} // namespace ck_tile
