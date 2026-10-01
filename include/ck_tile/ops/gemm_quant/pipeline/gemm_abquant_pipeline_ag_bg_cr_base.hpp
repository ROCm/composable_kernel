// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_base.hpp"

namespace ck_tile {

template <typename Problem, typename Policy>
struct GemmABQuantPipelineAgBgCrImplBase : public GemmPipelineAgBgCrImplBase<Problem, Policy>
{
    using AQuantBase = GemmAQuantPipelineAgBgCrImplBase<Problem, Policy>;
    using BQuantBase = GemmBQuantPipelineAgBgCrImplBase<Problem, Policy>;

    template <typename AQDramBlockWindowTmp>
    CK_TILE_DEVICE constexpr auto
    GetAQDramLoadWindow(const AQDramBlockWindowTmp& aq_dram_block_window_tmp) const
    {
        if constexpr(std::is_same_v<typename Problem::AQLayout, tensor_layout::gemm::ColumnMajor> &&
                     !Problem::Traits::APreshuffleQuant)
        {
            // The ABQuant kernel supplies AQ as [M, QK] for either storage
            // layout. CompV3's ColumnMajor distribution and K-step use [QK, M].
            // Transpose this consumer's view, tile lengths, and origin together;
            // the EightWaves consumer retains its original [M, QK] convention.
            constexpr auto I0 = number<0>{};
            constexpr auto I1 = number<1>{};
            const auto& view  = aq_dram_block_window_tmp.get_bottom_tensor_view();
            const auto& desc  = view.get_tensor_descriptor();
            auto transposed_view =
                transform_tensor_view(view,
                                      make_tuple(make_pass_through_transform(desc.get_length(I0)),
                                                 make_pass_through_transform(desc.get_length(I1))),
                                      make_tuple(sequence<0>{}, sequence<1>{}),
                                      make_tuple(sequence<1>{}, sequence<0>{}));
            const auto& lengths = aq_dram_block_window_tmp.get_window_lengths();
            const auto& origin  = aq_dram_block_window_tmp.get_window_origin();
            return make_tile_window(transposed_view,
                                    make_tuple(lengths[I1], lengths[I0]),
                                    make_array(origin[I1], origin[I0]),
                                    Policy::template MakeAQDramTileDistribution<Problem>());
        }
        else
        {
            return AQuantBase{}.GetAQDramLoadWindow(aq_dram_block_window_tmp);
        }
    }

    template <typename BQDramBlockWindowTmp>
    CK_TILE_DEVICE constexpr auto
    GetBQDramLoadWindow(const BQDramBlockWindowTmp& bq_dram_block_window_tmp) const
    {
        return BQuantBase{}.GetBQDramLoadWindow(bq_dram_block_window_tmp);
    }
};

} // namespace ck_tile
