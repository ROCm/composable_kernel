// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/load_tile.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/tile/slice_tile.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck/tile_program/block_tile/block_reduce.hpp"
#include "ck/tile_program/tile/shuffle_distributed_tensor.hpp"

namespace ck {
namespace tile_program {
namespace block {

template <typename Problem, typename Policy = BlockFmhaBwdPipelineDefaultPolicy>
struct BlockFmhaBwdOGradDotO
{
    using ODataType     = remove_cvref_t<typename Problem::ODataType>;
    using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;
    using DDataType     = remove_cvref_t<typename Problem::DDataType>;

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;
    static constexpr index_t kVHeaddim   = Problem::kVHeaddim;

    static constexpr bool kIsGroupMode     = Problem::kIsGroupMode;
    static constexpr bool kM0NeedPadding   = Problem::kM0NeedPadding;
    static constexpr bool kK0N1NeedPadding = Problem::kK0N1NeedPadding;

    __host__ __device__ static constexpr ck::index_t GetSmemSize() { return 0; }

    template <typename ODramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename DDramBlockWindowTmp>
    __host__ __device__ void operator()(const ODramBlockWindowTmp& o_dram_block_window_tmp,
                                        const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
                                        DDramBlockWindowTmp& d_dram_block_window_tmp) const
    {
        static_assert(
            is_same_v<ODataType, remove_cvref_t<typename ODramBlockWindowTmp::DataType>> &&
                is_same_v<OGradDataType,
                          remove_cvref_t<typename OGradDramBlockWindowTmp::DataType>> &&
                is_same_v<DDataType, remove_cvref_t<typename DDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kBlockSize == ODramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kBlockSize == OGradDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kBlockSize == DDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}],
                      "wrong!");

        auto o_dram_window =
            make_tile_window(o_dram_block_window_tmp.GetBottomTensorView(),
                             o_dram_block_window_tmp.GetWindowLengths(),
                             o_dram_block_window_tmp.GetWindowOrigin(),
                             Policy::template MakePreODramTileDistribution<Problem>());

        auto o = load_tile(o_dram_window);

        auto do_dram_window =
            make_tile_window(do_dram_block_window_tmp.GetBottomTensorView(),
                             do_dram_block_window_tmp.GetWindowLengths(),
                             do_dram_block_window_tmp.GetWindowOrigin(),
                             Policy::template MakePreOGradDramTileDistribution<Problem>());

        auto do_ = load_tile(do_dram_window);

        // declare d
        constexpr auto d_dstr =
            make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
                o.GetTileDistribution().GetStaticTileDistributionEncoding(), Sequence<1>{}));

        auto d = make_static_distributed_tensor<DDataType>(d_dstr);

        // auto d = make_static_distributed_tensor<DDataType>(
        //     Policy::template MakePreDDramTileDistribution<Problem>());

        tile_elementwise_inout([](auto& c) { c = 0; }, d); // Initialize D

        constexpr auto o_spans = decltype(o)::GetDistributedSpans();
        sweep_tile_span(o_spans[Number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            sweep_tile_span(o_spans[Number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                d(i_idx) +=
                    (type_convert<DDataType>(o[i_j_idx]) * type_convert<DDataType>(do_[i_j_idx]));
            });
        });

        store_tile(d_dram_block_window_tmp, d);
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
