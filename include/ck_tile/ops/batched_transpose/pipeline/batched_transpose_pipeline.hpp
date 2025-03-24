// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batched_transpose/pipeline/batched_transpose_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = BatchedTransposePolicy>
struct BatchedTransposePipeline
{
    // TODO: this kernel only support warp per row
    using Problem   = remove_cvref_t<Problem_>;
    using Policy    = remove_cvref_t<Policy_>;
    using InputType = ck_tile::remove_cvref_t<typename Problem::InputType>;
    static constexpr ck_tile::index_t kMPerBlock = Problem::kMPerBlock;
    static constexpr ck_tile::index_t kNPerBlock = Problem::kNPerBlock;
    static constexpr index_t AlignmentM          = Problem::AlignmentM;
    static constexpr index_t AlignmentN          = Problem::AlignmentN;
    static constexpr bool kPadM                  = Problem::kPadM;
    static constexpr bool kPadN                  = Problem::kPadN;

    template <typename InputWindow, typename OutputWindow>
    CK_TILE_DEVICE auto operator()(const InputWindow& input_window, OutputWindow& out_window)
    {
        // FIGURE THE RIGHT x value to pass to the function

        auto inp_win =
            make_tile_window(input_window, Policy::template MakeInputDistribution<Problem>());

        auto input_tile = load_tile(inp_win); // x->thread input_win->block

        auto output_tile = make_static_distributed_tensor<InputType>(Policy::template MakeOutputDistribution<Problem>());

        transpose_tile2d(output_tile, input_tile);

        auto out_win =
            make_tile_window(out_window, Policy::template MakeOutputDistribution<Problem>());

        store_tile(out_win, output_tile);

        /*auto inp_win =
            make_tile_window(input_window, Policy::template MakeInputDistribution<Problem>());


        auto x = load_tile(inp_win); // x->thread input_win->block

        auto y = make_static_distributed_tensor<InputType>(
            Policy::template MakeOutputDistribution<Problem>());

        constexpr auto span_2d_x = decltype(x)::get_distributed_spans();
        constexpr auto span_2d_y = decltype(y)::get_distributed_spans();

        constexpr auto element_byte    = sizeof(InputType);
        constexpr auto padding_element = 4 / element_byte;
        constexpr auto smem_stride     = 16 + padding_element;

        __shared__ InputType smem[16 * smem_stride];

        __syncthreads();

        sweep_tile_span(span_2d_x[number<0>{}], [&](auto idx0) {
            sweep_tile_span(span_2d_x[number<1>{}], [&](auto idx1) {
                uint32_t i_src_w                      = get_thread_id() & 15;
                uint32_t i_src_h                      = get_thread_id() >> 4;
                constexpr auto i_j_idx                = make_tuple(idx0, idx1);
                smem[i_src_h * smem_stride + i_src_w] = x(i_j_idx);
            });
        });

        __syncthreads();

        sweep_tile_span(span_2d_y[number<0>{}], [&](auto idx0) {
            sweep_tile_span(span_2d_y[number<1>{}], [&](auto idx1) {
                uint32_t i_src_w       = get_thread_id() & 15;
                uint32_t i_src_h       = get_thread_id() >> 4;
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                y(i_j_idx)             = smem[i_src_h * smem_stride + i_src_w];
            });
        });

        auto out_win =
            make_tile_window(out_window, Policy::template MakeOutputDistribution<Problem>());

        store_tile(out_win, y);*/
    }
};
} // namespace ck_tile
