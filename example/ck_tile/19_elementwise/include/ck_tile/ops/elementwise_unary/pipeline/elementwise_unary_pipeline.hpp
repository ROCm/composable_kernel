// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise_unary/pipeline/elementwise_unary_policy.hpp"
#include <string>
#include <type_traits>

#ifndef TOPK_SOFTMAX_USE_RAW_TILE_WINDOW
#define TOPK_SOFTMAX_USE_RAW_TILE_WINDOW 1
#endif

namespace ck_tile {

template <typename Problem_, typename Policy_ = ElementwiseUnaryPolicy>
struct ElementwiseUnaryipeline
{
    // TODO: this kernel only support warp per row
    using Problem      = remove_cvref_t<Problem_>;
    using Policy       = remove_cvref_t<Policy_>;
    using UnaryFunctor = typename Problem::UnaryFunctor;

    template <typename InputWindow, typename OutputWindow>
    CK_TILE_DEVICE auto
    operator()(const InputWindow& inp_window, OutputWindow& out_window, index_t loop_stride)
    {
        auto inp_win = make_tile_window(inp_window.get_bottom_tensor_view(),
                                        inp_window.get_window_lengths(),
                                        inp_window.get_window_origin(),
                                        Policy::template MakeInputDistribution<Problem>());
        auto out_win = make_tile_window(out_window.get_bottom_tensor_view(),
                                        out_window.get_window_lengths(),
                                        out_window.get_window_origin(),
                                        Policy::template MakeOutputDistribution<Problem>());

        static_for<0, Problem::Chunks, 1>{}([&](auto) {
            auto x = load_tile(inp_win);
            auto y = make_static_distributed_tensor<typename Problem::OutputType>(
                x.get_tile_distribution());

            tile_elementwise_inout(UnaryFunctor{}, y, x);
            store_tile(out_win, y);

            move_tile_window(inp_win, {loop_stride});
            move_tile_window(out_win, {loop_stride});
        });
    }
};
} // namespace ck_tile
