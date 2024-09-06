// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/topk_softmax/pipeline/topk_softmax_warp_per_row_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = TopkSoftmaxWarpPerRowPolicy>
struct TopkSoftmaxWarpPerRowPipeline
{
    // TODO: this kernel only support warp per row
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    template <typename InputWindow, typename OutputWindow, typename IndexWindow>
    CK_TILE_DEVICE auto operator()(const InputWindow& input_window,
                                   OutputWindow& out_window,
                                   IndexWindow& idx_window,
                                   index_t k)
    {
        auto input_win = make_tile_window(input_window.get_bottom_tensor_view(),
                                          input_window.get_window_lengths(),
                                          input_window.get_window_origin(),
                                          Policy::template MakeInputDistribution<Problem>());

        auto x = load_tile(input_win);
        auto w = cast_tile<typename Problem::WeightType>(x);

        auto softmax = Policy::template GetSoftmax<Problem>();

        // softmax
        auto y = softmax(w);

        auto topk = Policy::template GetTopk<Problem>();

        auto out_win = make_tile_window(out_window.get_bottom_tensor_view(),
                                        out_window.get_window_lengths(),
                                        out_window.get_window_origin(),
                                        Policy::template MakeOutputDistribution<Problem>());
        auto idx_win = make_tile_window(idx_window.get_bottom_tensor_view(),
                                        idx_window.get_window_lengths(),
                                        idx_window.get_window_origin(),
                                        Policy::template MakeOutputDistribution<Problem>());

        topk(y, out_win, idx_win, k);
    }
};
} // namespace ck_tile
