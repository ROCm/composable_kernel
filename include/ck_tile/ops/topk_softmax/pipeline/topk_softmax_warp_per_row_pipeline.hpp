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
    using Problem    = remove_cvref_t<Problem_>;
    using Policy     = remove_cvref_t<Policy_>;
    using WeightType = typename Problem::WeightType;

    template <typename InputWindow, typename OutputWindow, typename IndexWindow>
    CK_TILE_DEVICE auto operator()(const InputWindow& input_window,
                                   OutputWindow& out_window,
                                   IndexWindow& idx_window,
                                   index_t k,
                                   index_t experts)
    {
        auto input_win = make_tile_window(input_window.get_bottom_tensor_view(),
                                          input_window.get_window_lengths(),
                                          input_window.get_window_origin(),
                                          Policy::template MakeInputDistribution<Problem>());

        auto x = load_tile(input_win);

        // cast and pad input data
        auto w = [&]() {
            auto w_ = cast_tile<WeightType>(x);

            constexpr auto span_2d = decltype(w_)::get_distributed_spans();
            sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    const auto x_indices =
                        get_x_indices_from_distributed_indices(w_.get_tile_distribution(), i_j_idx);
                    const auto current_expert = x_indices.at(number<1>{});
                    // set to -INF if OOB so that later softmax can work properly
                    w_(i_j_idx) =
                        current_expert >= experts ? -numeric<WeightType>::infinity() : w_(i_j_idx);
                });
            });
            return w_;
        }();

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
