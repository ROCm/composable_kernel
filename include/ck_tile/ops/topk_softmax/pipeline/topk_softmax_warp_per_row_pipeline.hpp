// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/topk_softmax/pipeline/topk_softmax_warp_per_row_policy.hpp"
#include <string>
#include <type_traits>

#ifndef TOPK_SOFTMAX_USE_RAW_TILE_WINDOW
#define TOPK_SOFTMAX_USE_RAW_TILE_WINDOW 0
#endif

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
                                   index_t rows,
                                   index_t experts,
                                   index_t k,
                                   index_t block_row_id)
    {
#if TOPK_SOFTMAX_USE_RAW_TILE_WINDOW
        auto inp_win = make_tile_window_linear_raw(
            input_window, Policy::template MakeInputDistribution<Problem>(), sequence<0, 1>{});
#else
        auto inp_win = make_tile_window_linear(
            input_window, Policy::template MakeInputDistribution<Problem>(), sequence<0, 1>{});
#endif
        auto out_win = make_tile_window_linear(out_window.get_bottom_tensor_view(),
                                               out_window.get_window_lengths(),
                                               out_window.get_window_origin(),
                                               Policy::template MakeOutputDistribution<Problem>());
        auto idx_win = make_tile_window_linear(idx_window.get_bottom_tensor_view(),
                                               idx_window.get_window_lengths(),
                                               idx_window.get_window_origin(),
                                               Policy::template MakeOutputDistribution<Problem>());

        auto softmax = Policy::template GetSoftmax<Problem>();
        auto topk    = Policy::template GetTopk<Problem>();

        const index_t grid_rows_per_loop = gridDim.x * Problem::RowsPerBlock;

        while(1)
        {
#if TOPK_SOFTMAX_USE_RAW_TILE_WINDOW
            __builtin_amdgcn_sched_barrier(0);
            auto x =
                load_tile_raw(inp_win, number<-1>{}, bool_constant<true>{}, bool_constant<true>{});
            buffer_load_fence(number<0>{});
            __builtin_amdgcn_sched_barrier(0);
#else
            auto x = load_tile(inp_win);
#endif
            // cast and pad input data
            auto w = [&]() {
#if 0
                auto w_ = cast_tile<WeightType>(x);

                constexpr auto span_2d = decltype(w_)::get_distributed_spans();
                sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        const auto x_indices   = get_x_indices_from_distributed_indices(
                            w_.get_tile_distribution(), i_j_idx);
                        const auto current_expert = x_indices.at(number<1>{});
                        // set to -INF if OOB so that later softmax can work properly
                        w_(i_j_idx) = current_expert >= experts ? -numeric<WeightType>::infinity()
                                                                : w_(i_j_idx);
                    });
                });
                return w_;
#else
                auto w_  = make_static_distributed_tensor<WeightType>(x.get_tile_distribution());
                auto w_f = [&](auto idx) {
                    w_(idx) = type_convert<WeightType>(x(idx));
                    const auto x_indices =
                        get_x_indices_from_distributed_indices(w_.get_tile_distribution(), idx);
                    const auto current_expert = x_indices.at(number<1>{});
                    w_(idx) =
                        current_expert >= experts ? -numeric<WeightType>::infinity() : w_(idx);
                };
                tile_sweeper ts{w_, w_f};
                ts();
                return w_;
#endif
            }();

            // softmax
            // auto y = softmax(w);
            auto x_tmp = softmax(w);

            // -------------------------------------------------------------------------------------
            // Grouped topk part
            const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
            struct ArgmaxPacket
            {
                WeightType value;
                index_t arg;
            };

            // Step1. calculate group score
            int num_expert_group = 16;
            int topk_group = 2;
            int expert_per_group = experts / num_expert_group;
            constexpr auto p_compute_spans = decltype(x_tmp)::get_distributed_spans();
            auto group_scores = x_tmp;
            // init group_scores to inf
            sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    group_scores(i_j_idx) = -numeric<WeightType>::infinity();
                });
            });
            for (index_t n_group = 0; n_group < num_expert_group; n_group++) {
                // get group value matrix (masked other groups)
                auto group_tmp = x_tmp;
                sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            group_tmp.get_tile_distribution(), make_tuple(idx0, idx1));
                        auto col_id = tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        group_tmp(i_j_idx) = ((col_id >= (n_group * expert_per_group)) && (col_id < ((n_group + 1) * expert_per_group))) ? x_tmp(i_j_idx) : -numeric<WeightType>::infinity();
                    });
                });
                // get one column for group scores = rowmax(group_tmp)
                auto group_scores_col = block_tile_reduce<WeightType>(
                    group_tmp, sequence<1>{}, f_max, std::numeric_limits<WeightType>::lowest());

                block_tile_reduce_sync(group_scores_col, f_max);
                // get all group scores
                sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            x_tmp.get_tile_distribution(), make_tuple(idx0, idx1));
                        auto col_id = tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        group_scores(i_j_idx) = (col_id == n_group) ? group_scores_col(i_idx): group_scores(i_j_idx);
                    });
                });
            }

            // Step2: select topk group and cal mask score matrix
            // argmax for topk
            const auto f_argmax = [](ArgmaxPacket e0, ArgmaxPacket e1) {
                return e0.value > e1.value ? e0 : e1;
            };

            // topk_group_mask(1 for selected group scores, -inf for other group scores)
            auto topk_group_scores_mask = x_tmp;
            // init topk_group_scores_mask to -inf
            sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    topk_group_scores_mask(i_j_idx) = -numeric<WeightType>::infinity();
                });
            });

            for(index_t k_group = 0; k_group < topk_group; k_group++)
            {
                auto group_packet            = [&]() {
                    auto tmp = make_static_distributed_tensor<ArgmaxPacket>(x_tmp.get_tile_distribution());
                    sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                        sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                            const auto tile_idx = get_x_indices_from_distributed_indices(
                                tmp.get_tile_distribution(), make_tuple(idx0, idx1));
                            constexpr auto i_j_idx = make_tuple(idx0, idx1);
                            ArgmaxPacket t;
                            t.value    = group_scores(i_j_idx);
                            t.arg      = tile_idx.at(number<1>{});
                            tmp(i_j_idx) = t;
                        });
                    });
                    return tmp;
                }();

                auto argmax_init = ArgmaxPacket{-numeric<WeightType>::infinity(), 0};
                auto group_r = block_tile_reduce<ArgmaxPacket>(group_packet, sequence<1>{}, f_argmax, argmax_init);

                block_tile_reduce_xor_sync(group_r, f_argmax);

                sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            x_tmp.get_tile_distribution(), make_tuple(idx0, idx1));
                        // auto row_id = tile_idx.at(number<0>{});
                        auto col_id = tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        auto k_group_idx       = group_r(i_idx).arg;
                        topk_group_scores_mask(i_j_idx) = ((col_id >= (k_group_idx * expert_per_group)) && (col_id < ((k_group_idx + 1) * expert_per_group))) ? 1 : topk_group_scores_mask(i_j_idx);          
                    });
                });

                // update value
                sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            x_tmp.get_tile_distribution(), make_tuple(idx0, idx1));
                        auto col_id = tile_idx.at(number<1>{});

                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        group_scores(i_j_idx) = (col_id == group_r(i_idx).arg) ? -numeric<WeightType>::infinity()
                                                                        : group_scores(i_j_idx);
                    });
                });
            }
            // Step3: mask score matrix
            auto x_tmp_masked = x_tmp;
            sweep_tile_span(p_compute_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_compute_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    x_tmp_masked(i_j_idx) = x_tmp(i_j_idx) * topk_group_scores_mask(i_j_idx);
                });
            });
            // Step4: select topk values from masked score matrix
            topk(x_tmp_masked, out_win, idx_win, k);

            // check exit
            if constexpr(Problem::LaunchType == 0)
            {
                break;
            }
            else
            {
                block_row_id += grid_rows_per_loop;
                if(block_row_id >= rows)
                    break;
            }

            move_tile_window(inp_win, {grid_rows_per_loop, number<0>{}});
            move_tile_window(out_win, {grid_rows_per_loop, number<0>{}});
            move_tile_window(idx_win, {grid_rows_per_loop, number<0>{}});
        }
    }
};
} // namespace ck_tile
