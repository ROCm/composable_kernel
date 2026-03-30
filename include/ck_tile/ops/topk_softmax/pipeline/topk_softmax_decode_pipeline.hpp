// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/topk_softmax/pipeline/topk_softmax_warp_per_row_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

// M=1 decode variant of TopkSoftmaxWarpPerRowPipeline.
// Inlines the topk loop (same algorithm as BlockTopkStream2D) but writes
// results to shared memory instead of global memory tile windows. Then
// thread 0 directly emits moe_sorting-compatible sorted outputs.
template <typename Problem_, typename Policy_ = TopkSoftmaxWarpPerRowPolicy>
struct TopkSoftmaxDecodePipeline
{
    using Problem    = remove_cvref_t<Problem_>;
    using Policy     = remove_cvref_t<Policy_>;
    using WeightType = typename Problem::WeightType;
    using IndexType  = typename Problem::IndexType;

    static constexpr index_t kMaxTopk = 8;

    // Same struct as BlockTopkStream2D::ArgmaxPacket
    struct ArgmaxPacket
    {
        WeightType arg;
        index_t value;
    };

    template <typename InputWindow>
    CK_TILE_DEVICE auto operator()(const InputWindow& input_window,
                                   index_t experts,
                                   index_t k,
                                   bool renormalize,
                                   IndexType* __restrict__ p_sorted_token_ids,
                                   WeightType* __restrict__ p_sorted_weights,
                                   IndexType* __restrict__ p_sorted_expert_ids,
                                   IndexType* __restrict__ p_total_tokens_post_pad,
                                   void* __restrict__ p_moe_buf,
                                   index_t unit_size,
                                   index_t moe_buf_interm_dim,
                                   index_t moe_buf_elem_bytes)
    {
        auto inp_win = make_tile_window_linear(
            input_window, Policy::template MakeInputDistribution<Problem>(), sequence<0, 1>{});

        auto softmax = Policy::template GetSoftmax<Problem>();

        // --- Phase 1: Load input and compute softmax/sigmoid ---
        auto x = load_tile(inp_win);

        auto w = [&]() {
            auto w_  = make_static_distributed_tensor<WeightType>(x.get_tile_distribution());
            auto w_f = [&](auto idx) {
                w_(idx) = type_convert<WeightType>(x(idx));
                const auto x_indices =
                    get_x_indices_from_distributed_indices(w_.get_tile_distribution(), idx);
                const auto current_expert = x_indices.at(number<1>{});
                w_(idx) =
                    current_expert >= experts ? -numeric<WeightType>::infinity() : w_(idx);
                if constexpr(!Problem::ActivationIsSoftmax)
                {
                    w_(idx) = WeightType(1) / (WeightType(1) + exp(-w_(idx)));
                }
            };
            sweep_tile(w_, w_f);
            return w_;
        }();

        auto y = [&]() {
            if constexpr(Problem::ActivationIsSoftmax)
                return softmax(w);
            else
                return w;
        }();

        // --- Phase 2: Inline topk loop (same as BlockTopkStream2D but → shared mem) ---
        __shared__ IndexType s_expert_ids[kMaxTopk];
        __shared__ WeightType s_weights[kMaxTopk];
        __shared__ IndexType s_original_slots[kMaxTopk];

        const auto f_argmax = [](ArgmaxPacket e0, ArgmaxPacket e1) {
            return e0.arg > e1.arg ? e0 : e1;
        };

        // Exactly mirrors BlockTopkStream2D::operator() lines 45-100
        decltype(y) y_tmp = y;
        constexpr auto span_2d = decltype(y_tmp)::get_distributed_spans();

        for(index_t i_k = 0; i_k < k; i_k++)
        {
            // Build ArgmaxPacket distributed tensor (same as BlockTopkStream2D lines 56-71)
            auto packet = [&]() {
                auto tmp =
                    make_static_distributed_tensor<ArgmaxPacket>(y.get_tile_distribution());

                sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            tmp.get_tile_distribution(), make_tuple(idx0, idx1));
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        ArgmaxPacket t;
                        t.arg        = y_tmp(i_j_idx);
                        t.value      = tile_idx.at(number<1>{});
                        tmp(i_j_idx) = t;
                    });
                });
                return tmp;
            }();

            // Reduce to find argmax (same as BlockTopkStream2D lines 73-75)
            auto argmax_init =
                ArgmaxPacket{-numeric<WeightType>::infinity(), 0};
            auto r =
                block_tile_reduce<ArgmaxPacket>(packet, sequence<1>{}, f_argmax, argmax_init);
            block_tile_reduce_xor_sync(r, f_argmax);

            // Extract result and store to shared memory instead of tile windows.
            // After xor_sync, all threads have the same argmax. We use the same
            // r(i_j_idx) access pattern as BlockTopkStream2D line 82.
            sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    ArgmaxPacket winner    = r(i_j_idx);

                    if(threadIdx.x == 0)
                    {
                        s_expert_ids[i_k]     = static_cast<IndexType>(winner.value);
                        s_weights[i_k]        = winner.arg;
                        s_original_slots[i_k] = static_cast<IndexType>(i_k);
                    }
                });
            });

            // Mask out selected expert (same as BlockTopkStream2D lines 89-100)
            sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        y.get_tile_distribution(), make_tuple(idx0, idx1));
                    auto col_id = tile_idx.at(number<1>{});

                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    y_tmp(i_j_idx) = (col_id == r(i_j_idx).value)
                                         ? -numeric<WeightType>::infinity()
                                         : y_tmp(i_j_idx);
                });
            });
            __syncthreads();
        }

        // --- Phase 3: Produce sorted outputs (thread 0 only, trivial for M=1) ---
        if(threadIdx.x == 0)
        {
            if(renormalize)
            {
                WeightType sum = WeightType(0);
                for(index_t i = 0; i < k; i++)
                    sum += s_weights[i];
                if(sum != WeightType(0))
                {
                    WeightType inv_sum = WeightType(1) / sum;
                    for(index_t i = 0; i < k; i++)
                        s_weights[i] *= inv_sum;
                }
            }

            // Sort by expert_id (ascending). k <= 8, insertion sort.
            for(index_t i = 1; i < k; i++)
            {
                IndexType key_eid  = s_expert_ids[i];
                WeightType key_w   = s_weights[i];
                IndexType key_slot = s_original_slots[i];
                index_t j          = i - 1;
                while(j >= 0 && s_expert_ids[j] > key_eid)
                {
                    s_expert_ids[j + 1]     = s_expert_ids[j];
                    s_weights[j + 1]        = s_weights[j];
                    s_original_slots[j + 1] = s_original_slots[j];
                    j--;
                }
                s_expert_ids[j + 1]     = key_eid;
                s_weights[j + 1]        = key_w;
                s_original_slots[j + 1] = key_slot;
            }

            constexpr index_t num_tokens = 1;
            index_t write_offset    = 0;
            index_t expert_tile_idx = 0;

            IndexType sentinel =
                static_cast<uint32_t>((num_tokens & 0x00ffffff) | ((k & 0xff) << 24));

            for(index_t i = 0; i < k; i++)
            {
                IndexType expert_id = s_expert_ids[i];
                WeightType weight   = s_weights[i];
                IndexType topk_slot = s_original_slots[i];

                IndexType packed_id =
                    static_cast<uint32_t>((0 & 0x00ffffff) | ((topk_slot & 0xff) << 24));

                p_sorted_token_ids[write_offset] = packed_id;
                p_sorted_weights[write_offset]   = weight;

                for(index_t p = 1; p < unit_size; p++)
                {
                    p_sorted_token_ids[write_offset + p] = sentinel;
                    p_sorted_weights[write_offset + p]   = WeightType(0);
                }

                p_sorted_expert_ids[expert_tile_idx] = expert_id;

                write_offset += unit_size;
                expert_tile_idx++;
            }

            p_total_tokens_post_pad[0] = static_cast<IndexType>(k * unit_size);
            p_total_tokens_post_pad[1] = static_cast<IndexType>(num_tokens);
        }

        // --- Phase 4: Zero moe_buf cooperatively ---
        if(p_moe_buf != nullptr)
        {
            const index_t total_bytes = moe_buf_interm_dim * moe_buf_elem_bytes;
            const index_t total_elems = total_bytes / 16;

            using vector_type  = ext_vector_t<index_t, 4>;
            vector_type* p_buf = reinterpret_cast<vector_type*>(p_moe_buf);
            auto zero_         = vector_type{0};

            for(index_t i = threadIdx.x; i < total_elems; i += blockDim.x)
            {
                p_buf[i] = zero_;
            }
        }
    }
};
} // namespace ck_tile
