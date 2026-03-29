// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

struct TopkSoftmaxDecodeHostArgs
{
    // Input (gating logits)
    const void* p_input;
    index_t num_experts;
    index_t topk;
    index_t stride_input;
    bool renormalize;

    // Output (moe_sorting format)
    void* p_sorted_token_ids;
    void* p_sorted_weights;
    void* p_sorted_expert_ids;
    void* p_total_tokens_post_pad;

    // moe_buf zeroing
    void* p_moe_buf;
    index_t unit_size;
    index_t moe_buf_interm_dim;
    index_t moe_buf_elem_bytes;
};

template <typename Pipeline_>
struct TopkSoftmaxDecodeKernel
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using Problem  = remove_cvref_t<typename Pipeline::Problem>;

    using InputType  = typename Problem::InputType;
    using WeightType = typename Problem::WeightType;
    using IndexType  = typename Problem::IndexType;

    static constexpr index_t kBlockSize = Problem::BlockSize;

    struct Kargs
    {
        const void* p_input;
        index_t num_experts;
        index_t topk;
        index_t stride_input;
        bool renormalize;

        void* p_sorted_token_ids;
        void* p_sorted_weights;
        void* p_sorted_expert_ids;
        void* p_total_tokens_post_pad;

        void* p_moe_buf;
        index_t unit_size;
        index_t moe_buf_interm_dim;
        index_t moe_buf_elem_bytes;
    };

    using Hargs = TopkSoftmaxDecodeHostArgs;

    CK_TILE_HOST static constexpr auto GridSize(const Hargs&) { return dim3(1); }

    CK_TILE_HOST static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_input                 = h.p_input;
        k.num_experts             = h.num_experts;
        k.topk                    = h.topk;
        k.stride_input            = h.stride_input;
        k.renormalize             = h.renormalize;
        k.p_sorted_token_ids      = h.p_sorted_token_ids;
        k.p_sorted_weights        = h.p_sorted_weights;
        k.p_sorted_expert_ids     = h.p_sorted_expert_ids;
        k.p_total_tokens_post_pad = h.p_total_tokens_post_pad;
        k.p_moe_buf               = h.p_moe_buf;
        k.unit_size               = h.unit_size;
        k.moe_buf_interm_dim      = h.moe_buf_interm_dim;
        k.moe_buf_elem_bytes      = h.moe_buf_elem_bytes;
        return k;
    }

    CK_TILE_HOST_DEVICE static constexpr auto BlockSize() { return Problem::BlockSize; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        constexpr index_t num_rows = 1;

        const auto input_window = [&]() {
            const InputType* p_input = reinterpret_cast<const InputType*>(kargs.p_input);
            auto tmp = make_naive_tensor_view<address_space_enum::global>(
                p_input,
                make_tuple(num_rows, kargs.num_experts),
                make_tuple(kargs.stride_input, 1),
                number<Problem::VectorSize>{},
                number<1>{});
            auto view = pad_tensor_view(
                tmp,
                make_tuple(number<Problem::RowsPerBlock>{}, number<Problem::Experts>{}),
                sequence<0, 1>{});
            return make_tile_window(
                view,
                make_tuple(number<Problem::RowsPerBlock>{}, number<Problem::Experts>{}),
                {0, 0});
        }();

        Pipeline{}(input_window,
                   kargs.num_experts,
                   kargs.topk,
                   kargs.renormalize,
                   reinterpret_cast<IndexType*>(kargs.p_sorted_token_ids),
                   reinterpret_cast<WeightType*>(kargs.p_sorted_weights),
                   reinterpret_cast<IndexType*>(kargs.p_sorted_expert_ids),
                   reinterpret_cast<IndexType*>(kargs.p_total_tokens_post_pad),
                   kargs.p_moe_buf,
                   kargs.unit_size,
                   kargs.moe_buf_interm_dim,
                   kargs.moe_buf_elem_bytes);
    }
};
} // namespace ck_tile
