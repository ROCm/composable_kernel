// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

struct MoeSortingHostArgs
{
    const void* p_topk_ids;
    const void* p_weights;
    void* sorted_token_ids;
    void* sorted_weights;
    void* expert_ids;
    void* total_tokens_post_pad;
    index_t tokens;
    index_t unit_size;
    index_t num_experts;
    index_t topk;
};

template <typename Problem_>
struct MoeSortingKernel
{
    using Problem = remove_cvref_t<Problem_>;

    using IndexType  = typename Problem::IndexType;
    using WeightType = typename Problem::WeightType;

    typedef MoeSortingHostArgs MoeSortingKargs;

    using Hargs = MoeSortingHostArgs;

    struct Kargs
    {
        const void* p_topk_ids;
        const void* p_weights;
        void* sorted_token_ids;
        void* sorted_weights;
        void* expert_ids;
        void* total_tokens_post_pad;
        index_t tokens;
        index_t num_experts;

        index_t tokens_per_thread;
        mdiv unit_size_mdiv;
        mdiv topk_mdiv;
    };

    CK_TILE_HOST static constexpr auto GridSize(const Hargs&)
    {
        // TODO: assume num-experts not too much
        return dim3(1);
    }

    CK_TILE_HOST static constexpr auto BlockSize(const Hargs& h)
    {
        // TODO: need pad to multiply of warp size
        return dim3(ck_tile::max(h.num_experts, ck_tile::get_warp_size()));
    }

    // in byte
    CK_TILE_DEVICE static constexpr index_t GetSmemSize()
    {
        // const auto blocks = BlockSize(h);
        // return ((blockDim.x + 1) * k.num_experts + (k.num_experts + 1)) * sizeof(index_t);
        // TODO: can not use dynamic calculation. need use static to guide compiler
        return 65536;
    }

    CK_TILE_HOST static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_topk_ids            = h.p_topk_ids;
        k.p_weights             = h.p_weights;
        k.sorted_token_ids      = h.sorted_token_ids;
        k.sorted_weights        = h.sorted_weights;
        k.expert_ids            = h.expert_ids;
        k.total_tokens_post_pad = h.total_tokens_post_pad;
        k.tokens                = h.tokens;
        k.num_experts           = h.num_experts;

        const auto blocks   = BlockSize(h);
        k.tokens_per_thread = integer_divide_ceil(h.tokens * h.topk, blocks.x);
        k.unit_size_mdiv    = mdiv{static_cast<uint32_t>(h.unit_size)};
        k.topk_mdiv         = mdiv{static_cast<uint32_t>(h.topk)};
        return k;
    }

    CK_TILE_DEVICE index_t calc_index(index_t total_col, index_t row, index_t col) const
    {
        return row * total_col + col;
    }

    CK_TILE_DEVICE void moe_align_block_size_kernel(const IndexType* __restrict__ topk_id,
                                                    const WeightType* __restrict__ weights,
                                                    index_t* sorted_token_ids,
                                                    WeightType* sorted_weights,
                                                    index_t* expert_ids,
                                                    index_t* total_tokens_post_pad,
                                                    const index_t num_experts,
                                                    const index_t tokens_per_thread,
                                                    const index_t numel,
                                                    const mdiv unit_size_mdiv,
                                                    const mdiv topk_mdiv,
                                                    void* smem) const
    {
        const index_t tid       = static_cast<index_t>(threadIdx.x);
        const index_t start_idx = tid * tokens_per_thread;

        index_t* shared_mem = reinterpret_cast<index_t*>(smem);

        index_t* tokens_cnts = shared_mem; // 2d: (blockDim.x + 1, num_experts)
        index_t* cumsum      = shared_mem + (blockDim.x + 1) * num_experts; // 1: (num_experts + 1)

        for(int i = 0; i < num_experts; ++i)
        {
            tokens_cnts[calc_index(num_experts, tid + 1, i)] = 0;
        }

        for(int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i)
        {
            ++tokens_cnts[calc_index(num_experts, tid + 1, topk_id[i])];
        }
        __syncthreads();

        if(tid < num_experts)
        {
            tokens_cnts[calc_index(num_experts, 0, tid)] = 0;
            for(int i = 1; i <= static_cast<index_t>(blockDim.x); ++i)
            {
                tokens_cnts[calc_index(num_experts, i, tid)] +=
                    tokens_cnts[calc_index(num_experts, i - 1, tid)];
            }
        }

        // __syncthreads();
        if(tid == 0)
        {
            cumsum[0] = 0;
            for(int i = 1; i <= num_experts; ++i)
            {
                auto current_units = [&]() {
                    index_t x_ = tokens_cnts[calc_index(num_experts, blockDim.x, i - 1)] +
                                 unit_size_mdiv.divisor - 1;
                    index_t y_ = unit_size_mdiv.div(x_);
                    return max(y_, 1) * unit_size_mdiv.divisor;
                }();
                cumsum[i] = cumsum[i - 1] + current_units;
            }
            *total_tokens_post_pad = unit_size_mdiv.div(cumsum[num_experts]);
        }

        __syncthreads();
        if(tid < num_experts)
        {
            for(int i = cumsum[tid]; i < cumsum[tid + 1]; i += unit_size_mdiv.divisor)
            {
                expert_ids[unit_size_mdiv.div(i)] = tid;
            }
        }

        for(int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i)
        {
            index_t expert_id = topk_id[i];
            index_t rank_post_pad =
                tokens_cnts[calc_index(num_experts, tid, expert_id)] + cumsum[expert_id];
            sorted_token_ids[rank_post_pad] = topk_mdiv.div(i);
            sorted_weights[rank_post_pad]   = weights[i];
            ++tokens_cnts[calc_index(num_experts, tid, expert_id)];
        }

        const index_t prefill_token = topk_mdiv.div(numel);
        if(tid < num_experts)
        {
            index_t expert_offset =
                cumsum[tid] + tokens_cnts[calc_index(num_experts, blockDim.x, tid)];
            while(expert_offset < cumsum[tid + 1])
            {
                sorted_token_ids[expert_offset] = prefill_token;
                sorted_weights[expert_offset]   = static_cast<WeightType>(0.0);
                expert_offset++;
            }
        }
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const size_t numel = kargs.tokens * kargs.topk_mdiv.divisor;
        __shared__ char smem[GetSmemSize()];
        return moe_align_block_size_kernel(static_cast<const IndexType*>(kargs.p_topk_ids),
                                           static_cast<const WeightType*>(kargs.p_weights),
                                           static_cast<IndexType*>(kargs.sorted_token_ids),
                                           static_cast<WeightType*>(kargs.sorted_weights),
                                           static_cast<IndexType*>(kargs.expert_ids),
                                           static_cast<IndexType*>(kargs.total_tokens_post_pad),
                                           kargs.num_experts,
                                           kargs.tokens_per_thread,
                                           numel,
                                           kargs.unit_size_mdiv,
                                           kargs.topk_mdiv,
                                           smem);
    }
};
} // namespace ck_tile
