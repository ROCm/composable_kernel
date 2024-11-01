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
    // using Pipeline = remove_cvref_t<Pipeline_>;
    using Problem = remove_cvref_t<Problem_>;

    using IndexType  = typename Problem::IndexType;
    using WeightType = typename Problem::WeightType;

    typedef MoeSortingHostArgs MoeSortingKargs;

    using Kargs = MoeSortingKargs;
    using Hargs = MoeSortingHostArgs;

    CK_TILE_HOST static constexpr auto MakeKargs(const Hargs& h) { return h; }

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
                                                    const index_t unit_size,
                                                    const size_t numel,
                                                    const index_t topk) const
    {
        const size_t tokens_per_thread = integer_divide_ceil(numel, blockDim.x);
        const size_t start_idx         = threadIdx.x * tokens_per_thread;

        extern __shared__ index_t shared_mem[];

        index_t* tokens_cnts = shared_mem; // 2d: (blockDim.x + 1, num_experts)
        index_t* cumsum      = shared_mem + (blockDim.x + 1) * num_experts; // 1: (num_experts + 1)

        for(int i = 0; i < num_experts; ++i)
        {
            tokens_cnts[calc_index(num_experts, threadIdx.x + 1, i)] = 0;
        }

        for(int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i)
        {
            ++tokens_cnts[calc_index(num_experts, threadIdx.x + 1, topk_id[i])];
        }
        __syncthreads();

        if(threadIdx.x < num_experts)
        {
            tokens_cnts[calc_index(num_experts, 0, threadIdx.x)] = 0;
            for(int i = 1; i <= blockDim.x; ++i)
            {
                tokens_cnts[calc_index(num_experts, i, threadIdx.x)] +=
                    tokens_cnts[calc_index(num_experts, i - 1, threadIdx.x)];
            }
        }

        __syncthreads();
        if(threadIdx.x == 0)
        {
            cumsum[0] = 0;
            for(int i = 1; i <= num_experts; ++i)
            {
                cumsum[i] =
                    cumsum[i - 1] +
                    max(integer_divide_ceil(tokens_cnts[calc_index(num_experts, blockDim.x, i - 1)],
                                            unit_size),
                        1) *
                        unit_size;
            }
            *total_tokens_post_pad = cumsum[num_experts] / unit_size;
        }

        __syncthreads();
        if(threadIdx.x < num_experts)
        {
            for(int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += unit_size)
            {
                expert_ids[i / unit_size] = threadIdx.x;
            }
        }

        for(int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i)
        {
            index_t expert_id = topk_id[i];
            index_t rank_post_pad =
                tokens_cnts[calc_index(num_experts, threadIdx.x, expert_id)] + cumsum[expert_id];
            sorted_token_ids[rank_post_pad] = i / topk;
            sorted_weights[rank_post_pad]   = weights[i];
            ++tokens_cnts[calc_index(num_experts, threadIdx.x, expert_id)];
        }
        const index_t prefill_token = numel / topk;
        if(threadIdx.x < num_experts)
        {
            index_t expert_offset =
                cumsum[threadIdx.x] + tokens_cnts[calc_index(num_experts, blockDim.x, threadIdx.x)];
            while(expert_offset < cumsum[threadIdx.x + 1])
            {
                sorted_token_ids[expert_offset] = prefill_token;
                sorted_weights[expert_offset]   = static_cast<WeightType>(0.0);
                expert_offset++;
            }
        }
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const size_t numel = kargs.tokens * kargs.topk;
        return moe_align_block_size_kernel(static_cast<const IndexType*>(kargs.p_topk_ids),
                                           static_cast<const WeightType*>(kargs.p_weights),
                                           static_cast<IndexType*>(kargs.sorted_token_ids),
                                           static_cast<WeightType*>(kargs.sorted_weights),
                                           static_cast<IndexType*>(kargs.expert_ids),
                                           static_cast<IndexType*>(kargs.total_tokens_post_pad),
                                           kargs.num_experts,
                                           kargs.unit_size,
                                           numel,
                                           kargs.topk);
    }
};
} // namespace ck_tile
