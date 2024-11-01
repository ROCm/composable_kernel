// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

template <typename WeightType, typename IndexType = index_t>
CK_TILE_HOST void reference_moe_sorting(IndexType* sorted_token_ids_ptr,
                                        WeightType* sorted_weight_buf,
                                        IndexType* sorted_expert_ids_ptr,
                                        index_t& sub_x_cnt,
                                        const WeightType* weights_ptr,
                                        const IndexType* topk_ids_ptr,
                                        const index_t num_token,
                                        const index_t experts,
                                        const index_t topk,
                                        const index_t sub_x)
{
    std::vector<std::vector<IndexType>> expert_tokens(experts,
                                                      std::vector<IndexType>(sub_x, num_token));
    std::vector<std::vector<WeightType>> expert_token_weights(experts,
                                                              std::vector<WeightType>(sub_x, 0));
    std::vector<IndexType> expert_slices(experts, 1);
    std::vector<IndexType> expert_slice_idxs(experts, 0);

    for(index_t t = 0; t < num_token; t++)
    {
        for(index_t k = 0; k < topk; k++)
        {
            index_t e    = *(topk_ids_ptr + t * topk + k);
            WeightType w = *(weights_ptr + t * topk + k);
            index_t idx  = expert_slice_idxs[e];
            if(idx > expert_slices[e] * sub_x - 1)
            {
                expert_slices[e]++;
                index_t new_size = expert_slices[e] * sub_x;
                expert_tokens[e].resize(new_size);
                expert_token_weights[e].resize(new_size);
                for(index_t idx = (expert_slices[e] - 1) * sub_x; idx < new_size; idx++)
                {
                    expert_tokens[e][idx]        = num_token;
                    expert_token_weights[e][idx] = 0;
                }
            }

            expert_tokens[e][idx]        = t;
            expert_token_weights[e][idx] = w;
            expert_slice_idxs[e]++;
        }
    }

    IndexType* tokens   = sorted_token_ids_ptr;
    WeightType* weights = sorted_weight_buf;
    IndexType* erp_ids  = sorted_expert_ids_ptr;
    for(index_t e = 0; e < experts; e++)
    {
        memcpy(tokens, expert_tokens[e].data(), sizeof(index_t) * expert_slices[e] * sub_x);
        tokens += expert_slices[e] * sub_x;
        memcpy(
            weights, expert_token_weights[e].data(), sizeof(WeightType) * expert_slices[e] * sub_x);
        weights += expert_slices[e] * sub_x;

        for(index_t s = 0; s < expert_slices[e]; s++)
        {
            erp_ids[s] = e;
            sub_x_cnt++;
        }
        erp_ids += expert_slices[e];
    }

    return;
}
} // namespace ck_tile
