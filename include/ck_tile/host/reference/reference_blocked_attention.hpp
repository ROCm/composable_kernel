// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

template <typename AccT, typename T>
CK_TILE_HOST_DEVICE constexpr AccT to_acc(T value)
{
    if constexpr(std::is_same_v<T, ck_tile::bf16_t>)
    {
#if CK_TILE_USE_CUSTOM_DATA_TYPE
        return static_cast<AccT>(value);
#else
        return static_cast<AccT>(
            ck_tile::bf16_to_float_raw(ck_tile::bit_cast<ck_tile::bf16_raw_t>(value)));
#endif
    }
    else
    {
        return static_cast<AccT>(value);
    }
}

// Reference implementation: blocked attention (for sparse attention tests).
template <typename T, typename MaskT, typename AccT = float>
void reference_blocked_attention(
    const HostTensor<T>& q,                  // [B, H, S_q, D]
    const HostTensor<T>& k,                  // [B, H, S_k, D]
    const HostTensor<T>& v,                  // [B, H, S_k, D_v]
    const HostTensor<MaskT>& block_relation, // [B, H, Q_blocks, K_blocks]
    HostTensor<T>& output,                   // [B, H, S_q, D_v]
    index_t BLKQ,
    index_t BLKK,
    AccT scale)
{
    auto q_lengths   = q.get_lengths();
    index_t batch    = q_lengths[0];
    index_t nhead    = q_lengths[1];
    index_t seqlen_q = q_lengths[2];
    index_t hdim     = q_lengths[3];

    auto v_lengths   = v.get_lengths();
    index_t seqlen_k = v_lengths[2];
    index_t hdim_v   = v_lengths[3];

    index_t num_q_blocks = (seqlen_q + BLKQ - 1) / BLKQ;
    index_t num_k_blocks = (seqlen_k + BLKK - 1) / BLKK;

    for(index_t b = 0; b < batch; ++b)
    {
        for(index_t h = 0; h < nhead; ++h)
        {
            for(index_t qb = 0; qb < num_q_blocks; ++qb)
            {
                index_t q_start = qb * BLKQ;
                if(q_start >= seqlen_q)
                {
                    continue;
                }
                index_t q_end = std::min<index_t>(q_start + BLKQ, seqlen_q);

                std::vector<index_t> relevant_k_indices;
                for(index_t kb = 0; kb < num_k_blocks; ++kb)
                {
                    // Treat block_relation as boolean; >0.5 marks an active block.
                    if(static_cast<float>(block_relation(b, h, qb, kb)) > 0.5f)
                    {
                        relevant_k_indices.push_back(kb);
                    }
                }

                if(relevant_k_indices.empty())
                {
                    continue;
                }

                for(index_t sq = q_start; sq < q_end; ++sq)
                {
                    std::vector<AccT> scores;
                    AccT max_score = -std::numeric_limits<AccT>::infinity();

                    for(auto kb : relevant_k_indices)
                    {
                        index_t k_start = kb * BLKK;
                        if(k_start >= seqlen_k)
                        {
                            continue;
                        }
                        index_t k_end = std::min<index_t>(k_start + BLKK, seqlen_k);

                        for(index_t sk = k_start; sk < k_end; ++sk)
                        {
                            AccT score = 0.0f;
                            for(index_t d = 0; d < hdim; ++d)
                            {
                                score +=
                                    to_acc<AccT>(q(b, h, sq, d)) * to_acc<AccT>(k(b, h, sk, d));
                            }
                            score = score * scale;
                            scores.push_back(score);
                            max_score = std::max(max_score, score);
                        }
                    }

                    AccT sum_exp = 0.0f;
                    for(auto& s : scores)
                    {
                        s = std::exp(s - max_score);
                        sum_exp += s;
                    }
                    for(auto& s : scores)
                    {
                        s /= sum_exp;
                    }

                    for(index_t dv = 0; dv < hdim_v; ++dv)
                    {
                        AccT out_val     = 0.0f;
                        size_t score_idx = 0;

                        for(auto kb : relevant_k_indices)
                        {
                            index_t k_start = kb * BLKK;
                            if(k_start >= seqlen_k)
                            {
                                continue;
                            }
                            index_t k_end = std::min<index_t>(k_start + BLKK, seqlen_k);

                            for(index_t sk = k_start; sk < k_end; ++sk)
                            {
                                out_val += scores[score_idx] * to_acc<AccT>(v(b, h, sk, dv));
                                score_idx++;
                            }
                        }

                        output(b, h, sq, dv) = static_cast<T>(out_val);
                    }
                }
            }
        }
    }
}

} // namespace ck_tile
