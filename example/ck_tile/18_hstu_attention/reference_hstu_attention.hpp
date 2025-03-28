// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <thread>
#include <mutex>
#include <cassert>
#include <cmath>

#include <ck_tile/core.hpp>
#include <ck_tile/host/host_tensor.hpp>

#include "bool_switch.hpp"

namespace ck_tile {

// clang-format off
// Reference implementation of HSTUAttention problem, which does the following from input tensors:
// S[num_batch, num_head, seqlen, seqlen] = Q[num_batch, seqlen, num_head, hdim_qk] @ key^T[num_batch, seqlen, num_head, hdim_v]
// P[num_batch, num_head, seqlen, seqlen] = Masking(SiLu(S[num_batch, num_head, seqlen, seqlen]))
// O[num_batch, num_head, seqlen, hdim_v] = P[num_batch, num_head, seqlen, seqlen] @ value^T[num_batch, num_head, seqlen, hdim_v]
// The process is very similar to the generic attention, the difference is that SiLu is used rather than Softmax, and hstu masking 
// is much more complicated than the lower-triangular + disagonal-window based causal mask
// clang-format on

template <typename InOutDataType,
          typename GemmAccDataType,
          typename CompDataType,
          bool use_causal,
          bool use_local>
struct reference_hstu_attention
{
    struct hstu_mask
    {
        int max_attn_len;
        int contextual_seq_len;
        int min_full_attn_seq_len;
        int max_uih_len;

        hstu_mask(int max_attn_len_,
                  int contextual_seq_len_,
                  int min_full_attn_seq_len_,
                  int max_uih_len_)
        {
            max_attn_len          = max_attn_len_;
            contextual_seq_len    = contextual_seq_len_;
            min_full_attn_seq_len = min_full_attn_seq_len_;
            max_uih_len           = max_uih_len_;
        };

        bool IsPixelInsideMask(int row, int col)
        {
            if(row < contextual_seq_len)
                return true;

            bool result = false;
            if constexpr(use_local)
            {
                if constexpr(use_causal)
                    result = (row >= col) && (row - col <= max_attn_len);
                else
                    result = std::abs(row - col) <= max_attn_len;

                if(min_full_attn_seq_len > 0)
                    result = result || (row >= max_uih_len - min_full_attn_seq_len);
            }
            else
            {
                if constexpr(use_causal)
                    result = (row >= col);
            };

            return result;
        };
    };

    static void Run(const HostTensor<InOutDataType>& q_batch_seq_nhead_hdim,
                    const HostTensor<InOutDataType>& k_batch_seq_nhead_hdim,
                    const HostTensor<InOutDataType>& v_batch_seq_nhead_hdim,
                    HostTensor<InOutDataType>& o_batch_seq_nhead_hdim,
                    int num_batch,
                    float alpha,
                    std::vector<int> seq_offsets,
                    std::vector<int> num_targets, // define masking length at the end of token
                                                  // sequence to be excluded for attention
                    int max_attn_len,             // define the diagonal local window size
                    int contextual_seq_len,    // define masking length at the begin of query token
                                               // sequence to be included for attention
                    int min_full_attn_seq_len) // define masking length at the end of query token
                                               // sequence which is included for full attention
    {
        bool is_jagged = !seq_offsets.empty();

        if(is_jagged)
        {
            // check the number of batches
            assert(seq_offsets.size() == num_batch + 1);
            assert(q_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            assert(k_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            assert(v_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            assert(o_batch_seq_nhead_hdim.get_lengths()[0] == 1);
        }
        else
        {
            assert(q_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            assert(k_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            assert(v_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            assert(o_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
        };

        // check the sequence length
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == k_batch_seq_nhead_hdim.get_lengths()[1]);
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == v_batch_seq_nhead_hdim.get_lengths()[1]);
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == o_batch_seq_nhead_hdim.get_lengths()[1]);

        // check the number of heads
        int num_head = q_batch_seq_nhead_hdim.get_lengths()[2];
        assert(num_head == k_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == v_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == o_batch_seq_nhead_hdim.get_lengths()[2]);

        // check the hdim
        int hdim_qk = q_batch_seq_nhead_hdim.get_lengths()[3];
        int hdim_v  = v_batch_seq_nhead_hdim.get_lengths()[3];
        assert(hdim_qk == k_batch_seq_nhead_hdim.get_lengths()[3]);
        assert(hdim_v == o_batch_seq_nhead_hdim.get_lengths()[3]);

        auto silu = [](CompDataType x) {
            auto one = ck_tile::type_convert<CompDataType>(1.0f);

            auto sigmod_val = one / (one + std::exp(-x));

            return sigmod_val * x;
        };

        bool has_target = !num_targets.empty();

        if(has_target)
            assert(num_targets.size() == num_batch);

        auto f = [&](auto i_batch, auto i_head) {
            int seqlen = is_jagged ? (seq_offsets[i_batch + 1] - seq_offsets[i_batch])
                                   : q_batch_seq_nhead_hdim.get_lengths()[1];

            int max_uih_len = seqlen;

            if(contextual_seq_len > 0)
                max_uih_len -= contextual_seq_len - 1;

            if(has_target)
                max_uih_len -= num_targets[i_batch];

            hstu_mask mask{max_attn_len, contextual_seq_len, min_full_attn_seq_len, max_uih_len};

            // for all rows in the batch
            for(int sq = 0; sq < max_uih_len; sq++)
            {
                std::vector<CompDataType> locals;

                // for all cols in the batch
                for(int sk = 0; sk < max_uih_len; sk++)
                {
                    if(mask.IsPixelInsideMask(sq, sk))
                    {
                        GemmAccDataType dot_prod = 0.f;
                        for(int k = 0; k < hdim_qk; k++)
                        {
                            InOutDataType qreg = q_batch_seq_nhead_hdim(i_batch, sq, i_head, k);
                            InOutDataType kreg = k_batch_seq_nhead_hdim(i_batch, sk, i_head, k);

                            dot_prod += ck_tile::type_convert<GemmAccDataType>(qreg) *
                                        ck_tile::type_convert<GemmAccDataType>(kreg);
                        }

                        locals.push_back(ck_tile::type_convert<CompDataType>(dot_prod) *
                                         ck_tile::type_convert<CompDataType>(alpha));
                    }
                    else
                        locals.push_back(ck_tile::type_convert<CompDataType>(0.0f));
                };

                // SiLu element-wise
                for(CompDataType& elem : locals)
                    elem = silu(elem) / ck_tile::type_convert<CompDataType>(seqlen);

                // second Gemm
                for(int k = 0; k < hdim_v; k++)
                {
                    GemmAccDataType dot_prod = 0.f;

                    for(int sk = 0; sk < max_uih_len; sk++)
                    {
                        InOutDataType preg = ck_tile::type_convert<InOutDataType>(locals[sk]);
                        InOutDataType vreg = v_batch_seq_nhead_hdim(i_batch, sk, i_head, k);

                        dot_prod += ck_tile::type_convert<GemmAccDataType>(preg) *
                                    ck_tile::type_convert<GemmAccDataType>(vreg);
                    };

                    o_batch_seq_nhead_hdim(i_batch, sq, i_head, k) =
                        ck_tile::type_convert<InOutDataType>(dot_prod);
                };
            };
        };

        make_ParallelTensorFunctor(f, num_batch, num_head)(std::thread::hardware_concurrency());
    }
};

} // namespace ck_tile
