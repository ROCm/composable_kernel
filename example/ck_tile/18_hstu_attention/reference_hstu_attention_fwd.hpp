// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <thread>
#include <mutex>
#include <cassert>
#include <cmath>

#include <ck_tile/core.hpp>
#include <ck_tile/host/host_tensor.hpp>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_block_masking.hpp"

namespace ck_tile {

// clang-format off
// Reference implementation of HSTUAttention forward problem, which does the following from input tensors:
// S[num_batch, num_head, seqlen, seqlen] = Q[num_batch, seqlen, num_head, hdim_qk] @ K[num_batch, seqlen, num_head, hdim_v]
// P[num_batch, num_head, seqlen, seqlen] = SiLU(Masking(S[num_batch, num_head, seqlen, seqlen]))
// O[num_batch, num_head, seqlen, hdim_v] = P[num_batch, num_head, seqlen, seqlen] @ V^T[num_batch, num_head, seqlen, hdim_v]
// The process is very similar to the generic attention, the difference is that SiLU is used rather than Softmax, and hstu masking 
// is much more complicated than the lower-triangular + disagonal-window based causal mask
// clang-format on

template <typename InOutDataType,
          typename GemmAccDataType,
          typename CompDataType,
          bool kIsJagged,
          bool kUseCausal>
struct reference_no_group_hstu_attention_fwd
{
    static void Run(bool is_cross_attention,
                    bool use_softmax,
                    bool store_lse,
                    const HostTensor<InOutDataType>& q_batch_seq_nhead_hdim,
                    const HostTensor<InOutDataType>& k_batch_seq_nhead_hdim,
                    const HostTensor<InOutDataType>& v_batch_seq_nhead_hdim,
                    HostTensor<InOutDataType>& o_batch_seq_nhead_hdim,
                    HostTensor<CompDataType>& lse_batch_seq_nhead,
                    HostTensor<int8_t>& mask_batch_nhead_seq_seq,
                    int num_batch,
                    float alpha,
                    float attn_scale,
                    int max_seqlen_q,
                    int max_seqlen_kv, // only used as last dim of the tensor for saving the mask
                    std::vector<int> seq_q_offsets,
                    std::vector<int> seq_kv_offsets,
                    std::vector<int> num_targets, // define masking length at the end of token
                                                  // sequence to be excluded for attention
                    int contextual_seqlen,    // define masking length at the begin of query token
                                              // sequence to be included for attention
                    int window_size,          // define the diagonal local window size
                    int min_full_attn_seqlen) // define masking length at the end of query token
                                              // sequence which is included for full attention
    {
        if constexpr(kIsJagged)
        {
            // check the number of batches
            assert(!seq_q_offsets.empty() && seq_q_offsets.size() == num_batch + 1);
            assert(!seq_kv_offsets.empty() && seq_kv_offsets.size() == num_batch + 1);
            assert(q_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            assert(k_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            assert(v_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            assert(o_batch_seq_nhead_hdim.get_lengths()[0] == 1);

            if(store_lse)
            {
                assert(lse_batch_seq_nhead.get_lengths()[0] == 1);
            }
        }
        else
        {
            assert(seq_q_offsets.empty());
            assert(seq_kv_offsets.empty());
            assert(q_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            assert(k_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            assert(v_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            assert(o_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);

            if(store_lse)
            {
                assert(lse_batch_seq_nhead.get_lengths()[0] == num_batch);
            }
        };

        // check the sequence length
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == k_batch_seq_nhead_hdim.get_lengths()[1]);
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == v_batch_seq_nhead_hdim.get_lengths()[1]);
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == o_batch_seq_nhead_hdim.get_lengths()[1]);
        if(store_lse)
        {
            assert(q_batch_seq_nhead_hdim.get_lengths()[1] == lse_batch_seq_nhead.get_lengths()[1]);
        }

        // check the number of heads
        int num_head = q_batch_seq_nhead_hdim.get_lengths()[2];
        assert(num_head == k_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == v_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == o_batch_seq_nhead_hdim.get_lengths()[2]);
        if(store_lse)
        {
            assert(num_head == lse_batch_seq_nhead.get_lengths()[2]);
        }

        // check the hdim
        int hdim_qk = q_batch_seq_nhead_hdim.get_lengths()[3];
        int hdim_v  = v_batch_seq_nhead_hdim.get_lengths()[3];
        assert(hdim_qk == k_batch_seq_nhead_hdim.get_lengths()[3]);
        assert(hdim_v == o_batch_seq_nhead_hdim.get_lengths()[3]);

        bool save_mask = false;

        if(static_cast<int>(mask_batch_nhead_seq_seq.get_lengths()[0]) == num_batch &&
           static_cast<int>(mask_batch_nhead_seq_seq.get_lengths()[1]) == num_head &&
           static_cast<int>(mask_batch_nhead_seq_seq.get_lengths()[2]) == max_seqlen_q &&
           static_cast<int>(mask_batch_nhead_seq_seq.get_lengths()[3]) == max_seqlen_kv)
            save_mask = true;

        // check num_tagets
        assert(num_tagets.empty() || num_targets.size() == num_batch);

        auto silu = [&](CompDataType x) {
            const auto one = ck_tile::type_convert<CompDataType>(1.0f);

            return x / (one + std::exp(-x));
        };

        auto f = [&](auto i_batch, auto i_head) {
            int seqlen_q  = kIsJagged ? (seq_q_offsets[i_batch + 1] - seq_q_offsets[i_batch])
                                      : q_batch_seq_nhead_hdim.get_lengths()[1];
            int seqlen_kv = kIsJagged ? (seq_kv_offsets[i_batch + 1] - seq_kv_offsets[i_batch])
                                      : k_batch_seq_nhead_hdim.get_lengths()[1];

            int num_target = num_targets.empty() ? 0 : num_targets[i_batch];

            float scale_p = attn_scale ? attn_scale : 1.0f / static_cast<float>(max_seqlen_q);

            BOOL_SWITCH_2(window_size > 0, kHasLocal, is_cross_attention, kIsCrossAttention, [&] {
                using HstuMaskType =
                    typename HstuBlockMasking<kIsCrossAttention, kUseCausal, kHasLocal>::Type;

                HstuMaskType mask = [&]() {
                    if constexpr(kHasLocal)
                    {
                        if constexpr(kIsCrossAttention)
                        {
                            // need adjust the min_full_attn_seqlen passed to the HstuBlockMask() if
                            // the user passed min_full_attn_seqlen is bigger than max_uih_len
                            if(seqlen_q - num_target > min_full_attn_seqlen)
                                return ck_tile::make_hstu_cross_attention_block_mask_with_local<
                                    HstuMaskType>(true,
                                                  seqlen_q,
                                                  seqlen_kv,
                                                  contextual_seqlen,
                                                  num_target,
                                                  window_size,
                                                  min_full_attn_seqlen);
                            else
                                return ck_tile::make_hstu_cross_attention_block_mask_with_local<
                                    HstuMaskType>(true,
                                                  seqlen_q,
                                                  seqlen_kv,
                                                  contextual_seqlen,
                                                  num_target,
                                                  window_size,
                                                  seqlen_q - num_target);
                        }
                        else
                        {
                            // need adjust the min_full_attn_seqlen passed to the HstuBlockMask() if
                            // the user passed min_full_attn_seqlen is bigger than max_uih_len
                            if(seqlen_q - num_target > min_full_attn_seqlen)
                                return ck_tile::make_hstu_self_attention_block_mask_with_local<
                                    HstuMaskType>(true,
                                                  seqlen_q,
                                                  contextual_seqlen,
                                                  num_target,
                                                  window_size,
                                                  min_full_attn_seqlen);
                            else
                                return ck_tile::make_hstu_self_attention_block_mask_with_local<
                                    HstuMaskType>(true,
                                                  seqlen_q,
                                                  contextual_seqlen,
                                                  num_target,
                                                  window_size,
                                                  seqlen_q - num_target);
                        }
                    }
                    else
                    {
                        if constexpr(kIsCrossAttention)
                            return ck_tile::make_hstu_cross_attention_block_mask_without_local<
                                HstuMaskType>(seqlen_q, seqlen_kv, contextual_seqlen, num_target);
                        else
                            return ck_tile::make_hstu_self_attention_block_mask_without_local<
                                HstuMaskType>(seqlen_q, contextual_seqlen, num_target);
                    }
                }();

                if(save_mask)
                {
                    for(int sq = 0; sq < max_seqlen_q; sq++)
                        for(int sk = 0; sk < max_seqlen_kv; sk++)
                            mask_batch_nhead_seq_seq(i_batch, i_head, sq, sk) = 0;

                    for(int sq = 0; sq < seqlen_q; sq++)
                        for(int sk = 0; sk < seqlen_kv; sk++)
                            mask_batch_nhead_seq_seq(i_batch, i_head, sq, sk) =
                                static_cast<int8_t>(mask.IsTokenPairInsideMask(sq, sk));
                }

                // for all rows in the batch
                for(int sq = 0; sq < seqlen_q; sq++)
                {
                    CompDataType m =
                        -ck_tile::numeric<CompDataType>::infinity(); // max value of the row
                    CompDataType l =
                        ck_tile::type_convert<CompDataType>(0.0f); // sum of exp(x-m) of the row
                                                                   //
                    std::vector<CompDataType> locals;

                    // for all cols in the batch
                    for(int sk = 0; sk < seqlen_kv; sk++)
                    {
                        if(mask.IsTokenPairInsideMask(sq, sk))
                        {
                            GemmAccDataType dot_prod = 0.f;
                            for(int k = 0; k < hdim_qk; k++)
                            {
                                if constexpr(kIsJagged)
                                {
                                    InOutDataType qreg = q_batch_seq_nhead_hdim(
                                        0, seq_q_offsets[i_batch] + sq, i_head, k);
                                    InOutDataType kreg = k_batch_seq_nhead_hdim(
                                        0, seq_kv_offsets[i_batch] + sk, i_head, k);

                                    dot_prod += ck_tile::type_convert<GemmAccDataType>(qreg) *
                                                ck_tile::type_convert<GemmAccDataType>(kreg);
                                }
                                else
                                {
                                    InOutDataType qreg =
                                        q_batch_seq_nhead_hdim(i_batch, sq, i_head, k);
                                    InOutDataType kreg =
                                        k_batch_seq_nhead_hdim(i_batch, sk, i_head, k);

                                    dot_prod += ck_tile::type_convert<GemmAccDataType>(qreg) *
                                                ck_tile::type_convert<GemmAccDataType>(kreg);
                                };
                            }

                            locals.push_back(ck_tile::type_convert<CompDataType>(dot_prod) *
                                             ck_tile::type_convert<CompDataType>(alpha));
                        }
                        else
                        {
                            if(!use_softmax)
                                locals.push_back(ck_tile::type_convert<CompDataType>(0.0f));
                            else
                                locals.push_back(-ck_tile::numeric<CompDataType>::infinity());
                        };
                    };

                    if(!use_softmax)
                    {
                        // SiLu element-wise
                        for(CompDataType& elem : locals)
                            elem = silu(elem) * ck_tile::type_convert<CompDataType>(scale_p);
                    }
                    else
                    {
                        for(CompDataType elem : locals)
                            m = ck_tile::max(m, elem);

                        if(m == -ck_tile::numeric<CompDataType>::infinity())
                        {
                            for(CompDataType& elem : locals)
                                elem = ck_tile::type_convert<CompDataType>(0.0f);
                        }
                        else
                        {
                            // stabalized sum of exp()
                            for(CompDataType elem : locals)
                                l += std::exp(elem - m);

                            // normalization
                            for(CompDataType& elem : locals)
                                elem = std::exp(elem - m) / l;
                        }

                        if(store_lse)
                        {
                            if constexpr(kIsJagged)
                                lse_batch_seq_nhead(0, seq_q_offsets[i_batch] + sq, i_head) =
                                    std::log(l) + m;
                            else
                                lse_batch_seq_nhead(i_batch, sq, i_head) = std::log(l) + m;
                        }
                    };

                    // second Gemm
                    for(int k = 0; k < hdim_v; k++)
                    {
                        GemmAccDataType dot_prod = 0.f;

                        for(int sk = 0; sk < seqlen_kv; sk++)
                        {
                            if constexpr(kIsJagged)
                            {
                                InOutDataType preg =
                                    ck_tile::type_convert<InOutDataType>(locals[sk]);
                                InOutDataType vreg = v_batch_seq_nhead_hdim(
                                    0, seq_kv_offsets[i_batch] + sk, i_head, k);

                                dot_prod += ck_tile::type_convert<GemmAccDataType>(preg) *
                                            ck_tile::type_convert<GemmAccDataType>(vreg);
                            }
                            else
                            {
                                InOutDataType preg =
                                    ck_tile::type_convert<InOutDataType>(locals[sk]);
                                InOutDataType vreg = v_batch_seq_nhead_hdim(i_batch, sk, i_head, k);

                                dot_prod += ck_tile::type_convert<GemmAccDataType>(preg) *
                                            ck_tile::type_convert<GemmAccDataType>(vreg);
                            };
                        };

                        if constexpr(kIsJagged)
                            o_batch_seq_nhead_hdim(0, seq_q_offsets[i_batch] + sq, i_head, k) =
                                ck_tile::type_convert<InOutDataType>(dot_prod);
                        else
                            o_batch_seq_nhead_hdim(i_batch, sq, i_head, k) =
                                ck_tile::type_convert<InOutDataType>(dot_prod);
                    };
                };
            });
        };

        make_ParallelTensorFunctor(f, num_batch, num_head)(std::thread::hardware_concurrency());
    }
};

template <typename InOutDataType, typename GemmAccDataType, typename CompDataType, bool kUseCausal>
struct reference_group_hstu_attention_fwd
{
    static void
    Run(bool is_cross_attention,
        bool use_softmax,
        bool store_lse,
        const HostTensor<InOutDataType>& q_batch_seq_nhead_hdim,
        const HostTensor<InOutDataType>& k_batch_seq_nhead_hdim,
        const HostTensor<InOutDataType>& v_batch_seq_nhead_hdim,
        HostTensor<InOutDataType>& o_batch_seq_nhead_hdim,
        HostTensor<CompDataType>& lse_batch_seq_nhead,
        HostTensor<int8_t>& mask_batch_nhead_seq_seq,
        int num_batch,
        int num_batch_per_group,
        float alpha,
        int max_max_seqlen_q,  // the maximum of all groups's max_seqlen_q, only used as second last
                               // dim of the tensor for saving the mask
        int max_max_seqlen_kv, // the maximum of all groups's max_seqlen_k, only used as last dim of
                               // the tensor for saving the mask
        const std::vector<int>& seq_q_offsets,
        const std::vector<int>& seq_kv_offsets,
        const std::vector<int>& num_targets,         // define masking length at the end of token
                                                     // sequence to be excluded for attention
        const std::vector<int>& group_max_seqlens_q, // max seqlen_q list by groups
        const std::vector<int>& group_contextual_seqlens,    // contextual seqlen list by groups
        const std::vector<int>& group_window_sizes,          // window_size list by groups
        const std::vector<int>& group_min_full_attn_seqlens, // min_full_attn_seqlen list by groups
        const std::vector<float>& group_attn_scales)         // attn scale list by group
    {
        // check the number of batches
        assert(!seq_offsets.empty() && seq_offsets.size() == num_batch + 1);
        assert(q_batch_seq_nhead_hdim.get_lengths()[0] == 1);
        assert(k_batch_seq_nhead_hdim.get_lengths()[0] == 1);
        assert(v_batch_seq_nhead_hdim.get_lengths()[0] == 1);
        assert(o_batch_seq_nhead_hdim.get_lengths()[0] == 1);
        if(store_lse)
        {
            assert(lse_batch_seq_nhead.get_lengths()[0] == 1);
        }

        // check the sequence length
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == k_batch_seq_nhead_hdim.get_lengths()[1]);
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == v_batch_seq_nhead_hdim.get_lengths()[1]);
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == o_batch_seq_nhead_hdim.get_lengths()[1]);
        if(store_lse)
        {
            assert(q_batch_seq_nhead_hdim.get_lengths()[1] == lse_batch_seq_nhead.get_lengths()[1]);
        }

        // check the number of heads
        int num_head = q_batch_seq_nhead_hdim.get_lengths()[2];
        assert(num_head == k_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == v_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == o_batch_seq_nhead_hdim.get_lengths()[2]);
        if(store_lse)
        {
            assert(num_head == lse_batch_seq_nhead.get_lengths()[2]);
        }

        // check the hdim
        int hdim_qk = q_batch_seq_nhead_hdim.get_lengths()[3];
        int hdim_v  = v_batch_seq_nhead_hdim.get_lengths()[3];
        assert(hdim_qk == k_batch_seq_nhead_hdim.get_lengths()[3]);
        assert(hdim_v == o_batch_seq_nhead_hdim.get_lengths()[3]);

        bool save_mask = false;

        if(static_cast<int>(mask_batch_nhead_seq_seq.get_lengths()[0]) == num_batch &&
           static_cast<int>(mask_batch_nhead_seq_seq.get_lengths()[1]) == num_head &&
           static_cast<int>(mask_batch_nhead_seq_seq.get_lengths()[2]) == max_max_seqlen_q &&
           static_cast<int>(mask_batch_nhead_seq_seq.get_lengths()[3]) == max_max_seqlen_kv)
            save_mask = true;

        // check num_tagets
        assert(num_tagets.empty() || num_targets.size() == num_batch);

        auto silu = [&](CompDataType x) {
            const auto one = ck_tile::type_convert<CompDataType>(1.0f);

            return x / (one + std::exp(-x));
        };

        auto f = [&](auto i_batch, auto i_head) {
            int i_group   = i_batch / num_batch_per_group;
            int seqlen_q  = seq_q_offsets[i_batch + 1] - seq_q_offsets[i_batch];
            int seqlen_kv = seq_kv_offsets[i_batch + 1] - seq_kv_offsets[i_batch];

            int num_target = num_targets.empty() ? 0 : num_targets[i_batch];

            int max_seqlen_q = group_max_seqlens_q[i_group];
            float attn_scale = group_attn_scales[i_group];

            float scale_p = (attn_scale ? attn_scale : 1.0f / static_cast<float>(max_seqlen_q));

            int contextual_seqlen    = group_contextual_seqlens[i_group];
            int window_size          = group_window_sizes[i_group];
            int min_full_attn_seqlen = group_min_full_attn_seqlens[i_group];

            BOOL_SWITCH_2(window_size > 0, kHasLocal, is_cross_attention, kIsCrossAttention, [&] {
                using HstuMaskType =
                    typename HstuBlockMasking<kIsCrossAttention, kUseCausal, kHasLocal>::Type;

                HstuMaskType mask = [&]() {
                    if constexpr(kHasLocal)
                    {
                        if constexpr(kIsCrossAttention)
                        {
                            // need adjust the min_full_attn_seqlen passed to the HstuBlockMask() if
                            // the user passed min_full_attn_seqlen is bigger than max_uih_len
                            if(seqlen_q - num_target > min_full_attn_seqlen)
                                return ck_tile::make_hstu_cross_attention_block_mask_with_local<
                                    HstuMaskType>(true,
                                                  seqlen_q,
                                                  seqlen_kv,
                                                  contextual_seqlen,
                                                  num_target,
                                                  window_size,
                                                  min_full_attn_seqlen);
                            else
                                return ck_tile::make_hstu_cross_attention_block_mask_with_local<
                                    HstuMaskType>(true,
                                                  seqlen_q,
                                                  seqlen_kv,
                                                  contextual_seqlen,
                                                  num_target,
                                                  window_size,
                                                  seqlen_q - num_target);
                        }
                        else
                        {
                            // need adjust the min_full_attn_seqlen passed to the HstuBlockMask() if
                            // the user passed min_full_attn_seqlen is bigger than max_uih_len
                            if(seqlen_q - num_target > min_full_attn_seqlen)
                                return ck_tile::make_hstu_self_attention_block_mask_with_local<
                                    HstuMaskType>(true,
                                                  seqlen_q,
                                                  contextual_seqlen,
                                                  num_target,
                                                  window_size,
                                                  min_full_attn_seqlen);
                            else
                                return ck_tile::make_hstu_self_attention_block_mask_with_local<
                                    HstuMaskType>(true,
                                                  seqlen_q,
                                                  contextual_seqlen,
                                                  num_target,
                                                  window_size,
                                                  seqlen_q - num_target);
                        }
                    }
                    else
                    {
                        if constexpr(kIsCrossAttention)
                            return ck_tile::make_hstu_cross_attention_block_mask_without_local<
                                HstuMaskType>(seqlen_q, seqlen_kv, contextual_seqlen, num_target);
                        else
                            return ck_tile::make_hstu_self_attention_block_mask_without_local<
                                HstuMaskType>(seqlen_q, contextual_seqlen, num_target);
                    }
                }();

                if(save_mask)
                {
                    for(int sq = 0; sq < max_max_seqlen_q; sq++)
                        for(int sk = 0; sk < max_max_seqlen_kv; sk++)
                            mask_batch_nhead_seq_seq(i_batch, i_head, sq, sk) = 0;

                    for(int sq = 0; sq < seqlen_q; sq++)
                        for(int sk = 0; sk < seqlen_kv; sk++)
                            mask_batch_nhead_seq_seq(i_batch, i_head, sq, sk) =
                                static_cast<int8_t>(mask.IsTokenPairInsideMask(sq, sk));
                }

                // for all rows in the batch
                for(int sq = 0; sq < seqlen_q; sq++)
                {
                    CompDataType m =
                        -ck_tile::numeric<CompDataType>::infinity(); // max value of the row
                    CompDataType l =
                        ck_tile::type_convert<CompDataType>(0.0f); // sum of exp(x-m) of the row
                                                                   //
                    std::vector<CompDataType> locals;

                    // for all cols in the batch
                    for(int sk = 0; sk < seqlen_kv; sk++)
                    {
                        if(mask.IsTokenPairInsideMask(sq, sk))
                        {
                            GemmAccDataType dot_prod = 0.f;
                            for(int k = 0; k < hdim_qk; k++)
                            {
                                InOutDataType qreg = q_batch_seq_nhead_hdim(
                                    0, seq_q_offsets[i_batch] + sq, i_head, k);
                                InOutDataType kreg = k_batch_seq_nhead_hdim(
                                    0, seq_kv_offsets[i_batch] + sk, i_head, k);

                                dot_prod += ck_tile::type_convert<GemmAccDataType>(qreg) *
                                            ck_tile::type_convert<GemmAccDataType>(kreg);
                            }

                            locals.push_back(ck_tile::type_convert<CompDataType>(dot_prod) *
                                             ck_tile::type_convert<CompDataType>(alpha));
                        }
                        else
                        {
                            if(!use_softmax)
                                locals.push_back(ck_tile::type_convert<CompDataType>(0.0f));
                            else
                                locals.push_back(-ck_tile::numeric<CompDataType>::infinity());
                        };
                    };

                    if(!use_softmax)
                    {
                        // SiLu element-wise
                        for(CompDataType& elem : locals)
                            elem = silu(elem) * ck_tile::type_convert<CompDataType>(scale_p);
                    }
                    else
                    {
                        for(CompDataType elem : locals)
                            m = ck_tile::max(m, elem);

                        if(m == -ck_tile::numeric<CompDataType>::infinity())
                        {
                            for(CompDataType& elem : locals)
                                elem = ck_tile::type_convert<CompDataType>(0.0f);
                        }
                        else
                        {
                            // stabalized sum of exp()
                            for(CompDataType elem : locals)
                                l += std::exp(elem - m);

                            // normalization
                            for(CompDataType& elem : locals)
                                elem = std::exp(elem - m) / l;
                        }

                        if(store_lse)
                        {
                            lse_batch_seq_nhead(0, seq_q_offsets[i_batch] + sq, i_head) =
                                std::log(l) + m;
                        }
                    };

                    // second Gemm
                    for(int k = 0; k < hdim_v; k++)
                    {
                        GemmAccDataType dot_prod = 0.f;

                        for(int sk = 0; sk < seqlen_kv; sk++)
                        {
                            InOutDataType preg = ck_tile::type_convert<InOutDataType>(locals[sk]);
                            InOutDataType vreg =
                                v_batch_seq_nhead_hdim(0, seq_kv_offsets[i_batch] + sk, i_head, k);

                            dot_prod += ck_tile::type_convert<GemmAccDataType>(preg) *
                                        ck_tile::type_convert<GemmAccDataType>(vreg);
                        };

                        o_batch_seq_nhead_hdim(0, seq_q_offsets[i_batch] + sq, i_head, k) =
                            ck_tile::type_convert<InOutDataType>(dot_prod);
                    };
                };
            });
        };

        make_ParallelTensorFunctor(f, num_batch, num_head)(std::thread::hardware_concurrency());
    }
};

} // namespace ck_tile
