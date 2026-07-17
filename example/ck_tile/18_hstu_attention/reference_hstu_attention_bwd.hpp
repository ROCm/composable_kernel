// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <thread>
#include <cassert>
#include <cmath>
#include <vector>

#include <ck_tile/core.hpp>
#include <ck_tile/host/host_tensor.hpp>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_block_masking.hpp"

namespace ck_tile {

// clang-format off
// Reference implementation of HSTUAttention backward problem.
// Given dO, Q, K, V, LSE (and the same mask parameters as the forward), compute dQ, dK, dV.
//
// Forward recap (see reference_hstu_attention_fwd.hpp):
//   S = alpha * Q @ K    (A=Q[sq,hdim_qk], B=K[sk,hdim_qk]),  masked-in pairs, else 0 or -inf
//   P[sq,sk]  = silu(S[sq,sk]) * scale_p          (kUseSoftmax=false)
//             = softmax_row(S)[sq,sk]              (kUseSoftmax=true)
//   O = P @ V^T              (A=P[sq,sk], B=V^T[hdim_v,sk])
//   LSE[sq]   = log(sum_sk exp(S[sq,sk]))          (kUseSoftmax=true, saved during fwd)
//
// Backward derivation:
//   dV = P^T @ dO^T     (A=P^T[sk,sq], B=dO^T[hdim_v,sq])
//
//   dP = dO @ V          (A=dO[sq,hdim_v], B=V[sk,hdim_v])
//
//   kUseSoftmax=false (SiLU path):
//     dsilu(x)  = sigmoid(x) * (1 + x*(1 - sigmoid(x)))
//     dS[sq,sk] = dP[sq,sk] * scale_p * dsilu(S[sq,sk])   (masked-in), else 0
//     (LSE is not used for the SiLU path since S must be recomputed for dsilu)
//
//   kUseSoftmax=true (Softmax path):
//     With LSE[sq] = log(sum_sk exp(S[sq,sk])) available from the forward pass,
//     P can be recovered without a two-pass softmax reduction:
//       P[sq,sk] = exp(S[sq,sk] - LSE[sq])
//     (masked-out positions have S=-inf, so exp(-inf - LSE) = 0 naturally)
//
//     D[sq] = dO[sq] row(.) O[sq]
//           (equivalent to sum_sk dP[sq,sk]*P[sq,sk], proved by swapping summation:
//            sum_sk dP[sq,sk]*P[sq,sk] = sum_sk (dO[sq] row(.) V[sk])*P[sq,sk]
//                                      = sum_k dO[sq,k] * sum_sk P[sq,sk]*V[sk,k]
//                                      = sum_k dO[sq,k] * O[sq,k]  = dO[sq] row(.) O[sq])
//     dS[sq,sk] = P[sq,sk] * (dP[sq,sk] - D[sq])
//                 (masked-out positions have P=0, so they contribute 0 naturally)
//
//   dQ = alpha * dS @ K^T    (A=dS[sq,sk], B=K^T[hdim_qk,sk])
//
//   dK = alpha * dS^T @ Q^T   (A=dS^T[sk,sq], B=Q^T[hdim_qk,sq])
// clang-format on

template <typename InOutDataType,
          typename GemmAccDataType,
          typename CompDataType,
          bool kIsJagged,
          bool kUseCausal>
struct reference_no_group_hstu_attention_bwd
{
    static void Run(bool is_cross_attention,
                    bool use_softmax,
                    bool has_dropout,
                    const HostTensor<InOutDataType>& q_batch_seq_nhead_hdim,
                    const HostTensor<InOutDataType>& k_batch_seq_nhead_hdim,
                    const HostTensor<InOutDataType>& v_batch_seq_nhead_hdim,
                    const HostTensor<CompDataType>& lse_batch_seq_nhead,
                    const HostTensor<InOutDataType>& o_batch_seq_nhead_hdim,
                    const HostTensor<InOutDataType>& do_batch_seq_nhead_hdim,
                    HostTensor<InOutDataType>& dq_batch_seq_nhead_hdim,
                    HostTensor<InOutDataType>& dk_batch_seq_nhead_hdim,
                    HostTensor<InOutDataType>& dv_batch_seq_nhead_hdim,
                    int num_batch,
                    float alpha,
                    float attn_scale,
                    int max_seqlen_q,
                    std::vector<int> seq_q_offsets,
                    std::vector<int> seq_kv_offsets,
                    std::vector<int> num_targets,
                    int contextual_seqlen,
                    int window_size,
                    int min_full_attn_seqlen,
                    float p_drop,
                    HostTensor<uint8_t>& rand_val_batch_seq_nhead_seq)
    {
        if constexpr(kIsJagged)
        {
            assert(!seq_q_offsets.empty() && seq_q_offsets.size() == num_batch + 1);
            assert(!seq_kv_offsets.empty() && seq_kv_offsets.size() == num_batch + 1);
            assert(q_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            assert(k_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            assert(v_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            assert(o_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            assert(do_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            assert(dq_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            assert(dk_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            assert(dv_batch_seq_nhead_hdim.get_lengths()[0] == 1);
            if(use_softmax)
                assert(lse_batch_seq_nhead.get_lengths()[0] == 1);
        }
        else
        {
            assert(seq_q_offsets.empty());
            assert(seq_kv_offsets.empty());
            assert(q_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            assert(k_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            assert(v_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            assert(o_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            assert(do_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            assert(dq_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            assert(dk_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            assert(dv_batch_seq_nhead_hdim.get_lengths()[0] == num_batch);
            if(use_softmax)
                assert(lse_batch_seq_nhead.get_lengths()[0] == num_batch);
        }

        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == k_batch_seq_nhead_hdim.get_lengths()[1]);
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == v_batch_seq_nhead_hdim.get_lengths()[1]);
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == o_batch_seq_nhead_hdim.get_lengths()[1]);
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == do_batch_seq_nhead_hdim.get_lengths()[1]);

        int num_head = q_batch_seq_nhead_hdim.get_lengths()[2];
        assert(num_head == k_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == v_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == o_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == do_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == dq_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == dk_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == dv_batch_seq_nhead_hdim.get_lengths()[2]);
        if(use_softmax)
            assert(num_head == lse_batch_seq_nhead.get_lengths()[2]);

        int hdim_qk = q_batch_seq_nhead_hdim.get_lengths()[3];
        int hdim_v  = v_batch_seq_nhead_hdim.get_lengths()[3];
        assert(hdim_qk == k_batch_seq_nhead_hdim.get_lengths()[3]);
        assert(hdim_v == o_batch_seq_nhead_hdim.get_lengths()[3]);
        assert(hdim_v == do_batch_seq_nhead_hdim.get_lengths()[3]);
        assert(hdim_qk == dq_batch_seq_nhead_hdim.get_lengths()[3]);
        assert(hdim_qk == dk_batch_seq_nhead_hdim.get_lengths()[3]);
        assert(hdim_v == dv_batch_seq_nhead_hdim.get_lengths()[3]);

        assert(num_targets.empty() || num_targets.size() == num_batch);

        auto silu = [&](CompDataType x) {
            const auto one = ck_tile::type_convert<CompDataType>(1.0f);
            return x / (one + std::exp(-x));
        };

        // Derivative of silu(x) = x*sigmoid(x):
        //   dsilu/dx = sigmoid(x) * (1 + x*(1 - sigmoid(x)))
        auto dsilu = [&](CompDataType x) {
            const auto one   = ck_tile::type_convert<CompDataType>(1.0f);
            CompDataType sig = one / (one + std::exp(-x));
            return sig * (one + x * (one - sig));
        };

        float rp_undrop             = 1;
        uint8_t p_undrop_in_uint8_t = std::numeric_limits<uint8_t>::max();

        if(has_dropout)
        {
            float p_undrop = 1.0f - p_drop;
            p_undrop_in_uint8_t =
                uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
            rp_undrop = 1.0 / p_undrop;
        }

        auto f = [&](auto i_batch, auto i_head) {
            int seqlen_q  = kIsJagged ? (seq_q_offsets[i_batch + 1] - seq_q_offsets[i_batch])
                                      : q_batch_seq_nhead_hdim.get_lengths()[1];
            int seqlen_kv = kIsJagged ? (seq_kv_offsets[i_batch + 1] - seq_kv_offsets[i_batch])
                                      : k_batch_seq_nhead_hdim.get_lengths()[1];

            int num_target = num_targets.empty() ? 0 : num_targets[i_batch];
            float scale_p  = attn_scale ? attn_scale : 1.0f / static_cast<float>(max_seqlen_q);

            BOOL_SWITCH_2(window_size > 0, kHasLocal, is_cross_attention, kIsCrossAttention, [&] {
                using HstuMaskType =
                    typename HstuBlockMasking<kIsCrossAttention, kUseCausal, kHasLocal>::Type;

                // Build the same mask as in the forward pass
                HstuMaskType mask = [&]() {
                    if constexpr(kHasLocal)
                    {
                        if constexpr(kIsCrossAttention)
                        {
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

                // Local accumulators for dK and dV: these accumulate over all sq rows,
                // so we keep them in higher-precision GemmAccDataType and write back once.
                std::vector<std::vector<GemmAccDataType>> dk_acc(
                    seqlen_kv, std::vector<GemmAccDataType>(hdim_qk, 0.f));
                std::vector<std::vector<GemmAccDataType>> dv_acc(
                    seqlen_kv, std::vector<GemmAccDataType>(hdim_v, 0.f));

                for(int sq = 0; sq < seqlen_q; sq++)
                {
                    // ------------------------------------------------------------------
                    // Step 1: Recompute S[sq,:] and P[sq,:] (forward pass recomputation)
                    //   S = alpha * Q @ K      (A=Q[sq,hdim_qk], B=K[sk,hdim_qk])
                    //   P[sq,sk] = silu(S)*scale_p or softmax_row(S)
                    // ------------------------------------------------------------------
                    std::vector<CompDataType> locals_S(seqlen_kv);
                    std::vector<CompDataType> locals_P(seqlen_kv);

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
                                }
                            }
                            locals_S[sk] = ck_tile::type_convert<CompDataType>(dot_prod) *
                                           ck_tile::type_convert<CompDataType>(alpha);
                        }
                        else
                        {
                            // Masked-out: SiLU path uses S=0 (silu(0)=0); Softmax path uses
                            // S=-inf (exp(-inf - LSE)=0). silu(-inf) would be NaN, so the SiLU
                            // path must NOT use -inf here.
                            if(!use_softmax)
                                locals_S[sk] = ck_tile::type_convert<CompDataType>(0.0f);
                            else
                                locals_S[sk] = -ck_tile::numeric<CompDataType>::infinity();
                        }
                    }

                    if(!use_softmax)
                    {
                        for(int sk = 0; sk < seqlen_kv; sk++)
                            locals_P[sk] =
                                silu(locals_S[sk]) * ck_tile::type_convert<CompDataType>(scale_p);
                    }
                    else
                    {
                        // Use precomputed LSE from the forward pass to recover P without
                        // a two-pass softmax reduction:
                        //   LSE[sq] = log(sum_sk exp(S[sq,sk]))
                        //   P[sq,sk] = exp(S[sq,sk] - LSE[sq])
                        // Masked-out positions have S=-inf, so exp(-inf - LSE) = 0.
                        CompDataType lse_sq;
                        if constexpr(kIsJagged)
                            lse_sq = lse_batch_seq_nhead(0, seq_q_offsets[i_batch] + sq, i_head);
                        else
                            lse_sq = lse_batch_seq_nhead(i_batch, sq, i_head);

                        if(lse_sq == -ck_tile::numeric<CompDataType>::infinity())
                        {
                            for(CompDataType& elem : locals_P)
                                elem = ck_tile::type_convert<CompDataType>(0.0f);
                        }
                        else
                        {
                            for(int sk = 0; sk < seqlen_kv; sk++)
                                locals_P[sk] = std::exp(locals_S[sk] - lse_sq);
                        }
                    }

                    // Dropout scale per key position: rp_undrop for kept, 0 for dropped
                    // (1 when dropout is off). Kept as a SEPARATE factor rather than folded into
                    // locals_P, because the two consumers need different quantities:
                    //   - dV / dP use the *dropped* probabilities   P_drop = drop_scale * P
                    //   - the softmax dS jacobian needs the *pure* softmax P together with the
                    //     dropped dP:  dS = P * (drop_scale*dP - D)
                    std::vector<CompDataType> locals_drop_scale(
                        seqlen_kv, ck_tile::type_convert<CompDataType>(1.0f));
                    if(has_dropout)
                    {
                        for(int sk = 0; sk < seqlen_kv; sk++)
                        {
                            uint8_t rand_val;

                            if constexpr(kIsJagged)
                                rand_val = rand_val_batch_seq_nhead_seq(
                                    0, seq_q_offsets[i_batch] + sq, i_head, sk);
                            else
                                rand_val = rand_val_batch_seq_nhead_seq(i_batch, sq, i_head, sk);

                            locals_drop_scale[sk] =
                                (rand_val <= p_undrop_in_uint8_t)
                                    ? ck_tile::type_convert<CompDataType>(rp_undrop)
                                    : ck_tile::type_convert<CompDataType>(0.0f);
                        }
                    };

                    // ------------------------------------------------------------------
                    // Step 2: dV = P^T @ dO^T   (A=P^T[sk,sq], B=dO^T[hdim_v,sq])
                    // ------------------------------------------------------------------
                    for(int sk = 0; sk < seqlen_kv; sk++)
                    {
                        // dV uses the dropped probabilities P_drop = drop_scale * P
                        InOutDataType p_reg = ck_tile::type_convert<InOutDataType>(
                            locals_drop_scale[sk] * locals_P[sk]);
                        for(int k = 0; k < hdim_v; k++)
                        {
                            InOutDataType do_reg;
                            if constexpr(kIsJagged)
                                do_reg = do_batch_seq_nhead_hdim(
                                    0, seq_q_offsets[i_batch] + sq, i_head, k);
                            else
                                do_reg = do_batch_seq_nhead_hdim(i_batch, sq, i_head, k);

                            dv_acc[sk][k] += ck_tile::type_convert<GemmAccDataType>(p_reg) *
                                             ck_tile::type_convert<GemmAccDataType>(do_reg);
                        }
                    }

                    // ------------------------------------------------------------------
                    // Step 3: dP = dO @ V      (A=dO[sq,hdim_v], B=V[sk,hdim_v])
                    // ------------------------------------------------------------------
                    std::vector<CompDataType> locals_dP(seqlen_kv);
                    for(int sk = 0; sk < seqlen_kv; sk++)
                    {
                        GemmAccDataType acc = 0.f;
                        for(int k = 0; k < hdim_v; k++)
                        {
                            InOutDataType do_reg;
                            InOutDataType vreg;
                            if constexpr(kIsJagged)
                            {
                                do_reg = do_batch_seq_nhead_hdim(
                                    0, seq_q_offsets[i_batch] + sq, i_head, k);
                                vreg = v_batch_seq_nhead_hdim(
                                    0, seq_kv_offsets[i_batch] + sk, i_head, k);
                            }
                            else
                            {
                                do_reg = do_batch_seq_nhead_hdim(i_batch, sq, i_head, k);
                                vreg   = v_batch_seq_nhead_hdim(i_batch, sk, i_head, k);
                            }
                            acc += ck_tile::type_convert<GemmAccDataType>(do_reg) *
                                   ck_tile::type_convert<GemmAccDataType>(vreg);
                        }
                        locals_dP[sk] = ck_tile::type_convert<CompDataType>(acc);
                    }

                    // ------------------------------------------------------------------
                    // Step 4: Compute dS[sq,:] from dP[sq,:] via activation chain rule
                    //
                    //   kUseSoftmax=false (SiLU):
                    //     dS[sq,sk] = dP[sq,sk] * scale_p * dsilu(S[sq,sk])  (masked-in)
                    //               = 0                                        (masked-out)
                    //
                    //   kUseSoftmax=true (Softmax):
                    //     D[sq] = dO[sq] row(.) O[sq]   (uses forward output O directly)
                    //     dS[sq,sk] = P[sq,sk] * (dP[sq,sk] - D[sq])
                    // ------------------------------------------------------------------
                    std::vector<CompDataType> locals_dS(seqlen_kv);
                    if(!use_softmax)
                    {
                        for(int sk = 0; sk < seqlen_kv; sk++)
                        {
                            if(mask.IsTokenPairInsideMask(sq, sk))
                                // dS = (drop_scale * dP) * scale_p * dsilu(S); the dropout mask
                                // propagates through the chain rule into dP (not into S/dsilu).
                                locals_dS[sk] = locals_drop_scale[sk] * locals_dP[sk] *
                                                ck_tile::type_convert<CompDataType>(scale_p) *
                                                dsilu(locals_S[sk]);
                            else
                                locals_dS[sk] = ck_tile::type_convert<CompDataType>(0.0f);
                        }
                    }
                    else
                    {
                        // D[sq] = dO[sq] row(.) O[sq]
                        GemmAccDataType D_acc = 0.f;
                        for(int k = 0; k < hdim_v; k++)
                        {
                            InOutDataType do_reg;
                            InOutDataType o_reg;
                            if constexpr(kIsJagged)
                            {
                                do_reg = do_batch_seq_nhead_hdim(
                                    0, seq_q_offsets[i_batch] + sq, i_head, k);
                                o_reg = o_batch_seq_nhead_hdim(
                                    0, seq_q_offsets[i_batch] + sq, i_head, k);
                            }
                            else
                            {
                                do_reg = do_batch_seq_nhead_hdim(i_batch, sq, i_head, k);
                                o_reg  = o_batch_seq_nhead_hdim(i_batch, sq, i_head, k);
                            }
                            D_acc += ck_tile::type_convert<GemmAccDataType>(do_reg) *
                                     ck_tile::type_convert<GemmAccDataType>(o_reg);
                        }
                        CompDataType D = ck_tile::type_convert<CompDataType>(D_acc);
                        // dS = P * (drop_scale*dP - D). P is the PURE softmax output; the dropout
                        // mask multiplies dP only. D = dO.O already carries dropout (O is dropped).
                        for(int sk = 0; sk < seqlen_kv; sk++)
                            locals_dS[sk] =
                                locals_P[sk] * (locals_drop_scale[sk] * locals_dP[sk] - D);
                    }

                    // ------------------------------------------------------------------
                    // Step 5: dQ = alpha * dS @ K^T   (A=dS[sq,sk], B=K^T[hdim_qk,sk])
                    //   (computed fresh per sq row, no accumulation needed)
                    // ------------------------------------------------------------------
                    for(int k = 0; k < hdim_qk; k++)
                    {
                        GemmAccDataType acc = 0.f;
                        for(int sk = 0; sk < seqlen_kv; sk++)
                        {
                            InOutDataType ds_reg =
                                ck_tile::type_convert<InOutDataType>(locals_dS[sk]);
                            InOutDataType kreg;
                            if constexpr(kIsJagged)
                                kreg = k_batch_seq_nhead_hdim(
                                    0, seq_kv_offsets[i_batch] + sk, i_head, k);
                            else
                                kreg = k_batch_seq_nhead_hdim(i_batch, sk, i_head, k);

                            acc += ck_tile::type_convert<GemmAccDataType>(ds_reg) *
                                   ck_tile::type_convert<GemmAccDataType>(kreg);
                        }
                        if constexpr(kIsJagged)
                            dq_batch_seq_nhead_hdim(0, seq_q_offsets[i_batch] + sq, i_head, k) =
                                ck_tile::type_convert<InOutDataType>(acc * alpha);
                        else
                            dq_batch_seq_nhead_hdim(i_batch, sq, i_head, k) =
                                ck_tile::type_convert<InOutDataType>(acc * alpha);
                    }

                    // ------------------------------------------------------------------
                    // Step 6: dK = alpha * dS^T @ Q^T   (A=dS^T[sk,sq], B=Q^T[hdim_qk,sq])
                    // ------------------------------------------------------------------
                    for(int sk = 0; sk < seqlen_kv; sk++)
                    {
                        InOutDataType ds_reg = ck_tile::type_convert<InOutDataType>(locals_dS[sk]);
                        for(int k = 0; k < hdim_qk; k++)
                        {
                            InOutDataType qreg;
                            if constexpr(kIsJagged)
                                qreg = q_batch_seq_nhead_hdim(
                                    0, seq_q_offsets[i_batch] + sq, i_head, k);
                            else
                                qreg = q_batch_seq_nhead_hdim(i_batch, sq, i_head, k);

                            dk_acc[sk][k] += ck_tile::type_convert<GemmAccDataType>(ds_reg) *
                                             ck_tile::type_convert<GemmAccDataType>(qreg);
                        }
                    }
                }

                // Write back dK (multiplied by alpha) and dV
                for(int sk = 0; sk < seqlen_kv; sk++)
                {
                    for(int k = 0; k < hdim_qk; k++)
                    {
                        if constexpr(kIsJagged)
                            dk_batch_seq_nhead_hdim(0, seq_kv_offsets[i_batch] + sk, i_head, k) =
                                ck_tile::type_convert<InOutDataType>(dk_acc[sk][k] * alpha);
                        else
                            dk_batch_seq_nhead_hdim(i_batch, sk, i_head, k) =
                                ck_tile::type_convert<InOutDataType>(dk_acc[sk][k] * alpha);
                    }
                    for(int k = 0; k < hdim_v; k++)
                    {
                        if constexpr(kIsJagged)
                            dv_batch_seq_nhead_hdim(0, seq_kv_offsets[i_batch] + sk, i_head, k) =
                                ck_tile::type_convert<InOutDataType>(dv_acc[sk][k]);
                        else
                            dv_batch_seq_nhead_hdim(i_batch, sk, i_head, k) =
                                ck_tile::type_convert<InOutDataType>(dv_acc[sk][k]);
                    }
                }
            });
        };

        make_ParallelTensorFunctor(f, num_batch, num_head)(std::thread::hardware_concurrency());
    }
};

template <typename InOutDataType, typename GemmAccDataType, typename CompDataType, bool kUseCausal>
struct reference_group_hstu_attention_bwd
{
    static void Run(bool is_cross_attention,
                    bool use_softmax,
                    bool has_dropout,
                    const HostTensor<InOutDataType>& q_batch_seq_nhead_hdim,
                    const HostTensor<InOutDataType>& k_batch_seq_nhead_hdim,
                    const HostTensor<InOutDataType>& v_batch_seq_nhead_hdim,
                    const HostTensor<CompDataType>& lse_batch_seq_nhead,
                    const HostTensor<InOutDataType>& o_batch_seq_nhead_hdim,
                    const HostTensor<InOutDataType>& do_batch_seq_nhead_hdim,
                    HostTensor<InOutDataType>& dq_batch_seq_nhead_hdim,
                    HostTensor<InOutDataType>& dk_batch_seq_nhead_hdim,
                    HostTensor<InOutDataType>& dv_batch_seq_nhead_hdim,
                    int num_batch,
                    int num_batch_per_group,
                    float alpha,
                    const std::vector<int>& seq_q_offsets,
                    const std::vector<int>& seq_kv_offsets,
                    const std::vector<int>& num_targets,
                    const std::vector<int>& group_max_seqlens_q,
                    const std::vector<int>& group_contextual_seqlens,
                    const std::vector<int>& group_window_sizes,
                    const std::vector<int>& group_min_full_attn_seqlens,
                    const std::vector<float>& group_attn_scales,
                    float p_drop,
                    HostTensor<uint8_t>& rand_val_batch_seq_nhead_seq)
    {
        // All sequences are jagged-packed (batch dim = 1), same as group forward
        assert(!seq_q_offsets.empty() && seq_q_offsets.size() == num_batch + 1);
        assert(!seq_kv_offsets.empty() && seq_kv_offsets.size() == num_batch + 1);
        assert(q_batch_seq_nhead_hdim.get_lengths()[0] == 1);
        assert(k_batch_seq_nhead_hdim.get_lengths()[0] == 1);
        assert(v_batch_seq_nhead_hdim.get_lengths()[0] == 1);
        assert(o_batch_seq_nhead_hdim.get_lengths()[0] == 1);
        assert(do_batch_seq_nhead_hdim.get_lengths()[0] == 1);
        assert(dq_batch_seq_nhead_hdim.get_lengths()[0] == 1);
        assert(dk_batch_seq_nhead_hdim.get_lengths()[0] == 1);
        assert(dv_batch_seq_nhead_hdim.get_lengths()[0] == 1);
        if(use_softmax)
            assert(lse_batch_seq_nhead.get_lengths()[0] == 1);

        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == k_batch_seq_nhead_hdim.get_lengths()[1]);
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == v_batch_seq_nhead_hdim.get_lengths()[1]);
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == o_batch_seq_nhead_hdim.get_lengths()[1]);
        assert(q_batch_seq_nhead_hdim.get_lengths()[1] == do_batch_seq_nhead_hdim.get_lengths()[1]);

        int num_head = q_batch_seq_nhead_hdim.get_lengths()[2];
        assert(num_head == k_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == v_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == o_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == do_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == dq_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == dk_batch_seq_nhead_hdim.get_lengths()[2]);
        assert(num_head == dv_batch_seq_nhead_hdim.get_lengths()[2]);
        if(use_softmax)
            assert(num_head == lse_batch_seq_nhead.get_lengths()[2]);

        int hdim_qk = q_batch_seq_nhead_hdim.get_lengths()[3];
        int hdim_v  = v_batch_seq_nhead_hdim.get_lengths()[3];
        assert(hdim_qk == k_batch_seq_nhead_hdim.get_lengths()[3]);
        assert(hdim_v == o_batch_seq_nhead_hdim.get_lengths()[3]);
        assert(hdim_v == do_batch_seq_nhead_hdim.get_lengths()[3]);
        assert(hdim_qk == dq_batch_seq_nhead_hdim.get_lengths()[3]);
        assert(hdim_qk == dk_batch_seq_nhead_hdim.get_lengths()[3]);
        assert(hdim_v == dv_batch_seq_nhead_hdim.get_lengths()[3]);

        assert(num_targets.empty() || num_targets.size() == num_batch);

        auto silu = [&](CompDataType x) {
            const auto one = ck_tile::type_convert<CompDataType>(1.0f);
            return x / (one + std::exp(-x));
        };

        auto dsilu = [&](CompDataType x) {
            const auto one   = ck_tile::type_convert<CompDataType>(1.0f);
            CompDataType sig = one / (one + std::exp(-x));
            return sig * (one + x * (one - sig));
        };

        float rp_undrop             = 1;
        uint8_t p_undrop_in_uint8_t = std::numeric_limits<uint8_t>::max();

        if(has_dropout)
        {
            float p_undrop = 1.0f - p_drop;
            p_undrop_in_uint8_t =
                uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
            rp_undrop = 1.0 / p_undrop;
        }

        auto f = [&](auto i_batch, auto i_head) {
            // Resolve group index and look up group-level hyperparameters
            int i_group   = i_batch / num_batch_per_group;
            int seqlen_q  = seq_q_offsets[i_batch + 1] - seq_q_offsets[i_batch];
            int seqlen_kv = seq_kv_offsets[i_batch + 1] - seq_kv_offsets[i_batch];

            int num_target = num_targets.empty() ? 0 : num_targets[i_batch];

            int max_seqlen_q = group_max_seqlens_q[i_group];
            float attn_scale = group_attn_scales[i_group];
            float scale_p    = (attn_scale ? attn_scale : 1.0f / static_cast<float>(max_seqlen_q));

            int contextual_seqlen    = group_contextual_seqlens[i_group];
            int window_size          = group_window_sizes[i_group];
            int min_full_attn_seqlen = group_min_full_attn_seqlens[i_group];

            BOOL_SWITCH_2(window_size > 0, kHasLocal, is_cross_attention, kIsCrossAttention, [&] {
                using HstuMaskType =
                    typename HstuBlockMasking<kIsCrossAttention, kUseCausal, kHasLocal>::Type;

                // Build the same mask as in the group forward pass
                HstuMaskType mask = [&]() {
                    if constexpr(kHasLocal)
                    {
                        if constexpr(kIsCrossAttention)
                        {
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

                // Local accumulators for dK and dV: accumulate over all sq rows
                std::vector<std::vector<GemmAccDataType>> dk_acc(
                    seqlen_kv, std::vector<GemmAccDataType>(hdim_qk, 0.f));
                std::vector<std::vector<GemmAccDataType>> dv_acc(
                    seqlen_kv, std::vector<GemmAccDataType>(hdim_v, 0.f));

                for(int sq = 0; sq < seqlen_q; sq++)
                {
                    // ------------------------------------------------------------------
                    // Step 1: Recompute S[sq,:] and P[sq,:] (forward pass recomputation)
                    //   S = alpha * Q @ K      (A=Q[sq,hdim_qk], B=K[sk,hdim_qk])
                    //   P[sq,sk] = silu(S)*scale_p or softmax_row(S)
                    // ------------------------------------------------------------------
                    std::vector<CompDataType> locals_S(seqlen_kv);
                    std::vector<CompDataType> locals_P(seqlen_kv);

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
                            locals_S[sk] = ck_tile::type_convert<CompDataType>(dot_prod) *
                                           ck_tile::type_convert<CompDataType>(alpha);
                        }
                        else
                        {
                            if(!use_softmax)
                                locals_S[sk] = ck_tile::type_convert<CompDataType>(0.0f);
                            else
                                locals_S[sk] = -ck_tile::numeric<CompDataType>::infinity();
                        }
                    }

                    if(!use_softmax)
                    {
                        for(int sk = 0; sk < seqlen_kv; sk++)
                            locals_P[sk] =
                                silu(locals_S[sk]) * ck_tile::type_convert<CompDataType>(scale_p);
                    }
                    else
                    {
                        // Use precomputed LSE from the forward pass to recover P without
                        // a two-pass softmax reduction:
                        //   LSE[sq] = log(sum_sk exp(S[sq,sk]))
                        //   P[sq,sk] = exp(S[sq,sk] - LSE[sq])
                        // Masked-out positions have S=-inf, so exp(-inf - LSE) = 0.
                        CompDataType lse_sq =
                            lse_batch_seq_nhead(0, seq_q_offsets[i_batch] + sq, i_head);

                        if(lse_sq == -ck_tile::numeric<CompDataType>::infinity())
                        {
                            for(CompDataType& elem : locals_P)
                                elem = ck_tile::type_convert<CompDataType>(0.0f);
                        }
                        else
                        {
                            for(int sk = 0; sk < seqlen_kv; sk++)
                                locals_P[sk] = std::exp(locals_S[sk] - lse_sq);
                        }
                    }

                    // Dropout scale per key position: rp_undrop for kept, 0 for dropped
                    // (1 when dropout is off). Kept as a SEPARATE factor rather than folded into
                    // locals_P, because the two consumers need different quantities:
                    //   - dV / dP use the *dropped* probabilities   P_drop = drop_scale * P
                    //   - the softmax dS jacobian needs the *pure* softmax P together with the
                    //     dropped dP:  dS = P * (drop_scale*dP - D)
                    std::vector<CompDataType> locals_drop_scale(
                        seqlen_kv, ck_tile::type_convert<CompDataType>(1.0f));
                    if(has_dropout)
                    {
                        for(int sk = 0; sk < seqlen_kv; sk++)
                        {
                            uint8_t rand_val;

                            rand_val = rand_val_batch_seq_nhead_seq(
                                0, seq_q_offsets[i_batch] + sq, i_head, sk);

                            locals_drop_scale[sk] =
                                (rand_val <= p_undrop_in_uint8_t)
                                    ? ck_tile::type_convert<CompDataType>(rp_undrop)
                                    : ck_tile::type_convert<CompDataType>(0.0f);
                        }
                    };

                    // ------------------------------------------------------------------
                    // Step 2: dV = P^T @ dO^T   (A=P^T[sk,sq], B=dO^T[hdim_v,sq])
                    // ------------------------------------------------------------------
                    for(int sk = 0; sk < seqlen_kv; sk++)
                    {
                        // dV uses the dropped probabilities P_drop = drop_scale * P
                        InOutDataType p_reg = ck_tile::type_convert<InOutDataType>(
                            locals_drop_scale[sk] * locals_P[sk]);
                        for(int k = 0; k < hdim_v; k++)
                        {
                            InOutDataType do_reg =
                                do_batch_seq_nhead_hdim(0, seq_q_offsets[i_batch] + sq, i_head, k);
                            dv_acc[sk][k] += ck_tile::type_convert<GemmAccDataType>(p_reg) *
                                             ck_tile::type_convert<GemmAccDataType>(do_reg);
                        }
                    }

                    // ------------------------------------------------------------------
                    // Step 3: dP = dO @ V      (A=dO[sq,hdim_v], B=V[sk,hdim_v])
                    // ------------------------------------------------------------------
                    std::vector<CompDataType> locals_dP(seqlen_kv);
                    for(int sk = 0; sk < seqlen_kv; sk++)
                    {
                        GemmAccDataType acc = 0.f;
                        for(int k = 0; k < hdim_v; k++)
                        {
                            InOutDataType do_reg =
                                do_batch_seq_nhead_hdim(0, seq_q_offsets[i_batch] + sq, i_head, k);
                            InOutDataType vreg =
                                v_batch_seq_nhead_hdim(0, seq_kv_offsets[i_batch] + sk, i_head, k);
                            acc += ck_tile::type_convert<GemmAccDataType>(do_reg) *
                                   ck_tile::type_convert<GemmAccDataType>(vreg);
                        }
                        locals_dP[sk] = ck_tile::type_convert<CompDataType>(acc);
                    }

                    // ------------------------------------------------------------------
                    // Step 4: Compute dS[sq,:] from dP[sq,:] via activation chain rule
                    //
                    //   kUseSoftmax=false (SiLU):
                    //     dS[sq,sk] = dP[sq,sk] * scale_p * dsilu(S[sq,sk])  (masked-in)
                    //               = 0                                        (masked-out)
                    //
                    //   kUseSoftmax=true (Softmax):
                    //     D[sq] = dO[sq] row(.) O[sq]   (uses forward output O directly)
                    //     dS[sq,sk] = P[sq,sk] * (dP[sq,sk] - D[sq])
                    // ------------------------------------------------------------------
                    std::vector<CompDataType> locals_dS(seqlen_kv);
                    if(!use_softmax)
                    {
                        for(int sk = 0; sk < seqlen_kv; sk++)
                        {
                            if(mask.IsTokenPairInsideMask(sq, sk))
                                // dS = (drop_scale * dP) * scale_p * dsilu(S); the dropout mask
                                // propagates through the chain rule into dP (not into S/dsilu).
                                locals_dS[sk] = locals_drop_scale[sk] * locals_dP[sk] *
                                                ck_tile::type_convert<CompDataType>(scale_p) *
                                                dsilu(locals_S[sk]);
                            else
                                locals_dS[sk] = ck_tile::type_convert<CompDataType>(0.0f);
                        }
                    }
                    else
                    {
                        // D[sq] = dO[sq] row(.) O[sq]
                        GemmAccDataType D_acc = 0.f;
                        for(int k = 0; k < hdim_v; k++)
                        {
                            InOutDataType do_reg =
                                do_batch_seq_nhead_hdim(0, seq_q_offsets[i_batch] + sq, i_head, k);
                            InOutDataType o_reg =
                                o_batch_seq_nhead_hdim(0, seq_q_offsets[i_batch] + sq, i_head, k);
                            D_acc += ck_tile::type_convert<GemmAccDataType>(do_reg) *
                                     ck_tile::type_convert<GemmAccDataType>(o_reg);
                        }
                        CompDataType D = ck_tile::type_convert<CompDataType>(D_acc);
                        // dS = P * (drop_scale*dP - D). P is the PURE softmax output; the dropout
                        // mask multiplies dP only. D = dO.O already carries dropout (O is dropped).
                        for(int sk = 0; sk < seqlen_kv; sk++)
                            locals_dS[sk] =
                                locals_P[sk] * (locals_drop_scale[sk] * locals_dP[sk] - D);
                    }

                    // ------------------------------------------------------------------
                    // Step 5: dQ = alpha * dS @ K^T   (A=dS[sq,sk], B=K^T[hdim_qk,sk])
                    //   (computed fresh per sq row, no accumulation needed)
                    // ------------------------------------------------------------------
                    for(int k = 0; k < hdim_qk; k++)
                    {
                        GemmAccDataType acc = 0.f;
                        for(int sk = 0; sk < seqlen_kv; sk++)
                        {
                            InOutDataType ds_reg =
                                ck_tile::type_convert<InOutDataType>(locals_dS[sk]);
                            InOutDataType kreg =
                                k_batch_seq_nhead_hdim(0, seq_kv_offsets[i_batch] + sk, i_head, k);
                            acc += ck_tile::type_convert<GemmAccDataType>(ds_reg) *
                                   ck_tile::type_convert<GemmAccDataType>(kreg);
                        }
                        dq_batch_seq_nhead_hdim(0, seq_q_offsets[i_batch] + sq, i_head, k) =
                            ck_tile::type_convert<InOutDataType>(acc * alpha);
                    }

                    // ------------------------------------------------------------------
                    // Step 6: dK = alpha * dS^T @ Q^T   (A=dS^T[sk,sq], B=Q^T[hdim_qk,sq])
                    // ------------------------------------------------------------------
                    for(int sk = 0; sk < seqlen_kv; sk++)
                    {
                        InOutDataType ds_reg = ck_tile::type_convert<InOutDataType>(locals_dS[sk]);
                        for(int k = 0; k < hdim_qk; k++)
                        {
                            InOutDataType qreg =
                                q_batch_seq_nhead_hdim(0, seq_q_offsets[i_batch] + sq, i_head, k);
                            dk_acc[sk][k] += ck_tile::type_convert<GemmAccDataType>(ds_reg) *
                                             ck_tile::type_convert<GemmAccDataType>(qreg);
                        }
                    }
                }

                // Write back dK (multiplied by alpha) and dV
                for(int sk = 0; sk < seqlen_kv; sk++)
                {
                    for(int k = 0; k < hdim_qk; k++)
                        dk_batch_seq_nhead_hdim(0, seq_kv_offsets[i_batch] + sk, i_head, k) =
                            ck_tile::type_convert<InOutDataType>(dk_acc[sk][k] * alpha);
                    for(int k = 0; k < hdim_v; k++)
                        dv_batch_seq_nhead_hdim(0, seq_kv_offsets[i_batch] + sk, i_head, k) =
                            ck_tile::type_convert<InOutDataType>(dv_acc[sk][k]);
                }
            });
        };

        make_ParallelTensorFunctor(f, num_batch, num_head)(std::thread::hardware_concurrency());
    }
};

} // namespace ck_tile
