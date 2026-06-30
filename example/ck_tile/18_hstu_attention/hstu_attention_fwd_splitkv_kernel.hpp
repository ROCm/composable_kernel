// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "hstu_block_masking.hpp"
#include "hstu_attention_kernel_util.hpp"

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S''[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] @ V^T[hdim_v, seqlen_k]

namespace ck_tile {

template <typename HstuAttentionPipeline_, typename EpiloguePipeline_>
struct HstuAttentionFwdSplitKVKernel
{
    using HstuAttentionPipeline                   = ck_tile::remove_cvref_t<HstuAttentionPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = HstuAttentionPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = HstuAttentionPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);

    using QKVDataType =
        ck_tile::remove_cvref_t<typename HstuAttentionPipeline::Problem::QKVDataType>;
    using BiasDataType =
        ck_tile::remove_cvref_t<typename HstuAttentionPipeline::Problem::BiasDataType>;
    using OaccDataType =
        ck_tile::remove_cvref_t<typename HstuAttentionPipeline::Problem::OaccDataType>;
    using CompDataType =
        ck_tile::remove_cvref_t<typename HstuAttentionPipeline::Problem::CompDataType>;

    static constexpr bool kIsCrossAttention = HstuAttentionPipeline::Problem::kIsCrossAttention;
    static constexpr bool kUseGroup         = HstuAttentionPipeline::Problem::kUseGroup;
    static constexpr bool kIsJagged         = HstuAttentionPipeline::Problem::kIsJagged;
    static constexpr auto kHasBias          = HstuAttentionPipeline::Problem::kHasBias;
    static constexpr bool kHasDropout       = HstuAttentionPipeline::Problem::kHasDropout;
    static constexpr bool kHasCausalMask    = HstuAttentionPipeline::Problem::kHasCausal;
    static constexpr bool kUseSoftmax       = HstuAttentionPipeline::Problem::kUseSoftmax;

    static constexpr bool kPadSeqLenQ   = HstuAttentionPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK   = HstuAttentionPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQK = HstuAttentionPipeline::kPadHeadDimQK;
    static constexpr bool kPadHeadDimV  = HstuAttentionPipeline::kPadHeadDimV;

    static constexpr bool kUseTrLoad = detail::is_using_trload_v<HstuAttentionPipeline>;

    template <ck_tile::index_t I> // to avoid duplicated base class problem, introduce an template
                                  // arg
    struct HstuAttentionFwdEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct HstuAttentionNoGroupBatchedFwdBaseKargs
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;

        const int32_t* num_targets_ptr;

        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        void* o_acc_ptr;
        ck_tile::index_t num_splits;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_kv;
        ck_tile::index_t hdim_qk;
        ck_tile::index_t hdim_v;

        ck_tile::index_t seq_stride_q;
        ck_tile::index_t seq_stride_k;
        ck_tile::index_t seq_stride_v;

        ck_tile::index_t num_head;
        float scale_s; // scaling value exerted on the immediate Q@K result
        float scale_p; // scaling value exerted on the SiLU result

        bool almost_invariant_seqlen; // should always be true for batched mode

        ck_tile::index_t contextual_seqlen;
        ck_tile::index_t window_size;
        ck_tile::index_t min_full_attn_seqlen;
    };

    struct HstuAttentionNoGroupJaggedFwdBaseKargs
    {
        const int32_t* seq_q_offsets_ptr;
        const int32_t* seq_kv_offsets_ptr;

        ck_tile::index_t seq_stride_q;
        ck_tile::index_t seq_stride_k;
        ck_tile::index_t seq_stride_v;

        const int32_t* num_targets_ptr;

        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        void* o_acc_ptr;
        ck_tile::index_t num_splits;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;

        ck_tile::index_t hdim_qk;
        ck_tile::index_t hdim_v;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_kv;

        ck_tile::index_t num_head;
        float scale_s; // scaling value exerted on the immediate Q@K result
        float scale_p; // scaling value exerted on the SiLU result

        bool almost_invariant_seqlen;

        ck_tile::index_t contextual_seqlen;
        ck_tile::index_t window_size;
        ck_tile::index_t min_full_attn_seqlen;
    };

    struct HstuAttentionGroupFwdBaseKargs
    {
        ck_tile::index_t num_batch_per_group;

        const int32_t* seq_q_offsets_ptr;
        const int32_t* seq_kv_offsets_ptr;

        ck_tile::index_t seq_stride_q;
        ck_tile::index_t seq_stride_k;
        ck_tile::index_t seq_stride_v;

        const int32_t* num_targets_ptr;

        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        void* o_acc_ptr;
        ck_tile::index_t num_splits;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;

        ck_tile::index_t hdim_qk;
        ck_tile::index_t hdim_v;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_kv;

        ck_tile::index_t num_head;
        float scale_s; // scaling value exerted on the immediate Q@K result
        float scale_p; // scaling value exerted on the SiLU result

        bool almost_invariant_seqlen;

        int32_t contextual_seqlen;    // to be set by the per-group contextual_seqlen
        int32_t window_size;          // to be set by the per-group window_size
        int32_t min_full_attn_seqlen; // to be set by the per-group min_full_attn_seqlen

        const int32_t* group_max_seqlen_q_ptr;
        const int32_t* group_contextual_seqlen_ptr;
        const int32_t* group_window_size_ptr;
        const int32_t* group_min_full_attn_seqlen_ptr;
        const float* group_attn_scale_ptr;
    };

    struct HstuAttentionFwdBatchedBiasKargs
    {
        const void* bias_ptr;
        ck_tile::index_t seq_stride_bias;
        ck_tile::index_t nhead_stride_bias;
        ck_tile::index_t batch_stride_bias;
    };

    struct HstuAttentionFwdJaggedBiasKargs
    {
        const void* bias_ptr;
        ck_tile::index_t seq_stride_bias;
        ck_tile::index_t nhead_stride_bias;
    };

    struct HstuAttentionFwdSoftmaxKargs
    {
        void* lse_acc_ptr;
    };

    struct HstuAttentionFwdCommonDropoutKargs
    {
        void init_dropout(float p_drop, uint64_t seed, uint64_t offset)
        {
            float p_undrop = 1.0 - p_drop;
            p_undrop_in_uint8_t =
                uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
            rp_undrop = 1.0 / p_undrop;

            this->drop_seed   = seed;
            this->drop_offset = offset;
        }

        // used for generating random numbers
        uint64_t drop_seed;
        uint64_t drop_offset;

        float rp_undrop             = 1;
        uint8_t p_undrop_in_uint8_t = std::numeric_limits<uint8_t>::max();
    };

    struct HstuAttentionNoGroupBatchedFwdKargs
        : HstuAttentionNoGroupBatchedFwdBaseKargs,
          std::conditional_t<kHasBias,
                             HstuAttentionFwdBatchedBiasKargs,
                             HstuAttentionFwdEmptyKargs<1>>,
          std::conditional_t<kHasDropout,
                             HstuAttentionFwdCommonDropoutKargs,
                             HstuAttentionFwdEmptyKargs<2>>,
          std::conditional_t<kUseSoftmax,
                             HstuAttentionFwdSoftmaxKargs,
                             HstuAttentionFwdEmptyKargs<3>>
    {
    };

    struct HstuAttentionNoGroupJaggedFwdKargs
        : HstuAttentionNoGroupJaggedFwdBaseKargs,
          std::conditional_t<kHasBias,
                             HstuAttentionFwdJaggedBiasKargs,
                             HstuAttentionFwdEmptyKargs<1>>,
          std::conditional_t<kHasDropout,
                             HstuAttentionFwdCommonDropoutKargs,
                             HstuAttentionFwdEmptyKargs<2>>,
          std::conditional_t<kUseSoftmax,
                             HstuAttentionFwdSoftmaxKargs,
                             HstuAttentionFwdEmptyKargs<3>>
    {
    };

    struct HstuAttentionGroupFwdKargs : HstuAttentionGroupFwdBaseKargs,
                                        std::conditional_t<kHasBias,
                                                           HstuAttentionFwdJaggedBiasKargs,
                                                           HstuAttentionFwdEmptyKargs<1>>,
                                        std::conditional_t<kHasDropout,
                                                           HstuAttentionFwdCommonDropoutKargs,
                                                           HstuAttentionFwdEmptyKargs<2>>,
                                        std::conditional_t<kUseSoftmax,
                                                           HstuAttentionFwdSoftmaxKargs,
                                                           HstuAttentionFwdEmptyKargs<3>>
    {
    };

    using Kargs = std::conditional_t<kUseGroup,
                                     HstuAttentionGroupFwdKargs,
                                     std::conditional_t<kIsJagged,
                                                        HstuAttentionNoGroupJaggedFwdKargs,
                                                        HstuAttentionNoGroupBatchedFwdKargs>>;

    static constexpr bool kUseNoGroupBatched = (!kUseGroup && !kIsJagged);
    static constexpr bool kUseNoGroupJagged  = (!kUseGroup && kIsJagged);

    template <bool Cond = kUseNoGroupBatched>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              void* o_acc_ptr,             // workspace for accumulation of o
              void* lse_acc_ptr,           // workspace for accumulation of lse
              ck_tile::index_t num_splits, // number of splitted seqlen_kv
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_kv,
              ck_tile::index_t hdim_qk,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head,
              float scale_s,
              float attn_scale,
              ck_tile::index_t seq_stride_q,
              ck_tile::index_t seq_stride_k,
              ck_tile::index_t seq_stride_v,
              ck_tile::index_t seq_stride_bias,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_bias,
              const void* num_targets_ptr,
              ck_tile::index_t contextual_seqlen,
              ck_tile::index_t window_size,
              ck_tile::index_t min_full_attn_seqlen,
              float p_drop,
              uint64_t philox_seed,
              uint64_t philox_offset)
    {
        Kargs kargs{
            {batch_stride_q,
             batch_stride_k,
             batch_stride_v,
             reinterpret_cast<const int32_t*>(num_targets_ptr),
             q_ptr,
             k_ptr,
             v_ptr,
             o_acc_ptr,
             num_splits,
             nhead_stride_q,
             nhead_stride_k,
             nhead_stride_v,
             seqlen_q,
             seqlen_kv,
             hdim_qk,
             hdim_v,
             seq_stride_q,
             seq_stride_k,
             seq_stride_v,
             num_head,
             scale_s,
             attn_scale ? attn_scale : 1.0f / static_cast<float>(seqlen_q),
             true, // almost_invariant_seqlen
             contextual_seqlen,
             window_size,
             min_full_attn_seqlen}, // args for common karg
            {},                     // placeholder for bias
            {},                     // placeholder for dropout
        };

        if constexpr(kHasBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.seq_stride_bias   = seq_stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        if constexpr(kHasDropout)
        {
            kargs.init_dropout(p_drop, philox_seed, philox_offset);
        }
        if constexpr(kUseSoftmax)
        {
            kargs.lse_acc_ptr = lse_acc_ptr;
        }

        return kargs;
    }

    template <bool Cond = kUseNoGroupJagged>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              void* o_acc_ptr,             // workspace for accumulation of o
              void* lse_acc_ptr,           // workspace for accumulation of lse
              ck_tile::index_t num_splits, // number of splitted seqlen_kv
              const void* seq_q_offsets_ptr,
              const void* seq_kv_offsets_ptr,
              ck_tile::index_t max_seqlen_q,
              ck_tile::index_t hdim_qk,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head,
              float scale_s,
              float attn_scale,
              bool almost_invariant_seqlen,
              ck_tile::index_t seq_stride_q,
              ck_tile::index_t seq_stride_k,
              ck_tile::index_t seq_stride_v,
              ck_tile::index_t seq_stride_bias,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              const void* num_targets_ptr,
              ck_tile::index_t contextual_seqlen,
              ck_tile::index_t window_size,
              ck_tile::index_t min_full_attn_seqlen,
              float p_drop,
              uint64_t philox_seed,
              uint64_t philox_offset)
    {
        Kargs kargs{
            {reinterpret_cast<const int32_t*>(seq_q_offsets_ptr),
             reinterpret_cast<const int32_t*>(seq_kv_offsets_ptr),
             seq_stride_q,
             seq_stride_k,
             seq_stride_v,
             reinterpret_cast<const int32_t*>(num_targets_ptr),
             q_ptr,
             k_ptr,
             v_ptr,
             o_acc_ptr,
             num_splits,
             nhead_stride_q,
             nhead_stride_k,
             nhead_stride_v,
             hdim_qk,
             hdim_v,
             -1, // seqlen_q will be updated by another pointer
             -1, // seqlen_kv will be updated by another pointer
             num_head,
             scale_s,
             attn_scale ? attn_scale : 1.0f / static_cast<float>(max_seqlen_q),
             almost_invariant_seqlen,
             contextual_seqlen,
             window_size,
             min_full_attn_seqlen}, // args for common karg
            {},                     // placeholder for bias
            {},                     // placeholder for dropout
        };

        if constexpr(kHasBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.seq_stride_bias   = seq_stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
        }
        if constexpr(kHasDropout)
        {
            kargs.init_dropout(p_drop, philox_seed, philox_offset);
        }
        if constexpr(kUseSoftmax)
        {
            kargs.lse_acc_ptr = lse_acc_ptr;
        }

        return kargs;
    }

    template <bool Cond = kUseGroup>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              void* o_acc_ptr,             // workspace for accumulation of o
              void* lse_acc_ptr,           // workspace for accumulation of lse
              ck_tile::index_t num_splits, // number of splitted seqlen_kv
              ck_tile::index_t num_batch_per_group,
              const void* seq_q_offsets_ptr,
              const void* seq_kv_offsets_ptr,
              const void* group_max_seqlen_q_ptr,
              const void* group_contextual_seqlen_ptr,
              const void* group_window_size_ptr,
              const void* group_min_full_attn_seqlen_ptr,
              const void* group_attn_scale_ptr,
              ck_tile::index_t hdim_qk,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head,
              float scale_s,
              bool almost_invariant_seqlen,
              ck_tile::index_t seq_stride_q,
              ck_tile::index_t seq_stride_k,
              ck_tile::index_t seq_stride_v,
              ck_tile::index_t seq_stride_bias,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              const void* num_targets_ptr,
              float p_drop,
              uint64_t philox_seed,
              uint64_t philox_offset)
    {
        Kargs kargs{
            {num_batch_per_group,
             reinterpret_cast<const int32_t*>(seq_q_offsets_ptr),
             reinterpret_cast<const int32_t*>(seq_kv_offsets_ptr),
             seq_stride_q,
             seq_stride_k,
             seq_stride_v,
             reinterpret_cast<const int32_t*>(num_targets_ptr),
             q_ptr,
             k_ptr,
             v_ptr,
             o_acc_ptr,
             num_splits,
             nhead_stride_q,
             nhead_stride_k,
             nhead_stride_v,
             hdim_qk,
             hdim_v,
             -1, // seqlen_q will be updated by another pointer
             -1, // seqlen_kv will be updated by another pointer
             num_head,
             scale_s,
             1.0f, // to be set according to the per-group attn_scale and max_seqlen
             almost_invariant_seqlen,
             0, // to be set by the per-group contextual_seqlen
             0, // to be set by the per-group window_size
             0, // to be set by the per-group min_full_attn_seqlen
             reinterpret_cast<const int32_t*>(group_max_seqlen_q_ptr),
             reinterpret_cast<const int32_t*>(group_contextual_seqlen_ptr),
             reinterpret_cast<const int32_t*>(group_window_size_ptr),
             reinterpret_cast<const int32_t*>(group_min_full_attn_seqlen_ptr),
             reinterpret_cast<const float*>(group_attn_scale_ptr)}, // args for common karg
            {},                                                     // placeholder for bias
            {},                                                     // placeholder for dropout
        };

        if constexpr(kHasBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.seq_stride_bias   = seq_stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
        }
        if constexpr(kHasDropout)
        {
            kargs.init_dropout(p_drop, philox_seed, philox_offset);
        }
        if constexpr(kUseSoftmax)
        {
            kargs.lse_acc_ptr = lse_acc_ptr;
        }

        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size_,
                                                ck_tile::index_t nhead_,
                                                ck_tile::index_t seqlen_,
                                                ck_tile::index_t hdim_v_,
                                                ck_tile::index_t num_splits,
                                                bool almost_invariant_seqlen,
                                                bool has_minfull_attn_seqlen = false)
    {
        // The Q sequence [0, seqlen) will be split to two parts for allocating workgroups:
        // 1) [0, seqlen - target - min_full_attn_seqlen)
        // 2) [seqlen - target - min_full_attn_seqlen, seqlen)
        ck_tile::index_t num_tile_in_seqlen =
            ck_tile::integer_divide_ceil(seqlen_, HstuAttentionPipeline::kM0);

        // when kHasDropout is false, the second split starts just from position seqlen_q -
        // num_target - min_full_attn_seqlen, so both the first split and the second split could
        // have a incomplete last tile, thus one additional workgroup should be allocated for each
        // seqlen_q
        if constexpr(!kHasDropout)
        {
            if constexpr(kUseGroup)
            {
                // always assume minfull_attn_seqlen > 0 since we don't know its exact value
                // for each group
                num_tile_in_seqlen += 1;
            }
            else
            {
                if(has_minfull_attn_seqlen)
                    num_tile_in_seqlen += 1;
            };
        }

        if(almost_invariant_seqlen)
        {
            if constexpr(HstuAttentionPipeline::kN1 < HstuAttentionPipeline::kQKHeaddim)
            {
                return dim3(batch_size_,
                            nhead_,
                            num_tile_in_seqlen * num_splits *
                                ck_tile::integer_divide_ceil(hdim_v_, HstuAttentionPipeline::kN1));
            }
            else
            {
                return dim3(batch_size_, nhead_, num_tile_in_seqlen * num_splits);
            }
        }
        else
        {
            if constexpr(HstuAttentionPipeline::kN1 < HstuAttentionPipeline::kQKHeaddim)
            {
                return dim3(num_tile_in_seqlen * num_splits *
                                ck_tile::integer_divide_ceil(hdim_v_, HstuAttentionPipeline::kN1),
                            nhead_,
                            batch_size_);
            }
            else
            {
                return dim3(num_tile_in_seqlen * num_splits, nhead_, batch_size_);
            }
        }
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
            return ck_tile::make_tuple(quotient, modulus);
        };

        if constexpr(HstuAttentionPipeline::kN1 < HstuAttentionPipeline::kQKHeaddim)
        {
            const index_t num_tile_n1 =
                ck_tile::integer_divide_ceil(kargs.hdim_v, HstuAttentionPipeline::kN1);

            if(kargs.almost_invariant_seqlen)
            {
                const index_t i_batch = blockIdx.x;
                const index_t i_nhead = blockIdx.y;
                const index_t i_block = blockIdx.z;

                auto [i_tile_m_i_split, i_tile_n] = f(i_block, num_tile_n1);
                auto [i_tile_m, i_split]          = f(i_tile_m_i_split, kargs.num_splits);

                i_tile_m = gridDim.z / num_tile_n1 / kargs.num_splits - 1 - i_tile_m;

                return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch, i_split);
            }
            else
            {
                const index_t i_block = blockIdx.x;
                const index_t i_nhead = blockIdx.y;
                const index_t i_batch = blockIdx.z;

                auto [i_tile_m_i_split, i_tile_n] = f(i_block, num_tile_n1);
                auto [i_tile_m, i_split]          = f(i_tile_m_i_split, kargs.num_splits);

                return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch, i_split);
            }
        }
        else
        {
            if(kargs.almost_invariant_seqlen)
            {
                const index_t i_batch = blockIdx.x;
                const index_t i_nhead = blockIdx.y;
                const index_t i_block = blockIdx.z;

                index_t i_tile_m_i_split = i_block;
                auto [i_tile_m, i_split] = f(i_tile_m_i_split, kargs.num_splits);
                i_tile_m                 = gridDim.z / kargs.num_splits - 1 - i_tile_m;
                const index_t i_tile_n   = 0;

                return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch, i_split);
            }
            else
            {
                const index_t i_block = blockIdx.x;
                const index_t i_nhead = blockIdx.y;
                const index_t i_batch = blockIdx.z;

                index_t i_tile_m_i_split = i_block;
                auto [i_tile_m, i_split] = f(i_tile_m_i_split, kargs.num_splits);
                const index_t i_tile_n   = 0;

                return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch, i_split);
            }
        }
    }

    CK_TILE_DEVICE static constexpr auto
    CalculateTileRangeAlongXForSplit(ck_tile::index_t global_seqlen_k_start,
                                     ck_tile::index_t global_seqlen_k_end,
                                     ck_tile::index_t num_splits,
                                     ck_tile::index_t i_split)
    {
        ck_tile::index_t num_tile = ck_tile::integer_divide_ceil(
            global_seqlen_k_end - global_seqlen_k_start, HstuAttentionPipeline::kN0);

        ck_tile::index_t num_tile_per_split = ck_tile::integer_divide_ceil(num_tile, num_splits);

        ck_tile::index_t my_seqlen_k_start =
            global_seqlen_k_start + num_tile_per_split * i_split * HstuAttentionPipeline::kN0;
        ck_tile::index_t my_seqlen_k_end = min(
            global_seqlen_k_start + num_tile_per_split * (i_split + 1) * HstuAttentionPipeline::kN0,
            global_seqlen_k_end);

        return ck_tile::make_tuple(my_seqlen_k_start, my_seqlen_k_end);
    }

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        if(is_wave32())
        {
            // it looks get_warp_size() always return 64 when called from host, so
            // halfing is needed to get actual BlockSize
            return dim3(kBlockSize / get_warp_size() * 32);
        }
        else
            return dim3(kBlockSize);
    }

    CK_TILE_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(HstuAttentionPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        const auto [i_tile_m, i_tile_n, i_nhead, i_batch, i_split] = GetTileIndex(kargs);

        long_index_t batch_offset_q       = 0;
        long_index_t batch_offset_k       = 0;
        long_index_t batch_offset_v       = 0;
        long_index_t batch_offset_bias    = 0;
        long_index_t batch_offset_o_acc   = 0;
        long_index_t batch_offset_lse_acc = 0;

        if constexpr(kIsJagged)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seq_q_offsets_ptr[i_batch];
            const long_index_t key_start   = kargs.seq_kv_offsets_ptr[i_batch];

            batch_offset_q = query_start * kargs.seq_stride_q;
            batch_offset_k = key_start * kargs.seq_stride_k;
            batch_offset_v = key_start * kargs.seq_stride_v;

            if constexpr(kHasBias)
            {
                batch_offset_bias = query_start * kargs.seq_stride_bias;
            }

            // assume o_acc is in compact shape of [batch_size, max_seqlen_q, num_head, num_splits,
            // hdim]
            batch_offset_o_acc = query_start * kargs.num_head * kargs.num_splits * kargs.hdim_v;

            // assume l_acc is in compact shape of [batch_size, max_seqlen_q, num_head, num_splits]
            if constexpr(kUseSoftmax)
            {
                batch_offset_lse_acc = query_start * kargs.num_head * kargs.num_splits;
            }

            kargs.seqlen_q =
                kargs.seq_q_offsets_ptr[i_batch + 1] - kargs.seq_q_offsets_ptr[i_batch];
            kargs.seqlen_kv =
                kargs.seq_kv_offsets_ptr[i_batch + 1] - kargs.seq_kv_offsets_ptr[i_batch];

            // read from device memory for the group specific mask and scaling parameters
            if constexpr(kUseGroup)
            {
                index_t i_group =
                    __builtin_amdgcn_readfirstlane(i_batch / kargs.num_batch_per_group);

                float attn_scale     = kargs.group_attn_scale_ptr[i_group];
                index_t max_seqlen_q = kargs.group_max_seqlen_q_ptr[i_group];
                kargs.scale_p = (attn_scale ? attn_scale : 1.0f / static_cast<float>(max_seqlen_q));
                kargs.contextual_seqlen    = kargs.group_contextual_seqlen_ptr[i_group];
                kargs.window_size          = kargs.group_window_size_ptr[i_group];
                kargs.min_full_attn_seqlen = kargs.group_min_full_attn_seqlen_ptr[i_group];
            };
        }
        else
        {
            batch_offset_q = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
            if constexpr(kHasBias)
            {
                batch_offset_bias = static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias;
            }
            // assume o_acc is in compact shape of [batch_size, seqlen_q, num_head, num_splits,
            // hdim]
            batch_offset_o_acc = static_cast<long_index_t>(i_batch) * kargs.seqlen_q *
                                 kargs.num_head * kargs.num_splits * kargs.hdim_v;

            // assume l_acc is in compact shape of [batch_size, seqlen_q, num_head, num_splits]
            if constexpr(kUseSoftmax)
            {
                batch_offset_lse_acc = static_cast<long_index_t>(i_batch) * kargs.seqlen_q *
                                       kargs.num_head * kargs.num_splits;
            }
        }

        int num_target = (kargs.num_targets_ptr == nullptr) ? 0 : kargs.num_targets_ptr[i_batch];

        index_t seqlen_in_first_split = kargs.seqlen_q;
        bool is_tile_in_first_split   = true;
        index_t i_m0;

        // 1) when kHasDropout is false, the second split starts just from position seqlen_q -
        // num_target - min_full_attn_seqlen; 2) when kHasDropout is true, the second split must
        // start from the first kM0 aligned position bigger or equal to seqlen_q - num_target -
        // min_full_attn_seqlen, since BlockDropout requires each workgroup has a kM0 aligned i_m0
        // position
        if(kargs.min_full_attn_seqlen > 0)
        {
            // need consider for cases where min_full_attn_seqlen be bigger than max_uih_len
            if(kargs.seqlen_q - num_target > kargs.min_full_attn_seqlen)
            {
                seqlen_in_first_split = kargs.seqlen_q - num_target - kargs.min_full_attn_seqlen;

                index_t num_tile_in_first_split =
                    __builtin_amdgcn_readfirstlane(ck_tile::integer_divide_ceil(
                        seqlen_in_first_split, HstuAttentionPipeline::kM0));

                is_tile_in_first_split = (i_tile_m < num_tile_in_first_split);

                if constexpr(!kHasDropout)
                {
                    // be careful i_m0 for second_split could be not aligned on kM0
                    i_m0 =
                        is_tile_in_first_split
                            ? __builtin_amdgcn_readfirstlane(i_tile_m * HstuAttentionPipeline::kM0)
                            : __builtin_amdgcn_readfirstlane((i_tile_m - num_tile_in_first_split) *
                                                             HstuAttentionPipeline::kM0) +
                                  seqlen_in_first_split;
                }
                else
                    i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * HstuAttentionPipeline::kM0);
            }
            else
            {
                seqlen_in_first_split  = 0;
                is_tile_in_first_split = false;

                // adjust the min_full_attn_seqlen to be passed to HstuBlockMask constructor
                kargs.min_full_attn_seqlen = kargs.seqlen_q - num_target;

                i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * HstuAttentionPipeline::kM0);
            };
        }
        else
            i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * HstuAttentionPipeline::kM0);

        const index_t i_n1 = __builtin_amdgcn_readfirstlane(i_tile_n * HstuAttentionPipeline::kN1);

        index_t seqlen_q_in_ctrl =
            kHasDropout ? kargs.seqlen_q
                        : (is_tile_in_first_split ? seqlen_in_first_split : kargs.seqlen_q);

        if(seqlen_q_in_ctrl <= i_m0)
            return;

        const QKVDataType* q_ptr = reinterpret_cast<const QKVDataType*>(kargs.q_ptr) +
                                   static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                   batch_offset_q;
        const QKVDataType* k_ptr = reinterpret_cast<const QKVDataType*>(kargs.k_ptr) +
                                   static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_k +
                                   batch_offset_k;
        const QKVDataType* v_ptr = reinterpret_cast<const QKVDataType*>(kargs.v_ptr) +
                                   static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_v +
                                   batch_offset_v;
        // assume o_acc is in compact shape of [batch_size, seqlen, num_head, num_splits, hdim]
        OaccDataType* o_acc_ptr =
            reinterpret_cast<OaccDataType*>(kargs.o_acc_ptr) +
            static_cast<long_index_t>(i_nhead) * kargs.num_splits * kargs.hdim_v +
            static_cast<long_index_t>(i_split) * kargs.hdim_v + batch_offset_o_acc;

        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(seqlen_q_in_ctrl, kargs.hdim_qk),
                make_tuple(kargs.seq_stride_q, 1),
                number<HstuAttentionPipeline::kAlignmentQ>{},
                number<1>{});
            return pad_tensor_view(q_dram_naive,
                                   make_tuple(number<HstuAttentionPipeline::kM0>{},
                                              number<HstuAttentionPipeline::kQKHeaddim>{}),
                                   sequence<false, kPadHeadDimQK>{});
        }();
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(kargs.seqlen_kv, kargs.hdim_qk),
                make_tuple(kargs.seq_stride_k, 1),
                number<HstuAttentionPipeline::kAlignmentK>{},
                number<1>{});

            return pad_tensor_view(k_dram_naive,
                                   make_tuple(number<HstuAttentionPipeline::kN0>{},
                                              number<HstuAttentionPipeline::kQKHeaddim>{}),
                                   sequence<false, kPadHeadDimQK>{});
        }();
        const auto v_dram = [&]() {
            const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                v_ptr,
                make_tuple(kargs.seqlen_kv, kargs.hdim_v),
                make_tuple(kargs.seq_stride_v, 1),
                number<HstuAttentionPipeline::kAlignmentV>{},
                number<1>{});

            if constexpr(!kUseTrLoad)
            {
                const auto v_dram_transposed =
                    transform_tensor_view(v_dram_naive,
                                          make_tuple(make_pass_through_transform(kargs.hdim_v),
                                                     make_pass_through_transform(kargs.seqlen_kv)),
                                          make_tuple(sequence<1>{}, sequence<0>{}),
                                          make_tuple(sequence<0>{}, sequence<1>{}));

                return pad_tensor_view(v_dram_transposed,
                                       make_tuple(number<HstuAttentionPipeline::kN1>{},
                                                  number<HstuAttentionPipeline::kK1>{}),
                                       sequence<kPadHeadDimV, false>{});
            }
            else
            {
                return pad_tensor_view(v_dram_naive,
                                       make_tuple(number<HstuAttentionPipeline::kK1>{},
                                                  number<HstuAttentionPipeline::kN1>{}),
                                       sequence<false, kPadHeadDimV>{});
            };
        }();

        auto q_dram_window =
            make_tile_window(q_dram,
                             [&]() {
                                 return make_tuple(number<HstuAttentionPipeline::kM0>{},
                                                   number<HstuAttentionPipeline::kQKHeaddim>{});
                             }(),
                             {i_m0, 0});

        auto k_dram_window =
            make_tile_window(k_dram,
                             make_tuple(number<HstuAttentionPipeline::kN0>{},
                                        number<HstuAttentionPipeline::kQKHeaddim>{}),
                             {0, 0});

        auto v_dram_window = make_tile_window(
            v_dram,
            make_tuple(number<HstuAttentionPipeline::kN1>{}, number<HstuAttentionPipeline::kK1>{}),
            {i_n1, 0});

        /// FIXME: Before C++20, capturing structured binding variables are not supported. Remove
        /// following copy capture of the 'i_nhead' if in C++20
        const auto bias_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto bias_dram_window_lengths = make_tuple(
                number<HstuAttentionPipeline::kM0>{}, number<HstuAttentionPipeline::kN0>{});
            if constexpr(kHasBias)
            {
                const BiasDataType* bias_ptr =
                    reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_bias +
                    batch_offset_bias;

                const auto bias_dram = [&]() {
                    const auto bias_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        bias_ptr,
                        make_tuple(seqlen_q_in_ctrl, kargs.seqlen_kv),
                        make_tuple(kargs.seq_stride_bias, 1),
                        number<HstuAttentionPipeline::kAlignmentBias>{},
                        number<1>{});

                    return pad_tensor_view(
                        bias_dram_naive, bias_dram_window_lengths, sequence<false, kPadSeqLenK>{});
                }();

                return make_tile_window(bias_dram, bias_dram_window_lengths, {i_m0, 0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        auto lse_acc_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto lse_acc_dram_window_lengths =
                make_tuple(number<HstuAttentionPipeline::kM0>{});

            if constexpr(kUseSoftmax)
            {
                // assume l_acc is in compact shape of [batch_size, seqlen_q, num_head, num_splits]
                CompDataType* lse_acc_ptr = reinterpret_cast<CompDataType*>(kargs.lse_acc_ptr) +
                                            static_cast<long_index_t>(i_nhead) * kargs.num_splits +
                                            batch_offset_lse_acc + i_split;

                // LSEacc DRAM and LSEacc DRAM window
                auto seq_stride_lse_acc       = kargs.num_head * kargs.num_splits;
                const auto lse_acc_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    lse_acc_ptr,
                    make_tuple(seqlen_q_in_ctrl),
                    make_tuple(seq_stride_lse_acc),
                    number<HstuAttentionPipeline::kAlignmentLSEacc>{},
                    number<1>{});

                const auto lse_acc_dram =
                    pad_tensor_view(lse_acc_dram_naive,
                                    make_tuple(number<HstuAttentionPipeline::kM0>{}),
                                    sequence<false>{});

                return make_tile_window(lse_acc_dram, lse_acc_dram_window_lengths, {i_m0});
            }
            else
            {
                return make_null_tile_window(lse_acc_dram_window_lengths);
            }
        }();

        auto null_randval_window = [&]() {
            if constexpr(kHasDropout)
            {
                // need to make a tile window from this null_randval_dram since the null_tile_window
                // does not have store_tile() over-loaded, which will cause compiling issue when
                // used inside BlockDropout::Run(). Also we need this dram window to provide
                // start_m0_idx used in BlockDropout::Run()
                const auto null_randval_dram = [&]() {
                    const auto null_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        static_cast<uint8_t*>(nullptr),
                        make_tuple(seqlen_q_in_ctrl, kargs.seqlen_kv),
                        make_tuple(kargs.seqlen_kv, 1),
                        number<1>{},
                        number<1>{});

                    return pad_tensor_view(null_dram_naive,
                                           make_tuple(number<HstuAttentionPipeline::kM0>{},
                                                      number<HstuAttentionPipeline::kN0>{}),
                                           sequence<true, true>{});
                }();

                return make_tile_window(null_randval_dram,
                                        make_tuple(number<HstuAttentionPipeline::kM0>{},
                                                   number<HstuAttentionPipeline::kN0>{}),
                                        {i_m0, 0});
            }
            else
                return make_null_tile_window(make_tuple(number<1>{}, number<1>{}));
        }();

        auto dropout = [&, i_nhead_ = i_nhead, i_batch_ = i_batch]() {
            if constexpr(kHasDropout)
            {
                // no need to save rand_val since we have separate kernel to generate them for the
                // host
                return BlockDropout{i_batch_,
                                    i_nhead_,
                                    kargs.num_head,
                                    kargs.drop_seed,
                                    kargs.drop_offset,
                                    kargs.rp_undrop,
                                    kargs.p_undrop_in_uint8_t,
                                    false};
            }
            else
            {
                return NullBlockDropout{};
            };
        }();

        auto o_acc_tile = [&]() {
            if(kargs.window_size > 0)
            {
                using HstuMaskType = typename ck_tile::
                    HstuBlockMasking<kIsCrossAttention, kHasCausalMask, true>::Type;

                auto mask = [&]() {
                    if constexpr(kIsCrossAttention)
                    {
                        return make_hstu_cross_attention_block_mask_with_local<HstuMaskType>(
                            is_tile_in_first_split,
                            kargs.seqlen_q,
                            kargs.seqlen_kv,
                            kargs.contextual_seqlen,
                            num_target,
                            kargs.window_size,
                            kargs.min_full_attn_seqlen);
                    }
                    else
                    {
                        return make_hstu_self_attention_block_mask_with_local<HstuMaskType>(
                            is_tile_in_first_split,
                            kargs.seqlen_q,
                            kargs.contextual_seqlen,
                            num_target,
                            kargs.window_size,
                            kargs.min_full_attn_seqlen);
                    };
                }();

                const auto [global_seqlen_k_start, global_seqlen_k_end] =
                    mask.GetTileRangeAlongX(bool_constant<kHasDropout>{},
                                            i_m0,
                                            number<HstuAttentionPipeline::kM0>{},
                                            number<HstuAttentionPipeline::kN0>{});

                const auto [seqlen_k_start, seqlen_k_end] = CalculateTileRangeAlongXForSplit(
                    global_seqlen_k_start, global_seqlen_k_end, kargs.num_splits, i_split);

                if constexpr(!kUseSoftmax)
                {
                    return HstuAttentionPipeline{}(q_dram_window,
                                                   k_dram_window,
                                                   v_dram_window,
                                                   bias_dram_window,
                                                   null_randval_window,
                                                   seqlen_k_start,
                                                   seqlen_k_end,
                                                   mask,
                                                   kargs.scale_s,
                                                   kargs.scale_p,
                                                   smem_ptr,
                                                   dropout);
                }
                else
                {
                    return HstuAttentionPipeline{}(q_dram_window,
                                                   k_dram_window,
                                                   v_dram_window,
                                                   bias_dram_window,
                                                   lse_acc_dram_window,
                                                   null_randval_window,
                                                   seqlen_k_start,
                                                   seqlen_k_end,
                                                   mask,
                                                   kargs.scale_s,
                                                   kargs.scale_p,
                                                   smem_ptr,
                                                   dropout);
                }
            }
            else
            {
                using HstuMaskType = typename ck_tile::
                    HstuBlockMasking<kIsCrossAttention, kHasCausalMask, false>::Type;

                auto mask = [&]() {
                    if constexpr(kIsCrossAttention)
                    {
                        return make_hstu_cross_attention_block_mask_without_local<HstuMaskType>(
                            kargs.seqlen_q, kargs.seqlen_kv, kargs.contextual_seqlen, num_target);
                    }
                    else
                    {
                        return make_hstu_self_attention_block_mask_without_local<HstuMaskType>(
                            kargs.seqlen_q, kargs.contextual_seqlen, num_target);
                    };
                }();

                const auto [global_seqlen_k_start, global_seqlen_k_end] =
                    mask.GetTileRangeAlongX(i_m0,
                                            number<HstuAttentionPipeline::kM0>{},
                                            number<HstuAttentionPipeline::kN0>{});

                const auto [seqlen_k_start, seqlen_k_end] = CalculateTileRangeAlongXForSplit(
                    global_seqlen_k_start, global_seqlen_k_end, kargs.num_splits, i_split);

                if constexpr(!kUseSoftmax)
                {
                    return HstuAttentionPipeline{}(q_dram_window,
                                                   k_dram_window,
                                                   v_dram_window,
                                                   bias_dram_window,
                                                   null_randval_window,
                                                   seqlen_k_start,
                                                   seqlen_k_end,
                                                   mask,
                                                   kargs.scale_s,
                                                   kargs.scale_p,
                                                   smem_ptr,
                                                   dropout);
                }
                else
                {
                    return HstuAttentionPipeline{}(q_dram_window,
                                                   k_dram_window,
                                                   v_dram_window,
                                                   bias_dram_window,
                                                   lse_acc_dram_window,
                                                   null_randval_window,
                                                   seqlen_k_start,
                                                   seqlen_k_end,
                                                   mask,
                                                   kargs.scale_s,
                                                   kargs.scale_p,
                                                   smem_ptr,
                                                   dropout);
                }
            }
        }();

        // Oacc DRAM and Oacc DRAM window
        auto o_acc_dram = [&]() {
            auto seq_stride_o_acc       = kargs.num_head * kargs.num_splits * kargs.hdim_v;
            const auto o_acc_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_acc_ptr,
                make_tuple(seqlen_q_in_ctrl, kargs.hdim_v),
                make_tuple(seq_stride_o_acc, 1),
                number<HstuAttentionPipeline::kAlignmentO>{},
                number<1>{});

            return pad_tensor_view(o_acc_dram_naive,
                                   make_tuple(number<HstuAttentionPipeline::kM0>{},
                                              number<HstuAttentionPipeline::kN1>{}),
                                   sequence<false, kPadHeadDimV>{});
        }();

        auto o_acc_dram_window = make_tile_window(
            o_acc_dram,
            make_tuple(number<HstuAttentionPipeline::kM0>{}, number<HstuAttentionPipeline::kN1>{}),
            {i_m0, i_n1});

        EpiloguePipeline{}(o_acc_dram_window, o_acc_tile, nullptr);
    }
};

} // namespace ck_tile
