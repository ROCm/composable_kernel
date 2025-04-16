// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "hstu_block_masking.hpp"

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S''[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] @ V^T[hdim_v, seqlen_k]

namespace ck_tile {

template <typename HstuAttentionPipeline_, typename EpiloguePipeline_>
struct HstuAttentionFwdKernel
{
    using HstuAttentionPipeline                   = ck_tile::remove_cvref_t<HstuAttentionPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = HstuAttentionPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = HstuAttentionPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);
    static constexpr ck_tile::index_t kBlockPerCuInput =
        HstuAttentionPipeline::Problem::kBlockPerCu;

    using QKVDataType  = ck_tile::remove_cvref_t<typename HstuAttentionPipeline::QKVDataType>;
    using BiasDataType = ck_tile::remove_cvref_t<typename HstuAttentionPipeline::BiasDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename HstuAttentionPipeline::ODataType>;

    using VLayout = ck_tile::remove_cvref_t<typename HstuAttentionPipeline::VLayout>;

    static constexpr bool kIsJagged     = HstuAttentionPipeline::kIsJagged;
    static constexpr bool kPadSeqLenQ   = HstuAttentionPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK   = HstuAttentionPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQK = HstuAttentionPipeline::kPadHeadDimQK;
    static constexpr bool kPadHeadDimV  = HstuAttentionPipeline::kPadHeadDimV;
    static constexpr auto kHasBias      = HstuAttentionPipeline::kHasBias;
    static constexpr bool kHasDropout   = HstuAttentionPipeline::kHasDropout;
    using HstuMask = ck_tile::remove_cvref_t<typename HstuAttentionPipeline::HstuMask>;
    static constexpr bool kHasLocalMask = HstuMask::kUseLocal;

    template <ck_tile::index_t I> // to avoid duplicated base class problem, introduce an template
                                  // arg
    struct HstuAttentionFwdEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct HstuAttentionFwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        void* o_ptr;

        ck_tile::index_t seqlen;
        ck_tile::index_t hdim_qk;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head;
        float scale_s;

        ck_tile::index_t seq_stride_q;
        ck_tile::index_t seq_stride_k;
        ck_tile::index_t seq_stride_v;
        ck_tile::index_t seq_stride_o;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_o;

        const int32_t* num_targets_ptr;
        ck_tile::index_t contextual_seqlen;
    };

    struct HstuAttentionFwdCommonBiasKargs
    {
        const void* bias_ptr               = nullptr;
        ck_tile::index_t seq_stride_bias   = 0;
        ck_tile::index_t nhead_stride_bias = 0;
    };

    struct HstuAttentionFwdBatchModeBiasKargs : HstuAttentionFwdCommonBiasKargs
    {
        ck_tile::index_t batch_stride_bias = 0;
    };

    struct HstuAttentionFwdMaskKargs
    {
        ck_tile::index_t window_size;
        ck_tile::index_t min_full_attn_seqlen;
    };

    struct HstuAttentionFwdDropoutSeedOffset
    {
        uint64_t drop_seed;
        uint64_t drop_offset;
    };

    struct HstuAttentionFwdCommonDropoutKargs : HstuAttentionFwdDropoutSeedOffset
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

        float rp_undrop             = 1;
        uint8_t p_undrop_in_uint8_t = std::numeric_limits<uint8_t>::max();
    };

    struct HstuAttentionFwdBatchModeKargs : HstuAttentionFwdCommonKargs,
                                            std::conditional_t<kHasBias,
                                                               HstuAttentionFwdBatchModeBiasKargs,
                                                               HstuAttentionFwdEmptyKargs<0>>,
                                            std::conditional_t<kHasLocalMask,
                                                               HstuAttentionFwdMaskKargs,
                                                               HstuAttentionFwdEmptyKargs<1>>,
                                            std::conditional_t<kHasDropout,
                                                               HstuAttentionFwdCommonDropoutKargs,
                                                               HstuAttentionFwdEmptyKargs<2>>
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_o;
    };

    struct HstuAttentionFwdJaggModeKargs : HstuAttentionFwdCommonKargs,
                                           std::conditional_t<kHasBias,
                                                              HstuAttentionFwdCommonBiasKargs,
                                                              HstuAttentionFwdEmptyKargs<0>>,
                                           std::conditional_t<kHasLocalMask,
                                                              HstuAttentionFwdMaskKargs,
                                                              HstuAttentionFwdEmptyKargs<1>>,
                                           std::conditional_t<kHasDropout,
                                                              HstuAttentionFwdCommonDropoutKargs,
                                                              HstuAttentionFwdEmptyKargs<2>>
    {
        const int32_t* seq_offsets_ptr;
    };

    using Kargs = std::
        conditional_t<kIsJagged, HstuAttentionFwdJaggModeKargs, HstuAttentionFwdBatchModeKargs>;

    template <bool Cond = !kIsJagged>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargsImpl(const void* q_ptr,
                  const void* k_ptr,
                  const void* v_ptr,
                  const void* bias_ptr,
                  void* o_ptr,
                  ck_tile::index_t seqlen,
                  ck_tile::index_t hdim_qk,
                  ck_tile::index_t hdim_v,
                  ck_tile::index_t num_head,
                  float scale_s,
                  ck_tile::index_t seq_stride_q,
                  ck_tile::index_t seq_stride_k,
                  ck_tile::index_t seq_stride_v,
                  ck_tile::index_t seq_stride_bias,
                  ck_tile::index_t seq_stride_o,
                  ck_tile::index_t nhead_stride_q,
                  ck_tile::index_t nhead_stride_k,
                  ck_tile::index_t nhead_stride_v,
                  ck_tile::index_t nhead_stride_bias,
                  ck_tile::index_t nhead_stride_o,
                  ck_tile::index_t batch_stride_q,
                  ck_tile::index_t batch_stride_k,
                  ck_tile::index_t batch_stride_v,
                  ck_tile::index_t batch_stride_bias,
                  ck_tile::index_t batch_stride_o,
                  const void* num_targets_ptr,
                  ck_tile::index_t contextual_seqlen,
                  ck_tile::index_t window_size,
                  ck_tile::index_t min_full_attn_seqlen,
                  float p_drop,
                  const std::pair<uint64_t, uint64_t>& drop_seed_offset)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     seqlen,
                     hdim_qk,
                     hdim_v,
                     num_head,
                     scale_s,
                     seq_stride_q,
                     seq_stride_k,
                     seq_stride_v,
                     seq_stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o,
                     reinterpret_cast<const int32_t*>(num_targets_ptr),
                     contextual_seqlen}, // args for common karg
                    {},                  // placeholder for bias
                    {},                  // placeholder for mask
                    {},                  // placeholder for dropout
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_o};

        if constexpr(kHasBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.seq_stride_bias   = seq_stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        if constexpr(kHasLocalMask)
        {
            kargs.window_size          = window_size;
            kargs.min_full_attn_seqlen = min_full_attn_seqlen;
        }
        if constexpr(kHasDropout)
        {
            auto seed   = std::get<0>(drop_seed_offset);
            auto offset = std::get<1>(drop_seed_offset);
            kargs.init_dropout(p_drop, seed, offset);
        }

        return kargs;
    }

    template <bool Cond = !kIsJagged>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              void* o_ptr,
              ck_tile::index_t seqlen,
              ck_tile::index_t hdim_qk,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head,
              float scale_s,
              ck_tile::index_t seq_stride_q,
              ck_tile::index_t seq_stride_k,
              ck_tile::index_t seq_stride_v,
              ck_tile::index_t seq_stride_bias,
              ck_tile::index_t seq_stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_bias,
              ck_tile::index_t batch_stride_o,
              const void* num_targets_ptr,
              ck_tile::index_t contextual_seqlen,
              ck_tile::index_t window_size,
              ck_tile::index_t min_full_attn_seqlen,
              float p_drop,
              uint64_t philox_seed,
              uint64_t philox_offset)
    {
        return MakeKargsImpl(q_ptr,
                             k_ptr,
                             v_ptr,
                             bias_ptr,
                             o_ptr,
                             seqlen,
                             hdim_qk,
                             hdim_v,
                             num_head,
                             scale_s,
                             seq_stride_q,
                             seq_stride_k,
                             seq_stride_v,
                             seq_stride_bias,
                             seq_stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_o,
                             num_targets_ptr,
                             contextual_seqlen,
                             window_size,
                             min_full_attn_seqlen,
                             p_drop,
                             std::make_pair(philox_seed, philox_offset));
    }

    template <bool Cond = kIsJagged>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargsImpl(const void* q_ptr,
                  const void* k_ptr,
                  const void* v_ptr,
                  const void* bias_ptr,
                  void* o_ptr,
                  const void* seq_offsets_ptr,
                  ck_tile::index_t hdim_qk,
                  ck_tile::index_t hdim_v,
                  ck_tile::index_t num_head,
                  float scale_s,
                  ck_tile::index_t seq_stride_q,
                  ck_tile::index_t seq_stride_k,
                  ck_tile::index_t seq_stride_v,
                  ck_tile::index_t seq_stride_bias,
                  ck_tile::index_t seq_stride_o,
                  ck_tile::index_t nhead_stride_q,
                  ck_tile::index_t nhead_stride_k,
                  ck_tile::index_t nhead_stride_v,
                  ck_tile::index_t nhead_stride_bias,
                  ck_tile::index_t nhead_stride_o,
                  const void* num_targets_ptr,
                  ck_tile::index_t contextual_seqlen,
                  ck_tile::index_t window_size,
                  ck_tile::index_t min_full_attn_seqlen,
                  float p_drop,
                  const std::pair<uint64_t, uint64_t>& drop_seed_offset)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     -1, // seqlen will be updated by another pointer
                     hdim_qk,
                     hdim_v,
                     num_head,
                     scale_s,
                     seq_stride_q,
                     seq_stride_k,
                     seq_stride_v,
                     seq_stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o,
                     reinterpret_cast<const int32_t*>(num_targets_ptr),
                     contextual_seqlen}, // args for common karg
                    {},                  // placeholder for bias
                    {},                  // placeholder for mask
                    {},                  // placeholder for dropout
                    reinterpret_cast<const int32_t*>(seq_offsets_ptr)};

        if constexpr(kHasBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.seq_stride_bias   = seq_stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
        }
        if constexpr(kHasLocalMask)
        {
            kargs.window_size          = window_size;
            kargs.min_full_attn_seqlen = min_full_attn_seqlen;
        }
        if constexpr(kHasDropout)
        {
            auto seed   = std::get<0>(drop_seed_offset);
            auto offset = std::get<1>(drop_seed_offset);
            kargs.init_dropout(p_drop, seed, offset);
        }

        return kargs;
    }

    template <bool Cond = kIsJagged>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              void* o_ptr,
              const void* seq_offsets_ptr,
              ck_tile::index_t hdim_qk,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head,
              float scale_s,
              ck_tile::index_t seq_stride_q,
              ck_tile::index_t seq_stride_k,
              ck_tile::index_t seq_stride_v,
              ck_tile::index_t seq_stride_bias,
              ck_tile::index_t seq_stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_o,
              const void* num_targets_ptr,
              ck_tile::index_t contextual_seqlen,
              ck_tile::index_t window_size,
              ck_tile::index_t min_full_attn_seqlen,
              float p_drop,
              uint64_t philox_seed,
              uint64_t philox_offset)
    {
        return MakeKargsImpl(q_ptr,
                             k_ptr,
                             v_ptr,
                             bias_ptr,
                             o_ptr,
                             seq_offsets_ptr,
                             hdim_qk,
                             hdim_v,
                             num_head,
                             scale_s,
                             seq_stride_q,
                             seq_stride_k,
                             seq_stride_v,
                             seq_stride_bias,
                             seq_stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_o,
                             num_targets_ptr,
                             contextual_seqlen,
                             window_size,
                             min_full_attn_seqlen,
                             p_drop,
                             std::make_pair(philox_seed, philox_offset));
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size_,
                                                ck_tile::index_t nhead_,
                                                ck_tile::index_t seqlen_,
                                                ck_tile::index_t hdim_v_)
    {
        // TODO: this may need tuning
        return dim3(ck_tile::integer_divide_ceil(seqlen_, HstuAttentionPipeline::kM0) *
                        ck_tile::integer_divide_ceil(hdim_v_, HstuAttentionPipeline::kN1),
                    nhead_,
                    batch_size_);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
        // const index_t num_tile_m0 = seqlen_q / kM0;
        const index_t num_tile_n1 =
            ck_tile::integer_divide_ceil(kargs.hdim_v, HstuAttentionPipeline::kN1);

        const index_t i_block = blockIdx.x;
        const index_t i_nhead = blockIdx.y;
        const index_t i_batch = blockIdx.z;

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
            return ck_tile::make_tuple(quotient, modulus);
        };

        const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

        return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(HstuAttentionPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_m, i_tile_n, i_nhead, i_batch] = GetTileIndex(kargs);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * HstuAttentionPipeline::kM0);
        const index_t i_n1 = __builtin_amdgcn_readfirstlane(i_tile_n * HstuAttentionPipeline::kN1);

        long_index_t batch_offset_q    = 0;
        long_index_t batch_offset_k    = 0;
        long_index_t batch_offset_v    = 0;
        long_index_t batch_offset_bias = 0;
        long_index_t batch_offset_o    = 0;

        if constexpr(kIsJagged)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seq_offsets_ptr[i_batch];
            const long_index_t key_start   = query_start;

            batch_offset_q = query_start * kargs.seq_stride_q;
            batch_offset_k = key_start * kargs.seq_stride_k;
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                batch_offset_v = key_start * kargs.seq_stride_v;
            }
            else
            {
                batch_offset_v = key_start;
            }
            if constexpr(kHasBias)
            {
                batch_offset_bias = query_start * kargs.seq_stride_bias;
            }
            batch_offset_o = query_start * kargs.seq_stride_o;

            kargs.seqlen = kargs.seq_offsets_ptr[i_batch + 1] - kargs.seq_offsets_ptr[i_batch];

            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen <= i_m0)
            {
                return;
            }
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
            batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
        }

        int num_target = (kargs.num_targets_ptr == nullptr) ? 0 : kargs.num_targets_ptr[i_batch];

        HstuMask mask = [&]() {
            if constexpr(kHasLocalMask)
                return make_hstu_block_mask_with_local<HstuMask>(kargs.seqlen,
                                                                 kargs.contextual_seqlen,
                                                                 num_target,
                                                                 kargs.window_size,
                                                                 kargs.min_full_attn_seqlen);
            else
                return make_hstu_block_mask_without_local<HstuMask>(
                    kargs.seqlen, kargs.contextual_seqlen, num_target);
        }();

        // for simplicity, batch stride we just modify the pointer
        const QKVDataType* q_ptr = reinterpret_cast<const QKVDataType*>(kargs.q_ptr) +
                                   static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                   batch_offset_q;
        const QKVDataType* k_ptr = reinterpret_cast<const QKVDataType*>(kargs.k_ptr) +
                                   static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_k +
                                   batch_offset_k;
        const QKVDataType* v_ptr = reinterpret_cast<const QKVDataType*>(kargs.v_ptr) +
                                   static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_v +
                                   batch_offset_v;
        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(mask.max_uih_len, kargs.hdim_qk),
                make_tuple(kargs.seq_stride_q, 1),
                number<HstuAttentionPipeline::kAlignmentQ>{},
                number<1>{});
            if constexpr(HstuAttentionPipeline::kQLoadOnce)
            {
                return pad_tensor_view(q_dram_naive,
                                       make_tuple(number<HstuAttentionPipeline::kM0>{},
                                                  number<HstuAttentionPipeline::kSubQKHeaddim>{}),
                                       sequence<false, kPadHeadDimQK>{});
            }
            else
            {
                return pad_tensor_view(q_dram_naive,
                                       make_tuple(number<HstuAttentionPipeline::kM0>{},
                                                  number<HstuAttentionPipeline::kK0>{}),
                                       sequence<false, kPadHeadDimQK>{});
            }
        }();
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(mask.max_uih_len, kargs.hdim_qk),
                make_tuple(kargs.seq_stride_k, 1),
                number<HstuAttentionPipeline::kAlignmentK>{},
                number<1>{});

            return pad_tensor_view(k_dram_naive,
                                   make_tuple(number<HstuAttentionPipeline::kN0>{},
                                              number<HstuAttentionPipeline::kK0>{}),
                                   sequence<false, kPadHeadDimQK>{});
        }();
        const auto v_dram = [&]() {
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(mask.max_uih_len, kargs.hdim_v),
                    make_tuple(kargs.seq_stride_v, 1),
                    number<HstuAttentionPipeline::kAlignmentV>{},
                    number<1>{});

                const auto v_dram_transposed =
                    transform_tensor_view(v_dram_naive,
                                          make_tuple(make_pass_through_transform(kargs.hdim_v),
                                                     make_pass_through_transform(kargs.seqlen)),
                                          make_tuple(sequence<1>{}, sequence<0>{}),
                                          make_tuple(sequence<0>{}, sequence<1>{}));

                return pad_tensor_view(v_dram_transposed,
                                       make_tuple(number<HstuAttentionPipeline::kN1>{},
                                                  number<HstuAttentionPipeline::kK1>{}),
                                       sequence<kPadHeadDimV, false>{});
            }
            else
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(kargs.hdim_v, kargs.seqlen),
                    make_tuple(kargs.seq_stride_v, 1),
                    number<HstuAttentionPipeline::kAlignmentV>{},
                    number<1>{});

                return pad_tensor_view(v_dram_naive,
                                       make_tuple(number<HstuAttentionPipeline::kN1>{},
                                                  number<HstuAttentionPipeline::kK1>{}),
                                       sequence<kPadHeadDimV, false>{});
            }
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            [&]() {
                if constexpr(HstuAttentionPipeline::kQLoadOnce)
                    return make_tuple(number<HstuAttentionPipeline::kM0>{},
                                      number<HstuAttentionPipeline::kSubQKHeaddim>{});
                else
                    return make_tuple(number<HstuAttentionPipeline::kM0>{},
                                      number<HstuAttentionPipeline::kK0>{});
            }(),
            {i_m0, 0});

        auto k_dram_window = make_tile_window(
            k_dram,
            make_tuple(number<HstuAttentionPipeline::kN0>{}, number<HstuAttentionPipeline::kK0>{}),
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
                        make_tuple(kargs.seqlen, kargs.seqlen),
                        make_tuple(kargs.seq_stride_bias, 1),
                        number<HstuAttentionPipeline::kAlignmentBias>{},
                        number<1>{});

                    return pad_tensor_view(bias_dram_naive,
                                           bias_dram_window_lengths,
                                           sequence<kPadSeqLenQ, kPadSeqLenK>{});
                }();

                return make_tile_window(bias_dram, bias_dram_window_lengths, {i_m0, 0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        auto dropout = [&, i_nhead_ = i_nhead, i_batch_ = i_batch]() {
            if constexpr(kHasDropout)
            {
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
            return HstuAttentionPipeline{}(q_dram_window,
                                           k_dram_window,
                                           v_dram_window,
                                           bias_dram_window,
                                           mask,
                                           kargs.scale_s,
                                           smem_ptr,
                                           dropout);
        }();

        // O DRAM and O DRAM window
        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(mask.max_uih_len, kargs.hdim_v),
                make_tuple(kargs.seq_stride_o, 1),
                number<HstuAttentionPipeline::kAlignmentO>{},
                number<1>{});

            return pad_tensor_view(o_dram_naive,
                                   make_tuple(number<HstuAttentionPipeline::kM0>{},
                                              number<HstuAttentionPipeline::kN1>{}),
                                   sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        auto o_dram_window = make_tile_window(
            o_dram,
            make_tuple(number<HstuAttentionPipeline::kM0>{}, number<HstuAttentionPipeline::kN1>{}),
            {i_m0, i_n1});

        EpiloguePipeline{}(o_dram_window, o_acc_tile);
    }
};

} // namespace ck_tile
