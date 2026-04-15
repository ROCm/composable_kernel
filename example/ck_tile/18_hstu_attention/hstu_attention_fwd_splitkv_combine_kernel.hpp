// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "hstu_block_masking.hpp"

#ifndef HSTU_SCHED_BATCH_AS_FIRST_GRID_DIM
#define HSTU_SCHED_BATCH_AS_FIRST_GRID_DIM 1
#endif

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S''[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] @ V^T[hdim_v, seqlen_k]

namespace ck_tile {

template <typename HstuAttentionPipeline_, typename EpiloguePipeline_>
struct HstuAttentionFwdSplitKVCombineKernel
{
    using HstuAttentionPipeline                   = ck_tile::remove_cvref_t<HstuAttentionPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = HstuAttentionPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = HstuAttentionPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);

    using OaccDataType =
        ck_tile::remove_cvref_t<typename HstuAttentionPipeline::Problem::OaccDataType>;
    using ODataType = ck_tile::remove_cvref_t<typename HstuAttentionPipeline::Problem::ODataType>;

    static constexpr bool kIsJagged = HstuAttentionPipeline::Problem::kIsJagged;

    static constexpr bool kPadSeqLenQ  = HstuAttentionPipeline::kPadSeqLenQ;
    static constexpr bool kPadHeadDimO = HstuAttentionPipeline::kPadHeadDimO;

    template <ck_tile::index_t I> // to avoid duplicated base class problem, introduce an template
                                  // arg
    struct HstuAttentionCombineEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct HstuAttentionBatchedCombineBaseKargs
    {
        const void* o_acc_ptr;
        void* o_ptr;

        ck_tile::index_t batch_stride_o;
        ck_tile::index_t seq_stride_o;
        ck_tile::index_t nhead_stride_o;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t num_head;
        ck_tile::index_t num_splits;
        ck_tile::index_t hdim_v;
    };

    struct HstuAttentionJaggedCombineBaseKargs
    {
        const void* o_acc_ptr;
        void* o_ptr;

        ck_tile::index_t seq_stride_o;
        ck_tile::index_t nhead_stride_o;

        const int32_t* seq_q_offsets_ptr;
        ck_tile::index_t num_head;
        ck_tile::index_t num_splits;
        ck_tile::index_t hdim_v;

        ck_tile::index_t seqlen_q;
    };

    struct HstuAttentionBatchedCombineKargs : HstuAttentionBatchedCombineBaseKargs
    {
    };

    struct HstuAttentionJaggedCombineKargs : HstuAttentionJaggedCombineBaseKargs
    {
    };

    using Kargs = std::
        conditional_t<kIsJagged, HstuAttentionJaggedCombineKargs, HstuAttentionBatchedCombineKargs>;

    template <bool Cond = !kIsJagged>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* o_acc_ptr, // workspace for accumulation of o
              void* o_ptr,
              ck_tile::index_t batch_stride_o,
              ck_tile::index_t seq_stride_o,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t num_head,
              ck_tile::index_t num_splits, // number of splitted seqlen_kv
              ck_tile::index_t hdim_v)
    {
        Kargs kargs{o_acc_ptr,
                    o_ptr,
                    batch_stride_o,
                    seq_stride_o,
                    nhead_stride_o,
                    seqlen_q,
                    num_head,
                    num_splits,
                    hdim_v};

        return kargs;
    }

    template <bool Cond = kIsJagged>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* o_acc_ptr, // workspace for accumulation of o
              void* o_ptr,
              ck_tile::index_t seq_stride_o,
              ck_tile::index_t nhead_stride_o,
              const void* seq_q_offsets_ptr,
              ck_tile::index_t num_head,
              ck_tile::index_t num_splits, // number of splitted seqlen_kv
              ck_tile::index_t hdim_v)
    {
        Kargs kargs{o_acc_ptr,
                    o_ptr,
                    seq_stride_o,
                    nhead_stride_o,
                    reinterpret_cast<const int32_t*>(seq_q_offsets_ptr),
                    num_head,
                    num_splits,
                    hdim_v,
                    0 /* seqlen_q will be updated later */};

        return kargs;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size_, ck_tile::index_t nhead_, ck_tile::index_t seqlen_)
    {
        ck_tile::index_t num_tile_in_seqlen =
            ck_tile::integer_divide_ceil(seqlen_, HstuAttentionPipeline::kM);

#if HSTU_SCHED_BATCH_AS_FIRST_GRID_DIM
        return dim3(batch_size_, nhead_, num_tile_in_seqlen);
#else
            return dim3(num_tile_in_seqlen),
                        nhead_,
                        batch_size_);
#endif
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
        ignore = kargs;

#if HSTU_SCHED_BATCH_AS_FIRST_GRID_DIM
        const index_t i_batch  = blockIdx.x;
        const index_t i_nhead  = blockIdx.y;
        const index_t i_tile_m = blockIdx.z;
#else
        const index_t i_tile_m = blockIdx.x;
        const index_t i_nhead  = blockIdx.y;
        const index_t i_batch  = blockIdx.z;
#endif
        return ck_tile::make_tuple(i_tile_m, i_nhead, i_batch);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(HstuAttentionPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const auto [i_tile_m, i_nhead, i_batch] = GetTileIndex(kargs);

        long_index_t batch_offset_o_acc = 0;
        long_index_t batch_offset_o     = 0;

        if constexpr(kIsJagged)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seq_q_offsets_ptr[i_batch];

            // assume o_acc is in compact shape of [batch_size, max_seqlen, num_head, num_splits,
            // hdim]
            batch_offset_o_acc = query_start * kargs.num_head * kargs.num_splits * kargs.hdim_v;

            batch_offset_o = query_start * kargs.seq_stride_o;

            kargs.seqlen_q =
                kargs.seq_q_offsets_ptr[i_batch + 1] - kargs.seq_q_offsets_ptr[i_batch];
        }
        else
        {
            // assume o_acc is in compact shape of [batch_size, seqlen_q, num_head, num_splits,
            // hdim]
            batch_offset_o_acc = static_cast<long_index_t>(i_batch) * kargs.seqlen_q *
                                 kargs.num_head * kargs.num_splits * kargs.hdim_v;

            batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
        }

        index_t i_m0;

        i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * HstuAttentionPipeline::kM);

        if(kargs.seqlen_q <= i_m0)
            return;

        // assume o_acc is in compact shape of [batch_size, seqlen, num_head, num_splits, hdim]
        const OaccDataType* o_acc_ptr =
            reinterpret_cast<const OaccDataType*>(kargs.o_acc_ptr) +
            static_cast<long_index_t>(i_nhead) * kargs.num_splits * kargs.hdim_v +
            batch_offset_o_acc;
        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // Oacc DRAM and Oacc DRAM window
        auto seq_stride_o_acc       = kargs.num_head * kargs.num_splits * kargs.hdim_v;
        const auto o_acc_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            o_acc_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_v),
            make_tuple(seq_stride_o_acc, 1),
            number<HstuAttentionPipeline::kAlignmentO>{},
            number<1>{});

        const auto o_acc_dram =
            pad_tensor_view(o_acc_dram_naive,
                            make_tuple(number<HstuAttentionPipeline::kM>{},
                                       number<HstuAttentionPipeline::kOHeaddim>{}),
                            sequence<false, kPadHeadDimO>{});

        auto o_acc_dram_window =
            make_tile_window(o_acc_dram,
                             make_tuple(number<HstuAttentionPipeline::kM>{},
                                        number<HstuAttentionPipeline::kOHeaddim>{}),
                             {i_m0, 0});

        auto o_acc_tile = [&]() {
            return HstuAttentionPipeline{}(o_acc_dram_window, kargs.hdim_v, kargs.num_splits);
        }();

        // O DRAM and O DRAM window
        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.seq_stride_o, 1),
                number<HstuAttentionPipeline::kAlignmentO>{},
                number<1>{});

            return pad_tensor_view(o_dram_naive,
                                   make_tuple(number<HstuAttentionPipeline::kM>{},
                                              number<HstuAttentionPipeline::kOHeaddim>{}),
                                   sequence<false, kPadHeadDimO>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(number<HstuAttentionPipeline::kM>{},
                                        number<HstuAttentionPipeline::kOHeaddim>{}),
                             {i_m0, 0});

        EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
    }
};

} // namespace ck_tile
