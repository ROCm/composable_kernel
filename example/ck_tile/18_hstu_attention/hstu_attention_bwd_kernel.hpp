// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#if defined(HSTU_BWD_SINGLE_KERNEL)
// Guarded: this header is only ever reached from the single-kernel `#if` blocks
// of the two bwd dispatch headers. Keeping the guard here prevents the single
// masking header from leaking into a flag=OFF TU if someone includes this file
// from outside such a block later on.
#include "hstu_block_masking_bwd_single.hpp"
#endif

// HSTU attention backward — kernel layer (DESIGN §1.1).
//
// 3-kernel pipeline (sequence-conditional launch):
//   [PRE]  D[sq]=rowsum(O.*dO)            -- softmax path only        (M5)
//   [MAIN] HstuAttentionBwdDQDKDVKernel   -- 5 GEMM/7 stage, dV/dK + float dq_acc  (this file)
//   [POST] hstu_bwd_convert_dq_kernel     -- dq_acc(float) -> dQ(bf16/fp16)         (this file)
//
// launch order (SiLU): MAIN -> POST.
//
// M1: MAIN is a thin HSTU wrapper around the FMHA bwd kernel body (batched, no
// bias/dropout/group), differing from FMHA in that it carries TWO scalars
// (alpha, scale_p) to the SiLU pipeline instead of FMHA's single raw_scale/scale.
// POST is the atomic-path convert-only (single dq_acc, nsplits=1); the
// reduce+convert deterministic path (BlockFmhaBwdConvertQGrad) is M6.
//
//   TODO(M2): mask geometry (GenericAttentionMask<false> -> HSTU 5-factor mask).
//   TODO(M3/M4): jagged / group indexing.
//   TODO(M5): PRE (HstuAttentionBwdOGradDotO) + LSE read for softmax path.
//   TODO(M6): deterministic POST (reduce+convert over dq_acc splits).

namespace ck_tile {

// NOTE: no `using namespace hstu_bwd_single;` here on purpose. A using-directive
// only makes those names *also* visible in `ck_tile`, it does not hide the base
// `ck_tile::make_hstu_*` overloads that arrive in the same TU via
// hstu_attention_bwd_kernel_1/_2.hpp -> hstu_block_masking.hpp. With both sets
// visible every unqualified call becomes ambiguous, so the 16 factory call
// sites below spell out `hstu_bwd_single::` explicitly.

// HSTU bwd MAIN kernel: batched, SiLU, no bias/dropout/group (M1 scope).
// Mirrors FMHA FmhaBwdDQDKDVKernel::operator() window setup, but passes
// (alpha, scale_p) to the HSTU SiLU pipeline and uses HSTU plain kargs.
template <typename HstuPipeline_, typename KGradEpiloguePipeline_, typename VGradEpiloguePipeline_>
struct HstuAttentionBwdDQDKDVKernel
{
    using HstuPipeline          = remove_cvref_t<HstuPipeline_>;
    using KGradEpiloguePipeline = remove_cvref_t<KGradEpiloguePipeline_>;
    using VGradEpiloguePipeline = remove_cvref_t<VGradEpiloguePipeline_>;

    static constexpr index_t kBlockSize  = HstuPipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = HstuPipeline::kBlockPerCu;

    using QDataType     = remove_cvref_t<typename HstuPipeline::QDataType>;
    using KDataType     = remove_cvref_t<typename HstuPipeline::KDataType>;
    using VDataType     = remove_cvref_t<typename HstuPipeline::VDataType>;
    using AccDataType   = remove_cvref_t<typename HstuPipeline::AccDataType>;
    using OGradDataType = remove_cvref_t<typename HstuPipeline::OGradDataType>;
    using KGradDataType = remove_cvref_t<typename HstuPipeline::KGradDataType>;
    using VGradDataType = remove_cvref_t<typename HstuPipeline::VGradDataType>;
    using FmhaMask      = remove_cvref_t<typename HstuPipeline::FmhaMask>;

    // ck_tile bwd: seqlen never padded (OOB via buffer_load); headdim pad is index_t (0/8/1)
    static constexpr index_t kPadHeadDimQ  = HstuPipeline::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV  = HstuPipeline::kPadHeadDimV;
    static constexpr bool kIsDeterministic = HstuPipeline::kIsDeterministic;

    struct Kargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* do_ptr;
        void* dk_ptr;
        void* dv_ptr;
        void* dq_acc_ptr; // float

        // jagged (variable-length) mode (M3). When is_jagged: dim0=1 packed tensors,
        // per-(batch) base offset = seq_*_offsets_ptr[i_batch]*seq_stride (token-major),
        // and seqlen_{q,kv} are derived per batch from the offsets (the scalar
        // seqlen_q/seqlen_kv below are then ignored). Batched mode: is_jagged=false.
        bool is_jagged;
        const void* seq_q_offsets_ptr;  // int32, size num_batch+1
        const void* seq_kv_offsets_ptr; // int32, size num_batch+1 (== q offsets for self-attn)

        index_t seqlen_q;
        index_t seqlen_kv;
        index_t hdim_qk;
        index_t hdim_v;
        index_t nhead_ratio_qk;

        float alpha;
        float scale_p;

        // HSTU mask params (M2). num_targets_ptr is per-batch (int32); null => num_target=0.
        const void* num_targets_ptr;
        index_t contextual_seqlen;
        index_t max_attn_len; // == window_size (local_len)
        index_t min_full_attn_seqlen;

        index_t stride_q;
        index_t stride_k;
        index_t stride_v;
        index_t stride_do;
        index_t stride_dk;
        index_t stride_dv;
        index_t stride_dq_acc;

        index_t nhead_stride_q;
        index_t nhead_stride_k;
        index_t nhead_stride_v;
        index_t nhead_stride_do;
        index_t nhead_stride_dk;
        index_t nhead_stride_dv;
        index_t nhead_stride_dq_acc;

        index_t batch_stride_q;
        index_t batch_stride_k;
        index_t batch_stride_v;
        index_t batch_stride_do;
        index_t batch_stride_dk;
        index_t batch_stride_dv;
        index_t batch_stride_dq_acc;
        index_t split_stride_dq_acc; // M6 deterministic: per-split slot stride (single-slot elems)
    };

    CK_TILE_HOST static constexpr Kargs MakeKargs(const void* q_ptr,
                                                  const void* k_ptr,
                                                  const void* v_ptr,
                                                  const void* do_ptr,
                                                  void* dk_ptr,
                                                  void* dv_ptr,
                                                  void* dq_acc_ptr,
                                                  bool is_jagged,
                                                  const void* seq_q_offsets_ptr,
                                                  const void* seq_kv_offsets_ptr,
                                                  index_t seqlen_q,
                                                  index_t seqlen_kv,
                                                  index_t hdim_qk,
                                                  index_t hdim_v,
                                                  index_t nhead_ratio_qk,
                                                  float alpha,
                                                  float scale_p,
                                                  const void* num_targets_ptr,
                                                  index_t contextual_seqlen,
                                                  index_t max_attn_len,
                                                  index_t min_full_attn_seqlen,
                                                  index_t stride_q,
                                                  index_t stride_k,
                                                  index_t stride_v,
                                                  index_t stride_do,
                                                  index_t stride_dk,
                                                  index_t stride_dv,
                                                  index_t stride_dq_acc,
                                                  index_t nhead_stride_q,
                                                  index_t nhead_stride_k,
                                                  index_t nhead_stride_v,
                                                  index_t nhead_stride_do,
                                                  index_t nhead_stride_dk,
                                                  index_t nhead_stride_dv,
                                                  index_t nhead_stride_dq_acc,
                                                  index_t batch_stride_q,
                                                  index_t batch_stride_k,
                                                  index_t batch_stride_v,
                                                  index_t batch_stride_do,
                                                  index_t batch_stride_dk,
                                                  index_t batch_stride_dv,
                                                  index_t batch_stride_dq_acc,
                                                  index_t split_stride_dq_acc)
    {
        Kargs k;
        k.q_ptr               = q_ptr;
        k.k_ptr               = k_ptr;
        k.v_ptr               = v_ptr;
        k.do_ptr              = do_ptr;
        k.dk_ptr              = dk_ptr;
        k.dv_ptr              = dv_ptr;
        k.dq_acc_ptr          = dq_acc_ptr;
        k.is_jagged           = is_jagged;
        k.seq_q_offsets_ptr   = seq_q_offsets_ptr;
        k.seq_kv_offsets_ptr  = seq_kv_offsets_ptr;
        k.seqlen_q            = seqlen_q;
        k.seqlen_kv           = seqlen_kv;
        k.hdim_qk             = hdim_qk;
        k.hdim_v              = hdim_v;
        k.nhead_ratio_qk      = nhead_ratio_qk;
        k.alpha               = alpha;
        k.scale_p             = scale_p;
        k.num_targets_ptr     = num_targets_ptr;
        k.contextual_seqlen   = contextual_seqlen;
        k.max_attn_len        = max_attn_len;
        k.min_full_attn_seqlen = min_full_attn_seqlen;
        k.stride_q            = stride_q;
        k.stride_k            = stride_k;
        k.stride_v            = stride_v;
        k.stride_do           = stride_do;
        k.stride_dk           = stride_dk;
        k.stride_dv           = stride_dv;
        k.stride_dq_acc       = stride_dq_acc;
        k.nhead_stride_q      = nhead_stride_q;
        k.nhead_stride_k      = nhead_stride_k;
        k.nhead_stride_v      = nhead_stride_v;
        k.nhead_stride_do     = nhead_stride_do;
        k.nhead_stride_dk     = nhead_stride_dk;
        k.nhead_stride_dv     = nhead_stride_dv;
        k.nhead_stride_dq_acc = nhead_stride_dq_acc;
        k.batch_stride_q      = batch_stride_q;
        k.batch_stride_k      = batch_stride_k;
        k.batch_stride_v      = batch_stride_v;
        k.batch_stride_do     = batch_stride_do;
        k.batch_stride_dk     = batch_stride_dk;
        k.batch_stride_dv     = batch_stride_dv;
        k.batch_stride_dq_acc = batch_stride_dq_acc;
        k.split_stride_dq_acc = split_stride_dq_acc;
        return k;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(index_t batch_size, index_t nhead, index_t seqlen_kv)
    {
        return dim3(integer_divide_ceil(seqlen_kv, HstuPipeline::kN0), nhead, batch_size);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex()
    {
        return make_tuple(static_cast<index_t>(blockIdx.x),
                          static_cast<index_t>(blockIdx.y),
                          static_cast<index_t>(blockIdx.z));
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(HstuPipeline::GetSmemSize(),
                   KGradEpiloguePipeline::GetSmemSize(),
                   VGradEpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        __shared__ char smem_ptr[GetSmemSize()];

        const auto [i_tile_n, i_nhead, i_batch] = GetTileIndex();
        const index_t i_n0 = __builtin_amdgcn_readfirstlane(i_tile_n * HstuPipeline::kN0);

        // Per-(batch) base offsets. Jagged (M3): dim0=1 packed tensors, base =
        // seq_*_offsets_ptr[i_batch]*seq_stride (token-major, mirrors fwd kernel),
        // and seqlen_{q,kv} are derived from the offsets so the rest of the kernel
        // (windows, mask, OOB fill) is layout-agnostic. Batched: i_batch*batch_stride.
        long_index_t batch_offset_q;
        long_index_t batch_offset_k;
        long_index_t batch_offset_v;
        long_index_t batch_offset_do;
        long_index_t batch_offset_dk;
        long_index_t batch_offset_dv;
        long_index_t batch_offset_dq_acc;

        if(kargs.is_jagged)
        {
            const auto* q_offsets  = reinterpret_cast<const int32_t*>(kargs.seq_q_offsets_ptr);
            const auto* kv_offsets = reinterpret_cast<const int32_t*>(kargs.seq_kv_offsets_ptr);
            const long_index_t query_start = q_offsets[i_batch];
            const long_index_t key_start   = kv_offsets[i_batch];

            batch_offset_q      = query_start * kargs.stride_q;
            batch_offset_k      = key_start * kargs.stride_k;
            batch_offset_v      = key_start * kargs.stride_v;
            batch_offset_do     = query_start * kargs.stride_do;
            batch_offset_dk     = key_start * kargs.stride_dk;
            batch_offset_dv     = key_start * kargs.stride_dv;
            batch_offset_dq_acc = query_start * kargs.stride_dq_acc;

            // per-batch sequence lengths (token-major packed); overrides the scalar kargs
            kargs.seqlen_q  = q_offsets[i_batch + 1] - q_offsets[i_batch];
            kargs.seqlen_kv = kv_offsets[i_batch + 1] - kv_offsets[i_batch];
        }
        else
        {
            batch_offset_q      = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k      = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v      = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
            batch_offset_do     = static_cast<long_index_t>(i_batch) * kargs.batch_stride_do;
            batch_offset_dk     = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dk;
            batch_offset_dv     = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dv;
            batch_offset_dq_acc = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dq_acc;
        }

        // jagged: grid.x is sized for the largest sequence, so KV tiles past this
        // batch's seqlen_kv have no work. (Batched grid is exact -> never triggers.)
        if(i_n0 >= kargs.seqlen_kv)
            return;

        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
                                      batch_offset_do;
        KGradDataType* dk_ptr = reinterpret_cast<KGradDataType*>(kargs.dk_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dk +
                                batch_offset_dk;
        VGradDataType* dv_ptr = reinterpret_cast<VGradDataType*>(kargs.dv_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dv +
                                batch_offset_dv;

        // Q/K/V/dO DRAM views + windows
        const auto q_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_qk),
                make_tuple(kargs.stride_q, 1),
                number<HstuPipeline::kAlignmentQ>{},
                number<1>{}),
            make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});

        const auto k_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(kargs.seqlen_kv, kargs.hdim_qk),
                make_tuple(kargs.stride_k, 1),
                number<HstuPipeline::kAlignmentK>{},
                number<1>{}),
            make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});

        const auto v_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                v_ptr,
                make_tuple(kargs.seqlen_kv, kargs.hdim_v),
                make_tuple(kargs.stride_v, 1),
                number<HstuPipeline::kAlignmentV>{},
                number<1>{}),
            make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});

        const auto do_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                do_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_do, 1),
                number<HstuPipeline::kAlignmentOGrad>{},
                number<1>{}),
            make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});

        auto q_dram_window = make_tile_window(
            q_dram, make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kQKHeaddim>{}),
            {0, 0});
        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kQKHeaddim>{}),
            {i_n0, 0});
        auto v_dram_window = make_tile_window(
            v_dram, make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kVHeaddim>{}),
            {i_n0, 0});
        auto do_dram_window = make_tile_window(
            do_dram, make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kVHeaddim>{}),
            {0, 0});

        // dQ_acc window. Deterministic (M6): plain store (set) into THIS KV-block's own
        // split slot (base += i_tile_n * split_stride_dq_acc) -> no atomic, bit-reproducible;
        // POST reduces over splits. Atomic (default, nsplits=1): atomic_add into one slot.
        auto dq_dram_window = [&, i_tile_n_ = i_tile_n, i_nhead_ = i_nhead]() {
            constexpr auto mop = kIsDeterministic ? memory_operation_enum::set
                                                  : memory_operation_enum::atomic_add;
            AccDataType* dq_acc_ptr =
                reinterpret_cast<AccDataType*>(kargs.dq_acc_ptr) +
                static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_dq_acc +
                batch_offset_dq_acc;
            if constexpr(kIsDeterministic)
                dq_acc_ptr +=
                    static_cast<long_index_t>(i_tile_n_) * kargs.split_stride_dq_acc;
            auto dq_acc_dram = pad_tensor_view(
                make_naive_tensor_view<address_space_enum::global, mop>(
                    dq_acc_ptr,
                    make_tuple(kargs.seqlen_q, kargs.hdim_qk),
                    make_tuple(kargs.stride_dq_acc, 1),
                    number<HstuPipeline::kAlignmentQGrad>{},
                    number<1>{}),
                make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kQKHeaddim>{}),
                sequence<false, (kPadHeadDimQ > 0)>{});
            return make_tile_window(
                dq_acc_dram,
                make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kQKHeaddim>{}),
                {0, 0});
        }();

        // Build the HSTU mask identically to fwd/reference (self-attention; M2 batched).
        // is_tile_in_first_split=true (conservative: disables the IsFullTileInsideMask
        // fast-path so every edge tile is per-pixel checked; the tile-level first-split
        // optimization is a later perf item, and IsTokenPairInsideMask is self-contained).
        const int num_target =
            (kargs.num_targets_ptr != nullptr)
                ? reinterpret_cast<const int32_t*>(kargs.num_targets_ptr)[i_batch]
                : 0;
        auto mask = [&]() {
            if constexpr(FmhaMask::kUseLocal)
            {
                // clamp min_full like reference (reference_hstu_attention_bwd.hpp:177/198)
                const int eff_min_full =
                    (kargs.seqlen_q - num_target > kargs.min_full_attn_seqlen)
                        ? kargs.min_full_attn_seqlen
                        : (kargs.seqlen_q - num_target);
                // cross-attention: feed seqlen_kv into the seqlen_k slot (must go through the
                // make_hstu_cross_attention_* wrapper -> correct ctor arg order, see draft R2/R3).
                // if constexpr so the self (false) leg dead-code-elims byte-identical to M7c.
                if constexpr(FmhaMask::kIsCrossAttention)
                {
                    return hstu_bwd_single::make_hstu_cross_attention_block_mask_with_local<FmhaMask>(
                        /*is_tile_in_first_split=*/true,
                        kargs.seqlen_q,
                        kargs.seqlen_kv,
                        kargs.contextual_seqlen,
                        num_target,
                        kargs.max_attn_len,
                        eff_min_full);
                }
                else
                {
                    return hstu_bwd_single::make_hstu_self_attention_block_mask_with_local<FmhaMask>(
                        /*is_tile_in_first_split=*/true,
                        kargs.seqlen_q,
                        kargs.contextual_seqlen,
                        num_target,
                        kargs.max_attn_len,
                        eff_min_full);
                }
            }
            else
            {
                if constexpr(FmhaMask::kIsCrossAttention)
                {
                    return hstu_bwd_single::make_hstu_cross_attention_block_mask_without_local<FmhaMask>(
                        kargs.seqlen_q, kargs.seqlen_kv, kargs.contextual_seqlen, num_target);
                }
                else
                {
                    return hstu_bwd_single::make_hstu_self_attention_block_mask_without_local<FmhaMask>(
                        kargs.seqlen_q, kargs.contextual_seqlen, num_target);
                }
            }
        }();

        auto [dk_acc_tile, dv_acc_tile] = HstuPipeline{}(q_dram_window,
                                                         k_dram_window,
                                                         v_dram_window,
                                                         do_dram_window,
                                                         dq_dram_window,
                                                         mask,
                                                         kargs.alpha,
                                                         kargs.scale_p,
                                                         smem_ptr);

        auto dk_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                dk_ptr,
                make_tuple(kargs.seqlen_kv, kargs.hdim_qk),
                make_tuple(kargs.stride_dk, 1),
                number<HstuPipeline::kAlignmentKGrad>{},
                number<1>{}),
            make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});

        auto dv_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                dv_ptr,
                make_tuple(kargs.seqlen_kv, kargs.hdim_v),
                make_tuple(kargs.stride_dv, 1),
                number<HstuPipeline::kAlignmentVGrad>{},
                number<1>{}),
            make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});

        auto dk_dram_window = make_tile_window(
            dk_dram, make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kQKHeaddim>{}),
            {i_n0, 0});
        auto dv_dram_window = make_tile_window(
            dv_dram, make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kVHeaddim>{}),
            {i_n0, 0});

        KGradEpiloguePipeline{}(dk_dram_window, dk_acc_tile, nullptr);
        VGradEpiloguePipeline{}(dv_dram_window, dv_acc_tile, nullptr);
    }
};

// ---------------------------------------------------------------------------
// HSTU bwd MAIN kernel — SOFTMAX path (M5; batched + jagged, no group/bias/dropout).
// Same window setup as the SiLU kernel, but additionally builds the LSE + D dram
// windows ([batch,head,seq] seq-continuous; jagged: token base via q offsets) and
// passes (alpha) — no scale_p — to the softmax pipeline. D is produced by the PRE
// hstu_bwd_dot_do_o_kernel before MAIN.
template <typename HstuPipeline_, typename KGradEpiloguePipeline_, typename VGradEpiloguePipeline_>
struct HstuAttentionBwdDQDKDVSoftmaxKernel
{
    using HstuPipeline          = remove_cvref_t<HstuPipeline_>;
    using KGradEpiloguePipeline = remove_cvref_t<KGradEpiloguePipeline_>;
    using VGradEpiloguePipeline = remove_cvref_t<VGradEpiloguePipeline_>;

    static constexpr index_t kBlockSize  = HstuPipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = HstuPipeline::kBlockPerCu;

    using QDataType     = remove_cvref_t<typename HstuPipeline::QDataType>;
    using KDataType     = remove_cvref_t<typename HstuPipeline::KDataType>;
    using VDataType     = remove_cvref_t<typename HstuPipeline::VDataType>;
    using AccDataType   = remove_cvref_t<typename HstuPipeline::AccDataType>;
    using OGradDataType = remove_cvref_t<typename HstuPipeline::OGradDataType>;
    using KGradDataType = remove_cvref_t<typename HstuPipeline::KGradDataType>;
    using VGradDataType = remove_cvref_t<typename HstuPipeline::VGradDataType>;
    using LSEDataType   = remove_cvref_t<typename HstuPipeline::LSEDataType>;
    using DDataType     = remove_cvref_t<typename HstuPipeline::DDataType>;
    using FmhaMask      = remove_cvref_t<typename HstuPipeline::FmhaMask>;

    static constexpr index_t kPadHeadDimQ  = HstuPipeline::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV  = HstuPipeline::kPadHeadDimV;
    static constexpr bool kIsDeterministic = HstuPipeline::kIsDeterministic;

    struct Kargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* do_ptr;
        const void* lse_ptr; // float, [batch,head,seq] seq-continuous (jagged: [head,ΣL])
        const void* d_ptr;   // float, same layout as lse (PRE output)
        void* dk_ptr;
        void* dv_ptr;
        void* dq_acc_ptr; // float

        bool is_jagged;
        const void* seq_q_offsets_ptr;
        const void* seq_kv_offsets_ptr;

        index_t seqlen_q;
        index_t seqlen_kv;
        index_t hdim_qk;
        index_t hdim_v;
        index_t nhead_ratio_qk;

        float alpha;

        const void* num_targets_ptr;
        index_t contextual_seqlen;
        index_t max_attn_len;
        index_t min_full_attn_seqlen;

        index_t stride_q;
        index_t stride_k;
        index_t stride_v;
        index_t stride_do;
        index_t stride_dk;
        index_t stride_dv;
        index_t stride_dq_acc;

        index_t nhead_stride_q;
        index_t nhead_stride_k;
        index_t nhead_stride_v;
        index_t nhead_stride_do;
        index_t nhead_stride_dk;
        index_t nhead_stride_dv;
        index_t nhead_stride_dq_acc;
        // LSE [b,s,h] (fwd layout, seq_stride = num_head) and D [b,h,s] (packed,
        // seq_stride = 1) have DIFFERENT layouts -> split into 6 independent strides.
        index_t seq_stride_lse;     // BASE fwd LSE [b,s,h]: = num_head (NOT 1)
        index_t nhead_stride_lse;   // = 1
        index_t seq_stride_delta;   // D packed [b,h,s]: = 1
        index_t nhead_stride_delta; // = seqlen_q

        index_t batch_stride_q;
        index_t batch_stride_k;
        index_t batch_stride_v;
        index_t batch_stride_do;
        index_t batch_stride_dk;
        index_t batch_stride_dv;
        index_t batch_stride_dq_acc;
        index_t batch_stride_lse;
        index_t batch_stride_delta;
        index_t split_stride_dq_acc; // M6 deterministic
    };

    CK_TILE_HOST static constexpr Kargs MakeKargs(const void* q_ptr,
                                                  const void* k_ptr,
                                                  const void* v_ptr,
                                                  const void* do_ptr,
                                                  const void* lse_ptr,
                                                  const void* d_ptr,
                                                  void* dk_ptr,
                                                  void* dv_ptr,
                                                  void* dq_acc_ptr,
                                                  bool is_jagged,
                                                  const void* seq_q_offsets_ptr,
                                                  const void* seq_kv_offsets_ptr,
                                                  index_t seqlen_q,
                                                  index_t seqlen_kv,
                                                  index_t hdim_qk,
                                                  index_t hdim_v,
                                                  index_t nhead_ratio_qk,
                                                  float alpha,
                                                  const void* num_targets_ptr,
                                                  index_t contextual_seqlen,
                                                  index_t max_attn_len,
                                                  index_t min_full_attn_seqlen,
                                                  index_t stride_q,
                                                  index_t stride_k,
                                                  index_t stride_v,
                                                  index_t stride_do,
                                                  index_t stride_dk,
                                                  index_t stride_dv,
                                                  index_t stride_dq_acc,
                                                  index_t nhead_stride_q,
                                                  index_t nhead_stride_k,
                                                  index_t nhead_stride_v,
                                                  index_t nhead_stride_do,
                                                  index_t nhead_stride_dk,
                                                  index_t nhead_stride_dv,
                                                  index_t nhead_stride_dq_acc,
                                                  index_t seq_stride_lse,
                                                  index_t nhead_stride_lse,
                                                  index_t seq_stride_delta,
                                                  index_t nhead_stride_delta,
                                                  index_t batch_stride_q,
                                                  index_t batch_stride_k,
                                                  index_t batch_stride_v,
                                                  index_t batch_stride_do,
                                                  index_t batch_stride_dk,
                                                  index_t batch_stride_dv,
                                                  index_t batch_stride_dq_acc,
                                                  index_t batch_stride_lse,
                                                  index_t batch_stride_delta,
                                                  index_t split_stride_dq_acc)
    {
        Kargs k;
        k.q_ptr                = q_ptr;
        k.k_ptr                = k_ptr;
        k.v_ptr                = v_ptr;
        k.do_ptr               = do_ptr;
        k.lse_ptr              = lse_ptr;
        k.d_ptr                = d_ptr;
        k.dk_ptr               = dk_ptr;
        k.dv_ptr               = dv_ptr;
        k.dq_acc_ptr           = dq_acc_ptr;
        k.is_jagged            = is_jagged;
        k.seq_q_offsets_ptr    = seq_q_offsets_ptr;
        k.seq_kv_offsets_ptr   = seq_kv_offsets_ptr;
        k.seqlen_q             = seqlen_q;
        k.seqlen_kv            = seqlen_kv;
        k.hdim_qk              = hdim_qk;
        k.hdim_v               = hdim_v;
        k.nhead_ratio_qk       = nhead_ratio_qk;
        k.alpha                = alpha;
        k.num_targets_ptr      = num_targets_ptr;
        k.contextual_seqlen    = contextual_seqlen;
        k.max_attn_len         = max_attn_len;
        k.min_full_attn_seqlen = min_full_attn_seqlen;
        k.stride_q             = stride_q;
        k.stride_k             = stride_k;
        k.stride_v             = stride_v;
        k.stride_do            = stride_do;
        k.stride_dk            = stride_dk;
        k.stride_dv            = stride_dv;
        k.stride_dq_acc        = stride_dq_acc;
        k.nhead_stride_q       = nhead_stride_q;
        k.nhead_stride_k       = nhead_stride_k;
        k.nhead_stride_v       = nhead_stride_v;
        k.nhead_stride_do      = nhead_stride_do;
        k.nhead_stride_dk      = nhead_stride_dk;
        k.nhead_stride_dv      = nhead_stride_dv;
        k.nhead_stride_dq_acc  = nhead_stride_dq_acc;
        k.seq_stride_lse       = seq_stride_lse;
        k.nhead_stride_lse     = nhead_stride_lse;
        k.seq_stride_delta     = seq_stride_delta;
        k.nhead_stride_delta   = nhead_stride_delta;
        k.batch_stride_q       = batch_stride_q;
        k.batch_stride_k       = batch_stride_k;
        k.batch_stride_v       = batch_stride_v;
        k.batch_stride_do      = batch_stride_do;
        k.batch_stride_dk      = batch_stride_dk;
        k.batch_stride_dv      = batch_stride_dv;
        k.batch_stride_dq_acc  = batch_stride_dq_acc;
        k.batch_stride_lse     = batch_stride_lse;
        k.batch_stride_delta   = batch_stride_delta;
        k.split_stride_dq_acc  = split_stride_dq_acc;
        return k;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(index_t batch_size, index_t nhead, index_t seqlen_kv)
    {
        return dim3(integer_divide_ceil(seqlen_kv, HstuPipeline::kN0), nhead, batch_size);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex()
    {
        return make_tuple(static_cast<index_t>(blockIdx.x),
                          static_cast<index_t>(blockIdx.y),
                          static_cast<index_t>(blockIdx.z));
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(HstuPipeline::GetSmemSize(),
                   KGradEpiloguePipeline::GetSmemSize(),
                   VGradEpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        __shared__ char smem_ptr[GetSmemSize()];

        const auto [i_tile_n, i_nhead, i_batch] = GetTileIndex();
        const index_t i_n0 = __builtin_amdgcn_readfirstlane(i_tile_n * HstuPipeline::kN0);

        long_index_t batch_offset_q;
        long_index_t batch_offset_k;
        long_index_t batch_offset_v;
        long_index_t batch_offset_do;
        long_index_t batch_offset_dk;
        long_index_t batch_offset_dv;
        long_index_t batch_offset_dq_acc;
        long_index_t batch_offset_lse;   // LSE [b,s,h] per-(batch) base
        long_index_t batch_offset_delta; // D [b,h,s] per-(batch) base

        if(kargs.is_jagged)
        {
            const auto* q_offsets  = reinterpret_cast<const int32_t*>(kargs.seq_q_offsets_ptr);
            const auto* kv_offsets = reinterpret_cast<const int32_t*>(kargs.seq_kv_offsets_ptr);
            const long_index_t query_start = q_offsets[i_batch];
            const long_index_t key_start   = kv_offsets[i_batch];

            batch_offset_q      = query_start * kargs.stride_q;
            batch_offset_k      = key_start * kargs.stride_k;
            batch_offset_v      = key_start * kargs.stride_v;
            batch_offset_do     = query_start * kargs.stride_do;
            batch_offset_dk     = key_start * kargs.stride_dk;
            batch_offset_dv     = key_start * kargs.stride_dv;
            batch_offset_dq_acc = query_start * kargs.stride_dq_acc;
            batch_offset_lse    = query_start * kargs.seq_stride_lse;
            batch_offset_delta  = query_start; // D seq stride 1 (packed)

            kargs.seqlen_q  = q_offsets[i_batch + 1] - q_offsets[i_batch];
            kargs.seqlen_kv = kv_offsets[i_batch + 1] - kv_offsets[i_batch];
        }
        else
        {
            batch_offset_q      = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k      = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v      = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
            batch_offset_do     = static_cast<long_index_t>(i_batch) * kargs.batch_stride_do;
            batch_offset_dk     = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dk;
            batch_offset_dv     = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dv;
            batch_offset_dq_acc = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dq_acc;
            batch_offset_lse    = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse;
            batch_offset_delta  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_delta;
        }

        if(i_n0 >= kargs.seqlen_kv)
            return;

        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
                                      batch_offset_do;
        const LSEDataType* lse_ptr = reinterpret_cast<const LSEDataType*>(kargs.lse_ptr) +
                                     static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lse +
                                     batch_offset_lse;
        const DDataType* d_ptr = reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_delta +
                                 batch_offset_delta;
        KGradDataType* dk_ptr = reinterpret_cast<KGradDataType*>(kargs.dk_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dk +
                                batch_offset_dk;
        VGradDataType* dv_ptr = reinterpret_cast<VGradDataType*>(kargs.dv_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dv +
                                batch_offset_dv;

        const auto q_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                q_ptr, make_tuple(kargs.seqlen_q, kargs.hdim_qk), make_tuple(kargs.stride_q, 1),
                number<HstuPipeline::kAlignmentQ>{}, number<1>{}),
            make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});
        const auto k_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                k_ptr, make_tuple(kargs.seqlen_kv, kargs.hdim_qk), make_tuple(kargs.stride_k, 1),
                number<HstuPipeline::kAlignmentK>{}, number<1>{}),
            make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});
        const auto v_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                v_ptr, make_tuple(kargs.seqlen_kv, kargs.hdim_v), make_tuple(kargs.stride_v, 1),
                number<HstuPipeline::kAlignmentV>{}, number<1>{}),
            make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});
        const auto do_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                do_ptr, make_tuple(kargs.seqlen_q, kargs.hdim_v), make_tuple(kargs.stride_do, 1),
                number<HstuPipeline::kAlignmentOGrad>{}, number<1>{}),
            make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});

        // LSE [b,s,h]: seq stride = seq_stride_lse (= num_head, NOT 1) -> explicit
        // strided 1-D view. D [b,h,s]: seq stride = seq_stride_delta (= 1) -> packed.
        const auto lse_dram = make_naive_tensor_view<address_space_enum::global>(
            lse_ptr, make_tuple(kargs.seqlen_q), make_tuple(kargs.seq_stride_lse),
            number<1>{}, number<1>{});
        const auto d_dram = make_naive_tensor_view<address_space_enum::global>(
            d_ptr, make_tuple(kargs.seqlen_q), make_tuple(kargs.seq_stride_delta),
            number<1>{}, number<1>{});

        auto q_dram_window = make_tile_window(
            q_dram, make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kQKHeaddim>{}),
            {0, 0});
        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kQKHeaddim>{}),
            {i_n0, 0});
        auto v_dram_window = make_tile_window(
            v_dram, make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kVHeaddim>{}),
            {i_n0, 0});
        auto do_dram_window = make_tile_window(
            do_dram, make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kVHeaddim>{}),
            {0, 0});
        auto lse_dram_window =
            make_tile_window(lse_dram, make_tuple(number<HstuPipeline::kM0>{}), {0});
        auto d_dram_window =
            make_tile_window(d_dram, make_tuple(number<HstuPipeline::kM0>{}), {0});

        // dQ_acc window — deterministic (set + per-split slot) vs atomic (M6, see SiLU kernel).
        auto dq_dram_window = [&, i_tile_n_ = i_tile_n, i_nhead_ = i_nhead]() {
            constexpr auto mop = kIsDeterministic ? memory_operation_enum::set
                                                  : memory_operation_enum::atomic_add;
            AccDataType* dq_acc_ptr =
                reinterpret_cast<AccDataType*>(kargs.dq_acc_ptr) +
                static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_dq_acc +
                batch_offset_dq_acc;
            if constexpr(kIsDeterministic)
                dq_acc_ptr +=
                    static_cast<long_index_t>(i_tile_n_) * kargs.split_stride_dq_acc;
            auto dq_acc_dram = pad_tensor_view(
                make_naive_tensor_view<address_space_enum::global, mop>(
                    dq_acc_ptr, make_tuple(kargs.seqlen_q, kargs.hdim_qk),
                    make_tuple(kargs.stride_dq_acc, 1), number<HstuPipeline::kAlignmentQGrad>{},
                    number<1>{}),
                make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kQKHeaddim>{}),
                sequence<false, (kPadHeadDimQ > 0)>{});
            return make_tile_window(
                dq_acc_dram,
                make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kQKHeaddim>{}),
                {0, 0});
        }();

        const int num_target =
            (kargs.num_targets_ptr != nullptr)
                ? reinterpret_cast<const int32_t*>(kargs.num_targets_ptr)[i_batch]
                : 0;
        auto mask = [&]() {
            if constexpr(FmhaMask::kUseLocal)
            {
                const int eff_min_full =
                    (kargs.seqlen_q - num_target > kargs.min_full_attn_seqlen)
                        ? kargs.min_full_attn_seqlen
                        : (kargs.seqlen_q - num_target);
                // cross-attention: seqlen_kv into seqlen_k slot (see SiLU site / draft R2/R3).
                if constexpr(FmhaMask::kIsCrossAttention)
                {
                    return hstu_bwd_single::make_hstu_cross_attention_block_mask_with_local<FmhaMask>(
                        /*is_tile_in_first_split=*/true, kargs.seqlen_q, kargs.seqlen_kv,
                        kargs.contextual_seqlen, num_target, kargs.max_attn_len, eff_min_full);
                }
                else
                {
                    return hstu_bwd_single::make_hstu_self_attention_block_mask_with_local<FmhaMask>(
                        /*is_tile_in_first_split=*/true, kargs.seqlen_q, kargs.contextual_seqlen,
                        num_target, kargs.max_attn_len, eff_min_full);
                }
            }
            else
            {
                if constexpr(FmhaMask::kIsCrossAttention)
                {
                    return hstu_bwd_single::make_hstu_cross_attention_block_mask_without_local<FmhaMask>(
                        kargs.seqlen_q, kargs.seqlen_kv, kargs.contextual_seqlen, num_target);
                }
                else
                {
                    return hstu_bwd_single::make_hstu_self_attention_block_mask_without_local<FmhaMask>(
                        kargs.seqlen_q, kargs.contextual_seqlen, num_target);
                }
            }
        }();

        auto [dk_acc_tile, dv_acc_tile] = HstuPipeline{}(q_dram_window,
                                                         k_dram_window,
                                                         v_dram_window,
                                                         do_dram_window,
                                                         lse_dram_window,
                                                         d_dram_window,
                                                         dq_dram_window,
                                                         mask,
                                                         kargs.alpha,
                                                         smem_ptr);

        auto dk_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                dk_ptr, make_tuple(kargs.seqlen_kv, kargs.hdim_qk), make_tuple(kargs.stride_dk, 1),
                number<HstuPipeline::kAlignmentKGrad>{}, number<1>{}),
            make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});
        auto dv_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                dv_ptr, make_tuple(kargs.seqlen_kv, kargs.hdim_v), make_tuple(kargs.stride_dv, 1),
                number<HstuPipeline::kAlignmentVGrad>{}, number<1>{}),
            make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});

        auto dk_dram_window = make_tile_window(
            dk_dram, make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kQKHeaddim>{}),
            {i_n0, 0});
        auto dv_dram_window = make_tile_window(
            dv_dram, make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kVHeaddim>{}),
            {i_n0, 0});

        KGradEpiloguePipeline{}(dk_dram_window, dk_acc_tile, nullptr);
        VGradEpiloguePipeline{}(dv_dram_window, dv_acc_tile, nullptr);
    }
};

// ---------------------------------------------------------------------------
// HSTU bwd MAIN kernel — GROUP mode (M4). group = jagged superset:
//   * dim0=1 token-major packed + cu_seqlens (offset indexing reused from M3);
//   * per-group hyper-params indexed by i_group = i_batch / num_batch_per_group:
//       scale_p = group_attn_scale[i_group] ? : 1/group_max_seqlen_q[i_group];
//       window / contextual / min_full read per-group;
//   * alpha is GLOBAL (single scalar, D6); num_target is per-batch.
//
// Per-group window means kUseLocal cannot be fixed at compile time for the whole
// launch (different groups may be windowed or not). Mirroring the fwd kernel, we
// instantiate BOTH pipelines (with-local + without-local, same kUseCausal) and
// branch at runtime on (window_size > 0). The bwd pipeline only bakes
// FmhaMask::IsMasking, so the two pipelines differ solely in their mask object.
template <typename PipelineLocal_,
          typename PipelineNoLocal_,
          typename KGradEpiloguePipeline_,
          typename VGradEpiloguePipeline_>
struct HstuAttentionBwdDQDKDVGroupKernel
{
    using PipelineLocal         = remove_cvref_t<PipelineLocal_>;
    using PipelineNoLocal       = remove_cvref_t<PipelineNoLocal_>;
    using KGradEpiloguePipeline = remove_cvref_t<KGradEpiloguePipeline_>;
    using VGradEpiloguePipeline = remove_cvref_t<VGradEpiloguePipeline_>;

    // both pipelines share the same shape/Problem (only the Mask type differs)
    using P = PipelineLocal;

    static constexpr index_t kBlockSize  = P::kBlockSize;
    static constexpr index_t kBlockPerCu = P::kBlockPerCu;

    using QDataType     = remove_cvref_t<typename P::QDataType>;
    using KDataType     = remove_cvref_t<typename P::KDataType>;
    using VDataType     = remove_cvref_t<typename P::VDataType>;
    using AccDataType   = remove_cvref_t<typename P::AccDataType>;
    using OGradDataType = remove_cvref_t<typename P::OGradDataType>;
    using KGradDataType = remove_cvref_t<typename P::KGradDataType>;
    using VGradDataType = remove_cvref_t<typename P::VGradDataType>;
    using LocalMask     = remove_cvref_t<typename PipelineLocal::FmhaMask>;
    using NoLocalMask   = remove_cvref_t<typename PipelineNoLocal::FmhaMask>;

    static constexpr index_t kPadHeadDimQ  = P::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV  = P::kPadHeadDimV;
    static constexpr bool kIsDeterministic = P::kIsDeterministic;

    struct Kargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* do_ptr;
        void* dk_ptr;
        void* dv_ptr;
        void* dq_acc_ptr; // float

        const void* seq_q_offsets_ptr;  // int32, size num_batch+1
        const void* seq_kv_offsets_ptr; // int32, size num_batch+1

        // per-group hyper-params (device pointers), indexed by i_group
        const void* group_attn_scale_ptr;
        const void* group_max_seqlen_q_ptr;
        const void* group_window_size_ptr;
        const void* group_contextual_seqlen_ptr;
        const void* group_min_full_attn_seqlen_ptr;
        index_t num_batch_per_group;

        // per-batch num_target (int32); null => 0
        const void* num_targets_ptr;

        index_t hdim_qk;
        index_t hdim_v;
        index_t nhead_ratio_qk;

        float alpha; // global

        index_t stride_q;
        index_t stride_k;
        index_t stride_v;
        index_t stride_do;
        index_t stride_dk;
        index_t stride_dv;
        index_t stride_dq_acc;

        index_t nhead_stride_q;
        index_t nhead_stride_k;
        index_t nhead_stride_v;
        index_t nhead_stride_do;
        index_t nhead_stride_dk;
        index_t nhead_stride_dv;
        index_t nhead_stride_dq_acc;
        index_t split_stride_dq_acc; // M6b deterministic
    };

    CK_TILE_HOST static constexpr Kargs MakeKargs(const void* q_ptr,
                                                  const void* k_ptr,
                                                  const void* v_ptr,
                                                  const void* do_ptr,
                                                  void* dk_ptr,
                                                  void* dv_ptr,
                                                  void* dq_acc_ptr,
                                                  const void* seq_q_offsets_ptr,
                                                  const void* seq_kv_offsets_ptr,
                                                  const void* group_attn_scale_ptr,
                                                  const void* group_max_seqlen_q_ptr,
                                                  const void* group_window_size_ptr,
                                                  const void* group_contextual_seqlen_ptr,
                                                  const void* group_min_full_attn_seqlen_ptr,
                                                  index_t num_batch_per_group,
                                                  const void* num_targets_ptr,
                                                  index_t hdim_qk,
                                                  index_t hdim_v,
                                                  index_t nhead_ratio_qk,
                                                  float alpha,
                                                  index_t stride_q,
                                                  index_t stride_k,
                                                  index_t stride_v,
                                                  index_t stride_do,
                                                  index_t stride_dk,
                                                  index_t stride_dv,
                                                  index_t stride_dq_acc,
                                                  index_t nhead_stride_q,
                                                  index_t nhead_stride_k,
                                                  index_t nhead_stride_v,
                                                  index_t nhead_stride_do,
                                                  index_t nhead_stride_dk,
                                                  index_t nhead_stride_dv,
                                                  index_t nhead_stride_dq_acc,
                                                  index_t split_stride_dq_acc)
    {
        Kargs k;
        k.q_ptr                         = q_ptr;
        k.k_ptr                         = k_ptr;
        k.v_ptr                         = v_ptr;
        k.do_ptr                        = do_ptr;
        k.dk_ptr                        = dk_ptr;
        k.dv_ptr                        = dv_ptr;
        k.dq_acc_ptr                    = dq_acc_ptr;
        k.seq_q_offsets_ptr             = seq_q_offsets_ptr;
        k.seq_kv_offsets_ptr            = seq_kv_offsets_ptr;
        k.group_attn_scale_ptr          = group_attn_scale_ptr;
        k.group_max_seqlen_q_ptr        = group_max_seqlen_q_ptr;
        k.group_window_size_ptr         = group_window_size_ptr;
        k.group_contextual_seqlen_ptr   = group_contextual_seqlen_ptr;
        k.group_min_full_attn_seqlen_ptr = group_min_full_attn_seqlen_ptr;
        k.num_batch_per_group           = num_batch_per_group;
        k.num_targets_ptr               = num_targets_ptr;
        k.hdim_qk                       = hdim_qk;
        k.hdim_v                        = hdim_v;
        k.nhead_ratio_qk                = nhead_ratio_qk;
        k.alpha                         = alpha;
        k.stride_q                      = stride_q;
        k.stride_k                      = stride_k;
        k.stride_v                      = stride_v;
        k.stride_do                     = stride_do;
        k.stride_dk                     = stride_dk;
        k.stride_dv                     = stride_dv;
        k.stride_dq_acc                 = stride_dq_acc;
        k.nhead_stride_q                = nhead_stride_q;
        k.nhead_stride_k                = nhead_stride_k;
        k.nhead_stride_v                = nhead_stride_v;
        k.nhead_stride_do               = nhead_stride_do;
        k.nhead_stride_dk               = nhead_stride_dk;
        k.nhead_stride_dv               = nhead_stride_dv;
        k.nhead_stride_dq_acc           = nhead_stride_dq_acc;
        k.split_stride_dq_acc           = split_stride_dq_acc;
        return k;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(index_t batch_size, index_t nhead, index_t max_seqlen_kv)
    {
        return dim3(integer_divide_ceil(max_seqlen_kv, P::kN0), nhead, batch_size);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex()
    {
        return make_tuple(static_cast<index_t>(blockIdx.x),
                          static_cast<index_t>(blockIdx.y),
                          static_cast<index_t>(blockIdx.z));
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(PipelineLocal::GetSmemSize(),
                   PipelineNoLocal::GetSmemSize(),
                   KGradEpiloguePipeline::GetSmemSize(),
                   VGradEpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        __shared__ char smem_ptr[GetSmemSize()];

        const auto [i_tile_n, i_nhead, i_batch] = GetTileIndex();
        const index_t i_n0 = __builtin_amdgcn_readfirstlane(i_tile_n * P::kN0);

        // jagged offsets (token-major packed; same as M3)
        const auto* q_offsets  = reinterpret_cast<const int32_t*>(kargs.seq_q_offsets_ptr);
        const auto* kv_offsets = reinterpret_cast<const int32_t*>(kargs.seq_kv_offsets_ptr);
        const long_index_t query_start = q_offsets[i_batch];
        const long_index_t key_start   = kv_offsets[i_batch];
        const index_t seqlen_q  = q_offsets[i_batch + 1] - q_offsets[i_batch];
        const index_t seqlen_kv = kv_offsets[i_batch + 1] - kv_offsets[i_batch];

        // grid.x sized to the largest group seqlen -> early-exit OOB KV tiles
        if(i_n0 >= seqlen_kv)
            return;

        // per-group hyper-params (D6): i_group = i_batch / num_batch_per_group
        const index_t i_group =
            __builtin_amdgcn_readfirstlane(i_batch / kargs.num_batch_per_group);
        const float group_attn_scale =
            reinterpret_cast<const float*>(kargs.group_attn_scale_ptr)[i_group];
        const index_t group_max_seqlen_q =
            reinterpret_cast<const int32_t*>(kargs.group_max_seqlen_q_ptr)[i_group];
        const index_t window_size =
            reinterpret_cast<const int32_t*>(kargs.group_window_size_ptr)[i_group];
        const index_t contextual_seqlen =
            reinterpret_cast<const int32_t*>(kargs.group_contextual_seqlen_ptr)[i_group];
        const index_t min_full_attn_seqlen =
            reinterpret_cast<const int32_t*>(kargs.group_min_full_attn_seqlen_ptr)[i_group];
        const float scale_p =
            (group_attn_scale != 0.f) ? group_attn_scale
                                      : 1.0f / static_cast<float>(group_max_seqlen_q);

        const int num_target =
            (kargs.num_targets_ptr != nullptr)
                ? reinterpret_cast<const int32_t*>(kargs.num_targets_ptr)[i_batch]
                : 0;

        // per-(batch,head) base offsets
        const long_index_t batch_offset_q      = query_start * kargs.stride_q;
        const long_index_t batch_offset_k      = key_start * kargs.stride_k;
        const long_index_t batch_offset_v      = key_start * kargs.stride_v;
        const long_index_t batch_offset_do     = query_start * kargs.stride_do;
        const long_index_t batch_offset_dk     = key_start * kargs.stride_dk;
        const long_index_t batch_offset_dv     = key_start * kargs.stride_dv;
        const long_index_t batch_offset_dq_acc = query_start * kargs.stride_dq_acc;

        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
                                      batch_offset_do;
        KGradDataType* dk_ptr = reinterpret_cast<KGradDataType*>(kargs.dk_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dk +
                                batch_offset_dk;
        VGradDataType* dv_ptr = reinterpret_cast<VGradDataType*>(kargs.dv_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dv +
                                batch_offset_dv;

        const auto q_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                q_ptr, make_tuple(seqlen_q, kargs.hdim_qk), make_tuple(kargs.stride_q, 1),
                number<P::kAlignmentQ>{}, number<1>{}),
            make_tuple(number<P::kM0>{}, number<P::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});
        const auto k_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                k_ptr, make_tuple(seqlen_kv, kargs.hdim_qk), make_tuple(kargs.stride_k, 1),
                number<P::kAlignmentK>{}, number<1>{}),
            make_tuple(number<P::kN0>{}, number<P::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});
        const auto v_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                v_ptr, make_tuple(seqlen_kv, kargs.hdim_v), make_tuple(kargs.stride_v, 1),
                number<P::kAlignmentV>{}, number<1>{}),
            make_tuple(number<P::kN0>{}, number<P::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});
        const auto do_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                do_ptr, make_tuple(seqlen_q, kargs.hdim_v), make_tuple(kargs.stride_do, 1),
                number<P::kAlignmentOGrad>{}, number<1>{}),
            make_tuple(number<P::kM0>{}, number<P::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});

        auto q_dram_window = make_tile_window(
            q_dram, make_tuple(number<P::kM0>{}, number<P::kQKHeaddim>{}), {0, 0});
        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<P::kN0>{}, number<P::kQKHeaddim>{}), {i_n0, 0});
        auto v_dram_window = make_tile_window(
            v_dram, make_tuple(number<P::kN0>{}, number<P::kVHeaddim>{}), {i_n0, 0});
        auto do_dram_window = make_tile_window(
            do_dram, make_tuple(number<P::kM0>{}, number<P::kVHeaddim>{}), {0, 0});

        // dQ_acc window — determ (set + per-split slot) vs atomic (M6b; see no_group kernel).
        // group packed base = query_start*stride + i_nhead*nhead_stride; determ adds split slot.
        auto dq_dram_window = [&, i_tile_n_ = i_tile_n]() {
            constexpr auto mop = kIsDeterministic ? memory_operation_enum::set
                                                  : memory_operation_enum::atomic_add;
            AccDataType* dq_acc_ptr =
                reinterpret_cast<AccDataType*>(kargs.dq_acc_ptr) +
                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dq_acc +
                batch_offset_dq_acc;
            if constexpr(kIsDeterministic)
                dq_acc_ptr +=
                    static_cast<long_index_t>(i_tile_n_) * kargs.split_stride_dq_acc;
            auto dq_acc_dram = pad_tensor_view(
                make_naive_tensor_view<address_space_enum::global, mop>(
                    dq_acc_ptr, make_tuple(seqlen_q, kargs.hdim_qk),
                    make_tuple(kargs.stride_dq_acc, 1), number<P::kAlignmentQGrad>{}, number<1>{}),
                make_tuple(number<P::kM0>{}, number<P::kQKHeaddim>{}),
                sequence<false, (kPadHeadDimQ > 0)>{});
            return make_tile_window(
                dq_acc_dram, make_tuple(number<P::kM0>{}, number<P::kQKHeaddim>{}), {0, 0});
        }();

        // mask-independent dk/dv write-back (shared by both branches)
        auto write_dkdv = [&](auto& dk_acc_tile, auto& dv_acc_tile) {
            auto dk_dram = pad_tensor_view(
                make_naive_tensor_view<address_space_enum::global>(
                    dk_ptr, make_tuple(seqlen_kv, kargs.hdim_qk), make_tuple(kargs.stride_dk, 1),
                    number<P::kAlignmentKGrad>{}, number<1>{}),
                make_tuple(number<P::kN0>{}, number<P::kQKHeaddim>{}),
                sequence<false, (kPadHeadDimQ > 0)>{});
            auto dv_dram = pad_tensor_view(
                make_naive_tensor_view<address_space_enum::global>(
                    dv_ptr, make_tuple(seqlen_kv, kargs.hdim_v), make_tuple(kargs.stride_dv, 1),
                    number<P::kAlignmentVGrad>{}, number<1>{}),
                make_tuple(number<P::kN0>{}, number<P::kVHeaddim>{}),
                sequence<false, (kPadHeadDimV > 0)>{});
            auto dk_dram_window = make_tile_window(
                dk_dram, make_tuple(number<P::kN0>{}, number<P::kQKHeaddim>{}), {i_n0, 0});
            auto dv_dram_window = make_tile_window(
                dv_dram, make_tuple(number<P::kN0>{}, number<P::kVHeaddim>{}), {i_n0, 0});
            KGradEpiloguePipeline{}(dk_dram_window, dk_acc_tile, nullptr);
            VGradEpiloguePipeline{}(dv_dram_window, dv_acc_tile, nullptr);
        };

        // per-group window decides which mask type / pipeline runs at runtime
        if(window_size > 0)
        {
            // clamp min_full like the reference (reference_hstu_attention_bwd.hpp:198)
            const int eff_min_full = (seqlen_q - num_target > min_full_attn_seqlen)
                                         ? min_full_attn_seqlen
                                         : (seqlen_q - num_target);
            // cross-attention: feed seqlen_kv into the seqlen_k slot via the cross wrapper
            // (draft R2/R3); if constexpr keeps the self leg byte-identical to M7c.
            auto mask = [&]() {
                if constexpr(LocalMask::kIsCrossAttention)
                    return hstu_bwd_single::make_hstu_cross_attention_block_mask_with_local<LocalMask>(
                        /*is_tile_in_first_split=*/true, seqlen_q, seqlen_kv, contextual_seqlen,
                        num_target, window_size, eff_min_full);
                else
                    return hstu_bwd_single::make_hstu_self_attention_block_mask_with_local<LocalMask>(
                        /*is_tile_in_first_split=*/true, seqlen_q, contextual_seqlen, num_target,
                        window_size, eff_min_full);
            }();
            auto [dk_acc_tile, dv_acc_tile] =
                PipelineLocal{}(q_dram_window, k_dram_window, v_dram_window, do_dram_window,
                                dq_dram_window, mask, kargs.alpha, scale_p, smem_ptr);
            write_dkdv(dk_acc_tile, dv_acc_tile);
        }
        else
        {
            auto mask = [&]() {
                if constexpr(NoLocalMask::kIsCrossAttention)
                    return hstu_bwd_single::make_hstu_cross_attention_block_mask_without_local<NoLocalMask>(
                        seqlen_q, seqlen_kv, contextual_seqlen, num_target);
                else
                    return hstu_bwd_single::make_hstu_self_attention_block_mask_without_local<NoLocalMask>(
                        seqlen_q, contextual_seqlen, num_target);
            }();
            auto [dk_acc_tile, dv_acc_tile] =
                PipelineNoLocal{}(q_dram_window, k_dram_window, v_dram_window, do_dram_window,
                                  dq_dram_window, mask, kargs.alpha, scale_p, smem_ptr);
            write_dkdv(dk_acc_tile, dv_acc_tile);
        }
    }
};

// ---------------------------------------------------------------------------
// HSTU bwd MAIN kernel — GROUP + SOFTMAX (M5b). = M4 group kernel (per-group
// hyper-params via i_group + jagged offsets + runtime with-local/without-local
// pipeline branch) FUSED WITH M5 softmax (LSE/D dram windows + softmax pipeline,
// no scale_p). group is always packed (jagged), so LSE/D base = i_nhead*ΣL +
// query_start (seq stride 1), identical to the M5 jagged path.
template <typename PipelineLocal_,
          typename PipelineNoLocal_,
          typename KGradEpiloguePipeline_,
          typename VGradEpiloguePipeline_>
struct HstuAttentionBwdDQDKDVGroupSoftmaxKernel
{
    using PipelineLocal         = remove_cvref_t<PipelineLocal_>;
    using PipelineNoLocal       = remove_cvref_t<PipelineNoLocal_>;
    using KGradEpiloguePipeline = remove_cvref_t<KGradEpiloguePipeline_>;
    using VGradEpiloguePipeline = remove_cvref_t<VGradEpiloguePipeline_>;

    using P = PipelineLocal;

    static constexpr index_t kBlockSize  = P::kBlockSize;
    static constexpr index_t kBlockPerCu = P::kBlockPerCu;

    using QDataType     = remove_cvref_t<typename P::QDataType>;
    using KDataType     = remove_cvref_t<typename P::KDataType>;
    using VDataType     = remove_cvref_t<typename P::VDataType>;
    using AccDataType   = remove_cvref_t<typename P::AccDataType>;
    using OGradDataType = remove_cvref_t<typename P::OGradDataType>;
    using KGradDataType = remove_cvref_t<typename P::KGradDataType>;
    using VGradDataType = remove_cvref_t<typename P::VGradDataType>;
    using LSEDataType   = remove_cvref_t<typename P::LSEDataType>;
    using DDataType     = remove_cvref_t<typename P::DDataType>;
    using LocalMask     = remove_cvref_t<typename PipelineLocal::FmhaMask>;
    using NoLocalMask   = remove_cvref_t<typename PipelineNoLocal::FmhaMask>;

    static constexpr index_t kPadHeadDimQ  = P::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV  = P::kPadHeadDimV;
    static constexpr bool kIsDeterministic = P::kIsDeterministic;

    struct Kargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* do_ptr;
        const void* lse_ptr;
        const void* d_ptr;
        void* dk_ptr;
        void* dv_ptr;
        void* dq_acc_ptr; // float

        const void* seq_q_offsets_ptr;
        const void* seq_kv_offsets_ptr;

        const void* group_attn_scale_ptr;
        const void* group_max_seqlen_q_ptr;
        const void* group_window_size_ptr;
        const void* group_contextual_seqlen_ptr;
        const void* group_min_full_attn_seqlen_ptr;
        index_t num_batch_per_group;

        const void* num_targets_ptr;

        index_t hdim_qk;
        index_t hdim_v;
        index_t nhead_ratio_qk;

        float alpha; // global

        index_t stride_q;
        index_t stride_k;
        index_t stride_v;
        index_t stride_do;
        index_t stride_dk;
        index_t stride_dv;
        index_t stride_dq_acc;

        index_t nhead_stride_q;
        index_t nhead_stride_k;
        index_t nhead_stride_v;
        index_t nhead_stride_do;
        index_t nhead_stride_dk;
        index_t nhead_stride_dv;
        index_t nhead_stride_dq_acc;
        // group is always packed. LSE [ΣL,h] seq_stride = num_head; D [h,ΣL] seq_stride
        // = 1. Split into 4 strides (no batch stride: group uses jagged offsets).
        index_t seq_stride_lse;     // = num_head
        index_t nhead_stride_lse;   // = 1
        index_t seq_stride_delta;   // = 1
        index_t nhead_stride_delta; // = num_batch * max_seqlen_q
        index_t split_stride_dq_acc; // M6b deterministic
    };

    CK_TILE_HOST static constexpr Kargs MakeKargs(const void* q_ptr,
                                                  const void* k_ptr,
                                                  const void* v_ptr,
                                                  const void* do_ptr,
                                                  const void* lse_ptr,
                                                  const void* d_ptr,
                                                  void* dk_ptr,
                                                  void* dv_ptr,
                                                  void* dq_acc_ptr,
                                                  const void* seq_q_offsets_ptr,
                                                  const void* seq_kv_offsets_ptr,
                                                  const void* group_attn_scale_ptr,
                                                  const void* group_max_seqlen_q_ptr,
                                                  const void* group_window_size_ptr,
                                                  const void* group_contextual_seqlen_ptr,
                                                  const void* group_min_full_attn_seqlen_ptr,
                                                  index_t num_batch_per_group,
                                                  const void* num_targets_ptr,
                                                  index_t hdim_qk,
                                                  index_t hdim_v,
                                                  index_t nhead_ratio_qk,
                                                  float alpha,
                                                  index_t stride_q,
                                                  index_t stride_k,
                                                  index_t stride_v,
                                                  index_t stride_do,
                                                  index_t stride_dk,
                                                  index_t stride_dv,
                                                  index_t stride_dq_acc,
                                                  index_t nhead_stride_q,
                                                  index_t nhead_stride_k,
                                                  index_t nhead_stride_v,
                                                  index_t nhead_stride_do,
                                                  index_t nhead_stride_dk,
                                                  index_t nhead_stride_dv,
                                                  index_t nhead_stride_dq_acc,
                                                  index_t seq_stride_lse,
                                                  index_t nhead_stride_lse,
                                                  index_t seq_stride_delta,
                                                  index_t nhead_stride_delta,
                                                  index_t split_stride_dq_acc)
    {
        Kargs k;
        k.q_ptr                          = q_ptr;
        k.k_ptr                          = k_ptr;
        k.v_ptr                          = v_ptr;
        k.do_ptr                         = do_ptr;
        k.lse_ptr                        = lse_ptr;
        k.d_ptr                          = d_ptr;
        k.dk_ptr                         = dk_ptr;
        k.dv_ptr                         = dv_ptr;
        k.dq_acc_ptr                     = dq_acc_ptr;
        k.seq_q_offsets_ptr              = seq_q_offsets_ptr;
        k.seq_kv_offsets_ptr             = seq_kv_offsets_ptr;
        k.group_attn_scale_ptr           = group_attn_scale_ptr;
        k.group_max_seqlen_q_ptr         = group_max_seqlen_q_ptr;
        k.group_window_size_ptr          = group_window_size_ptr;
        k.group_contextual_seqlen_ptr    = group_contextual_seqlen_ptr;
        k.group_min_full_attn_seqlen_ptr = group_min_full_attn_seqlen_ptr;
        k.num_batch_per_group            = num_batch_per_group;
        k.num_targets_ptr                = num_targets_ptr;
        k.hdim_qk                        = hdim_qk;
        k.hdim_v                         = hdim_v;
        k.nhead_ratio_qk                 = nhead_ratio_qk;
        k.alpha                          = alpha;
        k.stride_q                       = stride_q;
        k.stride_k                       = stride_k;
        k.stride_v                       = stride_v;
        k.stride_do                      = stride_do;
        k.stride_dk                      = stride_dk;
        k.stride_dv                      = stride_dv;
        k.stride_dq_acc                  = stride_dq_acc;
        k.nhead_stride_q                 = nhead_stride_q;
        k.nhead_stride_k                 = nhead_stride_k;
        k.nhead_stride_v                 = nhead_stride_v;
        k.nhead_stride_do                = nhead_stride_do;
        k.nhead_stride_dk                = nhead_stride_dk;
        k.nhead_stride_dv                = nhead_stride_dv;
        k.nhead_stride_dq_acc            = nhead_stride_dq_acc;
        k.seq_stride_lse                 = seq_stride_lse;
        k.nhead_stride_lse               = nhead_stride_lse;
        k.seq_stride_delta               = seq_stride_delta;
        k.nhead_stride_delta             = nhead_stride_delta;
        k.split_stride_dq_acc            = split_stride_dq_acc;
        return k;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(index_t batch_size, index_t nhead, index_t max_seqlen_kv)
    {
        return dim3(integer_divide_ceil(max_seqlen_kv, P::kN0), nhead, batch_size);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex()
    {
        return make_tuple(static_cast<index_t>(blockIdx.x),
                          static_cast<index_t>(blockIdx.y),
                          static_cast<index_t>(blockIdx.z));
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(PipelineLocal::GetSmemSize(),
                   PipelineNoLocal::GetSmemSize(),
                   KGradEpiloguePipeline::GetSmemSize(),
                   VGradEpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        __shared__ char smem_ptr[GetSmemSize()];

        const auto [i_tile_n, i_nhead, i_batch] = GetTileIndex();
        const index_t i_n0 = __builtin_amdgcn_readfirstlane(i_tile_n * P::kN0);

        const auto* q_offsets  = reinterpret_cast<const int32_t*>(kargs.seq_q_offsets_ptr);
        const auto* kv_offsets = reinterpret_cast<const int32_t*>(kargs.seq_kv_offsets_ptr);
        const long_index_t query_start = q_offsets[i_batch];
        const long_index_t key_start   = kv_offsets[i_batch];
        const index_t seqlen_q  = q_offsets[i_batch + 1] - q_offsets[i_batch];
        const index_t seqlen_kv = kv_offsets[i_batch + 1] - kv_offsets[i_batch];

        if(i_n0 >= seqlen_kv)
            return;

        // per-group hyper-params (D6). softmax: scale_p unused, but window/contextual/
        // min_full are still per-group; scale_p replaced by LSE.
        const index_t i_group =
            __builtin_amdgcn_readfirstlane(i_batch / kargs.num_batch_per_group);
        const index_t window_size =
            reinterpret_cast<const int32_t*>(kargs.group_window_size_ptr)[i_group];
        const index_t contextual_seqlen =
            reinterpret_cast<const int32_t*>(kargs.group_contextual_seqlen_ptr)[i_group];
        const index_t min_full_attn_seqlen =
            reinterpret_cast<const int32_t*>(kargs.group_min_full_attn_seqlen_ptr)[i_group];

        const int num_target =
            (kargs.num_targets_ptr != nullptr)
                ? reinterpret_cast<const int32_t*>(kargs.num_targets_ptr)[i_batch]
                : 0;

        const long_index_t batch_offset_q      = query_start * kargs.stride_q;
        const long_index_t batch_offset_k      = key_start * kargs.stride_k;
        const long_index_t batch_offset_v      = key_start * kargs.stride_v;
        const long_index_t batch_offset_do     = query_start * kargs.stride_do;
        const long_index_t batch_offset_dk     = key_start * kargs.stride_dk;
        const long_index_t batch_offset_dv     = key_start * kargs.stride_dv;
        const long_index_t batch_offset_dq_acc = query_start * kargs.stride_dq_acc;
        const long_index_t batch_offset_lse    = query_start * kargs.seq_stride_lse;
        const long_index_t batch_offset_delta  = query_start; // D seq stride 1 (packed)

        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
                                      batch_offset_do;
        const LSEDataType* lse_ptr = reinterpret_cast<const LSEDataType*>(kargs.lse_ptr) +
                                     static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lse +
                                     batch_offset_lse;
        const DDataType* d_ptr = reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_delta +
                                 batch_offset_delta;
        KGradDataType* dk_ptr = reinterpret_cast<KGradDataType*>(kargs.dk_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dk +
                                batch_offset_dk;
        VGradDataType* dv_ptr = reinterpret_cast<VGradDataType*>(kargs.dv_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dv +
                                batch_offset_dv;

        const auto q_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                q_ptr, make_tuple(seqlen_q, kargs.hdim_qk), make_tuple(kargs.stride_q, 1),
                number<P::kAlignmentQ>{}, number<1>{}),
            make_tuple(number<P::kM0>{}, number<P::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});
        const auto k_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                k_ptr, make_tuple(seqlen_kv, kargs.hdim_qk), make_tuple(kargs.stride_k, 1),
                number<P::kAlignmentK>{}, number<1>{}),
            make_tuple(number<P::kN0>{}, number<P::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});
        const auto v_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                v_ptr, make_tuple(seqlen_kv, kargs.hdim_v), make_tuple(kargs.stride_v, 1),
                number<P::kAlignmentV>{}, number<1>{}),
            make_tuple(number<P::kN0>{}, number<P::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});
        const auto do_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                do_ptr, make_tuple(seqlen_q, kargs.hdim_v), make_tuple(kargs.stride_do, 1),
                number<P::kAlignmentOGrad>{}, number<1>{}),
            make_tuple(number<P::kM0>{}, number<P::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});

        // LSE seq stride = seq_stride_lse (= num_head); D seq stride = seq_stride_delta (= 1).
        const auto lse_dram = make_naive_tensor_view<address_space_enum::global>(
            lse_ptr, make_tuple(seqlen_q), make_tuple(kargs.seq_stride_lse),
            number<1>{}, number<1>{});
        const auto d_dram = make_naive_tensor_view<address_space_enum::global>(
            d_ptr, make_tuple(seqlen_q), make_tuple(kargs.seq_stride_delta),
            number<1>{}, number<1>{});

        auto q_dram_window = make_tile_window(
            q_dram, make_tuple(number<P::kM0>{}, number<P::kQKHeaddim>{}), {0, 0});
        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<P::kN0>{}, number<P::kQKHeaddim>{}), {i_n0, 0});
        auto v_dram_window = make_tile_window(
            v_dram, make_tuple(number<P::kN0>{}, number<P::kVHeaddim>{}), {i_n0, 0});
        auto do_dram_window = make_tile_window(
            do_dram, make_tuple(number<P::kM0>{}, number<P::kVHeaddim>{}), {0, 0});
        auto lse_dram_window = make_tile_window(lse_dram, make_tuple(number<P::kM0>{}), {0});
        auto d_dram_window   = make_tile_window(d_dram, make_tuple(number<P::kM0>{}), {0});

        // dQ_acc window — determ (set + per-split slot) vs atomic (M6b; see no_group kernel).
        // group packed base = query_start*stride + i_nhead*nhead_stride; determ adds split slot.
        auto dq_dram_window = [&, i_tile_n_ = i_tile_n]() {
            constexpr auto mop = kIsDeterministic ? memory_operation_enum::set
                                                  : memory_operation_enum::atomic_add;
            AccDataType* dq_acc_ptr =
                reinterpret_cast<AccDataType*>(kargs.dq_acc_ptr) +
                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dq_acc +
                batch_offset_dq_acc;
            if constexpr(kIsDeterministic)
                dq_acc_ptr +=
                    static_cast<long_index_t>(i_tile_n_) * kargs.split_stride_dq_acc;
            auto dq_acc_dram = pad_tensor_view(
                make_naive_tensor_view<address_space_enum::global, mop>(
                    dq_acc_ptr, make_tuple(seqlen_q, kargs.hdim_qk),
                    make_tuple(kargs.stride_dq_acc, 1), number<P::kAlignmentQGrad>{}, number<1>{}),
                make_tuple(number<P::kM0>{}, number<P::kQKHeaddim>{}),
                sequence<false, (kPadHeadDimQ > 0)>{});
            return make_tile_window(
                dq_acc_dram, make_tuple(number<P::kM0>{}, number<P::kQKHeaddim>{}), {0, 0});
        }();

        auto write_dkdv = [&](auto& dk_acc_tile, auto& dv_acc_tile) {
            auto dk_dram = pad_tensor_view(
                make_naive_tensor_view<address_space_enum::global>(
                    dk_ptr, make_tuple(seqlen_kv, kargs.hdim_qk), make_tuple(kargs.stride_dk, 1),
                    number<P::kAlignmentKGrad>{}, number<1>{}),
                make_tuple(number<P::kN0>{}, number<P::kQKHeaddim>{}),
                sequence<false, (kPadHeadDimQ > 0)>{});
            auto dv_dram = pad_tensor_view(
                make_naive_tensor_view<address_space_enum::global>(
                    dv_ptr, make_tuple(seqlen_kv, kargs.hdim_v), make_tuple(kargs.stride_dv, 1),
                    number<P::kAlignmentVGrad>{}, number<1>{}),
                make_tuple(number<P::kN0>{}, number<P::kVHeaddim>{}),
                sequence<false, (kPadHeadDimV > 0)>{});
            auto dk_dram_window = make_tile_window(
                dk_dram, make_tuple(number<P::kN0>{}, number<P::kQKHeaddim>{}), {i_n0, 0});
            auto dv_dram_window = make_tile_window(
                dv_dram, make_tuple(number<P::kN0>{}, number<P::kVHeaddim>{}), {i_n0, 0});
            KGradEpiloguePipeline{}(dk_dram_window, dk_acc_tile, nullptr);
            VGradEpiloguePipeline{}(dv_dram_window, dv_acc_tile, nullptr);
        };

        // per-group window decides which mask type / softmax pipeline runs at runtime
        if(window_size > 0)
        {
            const int eff_min_full = (seqlen_q - num_target > min_full_attn_seqlen)
                                         ? min_full_attn_seqlen
                                         : (seqlen_q - num_target);
            // cross-attention: feed seqlen_kv into the seqlen_k slot via the cross wrapper
            // (draft R2/R3); if constexpr keeps the self leg byte-identical to M7c.
            auto mask = [&]() {
                if constexpr(LocalMask::kIsCrossAttention)
                    return hstu_bwd_single::make_hstu_cross_attention_block_mask_with_local<LocalMask>(
                        /*is_tile_in_first_split=*/true, seqlen_q, seqlen_kv, contextual_seqlen,
                        num_target, window_size, eff_min_full);
                else
                    return hstu_bwd_single::make_hstu_self_attention_block_mask_with_local<LocalMask>(
                        /*is_tile_in_first_split=*/true, seqlen_q, contextual_seqlen, num_target,
                        window_size, eff_min_full);
            }();
            auto [dk_acc_tile, dv_acc_tile] =
                PipelineLocal{}(q_dram_window, k_dram_window, v_dram_window, do_dram_window,
                                lse_dram_window, d_dram_window, dq_dram_window, mask, kargs.alpha,
                                smem_ptr);
            write_dkdv(dk_acc_tile, dv_acc_tile);
        }
        else
        {
            auto mask = [&]() {
                if constexpr(NoLocalMask::kIsCrossAttention)
                    return hstu_bwd_single::make_hstu_cross_attention_block_mask_without_local<NoLocalMask>(
                        seqlen_q, seqlen_kv, contextual_seqlen, num_target);
                else
                    return hstu_bwd_single::make_hstu_self_attention_block_mask_without_local<NoLocalMask>(
                        seqlen_q, contextual_seqlen, num_target);
            }();
            auto [dk_acc_tile, dv_acc_tile] =
                PipelineNoLocal{}(q_dram_window, k_dram_window, v_dram_window, do_dram_window,
                                  lse_dram_window, d_dram_window, dq_dram_window, mask, kargs.alpha,
                                  smem_ptr);
            write_dkdv(dk_acc_tile, dv_acc_tile);
        }
    }
};

// PRE (softmax path): D[b,h,sq] = Σ_v O[b,sq,h,v] * dO[b,sq,h,v]  (DESIGN §1.1).
//
// PRECONDITION (M6b): the launch grid is bounded by `max_seqlen_q`; a thread whose sq is
// >= its batch's seqlen returns, BUT tokens in [max_seqlen_q, seqlen_q) are never visited.
// Therefore the caller MUST pass max_seqlen_q >= every batch's packed seqlen, otherwise the
// tail tokens of the longest batch get NO D written (d_ptr is intentionally not memset — a
// D=0 fallback would be deterministically-wrong, not correct) and dQ for those rows is wrong.
// The harness enforces this with an assert; group_max_seqlens_q must be the true per-batch max.
//
// One thread per (i_batch, i_nhead, sq) row. O/dO are in their [batch,seq,head,hdim]
// layout (jagged: [1,ΣL,head,hdim] via q offsets); D is written [batch,head,seq]
// (seq-continuous, d_nhead_stride = nhead_stride_delta, d_batch_stride =
// batch_stride_delta) — the exact layout the softmax MAIN reads. The caller must
// pass the delta (D) strides here, NOT the LSE strides (LSE has a different layout).
// hdim_v=64 is small, so a single-thread accumulate is fine.
template <typename InOutDataType, typename DDataType>
__global__ void hstu_bwd_dot_do_o_kernel(const InOutDataType* __restrict__ o_ptr,
                                         const InOutDataType* __restrict__ do_ptr,
                                         DDataType* __restrict__ d_ptr,
                                         bool is_jagged,
                                         const int32_t* __restrict__ seq_q_offsets,
                                         int num_batch,
                                         int num_head,
                                         int max_seqlen_q,
                                         int hdim_v,
                                         long_index_t o_seq_stride,
                                         long_index_t o_nhead_stride,
                                         long_index_t o_batch_stride,
                                         long_index_t d_nhead_stride,
                                         long_index_t d_batch_stride)
{
    const long_index_t tid =
        static_cast<long_index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // decode (i_batch, i_nhead, sq) over a [num_batch, num_head, max_seqlen_q] grid
    const long_index_t total = static_cast<long_index_t>(num_batch) * num_head * max_seqlen_q;
    if(tid >= total)
        return;
    const int sq      = static_cast<int>(tid % max_seqlen_q);
    const long_index_t t1 = tid / max_seqlen_q;
    const int i_nhead = static_cast<int>(t1 % num_head);
    const int i_batch = static_cast<int>(t1 / num_head);

    int seqlen_q;
    long_index_t o_base;   // base for O/dO at (batch, sq, head, 0)
    long_index_t d_base;   // base for D at (batch, head, sq)
    if(is_jagged)
    {
        const int q_start = seq_q_offsets[i_batch];
        seqlen_q          = seq_q_offsets[i_batch + 1] - q_start;
        const long_index_t token = q_start + sq;
        o_base = token * o_seq_stride + static_cast<long_index_t>(i_nhead) * o_nhead_stride;
        d_base = static_cast<long_index_t>(i_nhead) * d_nhead_stride + token; // seq stride 1
    }
    else
    {
        seqlen_q = max_seqlen_q;
        o_base   = static_cast<long_index_t>(i_batch) * o_batch_stride +
                 static_cast<long_index_t>(sq) * o_seq_stride +
                 static_cast<long_index_t>(i_nhead) * o_nhead_stride;
        d_base = static_cast<long_index_t>(i_batch) * d_batch_stride +
                 static_cast<long_index_t>(i_nhead) * d_nhead_stride + sq; // seq stride 1
    }
    if(sq >= seqlen_q)
        return;

    float acc = 0.f;
    for(int v = 0; v < hdim_v; ++v)
    {
        acc += type_convert<float>(o_ptr[o_base + v]) * type_convert<float>(do_ptr[o_base + v]);
    }
    d_ptr[d_base] = type_convert<DDataType>(acc);
}

// POST (atomic path): convert-only dq_acc(float) -> dQ(bf16/fp16).
// M1 atomic path: nsplits=1 and dq_acc shares dQ's layout, so the convert is a
// pure elementwise cast over the full contiguous buffer. Templated so it has
// vague linkage (safe to include in multiple TUs). The deterministic
// reduce+convert path (BlockFmhaBwdConvertQGrad) is M6.
template <typename QGradDataType, typename AccDataType>
__global__ void hstu_bwd_convert_dq_kernel(const AccDataType* __restrict__ dq_acc,
                                           QGradDataType* __restrict__ dq,
                                           long_index_t n)
{
    const long_index_t i =
        static_cast<long_index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n)
        dq[i] = type_convert<QGradDataType>(dq_acc[i]);
}

// POST (deterministic path): reduce dQ over the per-KV-block split slots, then convert.
// dq_acc holds num_splits stacked single-slot copies (stride = split_stride). The
// summation order is fixed (s = 0..num_splits-1) regardless of block scheduling, so
// the result is bit-reproducible. n = single-slot element count. Atomic path keeps the
// plain convert above (num_splits=1).
template <typename QGradDataType, typename AccDataType>
__global__ void hstu_bwd_reduce_convert_dq_kernel(const AccDataType* __restrict__ dq_acc,
                                                  QGradDataType* __restrict__ dq,
                                                  long_index_t n,
                                                  int num_splits,
                                                  long_index_t split_stride)
{
    const long_index_t i =
        static_cast<long_index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    float acc = 0.f;
    for(int s = 0; s < num_splits; ++s)
        acc += type_convert<float>(dq_acc[static_cast<long_index_t>(s) * split_stride + i]);
    dq[i] = type_convert<QGradDataType>(acc);
}

} // namespace ck_tile
