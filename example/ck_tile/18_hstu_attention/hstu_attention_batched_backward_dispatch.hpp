// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/host/hip_check_error.hpp>
#include <ck_tile/ops/epilogue.hpp>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_bwd_type_config.hpp"
#include "hstu_attention_bwd_setting.hpp"
#include "hstu_attention_bwd_helper.hpp"
#include "hstu_attention_params.hpp"
#include "hstu_attention_hdim_switch.hpp"
#include "hstu_attention_pipeline_problem.hpp"
#include "hstu_attention_traits.hpp"
#include "hstu_attention_no_softmax_bwd_pipeline_dq.hpp"
#include "hstu_attention_with_softmax_bwd_pipeline_dq_delta.hpp"
#include "hstu_attention_no_softmax_bwd_pipeline_dk_dv.hpp"
#include "hstu_attention_with_softmax_bwd_pipeline_dk_dv.hpp"
#include "hstu_attention_bwd_kernel_1.hpp"
#include "hstu_attention_bwd_kernel_2.hpp"
#include "hstu_attention_epilogue.hpp"

// Backward dispatch for batched (non-jagged, non-group) mode.
// Backward never uses split-kv: training always provides sufficient CU coverage.

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kHasDropout,
          ck_tile::index_t MaxK>
struct batched_backward_dispatch
{
    using HstuAttentionTileSettingForKernel1 =
        typename HstuAttentionBwdTileSettingForKernel1<MaxK>::Type;
    using HstuAttentionTileSettingForKernel2 =
        typename HstuAttentionBwdTileSettingForKernel2<MaxK>::Type;

    template <bool kIsCrossAttention>
    using HstuBwdBaseProblemTemp = ck_tile::HstuAttentionBwdPipelineBaseProblem<
        InOutDataType,
        typename HstuAttentionBwdTypeConfig<InOutDataType>::GemmAccDataType,
        typename HstuAttentionBwdTypeConfig<InOutDataType>::CompDataType,
        kIsCrossAttention,
        false, // kUseGroup
        false, // kIsJagged
        kHasBias,
        kUseCausal,
        kUseSoftmax,
        kHasDropout>;

    template <bool kIsCrossAttention>
    using HstuBwdPipelineProblemForKernel1Temp = ck_tile::HstuAttentionBwdPipelineProblemForKernel1<
        HstuBwdBaseProblemTemp<kIsCrossAttention>,
        HstuAttentionTileSettingForKernel1>;

    template <bool kIsCrossAttention>
    using HstuBwdPipelineProblemForKernel2Temp = ck_tile::HstuAttentionBwdPipelineProblemForKernel2<
        HstuBwdBaseProblemTemp<kIsCrossAttention>,
        HstuAttentionTileSettingForKernel2>;

    static void Run(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
    {
        BwdWorkspace ws;

        constexpr ck_tile::index_t occupancy_k1 = -1;
        constexpr ck_tile::index_t occupancy_k2 = -1;

        const bool pad_seqlen_k = !(param.seqlen_kv % HstuAttentionTileSettingForKernel1::kN0 == 0);
        const bool pad_headdim_qk =
            !(param.hdim_qk % HstuAttentionTileSettingForKernel1::kQKHeaddim == 0);
        const bool pad_headdim_v =
            !(param.hdim_v % HstuAttentionTileSettingForKernel1::kVHeaddim == 0);

        // seqlen_q is not along the fastest memory dim; OOB is handled by the buffer instructions
        constexpr bool kPadSeqLenQ = false;

        BOOL_SWITCH_3(
            pad_seqlen_k,
            kPadSeqLenK,
            pad_headdim_qk,
            kPadHeadDimQK,
            pad_headdim_v,
            kPadHeadDimV,
            [&] {
                using HstuTraits = ck_tile::HstuAttentionBwdTraits<kPadSeqLenQ,
                                                                   kPadSeqLenK,
                                                                   kPadHeadDimQK,
                                                                   kPadHeadDimV,
                                                                   occupancy_k1,
                                                                   occupancy_k2>;

                using HstuEpilogue =
                    ck_tile::NRepetitions2DEpilogue<ck_tile::Default2DEpilogueProblem<
                        typename HstuAttentionBwdTypeConfig<InOutDataType>::GemmAccDataType,
                        InOutDataType,
                        kPadSeqLenQ,
                        kPadHeadDimQK>>;

                BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, [&] {
                    using HstuPipelineProblemForKernel1 =
                        HstuBwdPipelineProblemForKernel1Temp<kIsCrossAttention>;
                    using HstuPipelineProblemForKernel2 =
                        HstuBwdPipelineProblemForKernel2Temp<kIsCrossAttention>;

                    using HstuPipelineKernel1 =
                        std::conditional_t<kUseSoftmax,
                                           ck_tile::HstuAttentionWithSoftmaxBwdPipelineQRKSVS_dQ_D<
                                               HstuPipelineProblemForKernel1,
                                               HstuTraits>,
                                           ck_tile::HstuAttentionNoSoftmaxBwdPipelineQRKSVS_dQ<
                                               HstuPipelineProblemForKernel1,
                                               HstuTraits>>;

                    using HstuPipelineKernel2 =
                        std::conditional_t<kUseSoftmax,
                                           ck_tile::HstuAttentionWithSoftmaxBwdPipelineKRVRQS_dK_dV<
                                               HstuPipelineProblemForKernel2,
                                               HstuTraits>,
                                           ck_tile::HstuAttentionNoSoftmaxBwdPipelineKRVRQS_dK_dV<
                                               HstuPipelineProblemForKernel2,
                                               HstuTraits>>;

                    using HstuKernel1 =
                        ck_tile::HstuAttentionBwdKernel1<HstuPipelineKernel1, HstuEpilogue>;
                    using HstuKernel2 =
                        ck_tile::HstuAttentionBwdKernel2<HstuPipelineKernel2, HstuEpilogue>;

                    RunWithKernels<HstuKernel1, HstuKernel2>(param, ws, stream);
                });
            });
    }

    template <typename HstuKernel1, typename HstuKernel2>
    static void
    RunWithKernels(HstuAttentionNoGroupBwdParams& param, BwdWorkspace& ws, hipStream_t stream)
    {
        // Delta layout: [num_batch, num_head, seqlen_q]
        // seq_stride=1, nhead_stride=seqlen_q, batch_stride=num_head*seqlen_q
        const ck_tile::index_t seq_stride_delta   = 1;
        const ck_tile::index_t nhead_stride_delta = param.seqlen_q;
        const ck_tile::index_t batch_stride_delta = param.num_head * param.seqlen_q;

        if constexpr(kUseSoftmax)
        {
            // delta is only written/read in the softmax path; allocate stream-ordered.
            const size_t delta_bytes = static_cast<size_t>(param.num_batch) *
                                       static_cast<size_t>(param.num_head) *
                                       static_cast<size_t>(param.seqlen_q) * sizeof(float);
            HIP_CHECK_ERROR(hipMallocAsync(&ws.delta_ptr, delta_bytes, stream));
        }

        // ---- Kernel 1: compute dQ (and delta = D[sq] for softmax path) ----
        {
            const auto kargs = [&] {
                return HstuKernel1::MakeKargs(param.q_ptr,
                                              param.k_ptr,
                                              param.v_ptr,
                                              param.bias_ptr,
                                              param.o_ptr,
                                              param.do_ptr,
                                              param.dq_ptr,
                                              param.lse_ptr,
                                              ws.delta_ptr,
                                              param.seqlen_q,
                                              param.is_cross_attention ? param.seqlen_kv
                                                                       : param.seqlen_q,
                                              param.hdim_qk,
                                              param.hdim_v,
                                              param.num_head,
                                              param.scale_s,
                                              param.attn_scale,
                                              param.seq_stride_q,
                                              param.seq_stride_k,
                                              param.seq_stride_v,
                                              param.seq_stride_o,
                                              param.seq_stride_do,
                                              param.seq_stride_dq,
                                              param.seq_stride_lse,
                                              seq_stride_delta,
                                              param.seq_stride_bias,
                                              param.nhead_stride_q,
                                              param.nhead_stride_k,
                                              param.nhead_stride_v,
                                              param.nhead_stride_o,
                                              param.nhead_stride_do,
                                              param.nhead_stride_dq,
                                              param.nhead_stride_lse,
                                              nhead_stride_delta,
                                              param.nhead_stride_bias,
                                              param.batch_stride_q,
                                              param.batch_stride_k,
                                              param.batch_stride_v,
                                              param.batch_stride_o,
                                              param.batch_stride_do,
                                              param.batch_stride_dq,
                                              param.batch_stride_lse,
                                              batch_stride_delta,
                                              param.batch_stride_bias,
                                              param.num_targets_ptr,
                                              param.contextual_seqlen,
                                              param.window_size,
                                              param.min_full_attn_seqlen,
                                              param.p_drop,
                                              param.philox_seed,
                                              param.philox_offset);
            }();

            bool has_minfull_attn_seqlen           = (param.min_full_attn_seqlen > 0);
            dim3 kGridSize                         = HstuKernel1::GridSize(param.num_batch,
                                                   param.num_head,
                                                   param.seqlen_q,
                                                   true /*almost_invariant_seqlen */,
                                                   has_minfull_attn_seqlen);
            dim3 kBlockSize                        = HstuKernel1::BlockSize();
            constexpr ck_tile::index_t kBlockPerCu = HstuKernel1::kBlockPerCu;

            (void)ck_tile::launch_kernel(
                ck_tile::stream_config{stream, false},
                ck_tile::make_kernel<kBlockPerCu>(HstuKernel1{}, kGridSize, kBlockSize, 0, kargs));
        }

        // ---- Kernel 2: compute dK and dV ----
        {
            const auto kargs = [&] {
                return HstuKernel2::MakeKargs(param.q_ptr,
                                              param.k_ptr,
                                              param.v_ptr,
                                              param.bias_ptr,
                                              param.do_ptr,
                                              param.dk_ptr,
                                              param.dv_ptr,
                                              param.lse_ptr,
                                              ws.delta_ptr,
                                              param.seqlen_q,
                                              param.is_cross_attention ? param.seqlen_kv
                                                                       : param.seqlen_q,
                                              param.hdim_qk,
                                              param.hdim_v,
                                              param.num_head,
                                              param.scale_s,
                                              param.attn_scale,
                                              param.seq_stride_q,
                                              param.seq_stride_k,
                                              param.seq_stride_v,
                                              param.seq_stride_do,
                                              param.seq_stride_dk,
                                              param.seq_stride_dv,
                                              param.seq_stride_lse,
                                              seq_stride_delta,
                                              param.seq_stride_bias,
                                              param.nhead_stride_q,
                                              param.nhead_stride_k,
                                              param.nhead_stride_v,
                                              param.nhead_stride_do,
                                              param.nhead_stride_dk,
                                              param.nhead_stride_dv,
                                              param.nhead_stride_lse,
                                              nhead_stride_delta,
                                              param.nhead_stride_bias,
                                              param.batch_stride_q,
                                              param.batch_stride_k,
                                              param.batch_stride_v,
                                              param.batch_stride_do,
                                              param.batch_stride_dk,
                                              param.batch_stride_dv,
                                              param.batch_stride_lse,
                                              batch_stride_delta,
                                              param.batch_stride_bias,
                                              param.num_targets_ptr,
                                              param.contextual_seqlen,
                                              param.window_size,
                                              param.min_full_attn_seqlen,
                                              param.p_drop,
                                              param.philox_seed,
                                              param.philox_offset);
            }();

            const ck_tile::index_t seqlen_kv =
                param.is_cross_attention ? param.seqlen_kv : param.seqlen_q;
            dim3 kGridSize = HstuKernel2::GridSize(
                param.num_batch, param.num_head, seqlen_kv, true /* almost_invariant_seqlen*/);
            dim3 kBlockSize                        = HstuKernel2::BlockSize();
            constexpr ck_tile::index_t kBlockPerCu = HstuKernel2::kBlockPerCu;

            (void)ck_tile::launch_kernel(
                ck_tile::stream_config{stream, false},
                ck_tile::make_kernel<kBlockPerCu>(HstuKernel2{}, kGridSize, kBlockSize, 0, kargs));
        }

        if constexpr(kUseSoftmax)
        {
            HIP_CHECK_ERROR(hipFreeAsync(ws.delta_ptr, stream));
        }
    }
};

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kHasDropout,
          ck_tile::index_t MaxK>
void run_batched_backward_dispatch(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
{
    batched_backward_dispatch<InOutDataType, kUseCausal, kUseSoftmax, kHasBias, kHasDropout, MaxK>::
        Run(param, stream);
}

#if defined(HSTU_BWD_SINGLE_KERNEL)
// ---- single-kernel bwd extra includes (base double-kernel path does not need these) ----
#include <stdexcept>
#include "ck_tile/core.hpp"
// Narrow fmha sub-headers instead of the `ck_tile/ops/fmha.hpp` aggregate: the
// aggregate pulls in kernel/fmha_fwd_kernel.hpp, whose `has_use_trload_flag`
// collides with the local copy in hstu_attention_kernel_util.hpp.
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_problem.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp"
#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_bwd_shape.hpp"
#include "hstu_attention_bwd_perf.hpp"
#include "hstu_attention_no_softmax_bwd_pipeline.hpp"
#include "hstu_attention_with_softmax_bwd_pipeline.hpp"
#include "hstu_attention_bwd_kernel.hpp"
// ===========================================================================
// Single-kernel bwd (OURS) — inlined behind HSTU_BWD_SINGLE_KERNEL.
//
// This is a straight port of OURS' batched_backward_dispatch
// (ck-dropout-wt @ a86529dc). The whole struct + free fn are RENAMED to the
// *_single_dispatch spelling so the mangled symbol never collides with the base
// batched_backward_dispatch above: base's 5th template axis is kHasDropout,
// ours is kIsDeterministic — same arity, opposite meaning, so identical names
// would ODR-clash. Renaming both layers is the fix.
//
// Field mapping vs OURS' own params (base params has no nhead_ratio_qk / alpha /
// d_ptr / perf_* / measure_perf; base LSE/D strides are already split, S1/S6):
//   * nhead_ratio_qk  -> constant 1     (HSTU is MHA, not GQA; base has no ratio field)
//   * alpha           -> param.scale_s  (Q@K scale)
//   * scale_p         -> param.attn_scale (SiLU-result scale; 0 -> 1/max_seqlen_q)
//   * d_ptr (PRE out) -> BwdWorkspace::delta_ptr (stream-ordered, softmax path only)
//   * softmax LSE/D   -> split fields (seq/nhead/batch stride, delta computed here)
//   * perf timing     -> dropped; launch directly like the base double-kernel path
// First version: batched + group only (this file is batched). Jagged stays a
// runtime branch (is_jagged) — preserved as-is, not exercised in v1.
// ===========================================================================
template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kIsDeterministic,
          ck_tile::index_t MaxK>
struct batched_backward_single_dispatch
{
    using FmhaBwdShape = typename HstuBwdShape<MaxK>::Type;

    using TC = HstuAttentionFwdTypeConfig<InOutDataType>;

    // Shared tail for both SiLU and softmax MAIN: zero dq_acc, launch MAIN, then POST.
    template <typename Pipeline, typename Kernel, typename Kargs>
    static void
    launch_main_and_post(HstuAttentionNoGroupBwdParams& param, hipStream_t stream, Kargs& kargs)
    {
        // single-slot element count (== atomic dq_acc size; == split_stride in determ)
        const size_t single =
            param.is_jagged
                ? static_cast<size_t>(param.batch_stride_dq_acc)
                : static_cast<size_t>(param.num_batch) *
                      static_cast<size_t>(param.batch_stride_dq_acc);

        // grid.x covers the largest seqlen_kv; split_idx = i_tile_n -> num_splits = grid.x.
        const ck_tile::index_t grid_seqlen_kv =
            param.is_jagged ? param.max_seqlen_kv : param.seqlen_kv;
        const int num_splits =
            kIsDeterministic
                ? static_cast<int>(ck_tile::integer_divide_ceil(grid_seqlen_kv, Pipeline::kN0))
                : 1;

        // ZERO dq_acc
        HIP_CHECK_ERROR(hipMemsetAsync(param.dq_acc_ptr,
                                       0,
                                       single * static_cast<size_t>(num_splits) *
                                           sizeof(typename TC::GemmAccDataType),
                                       stream));

        dim3 grid  = Kernel::GridSize(param.num_batch, param.num_head, grid_seqlen_kv);
        dim3 block = Kernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = Kernel::kBlockPerCu;
        // MAIN dqdkdv
        (void)ck_tile::launch_kernel(
            ck_tile::stream_config{stream, false},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grid, block, 0, kargs));

        // POST: dq_acc(float) -> dq. Determ reduces over splits (fixed order -> reproducible).
        const ck_tile::long_index_t n = static_cast<ck_tile::long_index_t>(single);
        constexpr int kPostThreads = 256;
        const int post_blocks      = static_cast<int>((n + kPostThreads - 1) / kPostThreads);
        if constexpr(kIsDeterministic)
        {
            hipLaunchKernelGGL(
                (ck_tile::hstu_bwd_reduce_convert_dq_kernel<InOutDataType,
                                                            typename TC::GemmAccDataType>),
                dim3(post_blocks),
                dim3(kPostThreads),
                0,
                stream,
                reinterpret_cast<const typename TC::GemmAccDataType*>(param.dq_acc_ptr),
                reinterpret_cast<InOutDataType*>(param.dq_ptr),
                n,
                num_splits,
                static_cast<ck_tile::long_index_t>(single));
        }
        else
        {
            hipLaunchKernelGGL(
                (ck_tile::hstu_bwd_convert_dq_kernel<InOutDataType,
                                                     typename TC::GemmAccDataType>),
                dim3(post_blocks),
                dim3(kPostThreads),
                0,
                stream,
                reinterpret_cast<const typename TC::GemmAccDataType*>(param.dq_acc_ptr),
                reinterpret_cast<InOutDataType*>(param.dq_ptr),
                n);
        }
    }

    template <typename Mask, bool kPadHeadDimQ, bool kPadHeadDimV>
    static void RunSilu(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
    {
        constexpr ck_tile::index_t occupancy = 1;

        using Traits = ck_tile::TileFmhaBwdTraits<kPadHeadDimQ,
                                                  kPadHeadDimV,
                                                  ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                  false, // kHasBiasGrad
                                                  occupancy>;

        using Dropout = ck_tile::BlockDropoutBwd<false, true, false>; // no-dropout

        using Problem = ck_tile::BlockFmhaBwdPipelineProblem<
            typename TC::ODataType,       // QDataType (== InOutDataType)
            typename TC::ODataType,       // KDataType
            typename TC::ODataType,       // VDataType
            typename TC::ODataType,       // GemmDataType
            typename TC::CompDataType,    // LSEDataType (dummy on SiLU path)
            typename TC::GemmAccDataType, // AccDataType (float)
            typename TC::CompDataType,    // DDataType (dummy on SiLU path)
            typename TC::BiasDataType,    // BiasDataType (dummy)
            uint8_t,                      // RandValOutputDataType (dummy)
            typename TC::ODataType,       // ODataType
            typename TC::ODataType,       // OGradDataType
            typename TC::ODataType,       // QGradDataType
            typename TC::ODataType,       // KGradDataType
            typename TC::ODataType,       // VGradDataType
            typename TC::BiasDataType,    // BiasGradDataType (dummy)
            FmhaBwdShape,
            false,            // kIsGroupMode
            kIsDeterministic, // deterministic (set+split) vs atomic (false)
            Mask,
            Dropout,
            false, // kUseTrLoad
            Traits>;

        using Pipeline = ck_tile::HstuAttentionBwdDQDKDVPipelineKRKTRVR<Problem>;

        using DKEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType,
            typename TC::ODataType,
            false,
            (kPadHeadDimQ > 0)>>;
        using DVEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType,
            typename TC::ODataType,
            false,
            (kPadHeadDimV > 0)>>;

        using Kernel = ck_tile::HstuAttentionBwdDQDKDVKernel<Pipeline, DKEpilogue, DVEpilogue>;

        // scale_p <- attn_scale (0 => 1/max_seqlen_q); alpha <- scale_s (see field-map note).
        const float scale_p =
            (param.attn_scale != 0.f) ? param.attn_scale
                                      : 1.0f / static_cast<float>(param.max_seqlen_q);

        auto kargs = Kernel::MakeKargs(param.q_ptr,
                                       param.k_ptr,
                                       param.v_ptr,
                                       param.do_ptr,
                                       param.dk_ptr,
                                       param.dv_ptr,
                                       param.dq_acc_ptr,
                                       param.is_jagged,
                                       param.seq_q_offsets_ptr,
                                       // self-attention: kv offsets == q offsets (mirrors fwd)
                                       param.is_cross_attention ? param.seq_kv_offsets_ptr
                                                                : param.seq_q_offsets_ptr,
                                       param.seqlen_q,
                                       param.seqlen_kv,
                                       param.hdim_qk,
                                       param.hdim_v,
                                       1, // nhead_ratio_qk (HSTU MHA, no GQA)
                                       param.scale_s, // alpha
                                       scale_p,
                                       param.num_targets_ptr,
                                       param.contextual_seqlen,
                                       param.window_size, // max_attn_len
                                       param.min_full_attn_seqlen,
                                       param.seq_stride_q,
                                       param.seq_stride_k,
                                       param.seq_stride_v,
                                       param.seq_stride_do,
                                       param.seq_stride_dk,
                                       param.seq_stride_dv,
                                       param.stride_dq_acc,
                                       param.nhead_stride_q,
                                       param.nhead_stride_k,
                                       param.nhead_stride_v,
                                       param.nhead_stride_do,
                                       param.nhead_stride_dk,
                                       param.nhead_stride_dv,
                                       param.nhead_stride_dq_acc,
                                       param.batch_stride_q,
                                       param.batch_stride_k,
                                       param.batch_stride_v,
                                       param.batch_stride_do,
                                       param.batch_stride_dk,
                                       param.batch_stride_dv,
                                       param.batch_stride_dq_acc,
                                       param.split_stride_dq_acc);

        launch_main_and_post<Pipeline, Kernel>(param, stream, kargs);
    }

    // softmax path: PRE (D=rowsum(O*dO)) -> memset dq_acc -> MAIN (softmax) -> POST.
    template <typename Mask, bool kPadHeadDimQ, bool kPadHeadDimV>
    static void RunSoftmax(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
    {
        constexpr ck_tile::index_t occupancy = 1;

        using Traits = ck_tile::TileFmhaBwdTraits<kPadHeadDimQ,
                                                  kPadHeadDimV,
                                                  ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                  false,
                                                  occupancy>;
        using Dropout = ck_tile::BlockDropoutBwd<false, true, false>;

        using Problem = ck_tile::BlockFmhaBwdPipelineProblem<
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::CompDataType,    // LSEDataType (real on softmax path)
            typename TC::GemmAccDataType,
            typename TC::CompDataType,    // DDataType (real on softmax path)
            typename TC::BiasDataType,
            uint8_t,
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::BiasDataType,
            FmhaBwdShape,
            false,            // kIsGroupMode
            kIsDeterministic, // deterministic (set+split) vs atomic (false)
            Mask,
            Dropout,
            false, // kUseTrLoad
            Traits>;

        using Pipeline = ck_tile::HstuAttentionWithSoftmaxBwdDQDKDVPipelineKRKTRVR<Problem>;

        using DKEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType, typename TC::ODataType, false, (kPadHeadDimQ > 0)>>;
        using DVEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType, typename TC::ODataType, false, (kPadHeadDimV > 0)>>;

        using Kernel =
            ck_tile::HstuAttentionBwdDQDKDVSoftmaxKernel<Pipeline, DKEpilogue, DVEpilogue>;

        // Delta layout: [num_batch, num_head, seqlen_q]
        // seq_stride=1, nhead_stride=seqlen_q, batch_stride=num_head*seqlen_q
        // (mirrors the base double-kernel path). BASE LSE layout is [b,s,h]:
        // seq_stride_lse=num_head, nhead_stride_lse=1 (taken from param).
        // jagged: D is packed [b,head,seq], seq_stride=1,
        // nhead_stride=num_batch*max_seqlen_q, no batch stride
        // (mirrors hstu_attention_group_backward_dispatch.hpp:514-519).
        const ck_tile::index_t seq_stride_delta   = 1;
        const ck_tile::index_t nhead_stride_delta =
            param.is_jagged ? param.num_batch * param.max_seqlen_q : param.seqlen_q;
        const ck_tile::index_t batch_stride_delta =
            param.is_jagged ? 0 : param.num_head * param.seqlen_q;

        // seqlen the PRE grid covers; D must be allocated for exactly this much.
        const ck_tile::index_t grid_seqlen =
            param.is_jagged ? param.max_seqlen_q : param.seqlen_q;

        // d_ptr comes from the base workspace (stream-ordered), not a param field.
        BwdWorkspace ws;
        const size_t delta_bytes = static_cast<size_t>(param.num_batch) *
                                   static_cast<size_t>(param.num_head) *
                                   static_cast<size_t>(grid_seqlen) *
                                   sizeof(typename TC::CompDataType);
        HIP_CHECK_ERROR(hipMallocAsync(&ws.delta_ptr, delta_bytes, stream));

        // ---- PRE: D = rowsum(O .* dO) -> ws.delta_ptr ([batch,head,seq] layout) ----
        {
            const ck_tile::long_index_t total =
                static_cast<ck_tile::long_index_t>(param.num_batch) * param.num_head * grid_seqlen;
            constexpr int kPreThreads = 256;
            const int pre_blocks = static_cast<int>((total + kPreThreads - 1) / kPreThreads);
            hipLaunchKernelGGL(
                (ck_tile::hstu_bwd_dot_do_o_kernel<InOutDataType, typename TC::CompDataType>),
                dim3(pre_blocks),
                dim3(kPreThreads),
                0,
                stream,
                reinterpret_cast<const InOutDataType*>(param.o_ptr),
                reinterpret_cast<const InOutDataType*>(param.do_ptr),
                reinterpret_cast<typename TC::CompDataType*>(ws.delta_ptr),
                param.is_jagged,
                reinterpret_cast<const int32_t*>(
                    param.is_jagged ? param.seq_q_offsets_ptr : nullptr),
                param.num_batch,
                param.num_head,
                grid_seqlen,
                param.hdim_v,
                static_cast<ck_tile::long_index_t>(param.seq_stride_o),
                static_cast<ck_tile::long_index_t>(param.nhead_stride_o),
                static_cast<ck_tile::long_index_t>(param.batch_stride_o),
                // PRE writes D with DELTA strides (not LSE): nhead=seqlen_q, batch=num_head*seqlen_q
                static_cast<ck_tile::long_index_t>(nhead_stride_delta),
                static_cast<ck_tile::long_index_t>(batch_stride_delta));
        }

        auto kargs = Kernel::MakeKargs(param.q_ptr,
                                       param.k_ptr,
                                       param.v_ptr,
                                       param.do_ptr,
                                       param.lse_ptr,
                                       ws.delta_ptr,
                                       param.dk_ptr,
                                       param.dv_ptr,
                                       param.dq_acc_ptr,
                                       param.is_jagged,
                                       param.seq_q_offsets_ptr,
                                       param.is_cross_attention ? param.seq_kv_offsets_ptr
                                                                : param.seq_q_offsets_ptr,
                                       param.seqlen_q,
                                       param.seqlen_kv,
                                       param.hdim_qk,
                                       param.hdim_v,
                                       1, // nhead_ratio_qk (HSTU MHA)
                                       param.scale_s, // alpha
                                       param.num_targets_ptr,
                                       param.contextual_seqlen,
                                       param.window_size,
                                       param.min_full_attn_seqlen,
                                       param.seq_stride_q,
                                       param.seq_stride_k,
                                       param.seq_stride_v,
                                       param.seq_stride_do,
                                       param.seq_stride_dk,
                                       param.seq_stride_dv,
                                       param.stride_dq_acc,
                                       param.nhead_stride_q,
                                       param.nhead_stride_k,
                                       param.nhead_stride_v,
                                       param.nhead_stride_do,
                                       param.nhead_stride_dk,
                                       param.nhead_stride_dv,
                                       param.nhead_stride_dq_acc,
                                       param.seq_stride_lse,   // BASE LSE layout [b,s,h] -> num_head
                                       param.nhead_stride_lse, // -> 1
                                       seq_stride_delta,       // 1
                                       nhead_stride_delta,     // seqlen_q
                                       param.batch_stride_q,
                                       param.batch_stride_k,
                                       param.batch_stride_v,
                                       param.batch_stride_do,
                                       param.batch_stride_dk,
                                       param.batch_stride_dv,
                                       param.batch_stride_dq_acc,
                                       param.batch_stride_lse,
                                       batch_stride_delta,
                                       param.split_stride_dq_acc);

        launch_main_and_post<Pipeline, Kernel>(param, stream, kargs);

        HIP_CHECK_ERROR(hipFreeAsync(ws.delta_ptr, stream));
    }

    static void Run(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
    {
        if(param.hdim_qk <= 0 || param.hdim_v <= 0 || param.hdim_qk > MaxK || param.hdim_v > MaxK)
            throw std::runtime_error(
                "HSTU bwd: hdim_qk/hdim_v must be in (0, MaxK]; hdim>256 unsupported");

        constexpr ck_tile::index_t kQKHeaddim = HstuBwdShape<MaxK>::kQKHeaddim;
        constexpr ck_tile::index_t kVHeaddim  = HstuBwdShape<MaxK>::kVHeaddim;
        const bool pad_qk    = !(param.hdim_qk % kQKHeaddim == 0);
        const bool pad_v     = !(param.hdim_v % kVHeaddim == 0);
        const bool use_local = (param.window_size > 0);

        BOOL_SWITCH_2(pad_qk, kPadHeadDimQ, pad_v, kPadHeadDimV, [&] {
            BOOL_SWITCH(use_local, kUseLocal, [&] {
                BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, [&] {
                    using Mask = typename ck_tile::
                        HstuBlockMasking<kIsCrossAttention, kUseCausal, kUseLocal, true>::Type;
                    if constexpr(kUseSoftmax)
                        RunSoftmax<Mask, kPadHeadDimQ, kPadHeadDimV>(param, stream);
                    else
                        RunSilu<Mask, kPadHeadDimQ, kPadHeadDimV>(param, stream);
                });
            });
        });
    }
};

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kIsDeterministic,
          ck_tile::index_t MaxK>
void run_batched_backward_single_dispatch(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
{
    batched_backward_single_dispatch<InOutDataType,
                                     kUseCausal,
                                     kUseSoftmax,
                                     kHasBias,
                                     kIsDeterministic,
                                     MaxK>::Run(param, stream);
}
#endif // HSTU_BWD_SINGLE_KERNEL
