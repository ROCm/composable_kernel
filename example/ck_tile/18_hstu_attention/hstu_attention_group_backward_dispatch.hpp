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
#include "hstu_attention_host_util.hpp"

// Backward dispatch for group (multi-group jagged) mode.
// Backward never uses split-kv: training always provides sufficient CU coverage.

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kHasDropout,
          ck_tile::index_t MaxK>
struct group_backward_dispatch
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
        true, // kUseGroup
        true, // kIsJagged
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

    static void Run(HstuAttentionGroupBwdParams& param, hipStream_t stream)
    {
        BwdWorkspace ws;

        constexpr ck_tile::index_t occupancy_k1 = -1;
        constexpr ck_tile::index_t occupancy_k2 = -1;

        const bool pad_headdim_qk =
            !(param.hdim_qk % HstuAttentionTileSettingForKernel1::kQKHeaddim == 0);
        const bool pad_headdim_v =
            !(param.hdim_v % HstuAttentionTileSettingForKernel1::kVHeaddim == 0);

        // seqlen_q is not along the fastest memory dim; OOB is handled by the buffer instructions.
        // In group/jagged mode seqlen_k always requires padding since it varies per sample.
        constexpr bool kPadSeqLenQ = false;
        constexpr bool kPadSeqLenK = true;

        BOOL_SWITCH_2(pad_headdim_qk, kPadHeadDimQK, pad_headdim_v, kPadHeadDimV, [&] {
            using HstuTraits = ck_tile::HstuAttentionBwdTraits<kPadSeqLenQ,
                                                               kPadSeqLenK,
                                                               kPadHeadDimQK,
                                                               kPadHeadDimV,
                                                               occupancy_k1,
                                                               occupancy_k2>;

            using HstuEpilogue = ck_tile::NRepetitions2DEpilogue<ck_tile::Default2DEpilogueProblem<
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
    RunWithKernels(HstuAttentionGroupBwdParams& param, BwdWorkspace& ws, hipStream_t stream)
    {
        // Delta layout in group/jagged mode: flat over the sequence dimension, strided by head.
        // seq_stride=1, nhead_stride=num_batch*max_seqlen_q -- no batch stride needed.
        const ck_tile::index_t seq_stride_delta   = 1;
        const ck_tile::index_t nhead_stride_delta = param.num_batch * param.max_seqlen_q;

        if constexpr(kUseSoftmax)
        {
            const size_t delta_bytes = static_cast<size_t>(param.num_batch) *
                                       static_cast<size_t>(param.num_head) *
                                       static_cast<size_t>(param.max_seqlen_q) * sizeof(float);
            HIP_CHECK_ERROR(hipMallocAsync(&ws.delta_ptr, delta_bytes, stream));
        }

        bool almost_invariant_seqlen = is_almost_invariant_seqlen(param);

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
                                              param.num_batch / param.num_group,
                                              param.seq_q_offsets_ptr,
                                              param.is_cross_attention ? param.seq_kv_offsets_ptr
                                                                       : param.seq_q_offsets_ptr,
                                              param.group_max_seqlen_q_ptr,
                                              param.group_contextual_seqlen_ptr,
                                              param.group_window_size_ptr,
                                              param.group_min_full_attn_seqlen_ptr,
                                              param.group_attn_scale_ptr,
                                              param.hdim_qk,
                                              param.hdim_v,
                                              param.num_head,
                                              param.scale_s,
                                              almost_invariant_seqlen,
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
                                              param.num_targets_ptr,
                                              param.p_drop,
                                              param.philox_seed,
                                              param.philox_offset);
            }();

            // Group kernel 1 always adds one sentinel tile for variable seqlen handling.
            dim3 kGridSize = HstuKernel1::GridSize(
                param.num_batch, param.num_head, param.max_seqlen_q, almost_invariant_seqlen);
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
                                              param.num_batch / param.num_group,
                                              param.seq_q_offsets_ptr,
                                              param.is_cross_attention ? param.seq_kv_offsets_ptr
                                                                       : param.seq_q_offsets_ptr,
                                              param.group_max_seqlen_q_ptr,
                                              param.group_contextual_seqlen_ptr,
                                              param.group_window_size_ptr,
                                              param.group_min_full_attn_seqlen_ptr,
                                              param.group_attn_scale_ptr,
                                              param.hdim_qk,
                                              param.hdim_v,
                                              param.num_head,
                                              param.scale_s,
                                              almost_invariant_seqlen,
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
                                              param.num_targets_ptr,
                                              param.p_drop,
                                              param.philox_seed,
                                              param.philox_offset);
            }();

            dim3 kGridSize = HstuKernel2::GridSize(
                param.num_batch, param.num_head, param.max_seqlen_kv, almost_invariant_seqlen);
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
void run_group_backward_dispatch(HstuAttentionGroupBwdParams& param, hipStream_t stream)
{
    group_backward_dispatch<InOutDataType, kUseCausal, kUseSoftmax, kHasBias, kHasDropout, MaxK>::
        Run(param, stream);
}

#if defined(HSTU_BWD_SINGLE_KERNEL)
// ============================================================================
// Single-kernel (fused) GROUP backward dispatch — HSTU_BWD_SINGLE_KERNEL path.
//
// Ported from OURS group dispatch (ck-dropout-wt a86529dc) onto the base
// infrastructure. Two-layer renamed to *_single_* to eliminate ODR clash with
// the base double-kernel group_backward_dispatch above:
//   * base 5th template axis = kHasDropout ; ours 5th axis = kIsDeterministic
//     (same arity, opposite semantics) -> MUST rename struct + free fn.
//
// Uses the base single-kernel group kernels inlined into
// hstu_attention_bwd_kernel.hpp (S1):
//   * SiLU    -> HstuAttentionBwdDQDKDVGroupKernel
//   * softmax -> HstuAttentionBwdDQDKDVGroupSoftmaxKernel (+ PRE dot_do_o,
//                POST convert / reduce_convert bare kernels, all in ck_tile ns).
//
// Param-name mapping (ck_qf HstuAttentionGroupBwdParams has no alpha /
// nhead_ratio_qk / d_ptr / measure_perf / perf_* fields; see "未覆盖处的决定"):
//   * alpha          <- param.scale_s   (global Q@K scale)
//   * nhead_ratio_qk <- 1               (HSTU group: Q/K share head count)
//   * d_ptr          <- ws.delta_ptr    (base BwdWorkspace, softmax path only)
//   * perf timing    <- hstu_bwd_perf::time_op(false, ...) (measure disabled;
//                        byte-identical host behavior to a bare single launch)
//   * scale_p        <- per-group group_attn_scale_ptr, resolved in-kernel
//   * lse/delta strides: group is packed [b,s,h] layout ->
//       seq_stride_lse=num_head, nhead_stride_lse=1,
//       seq_stride_delta=1,      nhead_stride_delta=num_batch*max_seqlen_q
//     (no batch stride: group uses jagged offsets).
// ============================================================================
#include <stdexcept>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_no_softmax_bwd_pipeline.hpp"
#include "hstu_attention_with_softmax_bwd_pipeline.hpp"
#include "hstu_attention_bwd_kernel.hpp"
#include "hstu_attention_bwd_shape.hpp"
#include "hstu_attention_bwd_perf.hpp"
#include "hstu_block_masking_bwd_single.hpp"

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kIsDeterministic,
          ck_tile::index_t MaxK>
struct group_backward_single_dispatch
{
    using FmhaBwdShape = typename HstuBwdShape<MaxK>::Type;

    using TC = HstuAttentionFwdTypeConfig<InOutDataType>;

    template <typename Mask, bool kPadHeadDimQ, bool kPadHeadDimV>
    using ProblemFor = ck_tile::BlockFmhaBwdPipelineProblem<
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::CompDataType,
        typename TC::GemmAccDataType,
        typename TC::CompDataType,
        typename TC::BiasDataType,
        uint8_t,
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::BiasDataType,
        FmhaBwdShape,
        false, // kIsGroupMode (FMHA group-mode flag; HSTU group uses jagged indexing in-kernel)
        kIsDeterministic, // deterministic (set+split) vs atomic (false)
        Mask,
        ck_tile::BlockDropoutBwd<false, true, false>,
        false, // kUseTrLoad
        ck_tile::TileFmhaBwdTraits<kPadHeadDimQ, kPadHeadDimV,
                                   ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                   false, 1>>;

    // Shared tail: zero dq_acc, launch MAIN, POST convert (atomic) / reduce_convert
    // (deterministic). perf timing disabled (params carry no measure_perf field).
    template <typename Pipeline, typename Kernel, typename Kargs>
    static void
    launch_main_and_post(HstuAttentionGroupBwdParams& param, hipStream_t stream, Kargs& kargs)
    {
        const size_t single = static_cast<size_t>(param.total_dq_acc_elems); // one packed slot
        const int num_splits =
            kIsDeterministic
                ? static_cast<int>(
                      ck_tile::integer_divide_ceil(param.max_seqlen_kv, Pipeline::kN0))
                : 1;

        (void)hstu_bwd_perf::time_op(false, stream, [&] {
            HIP_CHECK_ERROR(hipMemsetAsync(param.dq_acc_ptr,
                                           0,
                                           single * static_cast<size_t>(num_splits) *
                                               sizeof(typename TC::GemmAccDataType),
                                           stream));
        });

        dim3 grid  = Kernel::GridSize(param.num_batch, param.num_head, param.max_seqlen_kv);
        dim3 block = Kernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = Kernel::kBlockPerCu;
        (void)hstu_bwd_perf::time_op(false, stream, [&] {
            (void)ck_tile::launch_kernel(
                ck_tile::stream_config{stream, false},
                ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grid, block, 0, kargs));
        });

        const ck_tile::long_index_t n = static_cast<ck_tile::long_index_t>(single);
        constexpr int kPostThreads = 256;
        const int post_blocks      = static_cast<int>((n + kPostThreads - 1) / kPostThreads);
        (void)hstu_bwd_perf::time_op(false, stream, [&] {
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
        });
    }

    template <bool kPadHeadDimQ, bool kPadHeadDimV>
    static void RunSilu(HstuAttentionGroupBwdParams& param, hipStream_t stream)
    {
        BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, [&] {
        using LocalMask =
            typename ck_tile::HstuBlockMasking<kIsCrossAttention, kUseCausal, true>::Type;
        using NoLocalMask =
            typename ck_tile::HstuBlockMasking<kIsCrossAttention, kUseCausal, false>::Type;

        using PipelineLocal = ck_tile::HstuAttentionBwdDQDKDVPipelineKRKTRVR<
            ProblemFor<LocalMask, kPadHeadDimQ, kPadHeadDimV>>;
        using PipelineNoLocal = ck_tile::HstuAttentionBwdDQDKDVPipelineKRKTRVR<
            ProblemFor<NoLocalMask, kPadHeadDimQ, kPadHeadDimV>>;

        using DKEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType, typename TC::ODataType, false, (kPadHeadDimQ > 0)>>;
        using DVEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType, typename TC::ODataType, false, (kPadHeadDimV > 0)>>;

        using Kernel = ck_tile::HstuAttentionBwdDQDKDVGroupKernel<PipelineLocal,
                                                                  PipelineNoLocal,
                                                                  DKEpilogue,
                                                                  DVEpilogue>;

        auto kargs = Kernel::MakeKargs(param.q_ptr,
                                       param.k_ptr,
                                       param.v_ptr,
                                       param.do_ptr,
                                       param.dk_ptr,
                                       param.dv_ptr,
                                       param.dq_acc_ptr,
                                       param.seq_q_offsets_ptr,
                                       // self-attention: kv offsets == q offsets
                                       param.is_cross_attention ? param.seq_kv_offsets_ptr
                                                                : param.seq_q_offsets_ptr,
                                       param.group_attn_scale_ptr,
                                       param.group_max_seqlen_q_ptr,
                                       param.group_window_size_ptr,
                                       param.group_contextual_seqlen_ptr,
                                       param.group_min_full_attn_seqlen_ptr,
                                       param.num_batch / param.num_group,
                                       param.num_targets_ptr,
                                       param.hdim_qk,
                                       param.hdim_v,
                                       /*nhead_ratio_qk=*/1, // HSTU group: Q/K share head count
                                       /*alpha=*/param.scale_s,
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
                                       param.split_stride_dq_acc);

        launch_main_and_post<PipelineLocal, Kernel>(param, stream, kargs);
        }); // BOOL_SWITCH(is_cross_attention)
    }

    template <bool kPadHeadDimQ, bool kPadHeadDimV>
    static void RunSoftmax(HstuAttentionGroupBwdParams& param, hipStream_t stream)
    {
        BwdWorkspace ws;

        // group softmax: D (delta) layout [b,head,seq] packed, seq_stride=1,
        // nhead_stride=num_batch*max_seqlen_q, no batch stride.
        const ck_tile::index_t seq_stride_lse     = param.num_head; // LSE [b,s,h]
        const ck_tile::index_t nhead_stride_lse   = 1;
        const ck_tile::index_t seq_stride_delta   = 1;
        const ck_tile::index_t nhead_stride_delta = param.num_batch * param.max_seqlen_q;

        const size_t delta_bytes = static_cast<size_t>(param.num_batch) *
                                   static_cast<size_t>(param.num_head) *
                                   static_cast<size_t>(param.max_seqlen_q) * sizeof(float);
        HIP_CHECK_ERROR(hipMallocAsync(&ws.delta_ptr, delta_bytes, stream));

        BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, [&] {
        using LocalMask =
            typename ck_tile::HstuBlockMasking<kIsCrossAttention, kUseCausal, true>::Type;
        using NoLocalMask =
            typename ck_tile::HstuBlockMasking<kIsCrossAttention, kUseCausal, false>::Type;

        using PipelineLocal = ck_tile::HstuAttentionWithSoftmaxBwdDQDKDVPipelineKRKTRVR<
            ProblemFor<LocalMask, kPadHeadDimQ, kPadHeadDimV>>;
        using PipelineNoLocal = ck_tile::HstuAttentionWithSoftmaxBwdDQDKDVPipelineKRKTRVR<
            ProblemFor<NoLocalMask, kPadHeadDimQ, kPadHeadDimV>>;

        using DKEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType, typename TC::ODataType, false, (kPadHeadDimQ > 0)>>;
        using DVEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType, typename TC::ODataType, false, (kPadHeadDimV > 0)>>;

        using Kernel = ck_tile::HstuAttentionBwdDQDKDVGroupSoftmaxKernel<PipelineLocal,
                                                                         PipelineNoLocal,
                                                                         DKEpilogue,
                                                                         DVEpilogue>;

        // ---- PRE: D = rowsum(O .* dO). group is packed (jagged) -> token base via offsets.
        //      caller MUST pass delta strides here, NOT LSE strides.
        {
            const ck_tile::long_index_t total =
                static_cast<ck_tile::long_index_t>(param.num_batch) * param.num_head *
                param.max_seqlen_q;
            constexpr int kPreThreads = 256;
            const int pre_blocks = static_cast<int>((total + kPreThreads - 1) / kPreThreads);
            (void)hstu_bwd_perf::time_op(false, stream, [&] {
            hipLaunchKernelGGL(
                (ck_tile::hstu_bwd_dot_do_o_kernel<InOutDataType, typename TC::CompDataType>),
                dim3(pre_blocks),
                dim3(kPreThreads),
                0,
                stream,
                reinterpret_cast<const InOutDataType*>(param.o_ptr),
                reinterpret_cast<const InOutDataType*>(param.do_ptr),
                reinterpret_cast<typename TC::CompDataType*>(ws.delta_ptr),
                /*is_jagged=*/true,
                reinterpret_cast<const int32_t*>(param.seq_q_offsets_ptr),
                param.num_batch,
                param.num_head,
                param.max_seqlen_q,
                param.hdim_v,
                static_cast<ck_tile::long_index_t>(param.seq_stride_o),
                static_cast<ck_tile::long_index_t>(param.nhead_stride_o),
                static_cast<ck_tile::long_index_t>(0), // packed: no batch stride
                static_cast<ck_tile::long_index_t>(nhead_stride_delta), // delta nhead stride
                static_cast<ck_tile::long_index_t>(0)); // packed: no delta batch stride
            }); // time_op (PRE)
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
                                       param.seq_q_offsets_ptr,
                                       param.is_cross_attention ? param.seq_kv_offsets_ptr
                                                                : param.seq_q_offsets_ptr,
                                       param.group_attn_scale_ptr,
                                       param.group_max_seqlen_q_ptr,
                                       param.group_window_size_ptr,
                                       param.group_contextual_seqlen_ptr,
                                       param.group_min_full_attn_seqlen_ptr,
                                       param.num_batch / param.num_group,
                                       param.num_targets_ptr,
                                       param.hdim_qk,
                                       param.hdim_v,
                                       /*nhead_ratio_qk=*/1, // HSTU group: Q/K share head count
                                       /*alpha=*/param.scale_s,
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
                                       seq_stride_lse,
                                       nhead_stride_lse,
                                       seq_stride_delta,
                                       nhead_stride_delta,
                                       param.split_stride_dq_acc);

        launch_main_and_post<PipelineLocal, Kernel>(param, stream, kargs);
        }); // BOOL_SWITCH(is_cross_attention)

        HIP_CHECK_ERROR(hipFreeAsync(ws.delta_ptr, stream));
    }

    static void Run(HstuAttentionGroupBwdParams& param, hipStream_t stream)
    {
        if(param.hdim_qk <= 0 || param.hdim_v <= 0 || param.hdim_qk > MaxK || param.hdim_v > MaxK)
            throw std::runtime_error(
                "HSTU bwd group single: hdim_qk/hdim_v must be in (0, MaxK]; hdim>256 unsupported");

        constexpr ck_tile::index_t kQKHeaddim = HstuBwdShape<MaxK>::kQKHeaddim;
        constexpr ck_tile::index_t kVHeaddim  = HstuBwdShape<MaxK>::kVHeaddim;
        const bool pad_qk = !(param.hdim_qk % kQKHeaddim == 0);
        const bool pad_v  = !(param.hdim_v % kVHeaddim == 0);

        BOOL_SWITCH_2(pad_qk, kPadHeadDimQ, pad_v, kPadHeadDimV, [&] {
            if constexpr(kUseSoftmax)
                RunSoftmax<kPadHeadDimQ, kPadHeadDimV>(param, stream);
            else
                RunSilu<kPadHeadDimQ, kPadHeadDimV>(param, stream);
        });
    }
};

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kIsDeterministic,
          ck_tile::index_t MaxK>
void run_group_backward_single_dispatch(HstuAttentionGroupBwdParams& param, hipStream_t stream)
{
    group_backward_single_dispatch<InOutDataType,
                                   kUseCausal,
                                   kUseSoftmax,
                                   kHasBias,
                                   kIsDeterministic,
                                   MaxK>::Run(param, stream);
}
#endif // HSTU_BWD_SINGLE_KERNEL
