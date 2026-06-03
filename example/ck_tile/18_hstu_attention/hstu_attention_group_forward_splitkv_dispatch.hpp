// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/host/hip_check_error.hpp>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_fwd_setting.hpp"
#include "hstu_attention_fwd_splitkv_combine_setting.hpp"
#include "hstu_attention_params.hpp"
#include "hstu_attention_hdim_switch.hpp"
#include "hstu_attention_max_splits_switch.hpp"
#include "hstu_attention_pipeline_problem.hpp"
#include "hstu_attention_traits.hpp"
#include "hstu_attention_with_softmax_fwd_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_pipeline.hpp"
#include "hstu_attention_with_softmax_fwd_trload_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_trload_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_splitkv_combine_pipeline.hpp"
#include "hstu_attention_with_softmax_fwd_splitkv_combine_pipeline.hpp"
#include "hstu_attention_fwd_splitkv_kernel.hpp"
#include "hstu_attention_fwd_splitkv_combine_kernel.hpp"
#include "hstu_attention_splitkv_helper.hpp"

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kHasDropout,
          ck_tile::index_t MaxK,
          ck_tile::index_t MTile>
struct group_forward_splitkv_causal_softmax_bias_dropout_dispatch
{
    static_assert(MTile == 64, "MTile must be 64 to get to fwd splitkv path!");

    using HstuAttentionFwdTileSetting =
        typename std::conditional_t<kUseSoftmax,
                                    HstuAttentionWithSoftmaxFwdTileSetting<MaxK, MTile>,
                                    HstuAttentionNoSoftmaxFwdTileSetting<MaxK, MTile>>::Type;
    using HstuAttentionCombineTileSetting =
        typename HstuAttentionFwdSplitKVCombineTileSetting<MaxK>::Type;

#ifdef BUILD_HSTU_FOR_GFX95_ONLY
    static constexpr bool use_trload_pipeline = true;
#else
    static constexpr bool use_trload_pipeline = false;
#endif

    template <bool kIsCrossAttention>
    using HstuFwdPipelineProblemTemp = ck_tile::HstuAttentionFwdPipelineProblem<
        InOutDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::GemmAccDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::BiasDataType,
        kIsCrossAttention,
        true, // kUseGroup
        true, // kIsJagged
        kHasBias,
        kHasDropout,
        kUseCausal,
        kUseSoftmax,
        false, // kStoreLSE
        HstuAttentionFwdTileSetting>;

    using OaccDataType = HstuAttentionFwdTypeConfig<InOutDataType>::OaccDataType;
    using ODataType    = HstuAttentionFwdTypeConfig<InOutDataType>::ODataType;
    using LSEDataType  = HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType;

    template <ck_tile::index_t kMaxSplits>
    using HstuCombinePipelineProblemTemp =
        ck_tile::HstuAttentionFwdSplitKVCombinePipelineProblem<OaccDataType,
                                                               LSEDataType,
                                                               ODataType,
                                                               true /* kIsJagged */,
                                                               kUseSoftmax,
                                                               false, // kStoreLSE
                                                               HstuAttentionCombineTileSetting,
                                                               kMaxSplits>;

    static void Run(HstuAttentionGroupFwdParams& param, hipStream_t stream)
    {
        constexpr ck_tile::index_t occupancy = -1;

        SplitkvWorkspace ws;

        {
            const bool pad_headdim_qk =
                !(param.hdim_qk % HstuAttentionFwdTileSetting::kQKHeaddim == 0);
            const bool pad_headdim_v = !(param.hdim_v % HstuAttentionFwdTileSetting::kN1 == 0);

            // no need to check seqlen_q since it is not used as fastest dim,
            // buffer_load_dwordxx/buffer_store_dwordxx can handle oob access
            constexpr bool kPadSeqLenQ = false;

            constexpr bool kPadSeqLenK = true;

            BOOL_SWITCH_2(pad_headdim_qk, kPadHeadDimQK, pad_headdim_v, kPadHeadDimV, [&] {
                using HstuTraits = ck_tile::HstuAttentionFwdTraits<kPadSeqLenQ,
                                                                   kPadSeqLenK,
                                                                   kPadHeadDimQK,
                                                                   kPadHeadDimV,
                                                                   occupancy>;

                using HstuEpilogue = ck_tile::Default2DEpilogue<
                    ck_tile::Default2DEpilogueProblem<OaccDataType,
                                                      OaccDataType, // keep output as OaccDataType
                                                      kPadSeqLenQ,
                                                      kPadHeadDimV,
                                                      false>>;

                BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, [&] {
                    using HstuPipelineProblem = HstuFwdPipelineProblemTemp<kIsCrossAttention>;

                    if constexpr(!use_trload_pipeline)
                    {
                        using HstuPipeline = std::conditional_t<
                            kUseSoftmax,
                            ck_tile::HstuAttentionWithSoftmaxFwdPipelineQRKSVS<HstuPipelineProblem,
                                                                               HstuTraits>,
                            ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVS<HstuPipelineProblem,
                                                                             HstuTraits>>;

                        using HstuKernel =
                            ck_tile::HstuAttentionFwdSplitKVKernel<HstuPipeline, HstuEpilogue>;

                        RunWithFwdSplitKVKernel<HstuKernel>(param, ws, stream);
                    }
                    else
                    {
                        using HstuPipeline = std::conditional_t<
                            kUseSoftmax,
                            ck_tile::HstuAttentionWithSoftmaxFwdPipelineQRKSVSTrLoad<
                                HstuPipelineProblem,
                                HstuTraits>,
                            ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVSTrLoad<
                                HstuPipelineProblem,
                                HstuTraits>>;

                        using HstuKernel =
                            ck_tile::HstuAttentionFwdSplitKVKernel<HstuPipeline, HstuEpilogue>;

                        RunWithFwdSplitKVKernel<HstuKernel>(param, ws, stream);
                    };
                });
            });
        };

        if constexpr(kUseSoftmax)
        {
            static constexpr auto kM = HstuAttentionCombineTileSetting::kM;
            // ToDo: be carefule to use get_warp_size() on host-layer on Wave32
            static constexpr auto kBlockSize =
                HstuAttentionCombineTileSetting::NumWarps * ck_tile::get_warp_size();

            const bool pad_headdim_o =
                !(param.hdim_v % HstuAttentionCombineTileSetting::kOHeaddim == 0);

            // no need to check seqlen_q since it is not used as fastest dim,
            // buffer_load_dwordxx/buffer_store_dwordxx can handle oob access
            constexpr bool kPadSeqLenQ = false;

            MAX_SPLITS_SWITCH(ws.num_splits, TMP_MAX_SPLITS, [&] {
                constexpr ck_tile::index_t kMaxSplits = [&]() {
                    if constexpr(kM * TMP_MAX_SPLITS >= kBlockSize)
                        return TMP_MAX_SPLITS;
                    else if constexpr(kM * 2 * TMP_MAX_SPLITS >= kBlockSize)
                        return 2 * TMP_MAX_SPLITS;
                    else
                        return 4 * TMP_MAX_SPLITS;
                }();

                const bool pad_num_splits = (ws.num_splits < kMaxSplits);

                using HstuCombinePipelineProblem = HstuCombinePipelineProblemTemp<kMaxSplits>;

                BOOL_SWITCH_2(pad_headdim_o, kPadHeadDimO, pad_num_splits, kPadNumSplits, [&] {
                    using HstuTraits = ck_tile::HstuAttentionFwdSplitKVCombineTraits<kPadSeqLenQ,
                                                                                     kPadHeadDimO,
                                                                                     kPadNumSplits,
                                                                                     occupancy>;

                    using HstuEpilogue =
                        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<OaccDataType,
                                                                                     ODataType,
                                                                                     kPadSeqLenQ,
                                                                                     kPadHeadDimO,
                                                                                     false>>;

                    using HstuPipeline = ck_tile::HstuAttentionWithSoftmaxFwdSplitKVCombinePipeline<
                        HstuCombinePipelineProblem,
                        HstuTraits>;

                    using HstuKernel =
                        ck_tile::HstuAttentionFwdSplitKVCombineKernel<HstuPipeline, HstuEpilogue>;

                    RunWithFwdSplitKVCombineKernel<HstuKernel>(param, ws, stream);
                });
            });
        }
        else
        {
            const bool pad_headdim_o =
                !(param.hdim_v % HstuAttentionCombineTileSetting::kOHeaddim == 0);

            // no need to check seqlen_q since it is not used as fastest dim,
            // buffer_load_dwordxx/buffer_store_dwordxx can handle oob access
            constexpr bool kPadSeqLenQ = false;

            using HstuCombinePipelineProblem = HstuCombinePipelineProblemTemp<0>;

            BOOL_SWITCH(pad_headdim_o, kPadHeadDimO, [&] {
                using HstuTraits =
                    ck_tile::HstuAttentionFwdSplitKVCombineTraits<kPadSeqLenQ,
                                                                  kPadHeadDimO,
                                                                  false /*kPadNumSplits*/,
                                                                  occupancy>;

                using HstuEpilogue =
                    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<OaccDataType,
                                                                                 ODataType,
                                                                                 kPadSeqLenQ,
                                                                                 kPadHeadDimO,
                                                                                 false>>;

                using HstuPipeline = ck_tile::HstuAttentionNoSoftmaxFwdSplitKVCombinePipeline<
                    HstuCombinePipelineProblem,
                    HstuTraits>;

                using HstuKernel =
                    ck_tile::HstuAttentionFwdSplitKVCombineKernel<HstuPipeline, HstuEpilogue>;

                RunWithFwdSplitKVCombineKernel<HstuKernel>(param, ws, stream);
            });
        }
    };

    template <typename HstuKernel>
    static void RunWithFwdSplitKVKernel(HstuAttentionGroupFwdParams& param,
                                        SplitkvWorkspace& ws,
                                        hipStream_t stream)
    {
        ws.num_splits =
            get_suggested_num_splits(param.num_batch, param.num_head, param.max_seqlen_q);

        // assume the workspace for o_acc is in compact shape of [num_batch, max_seqlen, num_head,
        // num_splits, hdim]
        size_t workspace_bytes = static_cast<size_t>(param.num_batch) * param.max_seqlen_q *
                                 param.num_head * ws.num_splits * param.hdim_v *
                                 sizeof(OaccDataType);

        HIP_CHECK_ERROR(hipMallocAsync(&ws.o_acc_ptr, workspace_bytes, stream));

        if constexpr(kUseSoftmax)
        {
            // assume the workspace for l_acc is in compact shape of [num_batch, max_seqlen,
            // num_head, num_splits]
            workspace_bytes = static_cast<size_t>(param.num_batch) * param.max_seqlen_q *
                              param.num_head * ws.num_splits * sizeof(LSEDataType);

            HIP_CHECK_ERROR(hipMallocAsync(&ws.lse_acc_ptr, workspace_bytes, stream));
        }

        const auto kargs = [&] {
            return HstuKernel::MakeKargs(param.q_ptr,
                                         param.k_ptr,
                                         param.v_ptr,
                                         param.bias_ptr,
                                         ws.o_acc_ptr,
                                         ws.lse_acc_ptr,
                                         ws.num_splits,
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
                                         param.seq_stride_q,
                                         param.seq_stride_k,
                                         param.seq_stride_v,
                                         param.seq_stride_bias,
                                         param.nhead_stride_q,
                                         param.nhead_stride_k,
                                         param.nhead_stride_v,
                                         param.nhead_stride_bias,
                                         param.num_targets_ptr,
                                         param.p_drop,
                                         param.philox_seed,
                                         param.philox_offset);
        }();

        dim3 kGridSize = HstuKernel::GridSize(
            param.num_batch, param.num_head, param.max_seqlen_q, param.hdim_v, ws.num_splits);
        dim3 kBlockSize                        = HstuKernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = HstuKernel::kBlockPerCu;

        (void)ck_tile::launch_kernel(
            ck_tile::stream_config{stream, false},
            ck_tile::make_kernel<kBlockPerCu>(HstuKernel{}, kGridSize, kBlockSize, 0, kargs));
    };

    template <typename HstuKernel>
    static void RunWithFwdSplitKVCombineKernel(HstuAttentionGroupFwdParams& param,
                                               SplitkvWorkspace& ws,
                                               hipStream_t stream)
    {
        const auto kargs = [&] {
            return HstuKernel::MakeKargs(ws.o_acc_ptr,
                                         ws.lse_acc_ptr,
                                         param.o_ptr,
                                         nullptr, // lse_ptr
                                         param.seq_stride_o,
                                         0, // seq_stride_lse
                                         param.nhead_stride_o,
                                         0, // nhead_stride_lse
                                         param.seq_q_offsets_ptr,
                                         param.num_head,
                                         ws.num_splits,
                                         param.hdim_v);
        }();

        dim3 kGridSize  = HstuKernel::GridSize(param.num_batch, param.num_head, param.max_seqlen_q);
        dim3 kBlockSize = HstuKernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = HstuKernel::kBlockPerCu;

        (void)ck_tile::launch_kernel(
            ck_tile::stream_config{stream, false},
            ck_tile::make_kernel<kBlockPerCu>(HstuKernel{}, kGridSize, kBlockSize, 0, kargs));

        HIP_CHECK_ERROR(hipFreeAsync(ws.o_acc_ptr, stream));
        if constexpr(kUseSoftmax)
        {
            HIP_CHECK_ERROR(hipFreeAsync(ws.lse_acc_ptr, stream));
        }
    };
};
