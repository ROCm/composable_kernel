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
#include "hstu_attention_pipeline_problem.hpp"
#include "hstu_attention_traits.hpp"
#include "hstu_attention_with_softmax_fwd_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_pipeline.hpp"
#include "hstu_attention_with_softmax_fwd_trload_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_trload_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_splitkv_combine_pipeline.hpp"
#include "hstu_attention_fwd_splitkv_kernel.hpp"
#include "hstu_attention_fwd_splitkv_combine_kernel.hpp"

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kHasDropout,
          ck_tile::index_t MaxK,
          ck_tile::index_t MTile>
struct batched_forward_splitkv_causal_softmax_bias_dropout_dispatch
{
    static_assert(kUseSoftmax == false, "Softmax support is not enabled yet!");
    static_assert(MTile == 64, "MTile must be 64 to get to fwd splitkv path!");

    using HstuAttentionFwdTileSetting =
        typename std::conditional_t<kUseSoftmax,
                                    HstuAttentionWithSoftmaxFwdTileSetting<MaxK, MTile>,
                                    HstuAttentionNoSoftmaxFwdTileSetting<MaxK, MTile>>::Type;
    using HstuAttentionCombineTileSetting = HstuAttentionFwdSplitKVCombineTileSetting<MaxK>::Type;

#ifdef BUILD_HSTU_FOR_GFX95_ONLY
    static constexpr bool kUseTrLoad = true;
#else
    static constexpr bool kUseTrLoad = false;
#endif

    template <bool kIsCrossAttention>
    using HstuFwdPipelineProblemTemp = ck_tile::HstuAttentionFwdPipelineProblem<
        InOutDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::GemmAccDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::BiasDataType,
        kIsCrossAttention,
        false, // kUseGroup
        false, // kIsJagged
        kHasBias,
        kHasDropout,
        kUseCausal,
        kUseSoftmax,
        kUseTrLoad,
        HstuAttentionFwdTileSetting>;

    using OaccDataType = HstuAttentionFwdTypeConfig<InOutDataType>::OaccDataType;
    using ODataType    = HstuAttentionFwdTypeConfig<InOutDataType>::ODataType;

    static void Run(HstuAttentionNoGroupFwdParams& param, hipStream_t stream)
    {
        constexpr ck_tile::index_t occupancy = -1;

        {
            const bool pad_seqlen_k = !(param.seqlen_kv % HstuAttentionFwdTileSetting::kN0 == 0);
            const bool pad_headdim_qk =
                !(param.hdim_qk % HstuAttentionFwdTileSetting::kQKHeaddim == 0);
            const bool pad_headdim_v = !(param.hdim_v % HstuAttentionFwdTileSetting::kN1 == 0);

            // no need to check seqlen_q since it is not used as fastest dim,
            // buffer_load_dwordxx/buffer_store_dwordxx can handle oob access
            constexpr bool kPadSeqLenQ = false;

            BOOL_SWITCH_3(
                pad_seqlen_k,
                kPadSeqLenK,
                pad_headdim_qk,
                kPadHeadDimQK,
                pad_headdim_v,
                kPadHeadDimV,
                [&] {
                    using HstuTraits = ck_tile::HstuAttentionFwdTraits<kPadSeqLenQ,
                                                                       kPadSeqLenK,
                                                                       kPadHeadDimQK,
                                                                       kPadHeadDimV,
                                                                       occupancy>;

                    using HstuEpilogue =
                        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                            OaccDataType,
                            OaccDataType, // keep output as OaccDataType
                            kPadSeqLenQ,
                            kPadHeadDimV,
                            false>>;

                    BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, [&] {
                        using HstuPipelineProblem = HstuFwdPipelineProblemTemp<kIsCrossAttention>;

                        if constexpr(!kUseTrLoad)
                        {
                            using HstuPipeline = std::conditional_t<
                                kUseSoftmax,
                                ck_tile::HstuAttentionWithSoftmaxFwdPipelineQRKSVS<
                                    HstuPipelineProblem,
                                    HstuTraits>,
                                ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVS<
                                    HstuPipelineProblem,
                                    HstuTraits>>;

                            using HstuKernel =
                                ck_tile::HstuAttentionFwdSplitKVKernel<HstuPipeline, HstuEpilogue>;

                            RunWithFwdSplitKVKernel<HstuKernel>(param, stream);
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

                            RunWithFwdSplitKVKernel<HstuKernel>(param, stream);
                        };
                    });
                });
        };
        {
            using HstuCombinePipelineProblem =
                ck_tile::HstuAttentionFwdSplitKVCombinePipelineProblem<
                    OaccDataType,
                    ODataType,
                    false /* kIsJagged */,
                    kUseSoftmax,
                    HstuAttentionCombineTileSetting>;
            const bool pad_headdim_o =
                !(param.hdim_v % HstuAttentionCombineTileSetting::kOHeaddim == 0);

            // no need to check seqlen_q since it is not used as fastest dim,
            // buffer_load_dwordxx/buffer_store_dwordxx can handle oob access
            constexpr bool kPadSeqLenQ = false;

            BOOL_SWITCH(pad_headdim_o, kPadHeadDimO, [&] {
                using HstuTraits = ck_tile::
                    HstuAttentionFwdSplitKVCombineTraits<kPadSeqLenQ, kPadHeadDimO, occupancy>;

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

                RunWithFwdSplitKVCombineKernel<HstuKernel>(param, stream);
            });
        };
    };

    template <typename HstuKernel>
    static void RunWithFwdSplitKVKernel(HstuAttentionNoGroupFwdParams& param, hipStream_t stream)
    {
        param.num_splits =
            get_suggested_num_splits(param.num_batch, param.num_head, param.seqlen_q);

        // assume the workspace for o_acc is in compact shape of [num_batch, seqlen_q, num_head,
        // num_splits, hdim]
        size_t workspace_bytes = static_cast<size_t>(param.num_batch) * param.seqlen_q *
                                 param.num_head * param.num_splits * param.hdim_v *
                                 sizeof(OaccDataType);

        HIP_CHECK_ERROR(hipMallocAsync(&param.o_acc_ptr, workspace_bytes, stream));

        const auto kargs = [&] {
            return HstuKernel::MakeKargs(param.q_ptr,
                                         param.k_ptr,
                                         param.v_ptr,
                                         param.bias_ptr,
                                         param.o_acc_ptr,
                                         param.num_splits,
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
                                         param.seq_stride_bias,
                                         param.nhead_stride_q,
                                         param.nhead_stride_k,
                                         param.nhead_stride_v,
                                         param.nhead_stride_bias,
                                         param.batch_stride_q,
                                         param.batch_stride_k,
                                         param.batch_stride_v,
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
        dim3 kGridSize                         = HstuKernel::GridSize(param.num_batch,
                                              param.num_head,
                                              param.seqlen_q,
                                              param.hdim_v,
                                              param.num_splits,
                                              has_minfull_attn_seqlen);
        constexpr dim3 kBlockSize              = HstuKernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = HstuKernel::kBlockPerCu;

        (void)ck_tile::launch_kernel(
            ck_tile::stream_config{stream, false},
            ck_tile::make_kernel<kBlockPerCu>(HstuKernel{}, kGridSize, kBlockSize, 0, kargs));
    };

    template <typename HstuKernel>
    static void RunWithFwdSplitKVCombineKernel(HstuAttentionNoGroupFwdParams& param,
                                               hipStream_t stream)
    {
        const auto kargs = [&] {
            return HstuKernel::MakeKargs(param.o_acc_ptr,
                                         param.o_ptr,
                                         param.batch_stride_o,
                                         param.seq_stride_o,
                                         param.nhead_stride_o,
                                         param.seqlen_q,
                                         param.num_head,
                                         param.num_splits,
                                         param.hdim_v);
        }();

        dim3 kGridSize = HstuKernel::GridSize(param.num_batch, param.num_head, param.seqlen_q);
        constexpr dim3 kBlockSize              = HstuKernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = HstuKernel::kBlockPerCu;

        (void)ck_tile::launch_kernel(
            ck_tile::stream_config{stream, false},
            ck_tile::make_kernel<kBlockPerCu>(HstuKernel{}, kGridSize, kBlockSize, 0, kargs));

        HIP_CHECK_ERROR(hipFreeAsync(param.o_acc_ptr, stream));
    };
};
