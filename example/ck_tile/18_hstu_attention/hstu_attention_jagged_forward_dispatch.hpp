// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/ops/epilogue.hpp>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_fwd_setting.hpp"
#include "hstu_attention_params.hpp"
#include "hstu_attention_hdim_switch.hpp"
#include "hstu_attention_pipeline_problem.hpp"
#include "hstu_attention_traits.hpp"
#include "hstu_attention_with_softmax_fwd_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_pipeline.hpp"
#include "hstu_attention_with_softmax_fwd_trload_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_trload_pipeline.hpp"
#include "hstu_attention_fwd_kernel.hpp"
#include "hstu_attention_epilogue.hpp"

#include "hstu_attention_jagged_forward_splitkv_dispatch.hpp"

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kHasDropout,
          ck_tile::index_t MaxK,
          ck_tile::index_t MTile>
struct jagged_forward_causal_softmax_bias_dropout_dispatch
{
    using HstuAttentionTileSetting =
        typename std::conditional_t<kUseSoftmax,
                                    HstuAttentionWithSoftmaxFwdTileSetting<MaxK, MTile>,
                                    HstuAttentionNoSoftmaxFwdTileSetting<MaxK, MTile>>::Type;

#ifdef BUILD_HSTU_FOR_GFX95_ONLY
    static constexpr bool use_trload_pipeline = true;
#else
    static constexpr bool use_trload_pipeline = false;
#endif

    template <bool kIsCrossAttention>
    using HstuPipelineProblemTemp = ck_tile::HstuAttentionFwdPipelineProblem<
        InOutDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::GemmAccDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::BiasDataType,
        kIsCrossAttention,
        false, // kUseGroup
        true,  // kIsJagged
        kHasBias,
        kHasDropout,
        kUseCausal,
        kUseSoftmax,
        HstuAttentionTileSetting>;

    static void Run(HstuAttentionNoGroupFwdParams& param, hipStream_t stream)
    {
        constexpr ck_tile::index_t occupancy = -1;

        const bool pad_headdim_qk = !(param.hdim_qk % HstuAttentionTileSetting::kQKHeaddim == 0);
        const bool pad_headdim_v  = !(param.hdim_v % HstuAttentionTileSetting::kN1 == 0);

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

            using HstuEpilogue = ck_tile::NRepetitions2DEpilogue<ck_tile::Default2DEpilogueProblem<
                typename HstuAttentionFwdTypeConfig<InOutDataType>::OaccDataType,
                typename HstuAttentionFwdTypeConfig<InOutDataType>::ODataType,
                kPadSeqLenQ,
                kPadHeadDimV>>;

            BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, [&] {
                using HstuPipelineProblem = HstuPipelineProblemTemp<kIsCrossAttention>;

                if constexpr(!use_trload_pipeline)
                {
                    using HstuPipeline = std::conditional_t<
                        kUseSoftmax,
                        ck_tile::HstuAttentionWithSoftmaxFwdPipelineQRKSVS<HstuPipelineProblem,
                                                                           HstuTraits>,
                        ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVS<HstuPipelineProblem,
                                                                         HstuTraits>>;

                    using HstuKernel = ck_tile::HstuAttentionFwdKernel<HstuPipeline, HstuEpilogue>;

                    RunWithKernel<HstuKernel>(param, stream);
                }
                else
                {
                    using HstuPipeline = std::conditional_t<
                        kUseSoftmax,
                        ck_tile::HstuAttentionWithSoftmaxFwdPipelineQRKSVSTrLoad<
                            HstuPipelineProblem,
                            HstuTraits>,
                        ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVSTrLoad<HstuPipelineProblem,
                                                                               HstuTraits>>;

                    using HstuKernel = ck_tile::HstuAttentionFwdKernel<HstuPipeline, HstuEpilogue>;

                    RunWithKernel<HstuKernel>(param, stream);
                };
            });
        });
    };

    template <typename HstuKernel>
    static void RunWithKernel(HstuAttentionNoGroupFwdParams& param, hipStream_t stream)
    {
        const auto kargs = [&] {
            return HstuKernel::MakeKargs(param.q_ptr,
                                         param.k_ptr,
                                         param.v_ptr,
                                         param.bias_ptr,
                                         param.o_ptr,
                                         param.seq_q_offsets_ptr,
                                         param.is_cross_attention ? param.seq_kv_offsets_ptr
                                                                  : param.seq_q_offsets_ptr,
                                         param.max_seqlen_q,
                                         param.hdim_qk,
                                         param.hdim_v,
                                         param.num_head,
                                         param.scale_s,
                                         param.attn_scale,
                                         param.seq_stride_q,
                                         param.seq_stride_k,
                                         param.seq_stride_v,
                                         param.seq_stride_bias,
                                         param.seq_stride_o,
                                         param.nhead_stride_q,
                                         param.nhead_stride_k,
                                         param.nhead_stride_v,
                                         param.nhead_stride_bias,
                                         param.nhead_stride_o,
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
                                              param.max_seqlen_q,
                                              param.hdim_v,
                                              has_minfull_attn_seqlen);
        dim3 kBlockSize                        = HstuKernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = HstuKernel::kBlockPerCu;

        (void)ck_tile::launch_kernel(ck_tile::stream_config{stream, false},
                                     ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
                                         HstuKernel{}, kGridSize, kBlockSize, 0, kargs));
    };
};

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kHasDropout,
          ck_tile::index_t MaxK>
void run_jagged_forward_causal_softmax_bias_dropout_dispatch(HstuAttentionNoGroupFwdParams& param,
                                                             hipStream_t stream)
{
    if(get_hstu_attention_fwd_mtile(param.num_batch, param.num_head, param.max_seqlen_q) == 128)
        jagged_forward_causal_softmax_bias_dropout_dispatch<InOutDataType,
                                                            kUseCausal,
                                                            kUseSoftmax,
                                                            kHasBias,
                                                            kHasDropout,
                                                            MaxK,
                                                            128>::Run(param, stream);
    else
    {
        const bool disable_fwd_splitkv = []() {
            const char* env_p = std::getenv("HSTU_DISABLE_SPLITKV");
            if(env_p == nullptr)
                return false;
            return static_cast<bool>(atoi(env_p));
        }();

        if(!disable_fwd_splitkv &&
           shall_use_splitkv(param.num_batch, param.num_head, param.max_seqlen_q))
        {
            jagged_forward_splitkv_causal_softmax_bias_dropout_dispatch<InOutDataType,
                                                                        kUseCausal,
                                                                        kUseSoftmax,
                                                                        kHasBias,
                                                                        kHasDropout,
                                                                        MaxK,
                                                                        64>::Run(param, stream);
        }
        else
            jagged_forward_causal_softmax_bias_dropout_dispatch<InOutDataType,
                                                                kUseCausal,
                                                                kUseSoftmax,
                                                                kHasBias,
                                                                kHasDropout,
                                                                MaxK,
                                                                64>::Run(param, stream);
    };
};
