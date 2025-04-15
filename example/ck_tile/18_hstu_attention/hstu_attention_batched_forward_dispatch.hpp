/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/fmha.hpp>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_fwd_setting.hpp"
#include "hstu_attention_params.hpp"
#include "hstu_attention_hdim_switch.hpp"
#include "hstu_block_masking.hpp"
#include "hstu_attention_pipeline_problem.hpp"
#include "hstu_attention_traits.hpp"
#include "hstu_attention_fwd_pipeline.hpp"
#include "hstu_attention_fwd_kernel.hpp"

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseLocal,
          bool kHasBias,
          bool kHasDropout,
          ck_tile::index_t MaxK>
struct batched_forward_causal_local_bias_dropout_dispatch
{
    using HstuAttentionShape = typename HstuAttentionFwdShape<MaxK>::Type;
    using HstuMask           = typename ck_tile::HstuBlockMasking<kUseCausal, kUseLocal>::Type;

    template <typename HstuTraits>
    using HstuPipelineProblemTemp = ck_tile::HstuAttentionFwdPipelineProblem<
        InOutDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::GemmAccDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::BiasDataType,
        false, // kIsJagged
        kHasBias,
        kHasDropout,
        HstuMask,
        HstuAttentionShape,
        HstuTraits>;

    static void Run(HstuAttentionFwdParams& param, hipStream_t stream)
    {
        constexpr ck_tile::index_t occupancy = -1;

        const bool pad_seqlen_k   = !(param.seqlen % HstuAttentionShape::kN0 == 0);
        const bool pad_headdim_qk = !(param.hdim_qk % HstuAttentionShape::kSubQKHeaddim == 0);
        const bool pad_headdim_v  = !(param.hdim_v % HstuAttentionShape::kN1 == 0);

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

                using HstuPipelineProblem = HstuPipelineProblemTemp<HstuTraits>;

                using HstuEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                    typename HstuAttentionFwdTypeConfig<InOutDataType>::OaccDataType,
                    typename HstuAttentionFwdTypeConfig<InOutDataType>::ODataType,
                    kPadSeqLenQ,
                    kPadHeadDimV>>;

                using HstuPipeline = ck_tile::HstuAttentionFwdPipelineQRKSVS<HstuPipelineProblem>;
                using HstuKernel   = ck_tile::HstuAttentionFwdKernel<HstuPipeline, HstuEpilogue>;

                RunWithKernel<HstuKernel>(param, stream);
            });
    };

    template <typename HstuKernel>
    static void RunWithKernel(HstuAttentionFwdParams& param, hipStream_t stream)
    {
        const auto kargs = [&] {
            return HstuKernel::MakeKargs(param.q_ptr,
                                         param.k_ptr,
                                         param.v_ptr,
                                         param.bias_ptr,
                                         param.o_ptr,
                                         param.seqlen,
                                         param.hdim_qk,
                                         param.hdim_v,
                                         param.num_head,
                                         param.scale_s,
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
                                         param.batch_stride_q,
                                         param.batch_stride_k,
                                         param.batch_stride_v,
                                         param.batch_stride_bias,
                                         param.batch_stride_o,
                                         param.num_targets_ptr,
                                         param.contextual_seqlen,
                                         param.window_size,
                                         param.min_full_attn_seqlen,
                                         param.p_drop,
                                         param.philox_seed,
                                         param.philox_offset);
        }();

        dim3 kGridSize =
            HstuKernel::GridSize(param.num_batch, param.num_head, param.seqlen, param.hdim_v);
        constexpr dim3 kBlockSize              = HstuKernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = HstuKernel::kBlockPerCu;

        (void)ck_tile::launch_kernel(ck_tile::stream_config{stream, false},
                                     ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
                                         HstuKernel{}, kGridSize, kBlockSize, 0, kargs));
    };
};

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseLocal,
          bool kHasBias,
          bool kHasDropout,
          ck_tile::index_t MaxK>
void run_batched_forward_causal_local_bias_dropout_dispatch(HstuAttentionFwdParams& param,
                                                            hipStream_t stream)
{
    batched_forward_causal_local_bias_dropout_dispatch<InOutDataType,
                                                       kUseCausal,
                                                       kUseLocal,
                                                       kHasBias,
                                                       kHasDropout,
                                                       MaxK>::Run(param, stream);
};
