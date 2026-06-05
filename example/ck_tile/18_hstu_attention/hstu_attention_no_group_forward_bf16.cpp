// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include <stdexcept>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_hdim_switch.hpp"
#include "hstu_attention_batched_forward_dispatch.hpp"
#include "hstu_attention_jagged_forward_dispatch.hpp"

#include "instances/hstu_attention_batched_forward_bf16_instances_ref.hpp"
#include "instances/hstu_attention_jagged_forward_bf16_instances_ref.hpp"

void hstu_attention_no_group_forward_bf16(HstuAttentionNoGroupFwdParams& param, hipStream_t stream)
{
    const bool has_bias   = (param.bias_ptr != nullptr);
    const bool use_causal = param.use_causal;
    bool store_lse        = (param.use_softmax && param.is_training);

    BOOL_SWITCH_3(has_bias, kHasBias, use_causal, kUseCausal, param.use_softmax, kUseSoftmax, [&] {
        HDIM_SWITCH(param.hdim_qk, param.hdim_v, MaxK, [&] {
            BOOL_SWITCH(store_lse, kStoreLSE, [&] {
                if constexpr(kUseSoftmax || !kStoreLSE)
                {
                    if(param.is_jagged)
                        run_jagged_forward_causal_softmax_bias_dropout_dispatch<
                            ck_tile::bf16_t,
                            kUseCausal,
                            kUseSoftmax,
                            kStoreLSE,
                            kHasBias,
                            false, // kHasDropout
                            MaxK>(param, stream);
                    else
                        run_batched_forward_causal_softmax_bias_dropout_dispatch<
                            ck_tile::bf16_t,
                            kUseCausal,
                            kUseSoftmax,
                            kStoreLSE,
                            kHasBias,
                            false, // kHasDropout
                            MaxK>(param, stream);
                }
            });
        });
    });
};
