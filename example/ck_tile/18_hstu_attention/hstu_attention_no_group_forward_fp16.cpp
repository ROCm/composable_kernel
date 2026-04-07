// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include <stdexcept>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_hdim_switch.hpp"
#include "hstu_attention_batched_forward_dispatch.hpp"
#include "hstu_attention_jagged_forward_dispatch.hpp"

#include "instances/hstu_attention_batched_forward_fp16_instances_ref.hpp"
#include "instances/hstu_attention_jagged_forward_fp16_instances_ref.hpp"

void hstu_attention_no_group_forward_fp16(HstuAttentionNoGroupFwdParams& param, hipStream_t stream)
{
    const bool has_bias   = (param.bias_ptr != nullptr);
    const bool use_causal = param.use_causal;
    BOOL_SWITCH_2(has_bias, kHasBias, use_causal, kUseCausal, [&] {
        HDIM_SWITCH(param.hdim_qk, param.hdim_v, MaxK, [&] {
            BOOL_SWITCH(param.use_softmax, kUseSoftmax, [&] {
                if(param.is_jagged)
                    run_jagged_forward_causal_softmax_bias_dropout_dispatch<ck_tile::fp16_t,
                                                                            kUseCausal,
                                                                            kUseSoftmax,
                                                                            kHasBias,
                                                                            false, // kHasDropout
                                                                            MaxK>(param, stream);
                else
                    run_batched_forward_causal_softmax_bias_dropout_dispatch<ck_tile::fp16_t,
                                                                             kUseCausal,
                                                                             kUseSoftmax,
                                                                             kHasBias,
                                                                             false, // kHasDropout
                                                                             MaxK>(param, stream);
            });
        });
    });
};
