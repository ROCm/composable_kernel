// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include <stdexcept>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_hdim_switch.hpp"
#include "hstu_attention_batched_forward_dispatch.hpp"

#include "instances/hstu_attention_batched_forward_fp16_instances_ref.hpp"

void hstu_attention_batched_forward_fp16(HstuAttentionFwdParams& param, hipStream_t stream)
{
    const bool has_dropout = (param.p_drop > 0.0f);
    const bool has_bias    = (param.bias_ptr != nullptr);
    const bool use_causal  = param.use_causal;
    BOOL_SWITCH_3(has_bias, kHasBias, has_dropout, kHasDropout, use_causal, kUseCausal, [&] {
        HDIM_SWITCH(param.hdim_qk, param.hdim_v, MaxK, [&] {
            run_batched_forward_causal_bias_dropout_dispatch<ck_tile::fp16_t,
                                                             kUseCausal,
                                                             kHasBias,
                                                             kHasDropout,
                                                             MaxK>(param, stream);
        });
    });
};
