// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_hdim_switch.hpp"

#include "instances/hstu_attention_batched_backward_fp16_instances_ref.hpp"
#include "instances/hstu_attention_jagged_backward_fp16_instances_ref.hpp"

void hstu_attention_no_group_backward_fp16(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
{
    bool has_dropout = (param.p_drop > 0.0f);

    constexpr bool kHasBias = false;
    BOOL_SWITCH_3(param.use_causal,
                  kUseCausal,
                  param.use_softmax,
                  kUseSoftmax,
                  has_dropout,
                  kHasDropout,
                  [&] {
                      HDIM_SWITCH(param.hdim_qk, param.hdim_v, MaxK, [&] {
                          if(param.is_jagged)
                              run_jagged_backward_dispatch<ck_tile::fp16_t,
                                                           kUseCausal,
                                                           kUseSoftmax,
                                                           kHasBias,
                                                           kHasDropout,
                                                           MaxK>(param, stream);
                          else
                              run_batched_backward_dispatch<ck_tile::fp16_t,
                                                            kUseCausal,
                                                            kUseSoftmax,
                                                            kHasBias,
                                                            kHasDropout,
                                                            MaxK>(param, stream);
                      });
                  });
}
