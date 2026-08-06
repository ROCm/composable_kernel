// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_hdim_switch.hpp"

#include "instances/hstu_attention_group_backward_bf16_instances_ref.hpp"

void hstu_attention_group_backward_bf16(HstuAttentionGroupBwdParams& param, hipStream_t stream)
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
#if defined(HSTU_BWD_SINGLE_KERNEL)
                          // single kernel 没有 dropout。group 恒 packed，其 single
                          // dispatch 已内建 jagged offset / per-group scale 语义
                          // （见 hstu_attention_group_backward_dispatch.hpp:560-615），
                          // 故 group 侧无 is_jagged 分支，回落条件仅 has_dropout：
                          // p_drop>0 时必须走下面 base 双 kernel（带 kHasDropout 轴），
                          // 否则会静默算出错误结果。
                          if(!has_dropout)
                          {
                              // single 第 5 模板轴 = kIsDeterministic（与 base 第 5 轴
                              // kHasDropout 语义不同），值取自 param.kIsDeterministic，
                              // 经 BOOL_SWITCH 展开为编译期常量。
                              BOOL_SWITCH(param.kIsDeterministic, kIsDeterministic, [&] {
                                  run_group_backward_single_dispatch<ck_tile::bf16_t,
                                                                     kUseCausal,
                                                                     kUseSoftmax,
                                                                     kHasBias,
                                                                     kIsDeterministic,
                                                                     MaxK>(param, stream);
                              });
                              return;
                          }
#endif
                          run_group_backward_dispatch<ck_tile::bf16_t,
                                                      kUseCausal,
                                                      kUseSoftmax,
                                                      kHasBias,
                                                      kHasDropout,
                                                      MaxK>(param, stream);
                      });
                  });
}
