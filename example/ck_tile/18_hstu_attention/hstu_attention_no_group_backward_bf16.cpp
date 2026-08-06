// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_hdim_switch.hpp"

#include "instances/hstu_attention_batched_backward_bf16_instances_ref.hpp"
#include "instances/hstu_attention_jagged_backward_bf16_instances_ref.hpp"

void hstu_attention_no_group_backward_bf16(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
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
                          // single kernel 没有 dropout，也没有 jagged 分支（其 Run
                          // 无条件走 batched batch_stride_* 路，见
                          // hstu_attention_batched_backward_dispatch.hpp:687-712）。
                          // 故仅当 !has_dropout && !is_jagged 时才走 single；否则
                          // 落到下面 base 原路（base 有 dropout / jagged 两条支路）。
                          if(!has_dropout && !param.is_jagged)
                          {
                              // single 第 5 模板轴 = kIsDeterministic（与 base 第 5 轴
                              // kHasDropout 语义不同），值取自 param.kIsDeterministic，
                              // 经 BOOL_SWITCH 展开为编译期常量。
                              BOOL_SWITCH(param.kIsDeterministic, kIsDeterministic, [&] {
                                  run_batched_backward_single_dispatch<ck_tile::bf16_t,
                                                                       kUseCausal,
                                                                       kUseSoftmax,
                                                                       kHasBias,
                                                                       kIsDeterministic,
                                                                       MaxK>(param, stream);
                              });
                              return;
                          }
#endif
                          if(param.is_jagged)
                              run_jagged_backward_dispatch<ck_tile::bf16_t,
                                                           kUseCausal,
                                                           kUseSoftmax,
                                                           kHasBias,
                                                           kHasDropout,
                                                           MaxK>(param, stream);
                          else
                              run_batched_backward_dispatch<ck_tile::bf16_t,
                                                            kUseCausal,
                                                            kUseSoftmax,
                                                            kHasBias,
                                                            kHasDropout,
                                                            MaxK>(param, stream);
                      });
                  });
}
