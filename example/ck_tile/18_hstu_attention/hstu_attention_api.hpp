// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "hstu_attention_params.hpp"

extern void hstu_attention_no_group_forward_fp16(HstuAttentionNoGroupFwdParams& param,
                                                 hipStream_t stream);
extern void hstu_attention_no_group_forward_bf16(HstuAttentionNoGroupFwdParams& param,
                                                 hipStream_t stream);
extern void hstu_attention_group_forward_fp16(HstuAttentionGroupFwdParams& param,
                                              hipStream_t stream);
extern void hstu_attention_group_forward_bf16(HstuAttentionGroupFwdParams& param,
                                              hipStream_t stream);
