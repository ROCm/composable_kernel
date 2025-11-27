// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "hstu_attention_params.hpp"

extern void hstu_attention_batched_forward_fp16(HstuAttentionFwdParams& param, hipStream_t stream);
extern void hstu_attention_batched_forward_bf16(HstuAttentionFwdParams& param, hipStream_t stream);
extern void hstu_attention_jagged_forward_fp16(HstuAttentionFwdParams& param, hipStream_t stream);
extern void hstu_attention_jagged_forward_bf16(HstuAttentionFwdParams& param, hipStream_t stream);
