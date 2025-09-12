
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

// The file is automatically generated, don't modify!
// See the generator script
// `composable_kernel/example/ck_tile/18_hstu_attention/generate_instances.py`

#include <ck_tile/core/numeric/bfloat16.hpp>
#include "hstu_attention_jagged_forward_dispatch.hpp"

template void run_jagged_forward_causal_bias_dropout_dispatch<
    ck_tile::fp16_t,
    false,
    true,
    true,
    128>(HstuAttentionFwdParams& param, hipStream_t stream);
