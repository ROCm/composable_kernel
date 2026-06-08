
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

// The file is automatically generated, don't modify!
// See the generator script
// `composable_kernel/example/ck_tile/18_hstu_attention/generate_instances.py`

#include <ck_tile/core/numeric/bfloat16.hpp>
#include "hstu_attention_group_forward_dispatch.hpp"
#include "hstu_attention_params.hpp"

template void run_group_forward_dispatch<
    ck_tile::bf16_t,
    true,
    false,
    false,
    true,
    false,
    96>(HstuAttentionGroupFwdParams& param, hipStream_t stream);
