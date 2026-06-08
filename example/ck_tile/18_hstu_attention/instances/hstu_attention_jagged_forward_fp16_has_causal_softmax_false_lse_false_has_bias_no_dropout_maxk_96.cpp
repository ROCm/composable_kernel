
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

// The file is automatically generated, don't modify!
// See the generator script
// `composable_kernel/example/ck_tile/18_hstu_attention/generate_instances.py`

#include <ck_tile/core/numeric/half.hpp>
#include "hstu_attention_jagged_forward_dispatch.hpp"
#include "hstu_attention_params.hpp"

template void run_jagged_forward_dispatch<
    ck_tile::fp16_t,
    true,
    false,
    false,
    true,
    false,
    96>(HstuAttentionNoGroupFwdParams& param, hipStream_t stream);
