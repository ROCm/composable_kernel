
/*
  Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * The file is automatically generated, don't modify!
 * See the generator script
 * `composable_kernel/example/ck_tile/18_hstu_attention/generate_instances.py`
 */

#include <ck_tile/core/numeric/half.hpp>
#include "hstu_attention_jagged_forward_dispatch.hpp"

template void run_jagged_forward_causal_local_bias_dropout_dispatch<
    ck_tile::bf16_t,
    false,
    false,
    false,
    true,
    256>(HstuAttentionFwdParams& param, hipStream_t stream);
