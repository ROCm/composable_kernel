// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

using kernel_traits =
    unified_attention_decode_kernel_traits<unified_attention_args::data_type_enum::fp16, true, 64, 128, 8, 32, true>;  // Large cache: overflow checks enabled

INST_UNIFIED_ATTENTION_DISPATCH(kernel_traits)

} // namespace ck_tile
