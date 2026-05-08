// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Small-cache optimized variant: MaxNumBlocks=false (zero rebasing overhead)
using kernel_traits =
    unified_attention_decode_small_kernel_traits<unified_attention_args::data_type_enum::bf16, true, 64, 64, 8, 32, false>;

INST_UNIFIED_ATTENTION_DISPATCH_DECODE(kernel_traits)

} // namespace ck_tile
