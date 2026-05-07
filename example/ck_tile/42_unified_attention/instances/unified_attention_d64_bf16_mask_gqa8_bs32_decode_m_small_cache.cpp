// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Medium-tier small-cache optimized variant: MaxNumBlocks=100000 (zero rebasing overhead)
using kernel_traits =
    unified_attention_decode_kernel_traits<unified_attention_args::data_type_enum::bf16, true, 64, 128, 8, 32, 100000>;

INST_UNIFIED_ATTENTION_DISPATCH(kernel_traits)

} // namespace ck_tile
