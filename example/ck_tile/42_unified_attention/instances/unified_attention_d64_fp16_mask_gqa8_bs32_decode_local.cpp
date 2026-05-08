// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// d64 GQA-8 medium decode tier with BlockSize=32 (kBlockM=128, kBlockQ=16),
// IsMasking=true, IsLocal=true. fp16 sibling of the bf16 instance.
using kernel_traits =
    unified_attention_decode_kernel_traits<unified_attention_args::data_type_enum::fp16,
                                           /*IsMasking=*/true,
                                           /*HeadSize=*/64,
                                           /*BlockM=*/128,
                                           /*NumQPerKV=*/8,
                                           /*BlockSize=*/32,
                                           /*IsLocal=*/true>;

INST_UNIFIED_ATTENTION_DISPATCH(kernel_traits)

} // namespace ck_tile
