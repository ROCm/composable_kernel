// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// d64 GQA-8 tiny+bs32 decode tier (kBlockM=32, kBlockQ=4, BlockSize=32),
// IsMasking=true, IsLocal=true. Targets GPT-OSS decode shapes
// (q=1, page_blk_size=32, GQA-8) with sliding-window-attention.
using kernel_traits =
    unified_attention_decode_bs32_kernel_traits<unified_attention_args::data_type_enum::bf16,
                                                /*IsMasking=*/true,
                                                /*HeadSize=*/64,
                                                /*BlockM=*/32,
                                                /*NumQPerKV=*/8,
                                                /*BlockSize=*/32,
                                                /*IsLocal=*/true>;

INST_UNIFIED_ATTENTION_DISPATCH_DECODE(kernel_traits)

} // namespace ck_tile
