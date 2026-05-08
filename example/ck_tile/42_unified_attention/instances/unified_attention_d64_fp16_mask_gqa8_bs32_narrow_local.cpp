// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// d64 GQA-8 tiny+bs32 decode tier (kBlockM=32, kBlockQ=4, BlockSize=32),
// IsMasking=true, IsLocal=true. fp16 sibling of the bf16 instance used by
// the GPT-OSS decode SWA path.
using kernel_traits =
    unified_attention_decode_bs32_kernel_traits<unified_attention_args::data_type_enum::fp16,
                                                /*IsMasking=*/true,
                                                /*HeadSize=*/64,
                                                /*BlockM=*/32,
                                                /*NumQPerKV=*/8,
                                                /*BlockSize=*/32,
                                                /*IsLocal=*/true>;

INST_UNIFIED_ATTENTION_DISPATCH_DECODE(kernel_traits)

} // namespace ck_tile
