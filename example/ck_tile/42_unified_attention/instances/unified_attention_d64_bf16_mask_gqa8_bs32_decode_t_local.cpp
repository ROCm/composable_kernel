// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// d64 GQA-8 tiny decode tier with BlockSize=32 (kBlockM=16, kBlockQ=2),
// IsMasking=true, IsLocal=true. bf16 sibling of the fp16 instance.
using kernel_traits =
    unified_attention_decode_tiny_kernel_traits<unified_attention_args::data_type_enum::bf16,
                                                /*IsMasking=*/true,
                                                /*HeadSize=*/64,
                                                /*BlockM=*/16,
                                                /*NumQPerKV=*/8,
                                                /*BlockSize=*/32,
                                                /*IsLocal=*/true>;

INST_UNIFIED_ATTENTION_DISPATCH_DECODE(kernel_traits)

} // namespace ck_tile
