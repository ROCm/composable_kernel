// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// d64 GQA-8 tiny+bs32 decode (kBlockM=16, kBlockQ=2, BlockSize=32), masked causal (non-SWA).
using kernel_traits =
    unified_attention_decode_tiny_kernel_traits<unified_attention_args::data_type_enum::bf16, true, 64, 16, 8, 32>;

INST_UNIFIED_ATTENTION_DISPATCH_DECODE(kernel_traits)

} // namespace ck_tile
