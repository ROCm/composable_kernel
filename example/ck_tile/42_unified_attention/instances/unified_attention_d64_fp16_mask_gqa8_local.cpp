// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// d64 GQA-8 large/medium prefill tier, IsMasking=true, IsLocal=true.
using kernel_traits =
    unified_attention_kernel_traits<unified_attention_args::data_type_enum::fp16,
                                    /*IsMasking=*/true,
                                    /*HeadSize=*/64,
                                    /*BlockM=*/256,
                                    /*NumQPerKV=*/8,
                                    /*BlockSize=*/64,
                                    /*IsLocal=*/true>;

INST_UNIFIED_ATTENTION_DISPATCH(kernel_traits)

} // namespace ck_tile
