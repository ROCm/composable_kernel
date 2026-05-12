// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

INST_UNIFIED_ATTENTION_DISPATCH(decode_d128_mha_m32, fp16, true)

} // namespace ck_tile
