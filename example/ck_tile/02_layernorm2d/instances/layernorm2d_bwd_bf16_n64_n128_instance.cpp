
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "layernorm2d_bwd_instance_common.hpp"

// clang-format off
//                                                      rm  tm  tn   pd
template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  1,  64, true>>(const S&, A);
template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  1,  64, true>>(const S&, A);
// clang-format on
