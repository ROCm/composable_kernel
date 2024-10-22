
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "rmsnorm2d_fwd_instance_common.hpp"

// clang-format off
//                                                       rm  rn  tm  tn  vn  pd    rms     2p
template float rmsnorm2d_fwd_<trait_<ck_tile::fp16_t,  1, 2, 1,  256, 8,  true,  false, true>>(const S&, A);
template float rmsnorm2d_fwd_<trait_<ck_tile::fp16_t,  1, 4, 1,  256, 4,  true,  false, true>>(const S&, A);
template float rmsnorm2d_fwd_<trait_<ck_tile::fp16_t,  1, 2, 1, 1024, 2,  true,  false, true>>(const S&, A);
template float rmsnorm2d_fwd_<trait_<ck_tile::fp16_t,  1, 4, 1, 1024, 1,  true,  false, true>>(const S&, A);

// clang-format on
