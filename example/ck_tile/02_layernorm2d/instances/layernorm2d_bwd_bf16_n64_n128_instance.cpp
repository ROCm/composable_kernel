
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "layernorm2d_bwd_instance_common.hpp"

// clang-format off
//                                                      rm  rn  tm  tn   vm  vn  pd
// template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  1,  1,  64,  1,  1,  true>>(const S&, A);
// template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  1,  1,  64,  1,  1,  true>>(const S&, A);
// template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  1,  1,  128, 1,  1,  true>>(const S&, A);
// template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  1,  1,  128, 1,  1,  true>>(const S&, A);
// template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  1,  1,  128, 1,  8,  true>>(const S&, A);
// template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  1,  1,  128, 1,  8,  true>>(const S&, A);
// template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  1,  1,  256, 1,  1,  true>>(const S&, A);
// template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  1,  1,  256, 1,  1,  true>>(const S&, A);

// large m
// template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  2,  4,  16,  1,  8,  true,  false,  true>>(const S&, A);
// template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  2,  4,  16,  1,  8,  true,  false,  true>>(const S&, A);
// // template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  3,  8,  8, 1,  8,  true>>(const S&, A);
// // template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  3,  8,  8, 1,  8,  true>>(const S&, A);
// // template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  4,  32,  8, 1,  8,  true>>(const S&, A);
// // template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  4,  32,  8, 1,  8,  true>>(const S&, A);
// // template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  8,  64,  4, 1,  8,  true>>(const S&, A);
// // template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  8,  64,  4, 1,  8,  true>>(const S&, A);

// // large n
// // template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  32,  4,  16, 1,  8,  true>>(const S&, A);
// // template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  32,  4,  16, 1,  8,  true>>(const S&, A);
// // template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  4,  1,  64, 1,  8,  true>>(const S&, A);
// // template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  4,  1,  64, 1,  8,  true>>(const S&, A);

// // two pass
// template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  4,  2,  32,  1,  8,  true,  true,  true>>(const S&, A);
// template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  4,  2,  32,  1,  8,  true,  true,  true>>(const S&, A);


// Weight Grad
// template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  1,  64,  1,  1,  1,  true,  false,  false>>(const S&, A);
// template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  1,  64,  1,  1,  1,  true,  false,  false>>(const S&, A);
template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 2,  1,  32,  16,  8,  2,  true,  false,  false>>(const S&, A);
template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 2,  1,  32,  16,  8,  2,  true,  false,  false>>(const S&, A);
// template float layernorm2d_bwd_<trait_<ck_tile::bf16_t, 1,  1,  8,  32,  1,  2,  true,  false,  false>>(const S&, A);
// template float layernorm2d_bwd_<trait_<ck_tile::fp16_t, 1,  1,  8,  32,  1,  2,  true,  false,  false>>(const S&, A);
// clang-format on
