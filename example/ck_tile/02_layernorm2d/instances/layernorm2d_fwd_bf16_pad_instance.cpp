
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "layernorm2d_fwd_instance_common.hpp"

template <ck_tile::index_t NRepeat,
          ck_tile::index_t NThread,
          ck_tile::index_t VectorAccessSize,
          bool kTwoPass>
using t = layernorm2d_fwd_traits_<ck_tile::bf16_t,
                                  NRepeat,
                                  NThread,
                                  VectorAccessSize,
                                  true,
                                  false,
                                  kTwoPass>;

// Disable all vector 8fp16 read/write instances as it has performance issue regarding compiler
// template float layernorm2d_fwd_<t<1, 16, 8, false>>(const S&, A);
// template float layernorm2d_fwd_<t<1, 32, 8, false>>(const S&, A);
// template float layernorm2d_fwd_<t<1, 64, 8, false>>(const S&, A);
// template float layernorm2d_fwd_<t<2, 64, 8, false>>(const S&, A);
// template float layernorm2d_fwd_<t<4, 64, 8, false>>(const S&, A);
// template float layernorm2d_fwd_<t<4, 64, 8, true>>(const S&, A);

template float layernorm2d_fwd_<t<1, 32, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<1, 64, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<2, 64, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<4, 64, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<8, 64, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<8, 64, 4, true>>(const S&, A);

template float layernorm2d_fwd_<t<1, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<2, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<4, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<8, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<16, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<16, 64, 2, true>>(const S&, A);

template float layernorm2d_fwd_<t<32, 64, 1, false>>(const S&, A);
template float layernorm2d_fwd_<t<32, 64, 1, true>>(const S&, A);
