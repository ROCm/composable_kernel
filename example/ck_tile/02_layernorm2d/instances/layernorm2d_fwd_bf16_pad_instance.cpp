
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "layernorm2d_fwd_instance_common.hpp"

template <ck_tile::index_t kNRepeat,
          ck_tile::index_t kMThreadPerBlock,
          ck_tile::index_t kNThreadPerBlock,
          ck_tile::index_t kVectorAccessSize,
          bool kTwoPass>
using t = layernorm2d_fwd_traits_<ck_tile::bf16_t,
                                  kNRepeat,
                                  kMThreadPerBlock,
                                  kNThreadPerBlock,
                                  kVectorAccessSize,
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

template float layernorm2d_fwd_<t<1, 4, 32, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<1, 4, 64, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<2, 4, 64, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<4, 4, 64, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<8, 4, 64, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<8, 4, 64, 4, true>>(const S&, A);

template float layernorm2d_fwd_<t<1, 4, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<2, 4, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<4, 4, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<8, 4, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<16, 4, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<16, 4, 64, 2, true>>(const S&, A);

template float layernorm2d_fwd_<t<32, 4, 64, 1, false>>(const S&, A);
template float layernorm2d_fwd_<t<32, 4, 64, 1, true>>(const S&, A);
