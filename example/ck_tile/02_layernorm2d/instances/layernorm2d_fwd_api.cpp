// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "layernorm2d_fwd_instance_common.hpp"

template <typename data_type>
float layernorm2d_fwd_b16_(layernorm2d_fwd_traits /*t*/,
                           layernorm2d_fwd_args a,
                           const ck_tile::stream_config& s)
{
#if 1
    float r = -1;
    // clang-format off
    //                                            rm  rn  tm   tn  vn  pd     mv     2p
    if(a.n <= 64) {
            r = layernorm2d_fwd_<trait_<data_type, 1,  1,  4,  64, 1,  true,  false, false>>(s, a);
    }
    else if(a.n <= 128) {
        if (a.n % 2 == 0)
            r = layernorm2d_fwd_<trait_<data_type, 1,  1,  4,  64, 2,  true,  false, false>>(s, a);
        else
            r = layernorm2d_fwd_<trait_<data_type, 1,  2,  4,  64, 1,  true,  false, false>>(s, a);
    }
    else if(a.n <= 256) {
        if (a.n % 4 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 1,  4,  64, 4,  true,  false, false>>(s, a);
        else if (a.n % 2 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 2,  4,  64, 2,  true,  false, false>>(s, a);
        else
            r = layernorm2d_fwd_<trait_<data_type,  1, 4,  4,  64, 1,  true,  false, false>>(s, a);
    }
    else if(a.n <= 512) {
        if (a.n % 8 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 1,  4,  64, 8,  true,  false, false>>(s, a);
        else if (a.n % 4 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 2,  4,  64, 4,  true,  false, false>>(s, a);
        else if (a.n % 2 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 4,  4,  64, 2,  true,  false, false>>(s, a);
        else
            r = layernorm2d_fwd_<trait_<data_type,  1, 8,  4,  64, 1,  true,  false, false>>(s, a);
    }
    else if(a.n <= 768) {
        if (a.n % 4 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 3,  4,  64, 4,  true,  false, false>>(s, a);
        else if (a.n % 2 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 6,  4,  64, 2,  true,  false, false>>(s, a);
        else
            r = layernorm2d_fwd_<trait_<data_type,  1,12,  4,  64, 1,  true,  false, false>>(s, a);
    }
    else if(a.n <= 1024) {
        if (a.n % 8 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 1, 2,  128, 8,  true,  false, false>>(s, a);
        else if (a.n % 4 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 2, 2,  128, 4,  true,  false, false>>(s, a);
        else if (a.n % 2 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 4, 2,  128, 2,  true,  false, false>>(s, a);
        else
            r = layernorm2d_fwd_<trait_<data_type,  1, 4, 1,  256, 1,  true,  false, false>>(s, a);
    }
    else if(a.n <= 2048) {
        if (a.n % 8 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 1, 1,  256, 8,  true,  false, false>>(s, a);
        else if (a.n % 4 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 2, 1,  256, 4,  true,  false, false>>(s, a);
        else if (a.n % 2 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 4, 1,  256, 2,  true,  false, false>>(s, a);
        else
            r = layernorm2d_fwd_<trait_<data_type,  1, 8, 1,  256, 1,  true,  false, false>>(s, a);
    }
    else if(a.n <= 3072) {
        if (a.n % 8 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 3, 1,  128, 8,  true,  false, false>>(s, a);
        else if (a.n % 4 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 3, 1,  256, 4,  true,  false, false>>(s, a);
        else if (a.n % 2 == 0)
            r = layernorm2d_fwd_<trait_<data_type,  1, 6, 1,  256, 2,  true,  false, false>>(s, a);
        else
            r = layernorm2d_fwd_<trait_<data_type,  1, 3, 1, 1024, 1,  true,  false, false>>(s, a);
    }
    return r;
#else
    return layernorm2d_fwd_<trait_<data_type,  1, 1,  1,  256, 4,  true,  false, false>>(s, a);
#endif
    // clang-format on
}

float layernorm2d_fwd(layernorm2d_fwd_traits t,
                      layernorm2d_fwd_args a,
                      const ck_tile::stream_config& s)
{

    float r = -1;
    if(t.data_type.compare("fp16") == 0)
    {
        return layernorm2d_fwd_b16_<ck_tile::fp16_t>(t, a, s);
    }
    else if(t.data_type.compare("bf16") == 0)
    {
        return layernorm2d_fwd_b16_<ck_tile::bf16_t>(t, a, s);
    }
    if(r < 0)
        throw std::runtime_error("Without supported instances!");

    return r;
}
