// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "layernorm2d_fwd.hpp"

template <typename DataType,
          ck_tile::index_t kNRepeat,
          ck_tile::index_t kMThreadPerBlock,
          ck_tile::index_t kNThreadPerBlock,
          ck_tile::index_t kVectorAccessSize,
          bool kPadN,
          bool kTwoPass = false>
using trait_ = layernorm2d_fwd_traits_<DataType,
                                       kNRepeat,
                                       kMThreadPerBlock,
                                       kNThreadPerBlock,
                                       kVectorAccessSize,
                                       kPadN,
                                       false,
                                       kTwoPass>;

float layernorm2d_fwd(layernorm2d_fwd_traits t,
                      layernorm2d_fwd_args a,
                      const ck_tile::stream_config& s)
{
    float r = -1;
    if(t.data_type.compare("fp16") == 0)
    {
        if(a.N % 4 == 0)
        {
            if(a.N <= 128)
            {
                return a.N == 128
                           ? layernorm2d_fwd_<trait_<ck_tile::fp16_t, 1, 4, 32, 4, false>>(s, a)
                           : layernorm2d_fwd_<trait_<ck_tile::fp16_t, 1, 4, 32, 4, true>>(s, a);
            }
            else if(a.N <= 256)
            {
                return a.N == 256
                           ? layernorm2d_fwd_<trait_<ck_tile::fp16_t, 1, 4, 64, 4, false>>(s, a)
                           : layernorm2d_fwd_<trait_<ck_tile::fp16_t, 1, 4, 64, 4, true>>(s, a);
            }
            else if(a.N <= 512)
            {
                return a.N == 512
                           ? layernorm2d_fwd_<trait_<ck_tile::fp16_t, 2, 4, 64, 4, false>>(s, a)
                           : layernorm2d_fwd_<trait_<ck_tile::fp16_t, 2, 4, 64, 4, true>>(s, a);
            }
            else if(a.N <= 1024)
            {
                return a.N == 1024
                           ? layernorm2d_fwd_<trait_<ck_tile::fp16_t, 4, 4, 64, 4, false>>(s, a)
                           : layernorm2d_fwd_<trait_<ck_tile::fp16_t, 4, 4, 64, 4, true>>(s, a);
            }
            else if(a.N <= 2048)
            {
                return a.N == 2048
                           ? layernorm2d_fwd_<trait_<ck_tile::fp16_t, 8, 4, 64, 4, false>>(s, a)
                           : layernorm2d_fwd_<trait_<ck_tile::fp16_t, 8, 4, 64, 4, true>>(s, a);
            }
            else
            {
                return a.N % 2048 == 0
                           ? layernorm2d_fwd_<trait_<ck_tile::fp16_t, 8, 4, 64, 4, false, true>>(s,
                                                                                                 a)
                           : layernorm2d_fwd_<trait_<ck_tile::fp16_t, 8, 4, 64, 4, true, true>>(s,
                                                                                                a);
            }
        }
        else if(a.N % 2 == 0)
        {
            if(a.N <= 128)
            {
                return layernorm2d_fwd_<trait_<ck_tile::fp16_t, 1, 4, 64, 2, true>>(s, a);
            }
            else if(a.N <= 256)
            {
                return layernorm2d_fwd_<trait_<ck_tile::fp16_t, 2, 4, 64, 2, true>>(s, a);
            }
            else if(a.N <= 512)
            {
                return layernorm2d_fwd_<trait_<ck_tile::fp16_t, 4, 4, 64, 2, true>>(s, a);
            }
            else if(a.N <= 1024)
            {
                return layernorm2d_fwd_<trait_<ck_tile::fp16_t, 8, 4, 64, 2, true>>(s, a);
            }
            else if(a.N <= 2048)
            {
                return layernorm2d_fwd_<trait_<ck_tile::fp16_t, 16, 4, 64, 2, true>>(s, a);
            }
            else
            {
                return layernorm2d_fwd_<trait_<ck_tile::fp16_t, 16, 4, 64, 2, true, true>>(s, a);
            }
        }
        else
        {
            return a.N <= 2048
                       ? layernorm2d_fwd_<trait_<ck_tile::fp16_t, 32, 4, 64, 1, true, false>>(s, a)
                       : layernorm2d_fwd_<trait_<ck_tile::fp16_t, 32, 4, 64, 1, true, true>>(s, a);
        }
    }
    else if(t.data_type.compare("bf16") == 0)
    {
        if(a.N % 4 == 0)
        {
            if(a.N <= 128)
            {
                return a.N == 128
                           ? layernorm2d_fwd_<trait_<ck_tile::bf16_t, 1, 4, 32, 4, false>>(s, a)
                           : layernorm2d_fwd_<trait_<ck_tile::bf16_t, 1, 4, 32, 4, true>>(s, a);
            }
            else if(a.N <= 256)
            {
                return a.N == 256
                           ? layernorm2d_fwd_<trait_<ck_tile::bf16_t, 1, 4, 64, 4, false>>(s, a)
                           : layernorm2d_fwd_<trait_<ck_tile::bf16_t, 1, 4, 64, 4, true>>(s, a);
            }
            else if(a.N <= 512)
            {
                return a.N == 512
                           ? layernorm2d_fwd_<trait_<ck_tile::bf16_t, 2, 4, 64, 4, false>>(s, a)
                           : layernorm2d_fwd_<trait_<ck_tile::bf16_t, 2, 4, 64, 4, true>>(s, a);
            }
            else if(a.N <= 1024)
            {
                return a.N == 1024
                           ? layernorm2d_fwd_<trait_<ck_tile::bf16_t, 4, 4, 64, 4, false>>(s, a)
                           : layernorm2d_fwd_<trait_<ck_tile::bf16_t, 4, 4, 64, 4, true>>(s, a);
            }
            else if(a.N <= 2048)
            {
                return a.N == 2048
                           ? layernorm2d_fwd_<trait_<ck_tile::bf16_t, 8, 4, 64, 4, false>>(s, a)
                           : layernorm2d_fwd_<trait_<ck_tile::bf16_t, 8, 4, 64, 4, true>>(s, a);
            }
            else
            {
                return a.N % 2048 == 0
                           ? layernorm2d_fwd_<trait_<ck_tile::bf16_t, 8, 4, 64, 4, false, true>>(s,
                                                                                                 a)
                           : layernorm2d_fwd_<trait_<ck_tile::bf16_t, 8, 4, 64, 4, true, true>>(s,
                                                                                                a);
            }
        }
        else if(a.N % 2 == 0)
        {
            if(a.N <= 128)
            {
                return layernorm2d_fwd_<trait_<ck_tile::bf16_t, 1, 4, 64, 2, true>>(s, a);
            }
            else if(a.N <= 256)
            {
                return layernorm2d_fwd_<trait_<ck_tile::bf16_t, 2, 4, 64, 2, true>>(s, a);
            }
            else if(a.N <= 512)
            {
                return layernorm2d_fwd_<trait_<ck_tile::bf16_t, 4, 4, 64, 2, true>>(s, a);
            }
            else if(a.N <= 1024)
            {
                return layernorm2d_fwd_<trait_<ck_tile::bf16_t, 8, 4, 64, 2, true>>(s, a);
            }
            else if(a.N <= 2048)
            {
                return layernorm2d_fwd_<trait_<ck_tile::bf16_t, 16, 4, 64, 2, true>>(s, a);
            }
            else
            {
                return layernorm2d_fwd_<trait_<ck_tile::bf16_t, 16, 4, 64, 2, true, true>>(s, a);
            }
        }
        else
        {
            return a.N <= 2048
                       ? layernorm2d_fwd_<trait_<ck_tile::bf16_t, 32, 4, 64, 1, true, false>>(s, a)
                       : layernorm2d_fwd_<trait_<ck_tile::bf16_t, 32, 4, 64, 1, true, true>>(s, a);
        }
    }

    if(r < 0)
        throw std::runtime_error("Without supported instances!");

    return r;
}
