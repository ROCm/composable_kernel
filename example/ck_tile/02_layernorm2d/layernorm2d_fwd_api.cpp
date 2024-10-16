// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "layernorm_dispatch.hpp"

float layernorm2d_fwd(layernorm2d_fwd_traits t,
                      layernorm2d_fwd_args a,
                      const ck_tile::stream_config& s)
{
    float r = -1;
    if(t.data_type.compare("fp16") == 0)
    {
        // Disable all vector 8fp16 read/write instances as it has performance issue regarding
        // compiler
#if 0
    if(a.N % 8 == 0)
    {
        if(a.N <= 128)
        {
            return a.N == 128 ? run_layernorm<ck_tile::fp16_t, 1, 16, 8, false>(a, s)
                                  : run_layernorm<ck_tile::fp16_t, 1, 16, 8, true>(a, s);
        }
        else if(a.N <= 256)
        {
            return a.N == 256 ? run_layernorm<ck_tile::fp16_t, 1, 32, 8, false>(a, s)
                                  : run_layernorm<ck_tile::fp16_t, 1, 32, 8, true>(a, s);
        }
        else if(a.N <= 512)
        {
            return a.N == 512 ? run_layernorm<ck_tile::fp16_t, 1, 64, 8, false>(a, s)
                                  : run_layernorm<ck_tile::fp16_t, 1, 64, 8, true>(a, s);
        }
        else if(a.N <= 1024)
        {
            return a.N == 1024 ? run_layernorm<ck_tile::fp16_t, 2, 64, 8, false>(a, s)
                                   : run_layernorm<ck_tile::fp16_t, 2, 64, 8, true>(a, s);
        }
        else
        {
            return a.N == 2048 ? run_layernorm<ck_tile::fp16_t, 4, 64, 8, false>(a, s)
                                   : run_layernorm<ck_tile::fp16_t, 4, 64, 8, true>(a, s);
        }
    }
    else if(a.N % 4 == 0)
#endif
        if(a.N % 4 == 0)
        {
            if(a.N <= 128)
            {
                return a.N == 128 ? run_layernorm<ck_tile::fp16_t, 1, 32, 4, false>(a, s)
                                  : run_layernorm<ck_tile::fp16_t, 1, 32, 4, true>(a, s);
            }
            else if(a.N <= 256)
            {
                return a.N == 256 ? run_layernorm<ck_tile::fp16_t, 1, 64, 4, false>(a, s)
                                  : run_layernorm<ck_tile::fp16_t, 1, 64, 4, true>(a, s);
            }
            else if(a.N <= 512)
            {
                return a.N == 512 ? run_layernorm<ck_tile::fp16_t, 2, 64, 4, false>(a, s)
                                  : run_layernorm<ck_tile::fp16_t, 2, 64, 4, true>(a, s);
            }
            else if(a.N <= 1024)
            {
                return a.N == 1024 ? run_layernorm<ck_tile::fp16_t, 4, 64, 4, false>(a, s)
                                   : run_layernorm<ck_tile::fp16_t, 4, 64, 4, true>(a, s);
            }
            else if(a.N <= 2048)
            {
                return a.N == 2048 ? run_layernorm<ck_tile::fp16_t, 8, 64, 4, false>(a, s)
                                   : run_layernorm<ck_tile::fp16_t, 8, 64, 4, true>(a, s);
            }
            else
            {
                return a.N % 2048 == 0 ? run_layernorm<ck_tile::fp16_t, 8, 64, 4, false, true>(a, s)
                                       : run_layernorm<ck_tile::fp16_t, 8, 64, 4, true, true>(a, s);
            }
        }
        else if(a.N % 2 == 0)
        {
            if(a.N <= 128)
            {
                return a.N == 128 ? run_layernorm<ck_tile::fp16_t, 1, 64, 2, false>(a, s)
                                  : run_layernorm<ck_tile::fp16_t, 1, 64, 2, true>(a, s);
            }
            else if(a.N <= 256)
            {
                return a.N == 256 ? run_layernorm<ck_tile::fp16_t, 2, 64, 2, false>(a, s)
                                  : run_layernorm<ck_tile::fp16_t, 2, 64, 2, true>(a, s);
            }
            else if(a.N <= 512)
            {
                return a.N == 512 ? run_layernorm<ck_tile::fp16_t, 4, 64, 2, false>(a, s)
                                  : run_layernorm<ck_tile::fp16_t, 4, 64, 2, true>(a, s);
            }
            else if(a.N <= 1024)
            {
                return a.N == 1024 ? run_layernorm<ck_tile::fp16_t, 8, 64, 2, false>(a, s)
                                   : run_layernorm<ck_tile::fp16_t, 8, 64, 2, true>(a, s);
            }
            else if(a.N <= 2048)
            {
                return a.N == 2048 ? run_layernorm<ck_tile::fp16_t, 16, 64, 2, false>(a, s)
                                   : run_layernorm<ck_tile::fp16_t, 16, 64, 2, true>(a, s);
            }
            else
            {
                return a.N % 2048 == 0
                           ? run_layernorm<ck_tile::fp16_t, 16, 64, 2, false, true>(a, s)
                           : run_layernorm<ck_tile::fp16_t, 16, 64, 2, true, true>(a, s);
            }
        }
    }
    else if(t.data_type.compare("fp32") == 0)
    {
        if(a.N % 4 == 0)
        {
            if(a.N <= 128)
            {
                return a.N == 128 ? run_layernorm<float, 1, 32, 4, false>(a, s)
                                  : run_layernorm<float, 1, 32, 4, true>(a, s);
            }
            else if(a.N <= 256)
            {
                return a.N == 256 ? run_layernorm<float, 1, 64, 4, false>(a, s)
                                  : run_layernorm<float, 1, 64, 4, true>(a, s);
            }
            else if(a.N <= 512)
            {
                return a.N == 512 ? run_layernorm<float, 2, 64, 4, false>(a, s)
                                  : run_layernorm<float, 2, 64, 4, true>(a, s);
            }
            else if(a.N <= 1024)
            {
                return a.N == 1024 ? run_layernorm<float, 4, 64, 4, false>(a, s)
                                   : run_layernorm<float, 4, 64, 4, true>(a, s);
            }
            else if(a.N <= 2048)
            {
                return a.N == 2048 ? run_layernorm<float, 8, 64, 4, false>(a, s)
                                   : run_layernorm<float, 8, 64, 4, true>(a, s);
            }
            else
            {
                return a.N % 2048 == 0 ? run_layernorm<float, 8, 64, 4, false, true>(a, s)
                                       : run_layernorm<float, 8, 64, 4, true, true>(a, s);
            }
        }
        else if(a.N % 2 == 0)
        {
            if(a.N <= 128)
            {
                return a.N == 128 ? run_layernorm<float, 1, 64, 2, false>(a, s)
                                  : run_layernorm<float, 1, 64, 2, true>(a, s);
            }
            else if(a.N <= 256)
            {
                return a.N == 256 ? run_layernorm<float, 2, 64, 2, false>(a, s)
                                  : run_layernorm<float, 2, 64, 2, true>(a, s);
            }
            else if(a.N <= 512)
            {
                return a.N == 512 ? run_layernorm<float, 4, 64, 2, false>(a, s)
                                  : run_layernorm<float, 4, 64, 2, true>(a, s);
            }
            else if(a.N <= 1024)
            {
                return a.N == 1024 ? run_layernorm<float, 8, 64, 2, false>(a, s)
                                   : run_layernorm<float, 8, 64, 2, true>(a, s);
            }
            else if(a.N <= 2048)
            {
                return a.N == 2048 ? run_layernorm<float, 16, 64, 2, false>(a, s)
                                   : run_layernorm<float, 16, 64, 2, true>(a, s);
            }
            else
            {
                return a.N % 2048 == 0 ? run_layernorm<float, 16, 64, 2, false, true>(a, s)
                                       : run_layernorm<float, 16, 64, 2, true, true>(a, s);
            }
        }
    }

    return r;
}

