// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>

#include "layernorm_dispatch.hpp"

// clang-format off
// extern template float run_layernorm<ck_tile::fp16_t, 1, 16, 8, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 1, 32, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 1, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// extern template float run_layernorm<ck_tile::fp16_t, 1, 32, 8, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 1, 64, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 2, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// extern template float run_layernorm<ck_tile::fp16_t, 1, 64, 8, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 2, 64, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 4, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// extern template float run_layernorm<ck_tile::fp16_t, 2, 64, 8, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 4, 64, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 8, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// extern template float run_layernorm<ck_tile::fp16_t, 4, 64, 8, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 8, 64, 4, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 16, 64, 2, false>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 8, 64, 4, false, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 16, 64, 2, false, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// extern template float run_layernorm<ck_tile::fp16_t, 1, 16, 8, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 1, 32, 4, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 1, 64, 2, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// extern template float run_layernorm<ck_tile::fp16_t, 1, 32, 8, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 1, 64, 4, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 2, 64, 2, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// extern template float run_layernorm<ck_tile::fp16_t, 1, 64, 8, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 2, 64, 4, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 4, 64, 2, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// extern template float run_layernorm<ck_tile::fp16_t, 2, 64, 8, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 4, 64, 4, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 8, 64, 2, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// extern template float run_layernorm<ck_tile::fp16_t, 4, 64, 8, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 8, 64, 4, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 16, 64, 2, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 8, 64, 4, true, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern template float run_layernorm<ck_tile::fp16_t, 16, 64, 2, true, true>(const layernorm2d_fwd_args& param, ck_tile::stream_config stream);
// clang-format on

float layernorm2d_fwd_fp16(layernorm2d_fwd_args& param, ck_tile::stream_config stream)
{
    // Disable all vector 8fp16 read/write instances as it has performance issue regarding compiler
#if 0
    if(param.N % 8 == 0)
    {
        if(param.N <= 128)
        {
            return param.N == 128 ? run_layernorm<ck_tile::fp16_t, 1, 16, 8, false>(param, stream)
                                  : run_layernorm<ck_tile::fp16_t, 1, 16, 8, true>(param, stream);
        }
        else if(param.N <= 256)
        {
            return param.N == 256 ? run_layernorm<ck_tile::fp16_t, 1, 32, 8, false>(param, stream)
                                  : run_layernorm<ck_tile::fp16_t, 1, 32, 8, true>(param, stream);
        }
        else if(param.N <= 512)
        {
            return param.N == 512 ? run_layernorm<ck_tile::fp16_t, 1, 64, 8, false>(param, stream)
                                  : run_layernorm<ck_tile::fp16_t, 1, 64, 8, true>(param, stream);
        }
        else if(param.N <= 1024)
        {
            return param.N == 1024 ? run_layernorm<ck_tile::fp16_t, 2, 64, 8, false>(param, stream)
                                   : run_layernorm<ck_tile::fp16_t, 2, 64, 8, true>(param, stream);
        }
        else
        {
            return param.N == 2048 ? run_layernorm<ck_tile::fp16_t, 4, 64, 8, false>(param, stream)
                                   : run_layernorm<ck_tile::fp16_t, 4, 64, 8, true>(param, stream);
        }
    }
    else if(param.N % 4 == 0)
#endif
    if(param.N % 4 == 0)
    {
        if(param.N <= 128)
        {
            return param.N == 128 ? run_layernorm<ck_tile::fp16_t, 1, 32, 4, false>(param, stream)
                                  : run_layernorm<ck_tile::fp16_t, 1, 32, 4, true>(param, stream);
        }
        else if(param.N <= 256)
        {
            return param.N == 256 ? run_layernorm<ck_tile::fp16_t, 1, 64, 4, false>(param, stream)
                                  : run_layernorm<ck_tile::fp16_t, 1, 64, 4, true>(param, stream);
        }
        else if(param.N <= 512)
        {
            return param.N == 512 ? run_layernorm<ck_tile::fp16_t, 2, 64, 4, false>(param, stream)
                                  : run_layernorm<ck_tile::fp16_t, 2, 64, 4, true>(param, stream);
        }
        else if(param.N <= 1024)
        {
            return param.N == 1024 ? run_layernorm<ck_tile::fp16_t, 4, 64, 4, false>(param, stream)
                                   : run_layernorm<ck_tile::fp16_t, 4, 64, 4, true>(param, stream);
        }
        else if(param.N <= 2048)
        {
            return param.N == 2048 ? run_layernorm<ck_tile::fp16_t, 8, 64, 4, false>(param, stream)
                                   : run_layernorm<ck_tile::fp16_t, 8, 64, 4, true>(param, stream);
        }
        else
        {
            return param.N % 2048 == 0
                       ? run_layernorm<ck_tile::fp16_t, 8, 64, 4, false, true>(param, stream)
                       : run_layernorm<ck_tile::fp16_t, 8, 64, 4, true, true>(param, stream);
        }
    }
    else if(param.N % 2 == 0)
    {
        if(param.N <= 128)
        {
            return param.N == 128 ? run_layernorm<ck_tile::fp16_t, 1, 64, 2, false>(param, stream)
                                  : run_layernorm<ck_tile::fp16_t, 1, 64, 2, true>(param, stream);
        }
        else if(param.N <= 256)
        {
            return param.N == 256 ? run_layernorm<ck_tile::fp16_t, 2, 64, 2, false>(param, stream)
                                  : run_layernorm<ck_tile::fp16_t, 2, 64, 2, true>(param, stream);
        }
        else if(param.N <= 512)
        {
            return param.N == 512 ? run_layernorm<ck_tile::fp16_t, 4, 64, 2, false>(param, stream)
                                  : run_layernorm<ck_tile::fp16_t, 4, 64, 2, true>(param, stream);
        }
        else if(param.N <= 1024)
        {
            return param.N == 1024 ? run_layernorm<ck_tile::fp16_t, 8, 64, 2, false>(param, stream)
                                   : run_layernorm<ck_tile::fp16_t, 8, 64, 2, true>(param, stream);
        }
        else if(param.N <= 2048)
        {
            return param.N == 2048 ? run_layernorm<ck_tile::fp16_t, 16, 64, 2, false>(param, stream)
                                   : run_layernorm<ck_tile::fp16_t, 16, 64, 2, true>(param, stream);
        }
        else
        {
            return param.N % 2048 == 0
                       ? run_layernorm<ck_tile::fp16_t, 16, 64, 2, false, true>(param, stream)
                       : run_layernorm<ck_tile::fp16_t, 16, 64, 2, true, true>(param, stream);
        }
    }
    else
    {
        throw std::runtime_error("Sequence length sizes not supported!");
    }
};
