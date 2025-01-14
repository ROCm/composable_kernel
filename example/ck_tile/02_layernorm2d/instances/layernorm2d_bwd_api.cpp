// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "layernorm2d_bwd.hpp"

float layernorm2d_bwd(layernorm2d_bwd_traits t,
                      layernorm2d_bwd_args a,
                      const ck_tile::stream_config& s)
{

    float r = -1;
    if(t.data_type.compare("fp16") == 0)
    {
        return layernorm2d_bwd_b16_<ck_tile::fp16_t>{}(t, a, s);
    }
    else if(t.data_type.compare("bf16") == 0)
    {
        return layernorm2d_bwd_b16_<ck_tile::bf16_t>{}(t, a, s);
    }
    if(r < 0)
        throw std::runtime_error("Without supported instances!");

    return r;
}
