// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/layernorm2d.hpp"
#include <string>

struct layernorm2d_fwd_traits
{
    std::string data_type;
};

struct layernorm2d_fwd_args
{
    const void* p_x;
    const void* p_gamma;
    const void* p_beta;
    void* p_y;
    void* p_mean;
    void* p_invStd;
    float epsilon;
    ck_tile::index_t M;
    ck_tile::index_t N;
};

// host API
float layernorm2d_fwd(layernorm2d_fwd_traits, layernorm2d_fwd_args, const ck_tile::stream_config&);
