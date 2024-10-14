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

template <typename DataType>
struct LayerNormTypeConfig;

template <>
struct LayerNormTypeConfig<ck_tile::half_t>
{
    using XDataType     = ck_tile::half_t;
    using YDataType     = ck_tile::half_t;
    using GammaDataType = ck_tile::half_t;
    using BetaDataType  = ck_tile::half_t;
#ifdef SAVE_MEAN_INV_STD
    using MeanDataType   = ck_tile::half_t;
    using InvStdDataType = ck_tile::half_t;
#else
    using MeanDataType   = ck_tile::null_type;
    using InvStdDataType = ck_tile::null_type;
#endif
    using ComputeDataType = float;
};

template <>
struct LayerNormTypeConfig<float>
{
    using XDataType     = float;
    using YDataType     = float;
    using GammaDataType = float;
    using BetaDataType  = float;
#ifdef SAVE_MEAN_INV_STD
    using MeanDataType   = float;
    using InvStdDataType = float;
#else
    using MeanDataType   = ck_tile::null_type;
    using InvStdDataType = ck_tile::null_type;
#endif
    using ComputeDataType = float;
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
