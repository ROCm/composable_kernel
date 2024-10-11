// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"

enum class ActivationType
{
    Gelu = 0,
    Relu,
    Silu,
    Swiglu,
    Geglu,
    Identity,
    GeluNoneApproximate,
    GeGluNoneApproximate,
    InvalidType
};
struct GemmBiasAddArgs
{
    const void* mat_a;
    const void* mat_b;
    const void* mat_bias;
    void* mat_c;
    ck::index_t M;
    ck::index_t N;
    ck::index_t K;
};

float gemm_bias_add_fp16(const GemmBiasAddArgs& args, const StreamConfig& config);
