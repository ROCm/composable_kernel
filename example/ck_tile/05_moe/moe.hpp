// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"

#include <type_traits>

template <typename DataType>
struct MoeTypeConfig;

template <>
struct MoeTypeConfig<ck_tile::half_t>
{
    using ADataType             = ck_tile::half_t;
    using GDataType             = ck_tile::half_t;
    using UDataType             = ck_tile::half_t;
    using DDataType          = ck_tile::half_t;
    using AccDataType = float;
    using ScaleDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
   // data type for second gemm accumulation
    using ODataType             = ck_tile::half_t;
};

template <>
struct MoeTypeConfig<ck_tile::bf16_t>
{
    using ADataType             = ck_tile::bf16_t;
    using GDataType             = ck_tile::bf16_t;
    using UDataType             = ck_tile::bf16_t;
    using DDataType          = ck_tile::bf16_t;
    using AccDataType = float;
    using ScaleDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
   // data type for second gemm accumulation
    using ODataType             = ck_tile::bf16_t;
};

template <>
struct MoeTypeConfig<ck_tile::fp8_t>
{
    using ADataType             = ck_tile::fp8_t;
    using GDataType             = ck_tile::fp8_t;
    using UDataType             = ck_tile::fp8_t;
    using DDataType          = ck_tile::fp8_t;
    using AccDataType = float;
    using ScaleDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
   // data type for second gemm accumulation
    using ODataType             = ck_tile::fp8_t;
};

template <>
struct MoeTypeConfig<ck_tile::bf8_t>
{
    using ADataType             = ck_tile::bf8_t;
    using GDataType             = ck_tile::bf8_t;
    using UDataType             = ck_tile::bf8_t;
    using DDataType          = ck_tile::bf8_t;
    using AccDataType = float;
    using ScaleDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
   // data type for second gemm accumulation
    using ODataType             = ck_tile::bf8_t;
};

//float fmha_fwd(fmha_fwd_traits, fmha_fwd_args, const ck_tile::stream_config&);
