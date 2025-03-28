// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

// Type configuration
template <typename DataType>
struct HSTUAttentionTypeConfig;

template <>
struct HSTUAttentionTypeConfig<ck_tile::fp16_t>
{
    using GemmAccDataType   = float;
    using SMComputeDataType = float;
};

template <>
struct HSTUAttentionTypeConfig<ck_tile::bf16_t>
{
    using GemmAccDataType   = float;
    using SMComputeDataType = float;
};
