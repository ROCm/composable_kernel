/*
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>

template <typename InOutDataType>
struct HstuAttentionFwdTypeConfig;

template <>
struct HstuAttentionFwdTypeConfig<ck_tile::fp16_t>
{
    using BiasDataType    = ck_tile::fp16_t;
    using GemmAccDataType = float;
    using CompDataType    = float; // data type for non-linear calculation
    using OaccDataType    = GemmAccDataType;
    using ODataType       = ck_tile::fp16_t;
};

template <>
struct HstuAttentionFwdTypeConfig<ck_tile::bf16_t>
{
    using BiasDataType    = ck_tile::bf16_t;
    using GemmAccDataType = float;
    using CompDataType    = float; // data type for non-linear calculation
    using OaccDataType    = GemmAccDataType;
    using ODataType       = ck_tile::bf16_t;
};

static constexpr bool IsVLayoutRowMajor = true;
