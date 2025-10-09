// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

// DetermineWarpPrecType is a set of rules to determine the right precision type to use
// for the warp GEMM, given the other precision type. This gives rise to a type conversion:
// type conversions are sometimes needed to obtain a pair of types that are compatible with
// the hardware matrix operations available. A typical use case is mixed precision GEMMs.

namespace ck_tile {
// For the most general case, we default to no conversion.
template <typename PrecType, typename OtherPrecType>
struct DetermineWarpPrecType
{
    using prec_type = PrecType;
};

// For pk_int4_t, we convert to the other precision type.
template <typename OtherPrecType>
struct DetermineWarpPrecType<ck_tile::pk_int4_t, OtherPrecType>
{
    using prec_type = OtherPrecType;
};

// For pk_fp4_t, we convert to the other precision type.
template <typename OtherPrecType>
struct DetermineWarpPrecType<ck_tile::pk_fp4_t, OtherPrecType>
{
    using prec_type = OtherPrecType;
};

// For pk_fp4_raw_t, we convert to the other precision type.
template <typename OtherPrecType>
struct DetermineWarpPrecType<ck_tile::pk_fp4_raw_t, OtherPrecType>
{
    using prec_type = OtherPrecType;
};

// For fp8 x bf16 or bf16 x fp8, convert fp8 to float
template <>
struct DetermineWarpPrecType<ck_tile::fp8_t, ck_tile::bf16_t>
{
    using prec_type = float;
};

// For fp8 x bf16 or bf16 x fp8, convert bf16 to float
template <>
struct DetermineWarpPrecType<ck_tile::bf16_t, ck_tile::fp8_t>
{
    using prec_type = float;
};
}; // namespace ck_tile
