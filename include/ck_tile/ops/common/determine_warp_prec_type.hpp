// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
}; // namespace ck_tile
