// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

// DetermineWarpPrecType is a set of rules to determine the right precision type to use
// for the warp GEMM, given the other precision type. This gives rise to a type conversion:
// type conversions are sometimes needed to obtain a pair of types that are compatible with
// the hardware matrix operations available. A typical use case is mixed precision GEMMs.

namespace ck_tile {
// For the most general case, default to no conversion.
template <typename APrecType, typename BPrecType>
struct DetermineWarpPrecType
{
    using a_prec_type = APrecType;
    using b_prec_type = BPrecType;
};

// For pk_fp4_t x pk_fp4_t, keep pk_fp4_t
template <>
struct DetermineWarpPrecType<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t>
{
    using a_prec_type = ck_tile::pk_fp4_t;
    using b_prec_type = ck_tile::pk_fp4_t;
};

// For pk_int4_t x B, use the B type.
template <typename BPrecType>
struct DetermineWarpPrecType<ck_tile::pk_int4_t, BPrecType>
{
    using a_prec_type = BPrecType;
    using b_prec_type = BPrecType;
};

// For A x pk_int4_t, use the A type.
template <typename APrecType>
struct DetermineWarpPrecType<APrecType, ck_tile::pk_int4_t>
{
    using a_prec_type = APrecType;
    using b_prec_type = APrecType;
};

// For pk_fp4_t x B, use the B type.
template <typename BPrecType>
struct DetermineWarpPrecType<ck_tile::pk_fp4_t, BPrecType>
{
    using a_prec_type = BPrecType;
    using b_prec_type = BPrecType;
};

// For A x pk_fp4_t, use the A type.
template <typename APrecType>
struct DetermineWarpPrecType<APrecType, ck_tile::pk_fp4_t>
{
    using a_prec_type = APrecType;
    using b_prec_type = APrecType;
};

// For B x pk_fp4_raw_t, use the B type.
template <typename BPrecType>
struct DetermineWarpPrecType<ck_tile::pk_fp4_raw_t, BPrecType>
{
    using a_prec_type = BPrecType;
    using b_prec_type = BPrecType;
};

// For A x pk_fp4_raw_t, use the A type.
template <typename APrecType>
struct DetermineWarpPrecType<APrecType, ck_tile::pk_fp4_raw_t>
{
    using a_prec_type = APrecType;
    using b_prec_type = APrecType;
};

// For fp8 x bf16, use fp8
template <>
struct DetermineWarpPrecType<ck_tile::fp8_t, ck_tile::bf16_t>
{
    using a_prec_type = ck_tile::fp8_t;
    using b_prec_type = ck_tile::fp8_t;
};

// For bf16 x fp8, use bf16
template <>
struct DetermineWarpPrecType<ck_tile::bf16_t, ck_tile::fp8_t>
{
    using a_prec_type = ck_tile::bf16_t;
    using b_prec_type = ck_tile::bf16_t;
};

// For fp8 x fp16, use fp8
template <>
struct DetermineWarpPrecType<ck_tile::fp8_t, ck_tile::half_t>
{
    using a_prec_type = ck_tile::fp8_t;
    using b_prec_type = ck_tile::fp8_t;
};

// For fp16 x fp8, use fp16
template <>
struct DetermineWarpPrecType<ck_tile::half_t, ck_tile::fp8_t>
{
    using a_prec_type = ck_tile::half_t;
    using b_prec_type = ck_tile::half_t;
};
}; // namespace ck_tile
