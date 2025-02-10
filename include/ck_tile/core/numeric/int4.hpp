// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/numeric/numeric.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/random.hpp"
#include <stdint.h>
#include <type_traits>

#pragma once

namespace ck_tile {

// 8 bit int4
struct int4x2_t
{
    uint8_t raw;
    CK_TILE_HOST_DEVICE constexpr int4x2_t() : raw{uint8_t{}} {}
    // CK_TILE_HOST_DEVICE constexpr int4x2_t(uint8_t init) : raw{((init & 0x0f) << 4) | (init & 0x0f)}
    // {
    // }
};

CK_TILE_HOST_DEVICE
constexpr fp32x2_t int4x2_to_floatx2(const int4x2_t& x)
{
    auto x_u8 = x.raw;
    // naive implement
    float x_h = ((x_u8 & 0xf0) >> 4);
    if(x_h >= 8)
    {
        x_h -= 16;
    }

    float x_l = (x_u8 & 0x0f);
    if(x_l >= 8)
    {
        x_l -= 16;
    }

    return {x_h, x_l};
}

CK_TILE_HOST_DEVICE
constexpr int4x2_t floatx2_to_int4x2(const fp32x2_t& x)
{
    // naive implement
    int4x2_t res;
    auto x_l = static_cast<int8_t>(x.x);
    auto x_h = static_cast<int8_t>(x.y);

    res.raw = (x_l << 4) | (x_h & 0x0F);

    return res;
}

} // namespace ck_tile
