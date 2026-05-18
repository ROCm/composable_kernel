// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/scale_util.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

namespace ck_tile {

struct e4m3_bexp_t
{
    using raw_type = uint8_t;
    using type     = raw_type;

    raw_type data;

    CK_TILE_HOST_DEVICE constexpr e4m3_bexp_t() : data{type{0b01111111}} {}
    CK_TILE_HOST_DEVICE explicit constexpr e4m3_bexp_t(type init) : data{init} {}
    CK_TILE_HOST_DEVICE explicit constexpr e4m3_bexp_t(float scale)
        : e4m3_bexp_t(static_cast<type>(numeric_utils<float>::get_exponent(scale)))
    {
    }
    CK_TILE_HOST_DEVICE constexpr operator type() const { return data; }
    CK_TILE_HOST_DEVICE constexpr raw_type& get() { return data; }
    CK_TILE_HOST_DEVICE constexpr raw_type get() const { return data; }
    CK_TILE_HOST_DEVICE operator float() const;

    constexpr bool operator==(const e4m3_bexp_t& other) const { return data == other.data; }

    constexpr bool operator!=(const e4m3_bexp_t& other) const { return data != other.data; }
};

using e4m3_t     = e4m3_bexp_t;
using e4m3_raw_t = typename e4m3_t::raw_type;

template <>
struct numeric_traits<e4m3_t>
{
    using bitwise_type = e4m3_raw_t;

    static constexpr int exp        = 4;
    static constexpr int mant       = 3;
    static constexpr int bias       = 7;
    static constexpr int PackedSize = 1;
};

// limits
template <class T>
struct numeric;

template <>
struct numeric<e4m3_t>
{
    static constexpr e4m3_raw_t binary_min = 0b00000000;
    static constexpr e4m3_raw_t binary_max = 0b01111110;
    static constexpr e4m3_raw_t binary_nan = 0b01111111;
    CK_TILE_HOST_DEVICE static constexpr e4m3_t min() { return e4m3_t{binary_min}; }
    CK_TILE_HOST_DEVICE static constexpr e4m3_t max() { return e4m3_t{binary_max}; }
    CK_TILE_HOST_DEVICE static constexpr e4m3_t quiet_NaN() { return e4m3_t{binary_nan}; }
    CK_TILE_HOST_DEVICE static constexpr e4m3_t signaling_NaN() { return e4m3_t{binary_nan}; }
    CK_TILE_HOST_DEVICE static constexpr bool has_inf() { return false; }

    CK_TILE_HOST_DEVICE static constexpr e4m3_t epsilon() { return signaling_NaN(); }
    CK_TILE_HOST_DEVICE static constexpr e4m3_t round_error() { return signaling_NaN(); }
    CK_TILE_HOST_DEVICE static constexpr e4m3_t zero()
    {
        return e4m3_t{static_cast<e4m3_raw_t>(0b00000000)};
    }
    CK_TILE_HOST_DEVICE static constexpr e4m3_t infinity() { return signaling_NaN(); }
};

CK_TILE_HOST_DEVICE e4m3_bexp_t::operator float() const
{
#if defined(__gfx1250__)
    union
    {
        unsigned int i32val;
        uint8_t i8val[4];
    } val;
    val.i8val[0] = this->data;
    return __builtin_amdgcn_cvt_f32_fp8(val.i32val, false);
#else
    return ScaleUtils<4, 3>::decode(this->data);
#endif
}

} // namespace ck_tile
#pragma clang diagnostic pop
