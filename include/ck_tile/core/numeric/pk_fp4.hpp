// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/mxfp_convert.hpp"

namespace ck_tile {

using fp32x2_t = float __attribute__((ext_vector_type(2)));
// TODO: use standard method to convert number
struct pk_float4_e2m1_t
{
    static constexpr int exponent = 2;
    static constexpr int mantissa = 1;
    static constexpr int bias     = 1;
    using T                       = pk_float4_e2m1_t;
    using raw_type                = uint8_t;
    using type                    = raw_type;
    raw_type data;
    // Refer: ONNX 1.19 Documentation
    static constexpr float e2m1_to_fp32_table[16] = {
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6};
    static constexpr fp16_t e2m1_to_fp16_table[16] = {
        // Need to TEST this encoding.
        bit_cast<fp16_t>(static_cast<uint16_t>(0x0000)), //  0
        bit_cast<fp16_t>(static_cast<uint16_t>(0x3800)), //  0.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0x3C00)), //  1
        bit_cast<fp16_t>(static_cast<uint16_t>(0x3E00)), //  1.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4000)), //  2
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4200)), //  3
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4400)), //  4
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4600)), //  6
        bit_cast<fp16_t>(static_cast<uint16_t>(0x8000)), // -0
        bit_cast<fp16_t>(static_cast<uint16_t>(0xB800)), // -0.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0xBC00)), // -1
        bit_cast<fp16_t>(static_cast<uint16_t>(0xBE00)), // -1.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC000)), // -2
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC200)), // -3
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC400)), // -4
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC600))  // -6
    };

    CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t() : data{type{}} {}
    CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t(type init) : data{init} {}
    CK_TILE_HOST_DEVICE constexpr operator type() const { return data; }
    CK_TILE_HOST_DEVICE constexpr operator fp32x2_t() const;
    CK_TILE_HOST_DEVICE constexpr operator float() const;
    CK_TILE_HOST_DEVICE constexpr operator fp16x2_t() const;

    template <index_t I>
    __host__ __device__ inline pk_float4_e2m1_t unpack(number<I>) const
    {
        static_assert(I < 2, "Index is out of range.");
        if constexpr(I == 1)
            return (data >> 4);
        else
            return data & 0b00001111;
    }

    __host__ __device__ static inline pk_float4_e2m1_t pack(const type x0, const type x1)
    {
        return (x1 << 4) | (x0 & 0b00001111);
    }
};

using pk_fp4_t     = pk_float4_e2m1_t;
using pk_fp4_raw_t = typename pk_fp4_t::raw_type;

template <>
struct numeric_traits<pk_fp4_t>
{
    using bitwise_type = pk_fp4_raw_t;

    static constexpr int exp          = 2;
    static constexpr int mant         = 1;
    static constexpr int bias         = 1;
    static constexpr uint8_t abs_mask = 0b01110111;
    static constexpr int PackedSize   = 2;
};

// limits
template <class T>
struct numeric;

template <>
struct numeric<pk_fp4_t>
{
    static constexpr pk_fp4_raw_t binary_min_normal    = 0b00100010; // 1
    static constexpr pk_fp4_raw_t binary_max_normal    = 0b01110111; // 6
    static constexpr pk_fp4_raw_t binary_lowest_normal = 0b11111111; // -6
    static constexpr pk_fp4_raw_t binary_min_subnorm   = 0b00010001; // 0.5
    static constexpr pk_fp4_raw_t binary_max_subnorm   = 0b00010001; // 0.5
    static constexpr pk_fp4_raw_t binary_zero          = 0b00000000; // 0
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t min() { return binary_min_normal; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t max() { return binary_max_normal; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t lowest() { return binary_lowest_normal; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t epsilon() { return binary_min_subnorm; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t round_error() { return binary_min_subnorm; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t zero() { return binary_zero; }
    // TODO: change name denorm -> subnorm
    CK_TILE_HOST_DEVICE static constexpr fp8_t denorm_min() { return binary_min_subnorm; }

    // N/A
    CK_TILE_HOST_DEVICE static constexpr bool has_inf() { return false; }
    // N/A
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t infinity() { return max(); }
    // N/A
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t quiet_NaN() { return max(); }
    // N/A
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t signaling_NaN() { return max(); }
};

CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator fp32x2_t() const
{
    return fp32x2_t{e2m1_to_fp32_table[data & 0xf], e2m1_to_fp32_table[data >> 4]};
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator float() const
{
    return e2m1_to_fp32_table[data & 0xf];
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator fp16x2_t() const
{
    return fp16x2_t{e2m1_to_fp16_table[data & 0xf], e2m1_to_fp16_table[data >> 4]};
}
CK_TILE_HOST_DEVICE constexpr bool operator==(const pk_fp4_t& lhs, const pk_fp4_t& rhs)
{
    return pk_fp4_raw_t(lhs) == pk_fp4_raw_t(rhs);
}
CK_TILE_HOST_DEVICE constexpr bool operator!=(const pk_fp4_t& lhs, const pk_fp4_t& rhs)
{
    return pk_fp4_raw_t(lhs) != pk_fp4_raw_t(rhs);
}

CK_TILE_HOST_DEVICE constexpr pk_fp4_raw_t float_to_e2m1(float x)
{
    return convert_to_type<pk_fp4_t>(x);
    //    // {0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}
    //	pk_fp4_raw_t res = (x<0 ? 0b00001000 : 0);
    //	x = std::abs(x);
    //	if(x < 2.25 ) { res |= int((x + 0.25)*2); }
    //	else if(x < 2.5) { res |= 0b00000100; }
    //	else if(x < 3.5) { res |= 0b00000101; }
    //	else if(x < 5) { res |= 0b00000110; }
    //	else { res |= 0b00000111; }
    //	return res;
}
CK_TILE_HOST_DEVICE constexpr fp32x2_t pk_fp4_to_fp32x2(const pk_fp4_t& x) { return fp32x2_t(x); }
CK_TILE_HOST_DEVICE constexpr pk_fp4_t fp32x2_to_pk_fp4(const fp32x2_t& x)
{
    return pk_fp4_t::pack(float_to_e2m1(x[0]), float_to_e2m1(x[1]));
}

} // namespace ck_tile
