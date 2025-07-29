// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/type.hpp"

namespace ck_tile {

/**
 * @brief Unsigned representation of a conventional biased Float32 exponent.
 *
 * bias = 127;
 *
 * E8M0_1   = 0b01111111; => 2^(127-127) = 1
 * E8M0_2   = 0b10000000; => 2^(128-127) = 2^1 = 2
 * E8M0_3   = 0b10000010; => 2^(130-127) = 2^3 = 8
 * E8M0_135 = 0b10000111; => 2^(135-127) = 2^8 = 256
 * E8M0_142 = 0b10001110; => 2^(142-127) = 2^15 = 32768
 * E8M0_MIN = 0b00000000; => 2^-127
 * E8M0_MAX = 0b11111110; => 2^127
 * E8M0_NAN = 0b11111111; => NaN
 */
struct e8m0_bexp_t
{
    using type = uint8_t;
    type data;

    constexpr static type bias     = 127;
    constexpr static type nan_mask = 0xFF;

    __host__ __device__ constexpr e8m0_bexp_t() : data{type{}} {}
    __host__ __device__ constexpr e8m0_bexp_t(type init) : data{init} {}
    __host__ __device__ constexpr e8m0_bexp_t(int init) : data{static_cast<type>(init & nan_mask)}
    {
    }
    __host__ __device__ explicit constexpr e8m0_bexp_t(float scale)
        : data{static_cast<type>((bit_cast<uint32_t>(scale) & (nan_mask << 23)) >> 23)}
    {
    }

    __host__ __device__ explicit constexpr operator float() const
    {
        if(data == nan_mask || data == 0)
        {
            uint32_t bits = data << 1;
            bits |= 1;
            bits <<= 22;
            return bit_cast<float>(bits);
        }
        else
        {
            uint32_t bits = data << 23;
            return bit_cast<float>(bits);
        }
    }

    __host__ __device__ constexpr bool operator==(const e8m0_bexp_t& other) const
    {
        // strict IEEE compliance for NaN
        return data == other.data && data != nan_mask;
    }

    __host__ __device__ constexpr bool is_nan() const { return data == nan_mask; }
};

// limits
template <class T>
struct numeric;

template <>
struct numeric<e8m0_bexp_t>
{
    static constexpr e8m0_bexp_t binary_min  = 0x00; // 0b00000000
    static constexpr e8m0_bexp_t binary_max  = 0xFE; // 0b11111110
    static constexpr e8m0_bexp_t binary_qnan = 0xFF; // 0b11111111
    static constexpr e8m0_bexp_t binary_1    = 0x7F; // 0b01111111
    static constexpr e8m0_bexp_t binary_2    = 0x80; // 0b10000000
    static constexpr e8m0_bexp_t binary_3    = 0x82; // 0b10000010
    static constexpr e8m0_bexp_t binary_135  = 0x87; // 0b10000111
    static constexpr e8m0_bexp_t binary_142  = 0x8E; // 0b10001110

    CK_TILE_HOST_DEVICE static constexpr e8m0_bexp_t Min() { return e8m0_bexp_t(binary_min); }
    CK_TILE_HOST_DEVICE static constexpr e8m0_bexp_t Max() { return e8m0_bexp_t(binary_max); }
    CK_TILE_HOST_DEVICE static constexpr e8m0_bexp_t QuietNaN() { return e8m0_bexp_t(binary_qnan); }
    CK_TILE_HOST_DEVICE static constexpr e8m0_bexp_t Binary_1() { return e8m0_bexp_t(binary_1); }
    CK_TILE_HOST_DEVICE static constexpr e8m0_bexp_t Binary_2() { return e8m0_bexp_t(binary_2); }
    CK_TILE_HOST_DEVICE static constexpr e8m0_bexp_t Binary_3() { return e8m0_bexp_t(binary_3); }
    CK_TILE_HOST_DEVICE static constexpr e8m0_bexp_t Binary_135()
    {
        return e8m0_bexp_t(binary_135);
    }
    CK_TILE_HOST_DEVICE static constexpr e8m0_bexp_t Binary_142()
    {
        return e8m0_bexp_t(binary_142);
    }
};
}

template <>
struct numeric_traits<e8m0_bexp_t>
{
    static constexpr int PackedSize = 2;
};

namespace utils {

template <typename T>
__host__ __device__ inline constexpr int32_t get_exponent_value(T x);

template <>
__host__ __device__ inline constexpr int32_t get_exponent_value<e8m0_bexp_t>(e8m0_bexp_t x)
{
    return x.data;
}

} // namespace utils

} // namespace ck_tile
