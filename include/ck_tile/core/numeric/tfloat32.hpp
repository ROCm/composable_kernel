// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/numeric.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

#include <cmath>
#include <stdio.h>

namespace ck_tile {

enum class tf32_rounding_mode
{
    trunc = 0, // truncate
    rne   = 1  // round to nearest even (RNE)
};

class alignas(4) tfloat32_t
{
    public:
    using type     = tfloat32_t;
    using raw_type = uint32_t;

    raw_type data;

    // default constructor
    CK_TILE_HOST_DEVICE constexpr tfloat32_t() : data() {}

    // construct from float
    CK_TILE_HOST_DEVICE
    explicit constexpr tfloat32_t(const float& x) : data(to_raw_type(x)) {}

    // cast to float
    CK_TILE_HOST_DEVICE constexpr operator float() const { return bit_cast<float>(data); }

    private:
    template <tf32_rounding_mode rounding =
                  static_cast<tf32_rounding_mode>(CK_TILE_FLOAT_TO_TF32_DEFAULT)>
    static CK_TILE_HOST_DEVICE constexpr raw_type to_raw_type(float x)
    {
        constexpr raw_type f32_exp_mask = 0x7F800000u;
        constexpr raw_type bit_mask     = 0xFFFFE000u;

        raw_type i = bit_cast<raw_type>(x);
        if constexpr(rounding == tf32_rounding_mode::rne)
        {
            // RNE rounding.
            if((i & f32_exp_mask) != f32_exp_mask)
            {
                // Add rounding bias for round-to-nearest-even (RNE) before truncating:
                //  - 0xFFFu is the rounding bias corresponding to the 13 fraction bits that
                //    will be discarded.
                //  - (i >> 13) & 1u extracts the least significant of those discarded bits and
                //    adding it implements "ties to even" (round half-way cases to even).
                i += 0xFFFu + ((i >> 13) & 1u);
            }
        }
        // Zero out the lowest 13 fraction bits to form the TF32-like value.
        i &= bit_mask;
        return i;
    }
};

using tf32_t     = tfloat32_t;
using tf32_raw_t = tfloat32_t::raw_type;

template <>
CK_TILE_HOST_DEVICE constexpr tf32_t bit_cast(const tf32_raw_t& x)
{
    if(x & ~0xFFFFE000u)
    {
        int line = __LINE__ - 2;
        printf(
            "%s:%d: \033[31merror:\033[0m %s failed due to low 13 bits must be zero, got 0x%08x\n",
            __FILE__,
            line,
            __func__,
            x);
        printf("%5d |     \033[34mif\033[0m(\033[32mx & ~0xFFFFE000u\033[0m)\n", line);
        printf("%5s |        \033[32m^~~~~~~~~~~~~~~~\033[0m\n", "");
        __builtin_trap();
    }
    tf32_t y;
    y.data = x;
    return y;
}

template <typename>
struct native_t;

template <>
struct native_t<tf32_t>
{
    using type = float;
};

template <>
struct numeric<tf32_t>
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr tf32_t min() { return bit_cast<tf32_t>(0x00800000u); }

    // minimum finite value
    CK_TILE_HOST_DEVICE static constexpr tf32_t lowest() { return bit_cast<tf32_t>(0xFF7FE000u); }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr tf32_t max() { return bit_cast<tf32_t>(0x7F7FE000u); }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr tf32_t epsilon() { return bit_cast<tf32_t>(0x3A800000u); }

    CK_TILE_HOST_DEVICE static constexpr tf32_t round_error()
    {
        return bit_cast<tf32_t>(0x3F000000u);
    }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr tf32_t infinity() { return bit_cast<tf32_t>(0x7F800000u); }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr tf32_t quiet_NaN()
    {
        return bit_cast<tf32_t>(0x7FC00000u);
    }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr tf32_t signaling_NaN()
    {
        return bit_cast<tf32_t>(0x7F802000u);
    }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr tf32_t denorm_min()
    {
        return bit_cast<tf32_t>(0x00002000u);
    }

    CK_TILE_HOST_DEVICE static constexpr tf32_t zero() { return bit_cast<tf32_t>(0u); }

    CK_TILE_HOST_DEVICE static constexpr tf32_t one() { return bit_cast<tf32_t>(0x3F800000u); }

    CK_TILE_HOST_DEVICE static constexpr tf32_t log2e()
    {
        return static_cast<tf32_t>(1.44269504088896340736);
    }
};

template <>
struct numeric_traits<tf32_t>
{
    static constexpr int exp            = 8;
    static constexpr int mant           = 10;
    static constexpr int bias           = 127;
    static constexpr uint32_t nan_mask  = 0x7F800000u;
    static constexpr uint32_t head_mask = 0xFF800000u;
    static constexpr uint32_t mant_mask = 0x7FE000u;
    static constexpr uint32_t exp_mask  = 0xFFu;
    static constexpr uint32_t abs_mask  = 0x7FFFFFFFu;
    static constexpr uint32_t Inf       = 0x7F800000u;
    static constexpr uint32_t NegInf    = 0xFF800000u;
    static constexpr uint32_t NaN       = 0x7F802000u;
    static constexpr uint32_t Neg0      = 0x80000000u;
    static constexpr int PackedSize     = 1;
    using bitwise_type                  = tf32_raw_t;
};

CK_TILE_ARITHMETIC_USING_FLOAT(CK_TILE_HOST_DEVICE, tf32_t)

// math
CK_TILE_HOST_DEVICE
tf32_t abs(const tf32_t& x)
{
    return bit_cast<tf32_t>(bit_cast<tf32_raw_t>(x) & numeric_traits<tf32_t>::abs_mask);
}

CK_TILE_HOST_DEVICE constexpr bool isnan(const tf32_t& x)
{
    tf32_raw_t xx = bit_cast<tf32_raw_t>(x);
    return (xx & numeric_traits<tf32_t>::abs_mask) > numeric_traits<tf32_t>::Inf;
}

CK_TILE_HOST
tf32_t sqrt(tf32_t x) { return static_cast<tf32_t>(std::sqrtf(static_cast<float>(x))); };

CK_TILE_DEVICE
tf32_t sqrt(tf32_t x)
{
    return static_cast<tf32_t>(__builtin_amdgcn_sqrtf(static_cast<float>(x)));
};

CK_TILE_HOST
tf32_t exp(tf32_t x) { return static_cast<tf32_t>(std::expf(static_cast<float>(x))); };

CK_TILE_DEVICE
tf32_t exp(tf32_t x) { return static_cast<tf32_t>(__ocml_exp_f32(static_cast<float>(x))); };

CK_TILE_HOST
tf32_t exp2(tf32_t x) { return static_cast<tf32_t>(std::exp2f(static_cast<float>(x))); };

CK_TILE_DEVICE
tf32_t exp2(tf32_t x) { return static_cast<tf32_t>(__ocml_exp2_f32(static_cast<float>(x))); };

CK_TILE_HOST
tf32_t log(tf32_t x) { return static_cast<tf32_t>(std::logf(static_cast<float>(x))); };

CK_TILE_DEVICE
tf32_t log(tf32_t x) { return static_cast<tf32_t>(__logf(static_cast<float>(x))); };

} // namespace ck_tile
