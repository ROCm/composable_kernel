// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef CK_CODE_GEN_RTC

#include "ck/utility/type.hpp"

namespace ck {
namespace utils {

// IEEE 754 single precision float constants
struct Float32Constants
{
    static constexpr uint32_t bias      = 127;
    static constexpr uint32_t mant_bits = 23;
    static constexpr uint32_t exp_mask  = 0xFF;
    static constexpr uint32_t mant_mask = 0x7FFFFF;
};

template <int ExponentBits, int MantissaBits>
struct ScaleFormat
{
    using storage_t = uint8_t;

    static_assert(ExponentBits > 0, "ExponentBits must be positive");
    static_assert(MantissaBits >= 0, "MantissaBits must be non-negative");
    static_assert(ExponentBits + MantissaBits <= 8, "Format must fit into 8 bits");

    static constexpr int exponent_bits = ExponentBits;
    static constexpr int mantissa_bits = MantissaBits;
    static constexpr int total_bits    = ExponentBits + MantissaBits;
    static constexpr storage_t mantissa_mask =
        MantissaBits == 0 ? storage_t{0}
                          : static_cast<storage_t>((storage_t{1} << MantissaBits) - 1);
    static constexpr storage_t exponent_mask =
        static_cast<storage_t>((storage_t{1} << ExponentBits) - 1);
    static constexpr storage_t max_exponent = exponent_mask;
    static constexpr storage_t max_finite =
        static_cast<storage_t>((exponent_mask << MantissaBits) | mantissa_mask - 1);
    static constexpr storage_t nan_mask =
        static_cast<storage_t>((exponent_mask << MantissaBits) | mantissa_mask);
    static constexpr storage_t value_mask = storage_t{0xFF};
    static constexpr int bias             = (storage_t{1} << (ExponentBits - 1)) - 1;

    // Rounding constants for mantissa conversion
    static constexpr uint32_t mant_shift      = Float32Constants::mant_bits - MantissaBits;
    static constexpr uint32_t round_bit_shift = mant_shift - 1;
    static constexpr uint32_t sticky_mask     = (uint32_t{1} << round_bit_shift) - 1;
    static constexpr uint32_t mant_max        = (uint32_t{1} << MantissaBits) - 1;
    static constexpr uint32_t implicit_one    = uint32_t{1} << MantissaBits;

    // Minimum exponent for denormal representation
    static constexpr int32_t denorm_min_exp = -(MantissaBits - 1);

    __host__ __device__ static constexpr bool is_nan(storage_t bits)
    {
        return (bits & nan_mask) == nan_mask;
    }

    __host__ __device__ static constexpr int exponent(storage_t bits)
    {
        return static_cast<int>((bits & value_mask) >> MantissaBits);
    }

    /**
     * @brief Encode a float to this format using round-to-nearest-even
     */
    __host__ __device__ static inline storage_t encode(float value)
    {
        // Handle negative values - this is a positive-only format
        if(value < 0.0f)
        {
            return nan_mask;
        }

        // Handle zero
        if(value == 0.0f)
        {
            return 0;
        }

        // Reinterpret float bits
        uint32_t f_bits = bit_cast<uint32_t>(value);

        // Extract float components
        uint32_t f_exp  = (f_bits >> Float32Constants::mant_bits) & Float32Constants::exp_mask;
        uint32_t f_mant = f_bits & Float32Constants::mant_mask;

        // Handle NaN and Inf
        if(f_exp == Float32Constants::exp_mask)
        {
            return nan_mask;
        }

        // Handle denormal float input (flush to zero)
        if(f_exp == 0)
        {
            return 0;
        }

        // Convert exponent from float bias to target format bias
        int32_t exp_unbiased = static_cast<int32_t>(f_exp) - Float32Constants::bias;
        int32_t target_exp   = exp_unbiased + bias;

        // Round mantissa using round-to-nearest-even
        uint32_t target_mant = (f_mant >> mant_shift) & mant_max;
        uint32_t round_bit   = (f_mant >> round_bit_shift) & 0x1;
        uint32_t sticky_bits = f_mant & sticky_mask;

        // Round to nearest even
        bool round_up = false;
        if(round_bit)
        {
            if(sticky_bits != 0)
            {
                round_up = true; // > 0.5 ULP, round up
            }
            else
            {
                // Exactly 0.5 ULP, round to even (round up if LSB is 1)
                round_up = (target_mant & 1) != 0;
            }
        }

        if(round_up)
        {
            target_mant++;
            if(target_mant > mant_max)
            {
                target_mant = 0;
                target_exp++;
            }
        }

        // Handle underflow (exponent too small)
        if(target_exp <= 0)
        {
            // Denormal or underflow
            if(target_exp < denorm_min_exp)
            {
                // Too small, flush to zero
                return 0;
            }
            // Denormal: shift mantissa and set exponent to 0
            uint32_t full_mant = implicit_one + target_mant;
            int32_t shift      = 1 - target_exp;
            // Round the shifted mantissa
            uint32_t shifted_mant = full_mant >> shift;
            uint32_t round_bit_dn = (full_mant >> (shift - 1)) & 0x1;
            uint32_t sticky_dn    = full_mant & ((1 << (shift - 1)) - 1);
            if(round_bit_dn)
            {
                if(sticky_dn != 0 || (shifted_mant & 1))
                {
                    shifted_mant++;
                }
            }
            if(shifted_mant > mant_max)
            {
                // Rounded up to smallest normal
                return static_cast<storage_t>(1 << MantissaBits);
            }
            else
            {
                return static_cast<storage_t>(shifted_mant & mantissa_mask);
            }
        }

        // Handle overflow (exponent too large)
        if(target_exp > max_exponent || (target_exp == max_exponent && target_mant == mant_max))
        {
            return max_finite;
        }

        // Normal case: pack exponent and mantissa
        return static_cast<storage_t>((target_exp << MantissaBits) | target_mant);
    }

    /**
     * @brief Decode this format to float
     */
    __host__ __device__ static inline float decode(storage_t bits)
    {
        // Handle NaN
        if(is_nan(bits))
        {
            return std::numeric_limits<float>::quiet_NaN();
        }

        int32_t exp_field  = static_cast<int32_t>((bits >> MantissaBits) & exponent_mask);
        int32_t mant_field = static_cast<int32_t>(bits & mantissa_mask);

        float ulp = powf(2.0f, -static_cast<float>(mantissa_bits));

        // Handle denormal
        if(exp_field == 0)
        {
            int32_t exp_value = 1;
            return powf(2.0f, static_cast<float>(exp_value - bias)) *
                   static_cast<float>(mant_field) * ulp;
        }
        else
        {
            return powf(2.0f, static_cast<float>(exp_field - bias)) *
                   (1.0f + static_cast<float>(mant_field) * ulp);
        }
    }
};

template <typename T>
__host__ __device__ inline constexpr int32_t get_exponent_value(T x);

} // namespace utils
} // namespace ck

#endif
