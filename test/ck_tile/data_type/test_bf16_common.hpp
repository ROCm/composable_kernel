// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "test_common.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <cstdio>

namespace ck_tile_test {

// Bit pattern verification for bf16
inline uint16_t bf16_to_bits(ck_tile::bf16_t x) { return ck_tile::bit_cast<uint16_t>(x); }

inline ck_tile::bf16_t bits_to_bf16(uint16_t bits)
{
    return ck_tile::bit_cast<ck_tile::bf16_t>(bits);
}

// Extract sign, exponent, and mantissa from bf16 bit pattern
inline void decompose_bf16(uint16_t bits, bool& sign, uint8_t& exp, uint8_t& mant)
{
    sign = (bits >> 15) & 1;
    exp  = (bits >> 7) & 0xFF;
    mant = bits & 0x7F;
}

// Test data generators
inline std::vector<float> generate_test_floats()
{
    std::vector<float> values;

    // Special values
    values.push_back(0.0f);
    values.push_back(-0.0f);
    values.push_back(std::numeric_limits<float>::infinity());
    values.push_back(-std::numeric_limits<float>::infinity());
    values.push_back(std::numeric_limits<float>::quiet_NaN());
    values.push_back(std::numeric_limits<float>::signaling_NaN());

    // Powers of 2
    for(int i = -126; i <= 127; i++)
    {
        values.push_back(std::ldexp(1.0f, i));
        values.push_back(-std::ldexp(1.0f, i));
    }

    // Normal values
    values.push_back(1.0f);
    values.push_back(-1.0f);
    values.push_back(0.5f);
    values.push_back(-0.5f);
    values.push_back(2.0f);
    values.push_back(-2.0f);

    // Values near bf16 limits
    values.push_back(3.38953139e38f); // Near bf16 max
    values.push_back(-3.38953139e38f);
    values.push_back(1.175494e-38f); // Near bf16 min normal
    values.push_back(-1.175494e-38f);

    // Subnormal values
    values.push_back(std::numeric_limits<float>::denorm_min());
    values.push_back(-std::numeric_limits<float>::denorm_min());
    values.push_back(std::numeric_limits<float>::min());
    values.push_back(-std::numeric_limits<float>::min());

    // Values that require rounding in bf16
    values.push_back(1.001953125f); // Requires rounding
    values.push_back(-1.001953125f);
    values.push_back(0.99951171875f); // Close to 1.0
    values.push_back(-0.99951171875f);

    // Random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for(int i = 0; i < 100; i++)
    {
        values.push_back(dist(gen));
    }

    return values;
}

inline std::vector<ck_tile::bf16_t> generate_special_bf16_values()
{
    std::vector<ck_tile::bf16_t> values;

    // Using numeric traits
    values.push_back(ck_tile::numeric<ck_tile::bf16_t>::zero());
    values.push_back(ck_tile::numeric<ck_tile::bf16_t>::min());
    values.push_back(ck_tile::numeric<ck_tile::bf16_t>::max());
    values.push_back(ck_tile::numeric<ck_tile::bf16_t>::lowest());
    values.push_back(ck_tile::numeric<ck_tile::bf16_t>::epsilon());
    values.push_back(ck_tile::numeric<ck_tile::bf16_t>::round_error());
    values.push_back(ck_tile::numeric<ck_tile::bf16_t>::infinity());
    values.push_back(ck_tile::numeric<ck_tile::bf16_t>::quiet_NaN());
    values.push_back(ck_tile::numeric<ck_tile::bf16_t>::signaling_NaN());
    values.push_back(ck_tile::numeric<ck_tile::bf16_t>::denorm_min());

    // Add negative zero
    values.push_back(bits_to_bf16(0x8000));

    // Add some specific bit patterns
    values.push_back(bits_to_bf16(0x3F80)); // 1.0
    values.push_back(bits_to_bf16(0xBF80)); // -1.0
    values.push_back(bits_to_bf16(0x4000)); // 2.0
    values.push_back(bits_to_bf16(0xC000)); // -2.0
    values.push_back(bits_to_bf16(0x3F00)); // 0.5
    values.push_back(bits_to_bf16(0xBF00)); // -0.5

    return values;
}

// Helper function to check if two bf16 values are equal (IEEE-compliant: NaN != NaN)
inline bool bf16_equal(ck_tile::bf16_t a, ck_tile::bf16_t b)
{
    // IEEE 754: NaN is never equal to NaN (or anything else)
    if(ck_tile::isnan(a) || ck_tile::isnan(b))
    {
        return false;
    }
    return bf16_to_bits(a) == bf16_to_bits(b);
}

// Helper function for near equality with ULP tolerance (IEEE-compliant: NaN != NaN)
inline bool bf16_near_equal(ck_tile::bf16_t a, ck_tile::bf16_t b, int ulp_tolerance = 1)
{
    // IEEE 754: NaN is never equal to NaN (or anything else)
    if(ck_tile::isnan(a) || ck_tile::isnan(b))
    {
        return false;
    }
    // Use bf16_to_float for proper conversion regardless of bf16_t implementation
    float fa = ck_tile::bf16_to_float(a);
    float fb = ck_tile::bf16_to_float(b);
    return ulp_distance(fa, fb) <= static_cast<uint64_t>(ulp_tolerance);
}

// Helper to print bf16 value with bit pattern for debugging
inline std::string bf16_to_string(ck_tile::bf16_t x)
{
    uint16_t bits = bf16_to_bits(x);
    float f       = ck_tile::bf16_to_float(x); // Use the proper conversion function
    bool sign;
    uint8_t exp, mant;
    decompose_bf16(bits, sign, exp, mant);

    char buffer[256];
    snprintf(buffer,
             sizeof(buffer),
             "bf16(bits=0x%04X, sign=%d, exp=%u, mant=%u, float=%.6g)",
             bits,
             sign,
             exp,
             mant,
             f);
    return std::string(buffer);
}

// Test fixture base class
class Bf16TestBase : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        // Common setup if needed
    }

    void TearDown() override
    {
        // Common teardown if needed
    }
};

} // namespace ck_tile_test
