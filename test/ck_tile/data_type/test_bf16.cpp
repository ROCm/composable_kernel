// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_bf16_common.hpp"
#include <cmath>

using namespace ck_tile;
using namespace ck_tile_test;
#include <cmath>
#include <chrono>
#include <type_traits>
#include <hip/hip_runtime.h>
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/device_prop.hpp"

using namespace ck_tile;
using namespace ck_tile_test;

// ============================================================================
// Tests from test_bf16_conversion.cpp
// ============================================================================

class Bf16ConversionTest : public Bf16TestBase
{
};

// Test float to bf16 conversion with default rounding mode
TEST_F(Bf16ConversionTest, FloatToBf16Basic)
{
    // Test exact representable values
    {
        float f  = 1.0f;
        bf16_t b = float_to_bf16(f);
        EXPECT_EQ(bf16_to_bits(b), 0x3F80);
        EXPECT_EQ(static_cast<float>(b), 1.0f);
    }

    {
        float f  = -1.0f;
        bf16_t b = float_to_bf16(f);
        EXPECT_EQ(bf16_to_bits(b), 0xBF80);
        EXPECT_EQ(static_cast<float>(b), -1.0f);
    }

    {
        float f  = 2.0f;
        bf16_t b = float_to_bf16(f);
        EXPECT_EQ(bf16_to_bits(b), 0x4000);
        EXPECT_EQ(static_cast<float>(b), 2.0f);
    }

    {
        float f  = 0.5f;
        bf16_t b = float_to_bf16(f);
        EXPECT_EQ(bf16_to_bits(b), 0x3F00);
        EXPECT_EQ(static_cast<float>(b), 0.5f);
    }
}

// Test special values
TEST_F(Bf16ConversionTest, FloatToBf16SpecialValues)
{
    // Zero
    {
        bf16_t b = float_to_bf16(0.0f);
        EXPECT_EQ(bf16_to_bits(b), 0x0000);
        EXPECT_EQ(static_cast<float>(b), 0.0f);
    }

    // Negative zero
    {
        bf16_t b = float_to_bf16(-0.0f);
        EXPECT_EQ(bf16_to_bits(b), 0x8000);
        EXPECT_EQ(static_cast<float>(b), -0.0f);
    }

    // Infinity
    {
        bf16_t b = float_to_bf16(std::numeric_limits<float>::infinity());
        EXPECT_EQ(bf16_to_bits(b), 0x7F80);
        EXPECT_TRUE(std::isinf(static_cast<float>(b)));
        EXPECT_TRUE(static_cast<float>(b) > 0);
    }

    // Negative infinity
    {
        bf16_t b = float_to_bf16(-std::numeric_limits<float>::infinity());
        EXPECT_EQ(bf16_to_bits(b), 0xFF80);
        EXPECT_TRUE(std::isinf(static_cast<float>(b)));
        EXPECT_TRUE(static_cast<float>(b) < 0);
    }

    // NaN
    {
        bf16_t b = float_to_bf16(std::numeric_limits<float>::quiet_NaN());
        EXPECT_TRUE(isnan(b));
        EXPECT_TRUE((bf16_to_bits(b) & 0x7F80) == 0x7F80); // Exponent all 1s
        EXPECT_TRUE((bf16_to_bits(b) & 0x007F) != 0);      // Mantissa not zero
    }
}

// Test rounding behavior
TEST_F(Bf16ConversionTest, FloatToBf16Rounding)
{
    // Test round-to-nearest-even (default mode)
    {
        // Value that requires rounding, should round to nearest bf16 value
        float f      = 1.001953125f; // Between 1.0 and 1.0078125, closer to 1.0
        bf16_t b     = float_to_bf16(f);
        float result = static_cast<float>(b);
        // Should round to nearest: 1.0f (since 1.001953125 is closer to 1.0 than 1.0078125)
        EXPECT_EQ(result, 1.0f);
    }

    // Test values that require rounding
    {
        float f      = 1.0009765625f; // Not exactly representable in bf16, closer to 1.0
        bf16_t b     = float_to_bf16(f);
        float result = static_cast<float>(b);
        // Should round to nearest: 1.0f (since 1.0009765625 is closer to 1.0 than 1.0078125)
        EXPECT_EQ(result, 1.0f);
    }
}

// Test different rounding modes
TEST_F(Bf16ConversionTest, FloatToBf16RoundingModes)
{
    // Standard rounding (round-to-nearest-even)
    {
        bf16_t b = float_to_bf16(1.001953125f, constant<bf16_rounding_mode::standard>{});
        // Verify exact bit pattern to validate rounding mode semantics
        // 1.001953125f (0x3f804000) should round to 1.0f (0x3F80) with round-to-nearest
        EXPECT_EQ(bf16_to_bits(b), 0x3F80);
        float result = static_cast<float>(b);
        EXPECT_EQ(result, 1.0f);
    }

    // Truncation mode (round-toward-zero, no rounding)
    {
        bf16_t b = float_to_bf16(1.001953125f, constant<bf16_rounding_mode::truncate>{});
        // Verify exact bit pattern: truncate just shifts right by 16 bits
        // 1.001953125f (0x3f804000) >> 16 = 0x3f80 = 1.0f
        EXPECT_EQ(bf16_to_bits(b), 0x3F80);
        float result = static_cast<float>(b);
        EXPECT_EQ(result, 1.0f);
        // Truncation should not increase value (rounds toward zero)
        EXPECT_LE(result, 1.001953125f);
    }

    // Truncation with NaN preservation
    {
        bf16_t b = float_to_bf16(std::numeric_limits<float>::quiet_NaN(),
                                 constant<bf16_rounding_mode::truncate_with_nan>{});
        EXPECT_TRUE(isnan(b));
    }
}

// Test double to bf16 conversion
TEST_F(Bf16ConversionTest, DoubleToBf16)
{
    {
        double d = 1.0;
        bf16_t b = double_to_bf16(d);
        EXPECT_EQ(bf16_to_bits(b), 0x3F80);
        EXPECT_EQ(static_cast<float>(b), 1.0f);
    }

    {
        double d     = -3.141592653589793;
        bf16_t b     = double_to_bf16(d);
        float result = static_cast<float>(b);
        EXPECT_NEAR(result, -3.141592653589793, 0.01);
    }

    // Large double value
    {
        double d = 1e100; // Much larger than bf16 can represent
        bf16_t b = double_to_bf16(d);
        EXPECT_TRUE(std::isinf(static_cast<float>(b)));
    }
}

// Test integer to bf16 conversion
TEST_F(Bf16ConversionTest, IntToBf16)
{
#if 0 // FIXME: CK_TILE_USE_CUSTOM_DATA_TYPE is broken - causes compilation errors
    {
        int i = 42;
        bf16_t b(i);
        EXPECT_EQ(static_cast<float>(b), 42.0f);
    }

    {
        int i = -100;
        bf16_t b(i);
        EXPECT_EQ(static_cast<float>(b), -100.0f);
    }

    {
        int i = 0;
        bf16_t b(i);
        EXPECT_EQ(static_cast<float>(b), 0.0f);
    }

    // Large int that requires rounding in bf16
    {
        int i = 16777217; // 2^24 + 1, not exactly representable in float
        bf16_t b(i);
        float result = static_cast<float>(b);
        EXPECT_NEAR(result, static_cast<float>(i), 256.0f);
    }
#endif
}

// Test bf16 to float conversion
TEST_F(Bf16ConversionTest, Bf16ToFloat)
{
    // Test all special bf16 values
    auto special_values = generate_special_bf16_values();
    for(const auto& bf16_val : special_values)
    {
        uint16_t bits = bf16_to_bits(bf16_val);
        float f       = bf16_to_float(bf16_val);

        if(isnan(bf16_val))
        {
            // Debug: Check bit pattern and float value
            uint32_t f_bits = bit_cast<uint32_t>(f);
            EXPECT_TRUE(std::isnan(f))
                << "bf16 NaN (bits=0x" << std::hex << bits << std::dec
                << ") should convert to float NaN, but got float with bits=0x" << std::hex << f_bits
                << std::dec << " value=" << f;
        }
        else if(bits == 0x7F80)
        {
            EXPECT_TRUE(std::isinf(f) && f > 0) << "bf16 +inf should convert to float +inf";
        }
        else if(bits == 0xFF80)
        {
            EXPECT_TRUE(std::isinf(f) && f < 0) << "bf16 -inf should convert to float -inf";
        }
        else
        {
            // For normal values, conversion should be exact
            bf16_t b_back = float_to_bf16(f);
            EXPECT_EQ(bf16_to_bits(bf16_val), bf16_to_bits(b_back))
                << "Round-trip conversion should preserve bf16 value";
        }
    }
}

// Test bf16 to double conversion
TEST_F(Bf16ConversionTest, Bf16ToDouble)
{
    {
        bf16_t b = float_to_bf16(1.0f);
        double d = bf16_to_double(b);
        EXPECT_EQ(d, 1.0);
    }

    {
        bf16_t b = numeric<bf16_t>::infinity();
        double d = bf16_to_double(b);
        EXPECT_TRUE(std::isinf(d) && d > 0);
    }

    {
        bf16_t b = numeric<bf16_t>::quiet_NaN();
        double d = bf16_to_double(b);
        EXPECT_TRUE(std::isnan(d));
    }
}

// Test bf16 to int conversion
TEST_F(Bf16ConversionTest, Bf16ToInt)
{
#if 0 // FIXME: CK_TILE_USE_CUSTOM_DATA_TYPE is broken - causes compilation errors
    {
        bf16_t b = float_to_bf16(42.0f);
        int i    = static_cast<int>(b);
        EXPECT_EQ(i, 42);
    }

    {
        bf16_t b = float_to_bf16(-100.0f);
        int i    = static_cast<int>(b);
        EXPECT_EQ(i, -100);
    }

    {
        bf16_t b = float_to_bf16(0.0f);
        int i    = static_cast<int>(b);
        EXPECT_EQ(i, 0);
    }

    // Test rounding behavior
    {
        bf16_t b = float_to_bf16(42.7f);
        int i    = static_cast<int>(b);
        EXPECT_EQ(i, 42); // Should truncate
    }

    {
        bf16_t b = float_to_bf16(-42.7f);
        int i    = static_cast<int>(b);
        EXPECT_EQ(i, -42); // Should truncate towards zero
    }
#endif
}

// // Test fp16 to bf16 conversion
TEST_F(Bf16ConversionTest, Fp16ToBf16)
{
    {
        fp16_t h = static_cast<fp16_t>(1.0f);
        bf16_t b = fp16_to_bf16(h);
        EXPECT_EQ(static_cast<float>(b), 1.0f);
    }

    {
        fp16_t h = static_cast<fp16_t>(-0.5f);
        bf16_t b = fp16_to_bf16(h);
        EXPECT_EQ(static_cast<float>(b), -0.5f);
    }

    // fp16 infinity
    {
        fp16_t h = numeric<fp16_t>::infinity();
        bf16_t b = fp16_to_bf16(h);
        EXPECT_TRUE(std::isinf(static_cast<float>(b)));
    }

    // fp16 NaN
    {
        fp16_t h = numeric<fp16_t>::quiet_NaN();
        bf16_t b = fp16_to_bf16(h);
        EXPECT_TRUE(isnan(b));
    }
}

// // Test bf16 to fp16 conversion
TEST_F(Bf16ConversionTest, Bf16ToFp16)
{
    {
        bf16_t b = float_to_bf16(1.0f);
        fp16_t h = bf16_to_fp16(b);
        EXPECT_EQ(static_cast<float>(h), 1.0f);
    }

    // Test value that's representable in bf16 but may lose precision in fp16
    {
        bf16_t b = float_to_bf16(131072.0f); // 2^17
        fp16_t h = bf16_to_fp16(b);
        // fp16 max is 65504, so this should overflow to infinity
        EXPECT_TRUE(std::isinf(static_cast<float>(h)));
    }
}

// // Test round-trip conversions
TEST_F(Bf16ConversionTest, RoundTripConversions)
{
    // Generate test values
    auto test_floats = generate_test_floats();

    for(float f : test_floats)
    {
        // Skip if the float is too large for bf16
        if(std::abs(f) > 3.38953139e38f && !std::isinf(f) && !std::isnan(f))
        {
            continue;
        }

        // float -> bf16 -> float
        bf16_t b     = float_to_bf16(f);
        float f_back = static_cast<float>(b);

        if(std::isnan(f))
        {
            EXPECT_TRUE(std::isnan(f_back)) << "NaN should be preserved";
        }
        else if(std::isinf(f))
        {
            EXPECT_TRUE(std::isinf(f_back)) << "Infinity should be preserved";
            EXPECT_EQ(std::signbit(f), std::signbit(f_back)) << "Sign should be preserved";
        }
        else
        {
            // For normal values, check if round-trip preserves the bf16 value
            bf16_t b_back = float_to_bf16(f_back);
            EXPECT_EQ(bf16_to_bits(b), bf16_to_bits(b_back))
                << "Round-trip should preserve bf16 representation for " << f;
        }
    }
}

// // Test denormal handling
TEST_F(Bf16ConversionTest, DenormalHandling)
{
    // Float denormals are much smaller than bf16 denormals (float has 23 mantissa bits,
    // bf16 has 7), so float denormals flush to zero when converted to bf16.
    // Note: bf16 does support denormals (see numeric<bf16_t>::denorm_min() = 0x0001),
    // but float denormals are below the smallest representable bf16 value.
    {
        float f  = std::numeric_limits<float>::denorm_min();
        bf16_t b = float_to_bf16(f);
        EXPECT_EQ(bf16_to_bits(b), 0x0000) << "Float denormal should flush to zero in bf16";
    }

    {
        float f  = -std::numeric_limits<float>::denorm_min();
        bf16_t b = float_to_bf16(f);
        EXPECT_EQ(bf16_to_bits(b), 0x8000)
            << "Negative float denormal should flush to negative zero in bf16";
    }

    // Test smallest normal bf16 value
    {
        bf16_t b = numeric<bf16_t>::min();
        float f  = static_cast<float>(b);
        EXPECT_GT(f, 0.0f);
        EXPECT_TRUE(std::isnormal(f)) << "bf16 min should convert to normal float";
    }
}

// Test overflow handling
TEST_F(Bf16ConversionTest, OverflowHandling)
{
    // Note: BF16 has the same 8-bit exponent as float32, but only 7 mantissa bits vs 23.
    // This means bf16::max (0x7F7F) ≈ 3.39e38 is LESS than float::max ≈ 3.40e38.
    //
    // Hardware behavior differs by architecture:
    // - gfx950: RTN rounding -> float::max rounds to infinity (IEEE-754 compliant)
    // - gfx9 (gfx90a, gfx908, gfx942): Saturates -> float::max clamps to bf16::max
    // - gfx12/gfx1250: Saturates -> float::max clamps to bf16::max (faster, non-IEEE)

    // Test float max overflow behavior (architecture-dependent)
    {
        float f            = std::numeric_limits<float>::max();
        bf16_t b           = float_to_bf16(f);
        float result       = bf16_to_float(b);
        uint16_t bf16_bits = bf16_to_bits(b);

#ifdef CK_TILE_BF16_OVERFLOW_SATURATES
        // gfx9/gfx11/gfx12: Hardware saturates to bf16::max
        EXPECT_FALSE(std::isinf(result))
            << "gfx9/gfx11/gfx12: float::max should saturate to bf16::max (0x7f7f). Got bf16=0x"
            << std::hex << bf16_bits << std::dec << " result=" << result;
        EXPECT_EQ(bf16_bits, 0x7f7f)
            << "gfx9/gfx11/gfx12: Expected saturation to bf16::max (0x7f7f), got 0x" << std::hex
            << bf16_bits << std::dec;
#else
        // gfx950 and software: RTN rounding to infinity (IEEE-754 behavior)
        EXPECT_TRUE(std::isinf(result) && result > 0)
            << "gfx950/software: float::max should overflow to +infinity with RTN rounding. Got "
               "bf16=0x"
            << std::hex << bf16_bits << std::dec << " result=" << result;
        EXPECT_EQ(bf16_bits, 0x7f80)
            << "Expected +infinity (0x7f80), got 0x" << std::hex << bf16_bits << std::dec;
#endif
    }

    {
        float f            = -std::numeric_limits<float>::max();
        bf16_t b           = float_to_bf16(f);
        float result       = bf16_to_float(b);
        uint16_t bf16_bits = bf16_to_bits(b);

#ifdef CK_TILE_BF16_OVERFLOW_SATURATES
        // gfx9/gfx11/gfx12: Hardware saturates to -bf16::max
        EXPECT_FALSE(std::isinf(result))
            << "gfx9/gfx11/gfx12: -float::max should saturate to -bf16::max (0xff7f)";
        EXPECT_EQ(bf16_bits, 0xff7f)
            << "gfx9/gfx11/gfx12: Expected saturation to -bf16::max (0xff7f), got 0x" << std::hex
            << bf16_bits << std::dec;
#else
        // gfx950 and software: RTN rounding to -infinity (IEEE-754 behavior)
        EXPECT_TRUE(std::isinf(result) && result < 0)
            << "gfx950/software: -float::max should overflow to -infinity with RTN rounding";
        EXPECT_EQ(bf16_bits, 0xff80)
            << "Expected -infinity (0xff80), got 0x" << std::hex << bf16_bits << std::dec;
#endif
    }

    // Test infinity passthrough
    {
        float f      = std::numeric_limits<float>::infinity();
        bf16_t b     = float_to_bf16(f);
        float result = bf16_to_float(b);
        EXPECT_TRUE(std::isinf(result) && result > 0)
            << "Float +infinity should convert to bf16 +infinity";
    }

    {
        float f      = -std::numeric_limits<float>::infinity();
        bf16_t b     = float_to_bf16(f);
        float result = bf16_to_float(b);
        EXPECT_TRUE(std::isinf(result) && result < 0)
            << "Float -infinity should convert to bf16 -infinity";
    }
}

// ============================================================================
// Tests from test_bf16_numeric_traits.cpp
// ============================================================================

class Bf16NumericTraitsTest : public Bf16TestBase
{
};

// Test numeric_traits structure
TEST_F(Bf16NumericTraitsTest, NumericTraitsValues)
{
    // bf16 has 8-bit exponent and 7-bit mantissa
    EXPECT_EQ(numeric_traits<bf16_t>::exp, 8);
    EXPECT_EQ(numeric_traits<bf16_t>::mant, 7);
    EXPECT_EQ(numeric_traits<bf16_t>::PackedSize, 1);
}

// Test numeric<bf16_t>::min()
TEST_F(Bf16NumericTraitsTest, MinValue)
{
    bf16_t min_val = numeric<bf16_t>::min();
    uint16_t bits  = bf16_to_bits(min_val);

    // bf16 min normal: sign=0, exp=00000001, mant=0000000
    EXPECT_EQ(bits, 0x0080);

    // Should be smallest positive normal value
    float f = static_cast<float>(min_val);
    EXPECT_GT(f, 0.0f);
    EXPECT_TRUE(std::isnormal(f));

    // Verify it's approximately 2^-126 * (1 + 0/128) = 2^-126
    EXPECT_NEAR(f, std::ldexp(1.0f, -126), 1e-45f);
}

// Test numeric<bf16_t>::max()
TEST_F(Bf16NumericTraitsTest, MaxValue)
{
    bf16_t max_val = numeric<bf16_t>::max();
    uint16_t bits  = bf16_to_bits(max_val);

    // bf16 max normal: sign=0, exp=11111110, mant=1111111
    EXPECT_EQ(bits, 0x7F7F);

    // Should be largest finite value
    float f = static_cast<float>(max_val);
    EXPECT_GT(f, 0.0f);
    EXPECT_TRUE(std::isfinite(f));
    EXPECT_FALSE(std::isinf(f));

    // Verify it's approximately 2^127 * (1 + 127/128)
    float expected = std::ldexp(1.0f + 127.0f / 128.0f, 127);
    EXPECT_NEAR(f, expected, expected * 1e-6f);
}

// Test numeric<bf16_t>::lowest()
TEST_F(Bf16NumericTraitsTest, LowestValue)
{
    bf16_t lowest_val = numeric<bf16_t>::lowest();
    uint16_t bits     = bf16_to_bits(lowest_val);

    // bf16 lowest (most negative): sign=1, exp=11111110, mant=1111111
    EXPECT_EQ(bits, 0xFF7F);

    // Should be most negative finite value
    float f = static_cast<float>(lowest_val);
    EXPECT_LT(f, 0.0f);
    EXPECT_TRUE(std::isfinite(f));

    // Should be negative of max
    EXPECT_EQ(f, -static_cast<float>(numeric<bf16_t>::max()));
}

// Test numeric<bf16_t>::epsilon()
TEST_F(Bf16NumericTraitsTest, EpsilonValue)
{
    bf16_t epsilon_val = numeric<bf16_t>::epsilon();
    uint16_t bits      = bf16_to_bits(epsilon_val);

    // bf16 epsilon: 2^-7 (smallest increment from 1.0)
    // sign=0, exp=01111000, mant=0000000
    EXPECT_EQ(bits, 0x3C00);

    float f = static_cast<float>(epsilon_val);
    EXPECT_EQ(f, std::ldexp(1.0f, -7)); // 2^-7 = 1/128
    EXPECT_EQ(f, 0.0078125f);

    // Verify it's the difference between 1.0 and the next larger value
    bf16_t one          = float_to_bf16(1.0f);
    bf16_t one_plus_eps = float_to_bf16(1.0f + static_cast<float>(epsilon_val));
    EXPECT_NE(bf16_to_bits(one), bf16_to_bits(one_plus_eps));
}

// // Test numeric<bf16_t>::round_error()
TEST_F(Bf16NumericTraitsTest, RoundErrorValue)
{
    bf16_t round_error_val = numeric<bf16_t>::round_error();
    uint16_t bits          = bf16_to_bits(round_error_val);

    // bf16 round error: 0.5
    // sign=0, exp=01111110, mant=0000000
    EXPECT_EQ(bits, 0x3F00);

    float f = static_cast<float>(round_error_val);
    EXPECT_EQ(f, 0.5f);
}

// // Test numeric<bf16_t>::infinity()
TEST_F(Bf16NumericTraitsTest, InfinityValue)
{
    bf16_t inf_val = numeric<bf16_t>::infinity();
    uint16_t bits  = bf16_to_bits(inf_val);

    // bf16 infinity: sign=0, exp=11111111, mant=0000000
    EXPECT_EQ(bits, 0x7F80);

    float f = static_cast<float>(inf_val);
    EXPECT_TRUE(std::isinf(f));
    EXPECT_GT(f, 0.0f);
}

// // Test numeric<bf16_t>::quiet_NaN()
TEST_F(Bf16NumericTraitsTest, QuietNaNValue)
{
    bf16_t qnan_val = numeric<bf16_t>::quiet_NaN();
    uint16_t bits   = bf16_to_bits(qnan_val);

    // bf16 quiet NaN: sign=0, exp=11111111, mant=non-zero
    EXPECT_EQ(bits, 0x7FFF);
    EXPECT_EQ((bits & 0x7F80), 0x7F80); // All exponent bits set
    EXPECT_NE((bits & 0x007F), 0);      // Mantissa non-zero

    EXPECT_TRUE(isnan(qnan_val));
}

// // Test numeric<bf16_t>::signaling_NaN()
TEST_F(Bf16NumericTraitsTest, SignalingNaNValue)
{
    bf16_t snan_val = numeric<bf16_t>::signaling_NaN();
    uint16_t bits   = bf16_to_bits(snan_val);

    // bf16 signaling NaN: sign=0, exp=11111111, mant=non-zero
    // Note: The implementation returns the same bit pattern as quiet NaN
    EXPECT_EQ(bits, 0x7FFF);
    EXPECT_EQ((bits & 0x7F80), 0x7F80); // All exponent bits set
    EXPECT_NE((bits & 0x007F), 0);      // Mantissa non-zero

    EXPECT_TRUE(isnan(snan_val));
}

// // Test numeric<bf16_t>::denorm_min()
TEST_F(Bf16NumericTraitsTest, DenormMinValue)
{
    bf16_t denorm_min_val = numeric<bf16_t>::denorm_min();
    uint16_t bits         = bf16_to_bits(denorm_min_val);

    // bf16 smallest positive subnormal: sign=0, exp=00000000, mant=0000001
    EXPECT_EQ(bits, 0x0001);

    float f = bf16_to_float(denorm_min_val);
    EXPECT_GT(f, 0.0f);

    // bf16 subnormal with exponent=0, mantissa=1:
    // Value = 2^(1-127) * (0 + 1/128) = 2^-126 * 2^-7 = 2^-133
    // Note: For subnormals, the implicit leading bit is 0, not 1
    EXPECT_NEAR(f, std::ldexp(1.0f, -133), 1e-45f);
}

// // Test numeric<bf16_t>::zero()
// Test numeric<bf16_t>::zero() - verifies zero value has all bits zero and converts correctly
TEST_F(Bf16NumericTraitsTest, ZeroValue)
{
    bf16_t zero_val = numeric<bf16_t>::zero();
    uint16_t bits   = bf16_to_bits(zero_val);

    // bf16 zero: all bits zero
    EXPECT_EQ(bits, 0x0000);

    float f = static_cast<float>(zero_val);
    EXPECT_EQ(f, 0.0f);
    EXPECT_FALSE(std::signbit(f)); // Positive zero
}

// Test special value bit patterns - verifies IEEE 754 special values (zero, infinity, NaN) are
// correctly represented
TEST_F(Bf16NumericTraitsTest, SpecialValueBitPatterns)
{
    // Positive zero: sign=0, exp=0, mant=0 - verifies positive zero bit pattern and conversion
    {
        bf16_t val = bits_to_bf16(0x0000);
        EXPECT_EQ(static_cast<float>(val), 0.0f);
        EXPECT_FALSE(std::signbit(static_cast<float>(val)));
    }

    // Negative zero: sign=1, exp=0, mant=0 - verifies negative zero preserves sign bit
    {
        bf16_t val = bits_to_bf16(0x8000);
        EXPECT_EQ(static_cast<float>(val), -0.0f);
        EXPECT_TRUE(std::signbit(static_cast<float>(val)));
    }

    // Positive infinity: sign=0, exp=all 1s, mant=0 - verifies infinity representation
    {
        bf16_t val = bits_to_bf16(0x7F80);
        EXPECT_TRUE(std::isinf(static_cast<float>(val)));
        EXPECT_GT(static_cast<float>(val), 0.0f);
    }

    // Negative infinity: sign=1, exp=all 1s, mant=0 - verifies negative infinity representation
    {
        bf16_t val = bits_to_bf16(0xFF80);
        EXPECT_TRUE(std::isinf(static_cast<float>(val)));
        EXPECT_LT(static_cast<float>(val), 0.0f);
    }

    // Various NaN patterns: exp=all 1s, mant=non-zero - verifies all valid NaN bit patterns are
    // detected
    {
        // Quiet NaN with different mantissa bits - tests all positive NaN patterns (0x7F81 to
        // 0x7FFF)
        for(uint16_t mant = 1; mant <= 0x7F; mant++)
        {
            uint16_t bits = 0x7F80 | mant;
            bf16_t val    = bits_to_bf16(bits);
            EXPECT_TRUE(isnan(val)) << "Bits 0x" << std::hex << bits << " should be NaN";
        }

        // Negative NaN - tests all negative NaN patterns (0xFF81 to 0xFFFF)
        for(uint16_t mant = 1; mant <= 0x7F; mant++)
        {
            uint16_t bits = 0xFF80 | mant;
            bf16_t val    = bits_to_bf16(bits);
            EXPECT_TRUE(isnan(val)) << "Bits 0x" << std::hex << bits << " should be NaN";
        }
    }
}

// // Test relationships between special values
TEST_F(Bf16NumericTraitsTest, SpecialValueRelationships)
{
    // min < max
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::min()),
              static_cast<float>(numeric<bf16_t>::max()));

    // lowest < max
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::lowest()),
              static_cast<float>(numeric<bf16_t>::max()));

    // lowest == -max
    EXPECT_EQ(static_cast<float>(numeric<bf16_t>::lowest()),
              -static_cast<float>(numeric<bf16_t>::max()));

    // denorm_min < min
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::denorm_min()),
              static_cast<float>(numeric<bf16_t>::min()));

    // zero < denorm_min < min < 1.0 < max < infinity
    EXPECT_EQ(static_cast<float>(numeric<bf16_t>::zero()), 0.0f);
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::zero()),
              static_cast<float>(numeric<bf16_t>::denorm_min()));
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::denorm_min()),
              static_cast<float>(numeric<bf16_t>::min()));
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::min()), 1.0f);
    EXPECT_LT(1.0f, static_cast<float>(numeric<bf16_t>::max()));
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::max()),
              static_cast<float>(numeric<bf16_t>::infinity()));
}

// // Test edge cases and boundary values
TEST_F(Bf16NumericTraitsTest, EdgeCases)
{
    // Test values just above and below special boundaries

    // Value just above min
    {
        uint16_t bits = bf16_to_bits(numeric<bf16_t>::min()) + 1;
        bf16_t val    = bits_to_bf16(bits);
        EXPECT_GT(static_cast<float>(val), static_cast<float>(numeric<bf16_t>::min()));
        EXPECT_TRUE(std::isnormal(static_cast<float>(val)));
    }

    // Value just below max (not infinity)
    {
        uint16_t bits = bf16_to_bits(numeric<bf16_t>::max()) - 1;
        bf16_t val    = bits_to_bf16(bits);
        EXPECT_LT(static_cast<float>(val), static_cast<float>(numeric<bf16_t>::max()));
        EXPECT_TRUE(std::isfinite(static_cast<float>(val)));
    }

    // Largest value that's not infinity (max normal)
    {
        uint16_t bits = 0x7F7F; // Exponent = 254, mantissa = all 1s
        bf16_t val    = bits_to_bf16(bits);
        EXPECT_TRUE(std::isfinite(static_cast<float>(val)));
        EXPECT_FALSE(std::isinf(static_cast<float>(val)));
    }

    // Smallest value that is infinity
    {
        uint16_t bits = 0x7F80; // Exponent = 255, mantissa = 0
        bf16_t val    = bits_to_bf16(bits);
        EXPECT_TRUE(std::isinf(static_cast<float>(val)));
    }
}

// ============================================================================
// Tests from test_bf16_math.cpp
// ============================================================================

using namespace ck_tile;
using namespace ck_tile_test;

// Device kernels for testing math functions
__global__ void test_abs_kernel(const bf16_t* input, bf16_t* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        output[idx] = abs(input[idx]);
    }
}

__global__ void test_sqrt_kernel(const bf16_t* input, bf16_t* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        output[idx] = sqrt(input[idx]);
    }
}

__global__ void test_exp_kernel(const bf16_t* input, bf16_t* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        output[idx] = exp(input[idx]);
    }
}

__global__ void test_exp2_kernel(const bf16_t* input, bf16_t* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        output[idx] = exp2(input[idx]);
    }
}

__global__ void test_log_kernel(const bf16_t* input, bf16_t* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        output[idx] = log(input[idx]);
    }
}

class Bf16MathTest : public Bf16TestBase
{
    protected:
    void* d_input                     = nullptr;
    void* d_output                    = nullptr;
    static constexpr size_t test_size = 256;

    void SetUp() override
    {
        Bf16TestBase::SetUp();
        hip_check_error(hipMalloc(&d_input, test_size * sizeof(bf16_t)));
        hip_check_error(hipMalloc(&d_output, test_size * sizeof(bf16_t)));
    }

    void TearDown() override
    {
        if(d_input)
            hip_check_error(hipFree(d_input));
        if(d_output)
            hip_check_error(hipFree(d_output));
        Bf16TestBase::TearDown();
    }
};

// Test abs() function on host
TEST_F(Bf16MathTest, AbsHost)
{
    // Positive value
    {
        bf16_t x      = float_to_bf16(3.14159f);
        bf16_t result = abs(x);
        EXPECT_EQ(bf16_to_bits(result), bf16_to_bits(x));
    }

    // Negative value
    {
        bf16_t x      = float_to_bf16(-3.14159f);
        bf16_t result = abs(x);
        // abs() should clear the sign bit, giving the positive value
        bf16_t expected = float_to_bf16(3.14159f);
        EXPECT_NEAR(bf16_to_float(result), 3.14159f, 0.01f);
        EXPECT_EQ(bf16_to_bits(result), bf16_to_bits(expected)); // Should match positive conversion
    }

    // Zero
    {
        bf16_t x      = float_to_bf16(0.0f);
        bf16_t result = abs(x);
        EXPECT_EQ(bf16_to_bits(result), 0x0000);
    }

    // Negative zero
    {
        bf16_t x      = bits_to_bf16(0x8000);
        bf16_t result = abs(x);
        EXPECT_EQ(bf16_to_bits(result), 0x0000); // Should become positive zero
    }

    // Infinity
    {
        bf16_t x      = numeric<bf16_t>::infinity();
        bf16_t result = abs(x);
        EXPECT_EQ(bf16_to_bits(result), 0x7F80);
    }

    // Negative infinity
    {
        bf16_t x      = bits_to_bf16(0xFF80);
        bf16_t result = abs(x);
        EXPECT_EQ(bf16_to_bits(result), 0x7F80); // Should become positive infinity
    }

    // NaN
    {
        bf16_t x      = numeric<bf16_t>::quiet_NaN();
        bf16_t result = abs(x);
        // abs() should clear sign bit but preserve NaN
        EXPECT_TRUE(isnan(result));
        EXPECT_FALSE(bf16_to_bits(result) & 0x8000); // Sign bit should be clear
    }
}

// Test isnan() predicate
TEST_F(Bf16MathTest, IsNanPredicate)
{
    // Normal values - should not be NaN
    EXPECT_FALSE(isnan(float_to_bf16(1.0f)));
    EXPECT_FALSE(isnan(float_to_bf16(-1.0f)));
    EXPECT_FALSE(isnan(float_to_bf16(0.0f)));
    EXPECT_FALSE(isnan(bits_to_bf16(0x8000))); // -0.0

    // Infinity - should not be NaN
    EXPECT_FALSE(isnan(numeric<bf16_t>::infinity()));
    EXPECT_FALSE(isnan(bits_to_bf16(0xFF80))); // -infinity

    // Various NaN patterns - should be NaN
    EXPECT_TRUE(isnan(numeric<bf16_t>::quiet_NaN()));
    EXPECT_TRUE(isnan(numeric<bf16_t>::signaling_NaN()));

    // Test various NaN bit patterns
    for(uint16_t mant = 1; mant <= 0x7F; mant++)
    {
        // Positive NaN
        bf16_t pos_nan = bits_to_bf16(0x7F80 | mant);
        EXPECT_TRUE(isnan(pos_nan)) << "Bits 0x" << std::hex << (0x7F80 | mant) << " should be NaN";

        // Negative NaN
        bf16_t neg_nan = bits_to_bf16(0xFF80 | mant);
        EXPECT_TRUE(isnan(neg_nan)) << "Bits 0x" << std::hex << (0xFF80 | mant) << " should be NaN";
    }
}

// Test abs() function on device
TEST_F(Bf16MathTest, AbsDevice)
{
    std::vector<bf16_t> h_input;
    std::vector<bf16_t> h_output(test_size);

    // Generate test values
    h_input.push_back(float_to_bf16(1.0f));
    h_input.push_back(float_to_bf16(-1.0f));
    h_input.push_back(float_to_bf16(3.14159f));
    h_input.push_back(float_to_bf16(-3.14159f));
    h_input.push_back(float_to_bf16(0.0f));
    h_input.push_back(bits_to_bf16(0x8000)); // -0.0
    h_input.push_back(numeric<bf16_t>::infinity());
    h_input.push_back(bits_to_bf16(0xFF80)); // -infinity
    h_input.push_back(numeric<bf16_t>::quiet_NaN());
    h_input.push_back(numeric<bf16_t>::max());
    h_input.push_back(numeric<bf16_t>::lowest());

    // Fill remaining with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    while(h_input.size() < test_size)
    {
        h_input.push_back(float_to_bf16(dist(gen)));
    }

    // Copy to device
    hip_check_error(
        hipMemcpy(d_input, h_input.data(), test_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 block(256);
    dim3 grid((test_size + block.x - 1) / block.x);
    test_abs_kernel<<<grid, block>>>(
        static_cast<bf16_t*>(d_input), static_cast<bf16_t*>(d_output), test_size);

    // Copy back
    hip_check_error(
        hipMemcpy(h_output.data(), d_output, test_size * sizeof(bf16_t), hipMemcpyDeviceToHost));

    // Verify results
    for(size_t i = 0; i < test_size; i++)
    {
        float input_val  = static_cast<float>(h_input[i]);
        float output_val = static_cast<float>(h_output[i]);

        if(isnan(h_input[i]))
        {
            EXPECT_TRUE(isnan(h_output[i])) << "abs(NaN) should be NaN";
            // Sign bit should be cleared
            EXPECT_FALSE(bf16_to_bits(h_output[i]) & 0x8000);
        }
        else
        {
            EXPECT_EQ(output_val, std::abs(input_val))
                << "abs(" << input_val << ") = " << output_val << " at index " << i;
        }
    }
}

// Test sqrt() function on device
TEST_F(Bf16MathTest, SqrtDevice)
{
    std::vector<bf16_t> h_input;
    std::vector<bf16_t> h_output(test_size);

    // Generate test values
    h_input.push_back(float_to_bf16(0.0f));
    h_input.push_back(float_to_bf16(1.0f));
    h_input.push_back(float_to_bf16(4.0f));
    h_input.push_back(float_to_bf16(9.0f));
    h_input.push_back(float_to_bf16(16.0f));
    h_input.push_back(float_to_bf16(0.25f));
    h_input.push_back(float_to_bf16(0.5f));
    h_input.push_back(float_to_bf16(2.0f));
    h_input.push_back(numeric<bf16_t>::infinity());
    h_input.push_back(float_to_bf16(-1.0f)); // Should produce NaN
    h_input.push_back(bits_to_bf16(0xFF80)); // -infinity, should produce NaN
    h_input.push_back(numeric<bf16_t>::quiet_NaN());

    // Fill remaining with positive values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    while(h_input.size() < test_size)
    {
        h_input.push_back(float_to_bf16(dist(gen)));
    }

    // Copy to device
    hip_check_error(
        hipMemcpy(d_input, h_input.data(), test_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 block(256);
    dim3 grid((test_size + block.x - 1) / block.x);
    test_sqrt_kernel<<<grid, block>>>(
        static_cast<bf16_t*>(d_input), static_cast<bf16_t*>(d_output), test_size);

    // Copy back
    hip_check_error(
        hipMemcpy(h_output.data(), d_output, test_size * sizeof(bf16_t), hipMemcpyDeviceToHost));

    // Verify results
    for(size_t i = 0; i < test_size; i++)
    {
        float input_val  = static_cast<float>(h_input[i]);
        float output_val = static_cast<float>(h_output[i]);

        if(isnan(h_input[i]))
        {
            EXPECT_TRUE(isnan(h_output[i])) << "sqrt(NaN) should be NaN";
        }
        else if(input_val < 0.0f)
        {
            EXPECT_TRUE(isnan(h_output[i])) << "sqrt(negative) should be NaN";
        }
        else if(std::isinf(input_val))
        {
            EXPECT_TRUE(std::isinf(output_val) && output_val > 0) << "sqrt(+inf) should be +inf";
        }
        else
        {
            float expected = std::sqrt(input_val);
            // Allow for some error due to bf16 precision
            EXPECT_NEAR(output_val, expected, expected * 0.01f)
                << "sqrt(" << input_val << ") = " << output_val << " at index " << i;
        }
    }
}

// Test exp() function on device
TEST_F(Bf16MathTest, ExpDevice)
{
    std::vector<bf16_t> h_input;
    std::vector<bf16_t> h_output(test_size);

    // Generate test values
    h_input.push_back(float_to_bf16(0.0f));           // exp(0) = 1
    h_input.push_back(float_to_bf16(1.0f));           // exp(1) = e
    h_input.push_back(float_to_bf16(-1.0f));          // exp(-1) = 1/e
    h_input.push_back(float_to_bf16(2.0f));           // exp(2) = e^2
    h_input.push_back(float_to_bf16(std::log(2.0f))); // exp(ln(2)) = 2
    h_input.push_back(numeric<bf16_t>::infinity());
    h_input.push_back(bits_to_bf16(0xFF80)); // -infinity
    h_input.push_back(numeric<bf16_t>::quiet_NaN());

    // Add values that will overflow/underflow
    h_input.push_back(float_to_bf16(100.0f));  // Will overflow to infinity
    h_input.push_back(float_to_bf16(-100.0f)); // Will underflow to zero

    // Fill remaining with reasonable values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    while(h_input.size() < test_size)
    {
        h_input.push_back(float_to_bf16(dist(gen)));
    }

    // Copy to device
    hip_check_error(
        hipMemcpy(d_input, h_input.data(), test_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 block(256);
    dim3 grid((test_size + block.x - 1) / block.x);
    test_exp_kernel<<<grid, block>>>(
        static_cast<bf16_t*>(d_input), static_cast<bf16_t*>(d_output), test_size);

    // Copy back
    hip_check_error(
        hipMemcpy(h_output.data(), d_output, test_size * sizeof(bf16_t), hipMemcpyDeviceToHost));

    // Verify results
    for(size_t i = 0; i < test_size; i++)
    {
        float input_val  = static_cast<float>(h_input[i]);
        float output_val = static_cast<float>(h_output[i]);

        if(isnan(h_input[i]))
        {
            EXPECT_TRUE(isnan(h_output[i])) << "exp(NaN) should be NaN";
        }
        else if(input_val == std::numeric_limits<float>::infinity())
        {
            EXPECT_TRUE(std::isinf(output_val) && output_val > 0) << "exp(+inf) should be +inf";
        }
        else if(input_val == -std::numeric_limits<float>::infinity())
        {
            EXPECT_EQ(output_val, 0.0f) << "exp(-inf) should be 0";
        }
        else if(input_val > 80.0f)
        { // Will overflow in bf16
            EXPECT_TRUE(std::isinf(output_val) && output_val > 0)
                << "exp(large) should overflow to +inf";
        }
        else if(input_val < -80.0f)
        { // Will underflow in bf16
            EXPECT_EQ(output_val, 0.0f) << "exp(very negative) should underflow to 0";
        }
        else
        {
            float expected = std::exp(input_val);
            // Allow for significant error due to bf16 precision
            float rel_error = std::abs(output_val - expected) / (expected + 1e-10f);
            EXPECT_LT(rel_error, 0.02f) << "exp(" << input_val << ") = " << output_val
                                        << ", expected " << expected << " at index " << i;
        }
    }
}

// Test exp2() function on device
TEST_F(Bf16MathTest, Exp2Device)
{
    std::vector<bf16_t> h_input;
    std::vector<bf16_t> h_output(test_size);

    // Generate test values
    h_input.push_back(float_to_bf16(0.0f));  // 2^0 = 1
    h_input.push_back(float_to_bf16(1.0f));  // 2^1 = 2
    h_input.push_back(float_to_bf16(2.0f));  // 2^2 = 4
    h_input.push_back(float_to_bf16(-1.0f)); // 2^(-1) = 0.5
    h_input.push_back(float_to_bf16(10.0f)); // 2^10 = 1024
    h_input.push_back(numeric<bf16_t>::infinity());
    h_input.push_back(bits_to_bf16(0xFF80)); // -infinity
    h_input.push_back(numeric<bf16_t>::quiet_NaN());

    // Fill remaining with reasonable values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    while(h_input.size() < test_size)
    {
        h_input.push_back(float_to_bf16(dist(gen)));
    }

    // Copy to device
    hip_check_error(
        hipMemcpy(d_input, h_input.data(), test_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 block(256);
    dim3 grid((test_size + block.x - 1) / block.x);
    test_exp2_kernel<<<grid, block>>>(
        static_cast<bf16_t*>(d_input), static_cast<bf16_t*>(d_output), test_size);

    // Copy back
    hip_check_error(
        hipMemcpy(h_output.data(), d_output, test_size * sizeof(bf16_t), hipMemcpyDeviceToHost));

    // Verify results
    for(size_t i = 0; i < test_size; i++)
    {
        float input_val  = static_cast<float>(h_input[i]);
        float output_val = static_cast<float>(h_output[i]);

        if(isnan(h_input[i]))
        {
            EXPECT_TRUE(isnan(h_output[i])) << "exp2(NaN) should be NaN";
        }
        else if(input_val == std::numeric_limits<float>::infinity())
        {
            EXPECT_TRUE(std::isinf(output_val) && output_val > 0) << "exp2(+inf) should be +inf";
        }
        else if(input_val == -std::numeric_limits<float>::infinity())
        {
            EXPECT_EQ(output_val, 0.0f) << "exp2(-inf) should be 0";
        }
        else if(input_val > 120.0f)
        { // Will overflow in bf16
            EXPECT_TRUE(std::isinf(output_val) && output_val > 0)
                << "exp2(large) should overflow to +inf";
        }
        else if(input_val < -120.0f)
        { // Will underflow in bf16
            EXPECT_EQ(output_val, 0.0f) << "exp2(very negative) should underflow to 0";
        }
        else
        {
            float expected = std::exp2(input_val);
            // Allow for significant error due to bf16 precision
            float rel_error = std::abs(output_val - expected) / (expected + 1e-10f);
            EXPECT_LT(rel_error, 0.02f) << "exp2(" << input_val << ") = " << output_val
                                        << ", expected " << expected << " at index " << i;
        }
    }
}

// Test log() function on device
TEST_F(Bf16MathTest, LogDevice)
{
    std::vector<bf16_t> h_input;
    std::vector<bf16_t> h_output(test_size);

    // Generate test values
    h_input.push_back(float_to_bf16(1.0f));           // log(1) = 0
    h_input.push_back(float_to_bf16(std::exp(1.0f))); // log(e) = 1
    h_input.push_back(float_to_bf16(2.0f));           // log(2) = ln(2)
    h_input.push_back(float_to_bf16(10.0f));          // log(10) = ln(10)
    h_input.push_back(float_to_bf16(0.5f));           // log(0.5) = -ln(2)
    h_input.push_back(float_to_bf16(0.0f));           // log(0) = -inf
    h_input.push_back(float_to_bf16(-1.0f));          // log(negative) = NaN
    h_input.push_back(numeric<bf16_t>::infinity());
    h_input.push_back(bits_to_bf16(0xFF80)); // -infinity
    h_input.push_back(numeric<bf16_t>::quiet_NaN());

    // Fill remaining with positive values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.01f, 100.0f);
    while(h_input.size() < test_size)
    {
        h_input.push_back(float_to_bf16(dist(gen)));
    }

    // Copy to device
    hip_check_error(
        hipMemcpy(d_input, h_input.data(), test_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 block(256);
    dim3 grid((test_size + block.x - 1) / block.x);
    test_log_kernel<<<grid, block>>>(
        static_cast<bf16_t*>(d_input), static_cast<bf16_t*>(d_output), test_size);

    // Copy back
    hip_check_error(
        hipMemcpy(h_output.data(), d_output, test_size * sizeof(bf16_t), hipMemcpyDeviceToHost));

    // Verify results
    for(size_t i = 0; i < test_size; i++)
    {
        float input_val  = static_cast<float>(h_input[i]);
        float output_val = static_cast<float>(h_output[i]);

        if(isnan(h_input[i]))
        {
            EXPECT_TRUE(isnan(h_output[i])) << "log(NaN) should be NaN";
        }
        else if(input_val < 0.0f)
        {
            EXPECT_TRUE(isnan(h_output[i])) << "log(negative) should be NaN";
        }
        else if(input_val == 0.0f)
        {
            EXPECT_TRUE(std::isinf(output_val) && output_val < 0) << "log(0) should be -inf";
        }
        else if(std::isinf(input_val) && input_val > 0)
        {
            EXPECT_TRUE(std::isinf(output_val) && output_val > 0) << "log(+inf) should be +inf";
        }
        else if(std::isinf(input_val) && input_val < 0)
        {
            EXPECT_TRUE(isnan(h_output[i])) << "log(-inf) should be NaN";
        }
        else
        {
            float expected = std::log(input_val);
            // Allow for significant error due to bf16 precision
            float abs_error = std::abs(output_val - expected);
            float rel_error = abs_error / (std::abs(expected) + 1e-10f);
            // Use absolute error for values close to zero
            if(std::abs(expected) < 0.1f)
            {
                EXPECT_LT(abs_error, 0.01f) << "log(" << input_val << ") = " << output_val
                                            << ", expected " << expected << " at index " << i;
            }
            else
            {
                EXPECT_LT(rel_error, 0.02f) << "log(" << input_val << ") = " << output_val
                                            << ", expected " << expected << " at index " << i;
            }
        }
    }
}

// ============================================================================
// Tests from test_bf16_platform.cpp
// ============================================================================

#include "ck_tile/host/device_prop.hpp"

using namespace ck_tile;
using namespace ck_tile_test;

// Device kernel to test native bf16 operations
__global__ void test_native_conversion_kernel(const float* input, bf16_t* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        // This will use either software or hardware conversion based on compile flags
        output[idx] = float_to_bf16(input[idx]);
    }
}

// Device kernel for FMA accumulation test (bf16 * bf16 + fp32 accumulator)
__global__ void
test_fma_accumulate_kernel(const bf16_t* a, const bf16_t* b, float* acc, size_t n, int iterations)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        float result = acc[idx];
        bf16_t val_a = a[idx];
        bf16_t val_b = b[idx];
        // Simulate GEMM inner loop: repeated bf16 * bf16 accumulated in fp32
        for(int i = 0; i < iterations; i++)
        {
            result = fma(bf16_to_float(val_a), bf16_to_float(val_b), result);
        }
        acc[idx] = result;
    }
}

// Device kernel to test arithmetic performance
__global__ void
test_arithmetic_performance_kernel(const bf16_t* a, const bf16_t* b, bf16_t* c, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
#if 0 // FIXME: CK_TILE_USE_CUSTOM_DATA_TYPE is broken - causes compilation errors
      // Perform some arithmetic operations
        bf16_t val_a  = a[idx];
        bf16_t val_b  = b[idx];
        bf16_t result = val_a + val_b;
        result        = result * val_a;
        result        = result - val_b;
        c[idx]        = result;
#else
        // When custom data type is not used, convert to float, compute, convert back
        float val_a  = static_cast<float>(a[idx]);
        float val_b  = static_cast<float>(b[idx]);
        float result = val_a + val_b;
        result       = result * val_a;
        result       = result - val_b;
        c[idx]       = float_to_bf16(result);
#endif
    }
}

class Bf16PlatformTest : public Bf16TestBase
{
    protected:
    std::string device_name;

    void SetUp() override
    {
        Bf16TestBase::SetUp();
        device_name = get_device_name();
    }
};

// Test compile-time flags
TEST_F(Bf16PlatformTest, CompileTimeFlags)
{
    std::cout << "=== BF16 Platform Configuration ===" << std::endl;
    std::cout << "Device: " << device_name << std::endl;

#ifdef CK_TILE_USE_LLVM_BUILTIN_BF16
    std::cout << "CK_TILE_USE_LLVM_BUILTIN_BF16: " << CK_TILE_USE_LLVM_BUILTIN_BF16 << std::endl;
#else
    std::cout << "CK_TILE_USE_LLVM_BUILTIN_BF16: undefined (defaults to 0)" << std::endl;
#endif

// FIXME: CK_TILE_USE_CUSTOM_DATA_TYPE is broken - causes compilation errors
#ifdef CK_TILE_USE_CUSTOM_DATA_TYPE
    std::cout << "CK_TILE_USE_CUSTOM_DATA_TYPE: " << CK_TILE_USE_CUSTOM_DATA_TYPE
              << " (BROKEN - do not enable)" << std::endl;
#else
    std::cout << "CK_TILE_USE_CUSTOM_DATA_TYPE: undefined (defaults to 0)" << std::endl;
#endif

#ifdef __gfx950__
    std::cout << "__gfx950__ is defined" << std::endl;
#else
    std::cout << "__gfx950__ is NOT defined" << std::endl;
#endif

#ifdef CK_GFX950_SUPPORT
    std::cout << "CK_GFX950_SUPPORT is defined" << std::endl;
#else
    std::cout << "CK_GFX950_SUPPORT is NOT defined" << std::endl;
#endif

    std::cout << "===================================" << std::endl;
}

// Test type identification
TEST_F(Bf16PlatformTest, TypeIdentification)
{
    // Check what type bf16_t actually is
    std::cout << "sizeof(bf16_t): " << sizeof(bf16_t) << " bytes" << std::endl;
    std::cout << "sizeof(bfloat16_t): " << sizeof(bfloat16_t) << " bytes" << std::endl;

// FIXME: CK_TILE_USE_CUSTOM_DATA_TYPE is broken - causes compilation errors
#if 0 // CK_TILE_USE_CUSTOM_DATA_TYPE
    std::cout << "Using custom bf16 struct implementation" << std::endl;
    EXPECT_TRUE((std::is_same<bf16_t, bfloat16_t>::value));
    EXPECT_TRUE((std::is_class<bf16_t>::value));
#else
#if CK_TILE_USE_LLVM_BUILTIN_BF16
    std::cout << "Using LLVM __bf16 builtin type" << std::endl;
    EXPECT_TRUE((std::is_same<bfloat16_t, __bf16>::value));
#else
    std::cout << "Using ushort as bf16 type" << std::endl;
    EXPECT_TRUE((std::is_same<bfloat16_t, ushort>::value));
#endif
    EXPECT_TRUE((std::is_same<bf16_t, bfloat16_t>::value));
#endif

    // Always 2 bytes regardless of implementation
    EXPECT_EQ(sizeof(bf16_t), 2);
}

// Test native hardware conversion on gfx950
TEST_F(Bf16PlatformTest, NativeHardwareConversion)
{
#if CK_TILE_USE_LLVM_BUILTIN_BF16 && (defined(__gfx950__) || defined(CK_GFX950_SUPPORT))
    std::cout << "Testing native hardware bf16 conversion on " << device_name << std::endl;

    // Test that native conversion is being used
    {
        float f      = 3.14159f;
        bf16_t b     = float_to_bf16(f);
        float f_back = static_cast<float>(b);

        // The conversion should still work correctly
        EXPECT_NEAR(f_back, f, 0.01f);

        // Check that we're using the expected conversion path
        // When using native __bf16, the conversion should be a simple cast
        bf16_t b_native = static_cast<bf16_t>(f);
        EXPECT_EQ(bf16_to_bits(b), bf16_to_bits(b_native));
    }
#else
    std::cout << "Native hardware bf16 conversion not available on this platform" << std::endl;
    std::cout << "Using software conversion implementation" << std::endl;
#endif
}

// Test conversion accuracy across implementations
TEST_F(Bf16PlatformTest, ConversionAccuracyComparison)
{
    // Test values that stress different aspects of conversion
    std::vector<float> test_values = {1.0f,
                                      -1.0f,
                                      0.5f,
                                      -0.5f,
                                      3.14159f,
                                      -2.71828f,
                                      1.001953125f, // Requires rounding
                                      std::numeric_limits<float>::max(),
                                      std::numeric_limits<float>::min(),
                                      std::numeric_limits<float>::infinity(),
                                      -std::numeric_limits<float>::infinity(),
                                      std::numeric_limits<float>::quiet_NaN(),
                                      0.0f,
                                      -0.0f};

    std::cout << "\nConversion accuracy test:" << std::endl;

    for(float f : test_values)
    {
        // Standard conversion
        bf16_t b_standard = float_to_bf16(f, constant<bf16_rounding_mode::standard>{});

        // Truncation
        bf16_t b_truncate = float_to_bf16(f, constant<bf16_rounding_mode::truncate>{});

        if(!std::isnan(f))
        {
            // For non-NaN values, standard rounding should be more accurate
            float f_standard = static_cast<float>(b_standard);
            float f_truncate = static_cast<float>(b_truncate);

            if(!std::isinf(f) && !std::isinf(f_standard))
            {
                // Skip comparison when standard rounding overflows to infinity
                // (this can happen at boundaries like float_max where rounding
                // causes mantissa overflow that propagates to exponent)
                float err_standard = std::abs(f - f_standard);
                float err_truncate = std::abs(f - f_truncate);

                // Standard rounding should never be worse than truncation
                EXPECT_LE(err_standard, err_truncate + 1e-10f)
                    << "For value " << f << ", standard rounding error (" << err_standard
                    << ") should not exceed truncation error (" << err_truncate << ")";
            }
        }
    }
}

// Test device-side performance characteristics
TEST_F(Bf16PlatformTest, DevicePerformance)
{
    const size_t n       = 1024 * 1024; // 1M elements
    const int iterations = 100;

    float* d_float;
    bf16_t* d_bf16_a;
    bf16_t* d_bf16_b;
    bf16_t* d_bf16_c;

    hip_check_error(hipMalloc(&d_float, n * sizeof(float)));
    hip_check_error(hipMalloc(&d_bf16_a, n * sizeof(bf16_t)));
    hip_check_error(hipMalloc(&d_bf16_b, n * sizeof(bf16_t)));
    hip_check_error(hipMalloc(&d_bf16_c, n * sizeof(bf16_t)));

    // Initialize data
    std::vector<float> h_float(n);
    for(size_t i = 0; i < n; i++)
    {
        h_float[i] = static_cast<float>(i % 1000) / 1000.0f;
    }

    hip_check_error(hipMemcpy(d_float, h_float.data(), n * sizeof(float), hipMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    // Warm up
    for(int i = 0; i < 10; i++)
    {
        test_native_conversion_kernel<<<grid, block>>>(d_float, d_bf16_a, n);
    }
    hip_check_error(hipDeviceSynchronize());

    // Time float to bf16 conversion
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++)
    {
        test_native_conversion_kernel<<<grid, block>>>(d_float, d_bf16_a, n);
    }
    hip_check_error(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_time   = static_cast<double>(duration) / iterations;
    double throughput = (n * sizeof(float) + n * sizeof(bf16_t)) / (avg_time * 1e6); // GB/s

    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Float to BF16 conversion:" << std::endl;
    std::cout << "  Average time: " << avg_time << " μs" << std::endl;
    std::cout << "  Throughput: " << throughput << " GB/s" << std::endl;

    // Initialize bf16 data for arithmetic test
    test_native_conversion_kernel<<<grid, block>>>(d_float, d_bf16_b, n);
    hip_check_error(hipDeviceSynchronize());

    // Time bf16 arithmetic operations
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++)
    {
        test_arithmetic_performance_kernel<<<grid, block>>>(d_bf16_a, d_bf16_b, d_bf16_c, n);
    }
    hip_check_error(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();

    duration   = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    avg_time   = static_cast<double>(duration) / iterations;
    throughput = (3 * n * sizeof(bf16_t)) / (avg_time * 1e6); // GB/s

    std::cout << "\nBF16 arithmetic operations:" << std::endl;
    std::cout << "  Average time: " << avg_time << " μs" << std::endl;
    std::cout << "  Throughput: " << throughput << " GB/s" << std::endl;
    std::cout << "===========================" << std::endl;

    hip_check_error(hipFree(d_float));
    hip_check_error(hipFree(d_bf16_a));
    hip_check_error(hipFree(d_bf16_b));
    hip_check_error(hipFree(d_bf16_c));
}

// Test platform-specific edge cases
TEST_F(Bf16PlatformTest, PlatformEdgeCases)
{
    // Test that the implementation handles architecture-specific quirks correctly

    // Test subnormal handling
    {
        float subnormal = std::numeric_limits<float>::denorm_min();
        bf16_t b        = float_to_bf16(subnormal);

        // bf16 doesn't support subnormals, should flush to zero
        EXPECT_EQ(bf16_to_bits(b), 0x0000);
    }

    // Test NaN propagation with different NaN patterns
    {
        // Create different NaN bit patterns
        std::vector<uint32_t> nan_patterns = {
            0x7FC00000, // Quiet NaN (standard)
            0x7F800001, // Signaling NaN (smallest)
            0x7FFFFFFF, // All mantissa bits set
            0xFFC00000, // Negative quiet NaN
            0xFF800001, // Negative signaling NaN
        };

        for(uint32_t pattern : nan_patterns)
        {
            float f = bit_cast<float>(pattern);
            EXPECT_TRUE(std::isnan(f));

            // Use explicit standard rounding mode which preserves sNaN
            // The standard rounding implementation sets bit 16 when converting
            // NaN patterns with mantissa bits only in lower 16 bits
            bf16_t b =
                bit_cast<bf16_t>(float_to_bf16_raw(f, constant<bf16_rounding_mode::standard>{}));
            EXPECT_TRUE(isnan(b)) << "Pattern 0x" << std::hex << pattern
                                  << " should convert to bf16 NaN";
        }
    }

    // Test boundary values
    // Note: BF16 has the same 8-bit exponent as float32, so they have the same range.
    // BF16 max is approximately 3.39e38 (0x7F7F = 2^127 * (1 + 127/128))
    {
        // Value within bf16 range
        float large = 3.38953139e38f; // Close to bf16 max
        bf16_t b    = float_to_bf16(large);
        EXPECT_FALSE(std::isinf(bf16_to_float(b)));

        // Float infinity should convert to bf16 infinity
        float inf = std::numeric_limits<float>::infinity();
        b         = float_to_bf16(inf);
        EXPECT_TRUE(std::isinf(bf16_to_float(b)));

        // Float max with truncation should NOT overflow (same exponent range)
        // Note: Standard rounding CAN overflow float_max to infinity due to
        // mantissa rounding propagating to the exponent. Use truncation mode
        // to verify that the range is preserved without rounding effects.
        float float_max = std::numeric_limits<float>::max();
        b               = bit_cast<bf16_t>(
            float_to_bf16_raw(float_max, constant<bf16_rounding_mode::truncate>{}));
        EXPECT_FALSE(std::isinf(bf16_to_float(b)))
            << "Float max with truncation should NOT overflow to bf16 infinity";
    }
}

// Test FMA accumulation precision (critical for GEMM kernels)
TEST_F(Bf16PlatformTest, FmaAccumulationPrecision)
{
    // This test verifies that bf16 * bf16 accumulated in fp32 maintains precision.
    // This is the standard pattern used in matrix multiplication kernels.

    const size_t n       = 256;
    const int iterations = 1000;

    bf16_t* d_a;
    bf16_t* d_b;
    float* d_acc;

    hip_check_error(hipMalloc(&d_a, n * sizeof(bf16_t)));
    hip_check_error(hipMalloc(&d_b, n * sizeof(bf16_t)));
    hip_check_error(hipMalloc(&d_acc, n * sizeof(float)));

    std::vector<bf16_t> h_a(n);
    std::vector<bf16_t> h_b(n);
    std::vector<float> h_acc(n, 0.0f);

    // Test case 1: Small values that would underflow if accumulated in bf16
    // 0.001 * 0.001 = 0.000001, which is below bf16 precision
    // But accumulated 1000 times in fp32: 1000 * 0.000001 = 0.001
    {
        for(size_t i = 0; i < n; i++)
        {
            h_a[i]   = float_to_bf16(0.001f);
            h_b[i]   = float_to_bf16(0.001f);
            h_acc[i] = 0.0f;
        }

        hip_check_error(hipMemcpy(d_a, h_a.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));
        hip_check_error(hipMemcpy(d_b, h_b.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));
        hip_check_error(hipMemcpy(d_acc, h_acc.data(), n * sizeof(float), hipMemcpyHostToDevice));

        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        test_fma_accumulate_kernel<<<grid, block>>>(d_a, d_b, d_acc, n, iterations);
        hip_check_error(hipDeviceSynchronize());

        hip_check_error(hipMemcpy(h_acc.data(), d_acc, n * sizeof(float), hipMemcpyDeviceToHost));

        // Expected: iterations * (0.001 * 0.001) = 1000 * 0.000001 = 0.001
        // Note: bf16(0.001) is approximately 0.0009765625 due to rounding
        float bf16_val = bf16_to_float(float_to_bf16(0.001f));
        float expected = static_cast<float>(iterations) * bf16_val * bf16_val;

        for(size_t i = 0; i < n; i++)
        {
            EXPECT_NEAR(h_acc[i], expected, expected * 0.01f)
                << "FMA accumulation failed at index " << i;
        }

        std::cout << "FMA small value accumulation: " << iterations << " iterations of " << bf16_val
                  << " * " << bf16_val << " = " << h_acc[0] << " (expected: " << expected << ")"
                  << std::endl;
    }

    // Test case 2: Mixed signs to test catastrophic cancellation
    // Alternating +1 and -1 should sum to 0 (or close to it)
    {
        for(size_t i = 0; i < n; i++)
        {
            h_a[i]   = float_to_bf16(1.0f);
            h_b[i]   = float_to_bf16((i % 2 == 0) ? 1.0f : -1.0f);
            h_acc[i] = 0.0f;
        }

        hip_check_error(hipMemcpy(d_a, h_a.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));
        hip_check_error(hipMemcpy(d_b, h_b.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));
        hip_check_error(hipMemcpy(d_acc, h_acc.data(), n * sizeof(float), hipMemcpyHostToDevice));

        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        test_fma_accumulate_kernel<<<grid, block>>>(d_a, d_b, d_acc, n, iterations);
        hip_check_error(hipDeviceSynchronize());

        hip_check_error(hipMemcpy(h_acc.data(), d_acc, n * sizeof(float), hipMemcpyDeviceToHost));

        // Even indices: 1000 * (1.0 * 1.0) = 1000
        // Odd indices: 1000 * (1.0 * -1.0) = -1000
        for(size_t i = 0; i < n; i++)
        {
            float expected =
                (i % 2 == 0) ? static_cast<float>(iterations) : -static_cast<float>(iterations);
            EXPECT_EQ(h_acc[i], expected) << "FMA mixed sign accumulation failed at index " << i;
        }
    }

    // Test case 3: Large values near overflow boundary
    {
        // Use values that when squared approach bf16 max but don't overflow
        float large_val = 100.0f; // 100 * 100 = 10000, well within range
        for(size_t i = 0; i < n; i++)
        {
            h_a[i]   = float_to_bf16(large_val);
            h_b[i]   = float_to_bf16(large_val);
            h_acc[i] = 0.0f;
        }

        hip_check_error(hipMemcpy(d_a, h_a.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));
        hip_check_error(hipMemcpy(d_b, h_b.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));
        hip_check_error(hipMemcpy(d_acc, h_acc.data(), n * sizeof(float), hipMemcpyHostToDevice));

        // Only 10 iterations to avoid overflow: 10 * 10000 = 100000
        const int large_iterations = 10;
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        test_fma_accumulate_kernel<<<grid, block>>>(d_a, d_b, d_acc, n, large_iterations);
        hip_check_error(hipDeviceSynchronize());

        hip_check_error(hipMemcpy(h_acc.data(), d_acc, n * sizeof(float), hipMemcpyDeviceToHost));

        float bf16_large = bf16_to_float(float_to_bf16(large_val));
        float expected   = static_cast<float>(large_iterations) * bf16_large * bf16_large;

        for(size_t i = 0; i < n; i++)
        {
            EXPECT_NEAR(h_acc[i], expected, expected * 0.001f)
                << "FMA large value accumulation failed at index " << i;
        }

        std::cout << "FMA large value accumulation: " << large_iterations << " iterations of "
                  << bf16_large << " * " << bf16_large << " = " << h_acc[0]
                  << " (expected: " << expected << ")" << std::endl;
    }

    hip_check_error(hipFree(d_a));
    hip_check_error(hipFree(d_b));
    hip_check_error(hipFree(d_acc));
}

// Summary test
TEST_F(Bf16PlatformTest, PlatformSummary)
{
    std::cout << "\n=== BF16 Implementation Summary ===" << std::endl;
    std::cout << "Device: " << device_name << std::endl;

// FIXME: CK_TILE_USE_CUSTOM_DATA_TYPE is broken - causes compilation errors
#if 0 // CK_TILE_USE_CUSTOM_DATA_TYPE
    std::cout << "Implementation: Custom BF16 struct with software arithmetic" << std::endl;
#elif CK_TILE_USE_LLVM_BUILTIN_BF16
#if defined(__gfx950__) || defined(CK_GFX950_SUPPORT)
    std::cout << "Implementation: Hardware __bf16 with native conversion (gfx950)" << std::endl;
#else
    std::cout << "Implementation: LLVM __bf16 builtin type" << std::endl;
#endif
#else
    std::cout << "Implementation: ushort with software conversion" << std::endl;
#endif

    std::cout << "Vector types supported: bf16x2_t, bf16x4_t, bf16x8_t, etc." << std::endl;
#if defined(CK_TILE_USE_CUSTOM_DATA_TYPE)
    std::cout << "Arithmetic operators: "
              << (CK_TILE_USE_CUSTOM_DATA_TYPE ? "Available (BROKEN)" : "Not available")
              << std::endl;
#endif
    std::cout << "===================================" << std::endl;
}

// ============================================================================
// Tests from test_bf16_vector.cpp
// ============================================================================

using namespace ck_tile;
using namespace ck_tile_test;

// Device kernel for testing vector operations
__global__ void test_vector_conversion_kernel(const float* input, bf16_t* output, size_t n)
{
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if(idx + 1 < n)
    {
        fp32x2_t f32_vec;
        f32_vec.x = input[idx];
        f32_vec.y = input[idx + 1];

        bf16x2_t bf16_vec = fp32x2_to_bf16x2(f32_vec);
        output[idx]       = bf16_vec.x;
        output[idx + 1]   = bf16_vec.y;
    }
}

__global__ void test_vector_element_access_kernel(bf16_t* data, size_t n)
{
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if(idx + 3 < n)
    {
        // Test bf16x4_t element access
        bf16x4_t vec4;
        vec4.x = data[idx];
        vec4.y = data[idx + 1];
        vec4.z = data[idx + 2];
        vec4.w = data[idx + 3];

        // Test lo/hi access
        bf16x2_t lo = vec4.lo;
        bf16x2_t hi = vec4.hi;

        // Write back swapped
        data[idx]     = hi.x; // was vec4.z
        data[idx + 1] = hi.y; // was vec4.w
        data[idx + 2] = lo.x; // was vec4.x
        data[idx + 3] = lo.y; // was vec4.y
    }
}

class Bf16VectorTest : public Bf16TestBase
{
};

// Test vector type sizes and alignment
TEST_F(Bf16VectorTest, VectorTypeSizes)
{
    // Verify sizes
    EXPECT_EQ(sizeof(bf16x2_t), 4);    // 2 * 2 bytes
    EXPECT_EQ(sizeof(bf16x4_t), 8);    // 4 * 2 bytes
    EXPECT_EQ(sizeof(bf16x8_t), 16);   // 8 * 2 bytes
    EXPECT_EQ(sizeof(bf16x16_t), 32);  // 16 * 2 bytes
    EXPECT_EQ(sizeof(bf16x32_t), 64);  // 32 * 2 bytes
    EXPECT_EQ(sizeof(bf16x64_t), 128); // 64 * 2 bytes

    // Verify alignment
    EXPECT_EQ(alignof(bf16x2_t), 4);
    EXPECT_EQ(alignof(bf16x4_t), 8);
    EXPECT_EQ(alignof(bf16x8_t), 16);
    EXPECT_EQ(alignof(bf16x16_t), 32);
    EXPECT_EQ(alignof(bf16x32_t), 64);
    EXPECT_EQ(alignof(bf16x64_t), 128);
}

// Test fp32x2_to_bf16x2 conversion
TEST_F(Bf16VectorTest, Fp32x2ToBf16x2Conversion)
{
    // Basic conversion
    {
        fp32x2_t f32_vec;
        f32_vec.x = 1.0f;
        f32_vec.y = 2.0f;

        bf16x2_t bf16_vec = fp32x2_to_bf16x2(f32_vec);

        EXPECT_EQ(static_cast<float>(bf16_vec.x), 1.0f);
        EXPECT_EQ(static_cast<float>(bf16_vec.y), 2.0f);
    }

    // Special values
    {
        fp32x2_t f32_vec;
        f32_vec.x = std::numeric_limits<float>::infinity();
        f32_vec.y = -std::numeric_limits<float>::infinity();

        bf16x2_t bf16_vec = fp32x2_to_bf16x2(f32_vec);

        EXPECT_TRUE(std::isinf(static_cast<float>(bf16_vec.x)));
        EXPECT_GT(static_cast<float>(bf16_vec.x), 0.0f);
        EXPECT_TRUE(std::isinf(static_cast<float>(bf16_vec.y)));
        EXPECT_LT(static_cast<float>(bf16_vec.y), 0.0f);
    }

    // NaN values
    {
        fp32x2_t f32_vec;
        f32_vec.x = std::numeric_limits<float>::quiet_NaN();
        f32_vec.y = 3.14159f;

        bf16x2_t bf16_vec = fp32x2_to_bf16x2(f32_vec);

        EXPECT_TRUE(isnan(bf16_vec.x));
        EXPECT_NEAR(bf16_to_float(bf16_vec.y), 3.14159f, 0.01f);
    }
}

// Test different rounding modes for vector conversion
TEST_F(Bf16VectorTest, Fp32x2ToBf16x2RoundingModes)
{
    fp32x2_t f32_vec;
    f32_vec.x = 1.001953125f; // Requires rounding
    f32_vec.y = -1.001953125f;

    // Standard rounding
    {
        bf16x2_t bf16_vec = fp32x2_to_bf16x2<bf16_rounding_mode::standard>(f32_vec);
        float x_result    = static_cast<float>(bf16_vec.x);
        float y_result    = static_cast<float>(bf16_vec.y);
        EXPECT_NEAR(x_result, 1.001953125f, 0.01f);
        EXPECT_NEAR(y_result, -1.001953125f, 0.01f);
    }

    // Truncation mode
    {
        bf16x2_t bf16_vec = fp32x2_to_bf16x2<bf16_rounding_mode::truncate>(f32_vec);
        float x_result    = static_cast<float>(bf16_vec.x);
        float y_result    = static_cast<float>(bf16_vec.y);
        EXPECT_LE(x_result, 1.001953125f);
        EXPECT_GE(y_result, -1.001953125f);
    }
}

// Test vector element access
TEST_F(Bf16VectorTest, VectorElementAccess)
{
    // bf16x2_t element access
    {
        bf16x2_t vec2;
        vec2.x = float_to_bf16(1.0f);
        vec2.y = float_to_bf16(2.0f);

        EXPECT_EQ(static_cast<float>(vec2.x), 1.0f);
        EXPECT_EQ(static_cast<float>(vec2.y), 2.0f);

        // Modify elements
        vec2.x = float_to_bf16(3.0f);
        vec2.y = float_to_bf16(4.0f);

        EXPECT_EQ(static_cast<float>(vec2.x), 3.0f);
        EXPECT_EQ(static_cast<float>(vec2.y), 4.0f);
    }

    // bf16x4_t element access
    {
        bf16x4_t vec4;
        vec4.x = float_to_bf16(1.0f);
        vec4.y = float_to_bf16(2.0f);
        vec4.z = float_to_bf16(3.0f);
        vec4.w = float_to_bf16(4.0f);

        EXPECT_EQ(static_cast<float>(vec4.x), 1.0f);
        EXPECT_EQ(static_cast<float>(vec4.y), 2.0f);
        EXPECT_EQ(static_cast<float>(vec4.z), 3.0f);
        EXPECT_EQ(static_cast<float>(vec4.w), 4.0f);

        // Test lo/hi access
        bf16x2_t lo = vec4.lo;
        bf16x2_t hi = vec4.hi;

        EXPECT_EQ(static_cast<float>(lo.x), 1.0f);
        EXPECT_EQ(static_cast<float>(lo.y), 2.0f);
        EXPECT_EQ(static_cast<float>(hi.x), 3.0f);
        EXPECT_EQ(static_cast<float>(hi.y), 4.0f);
    }
}

// Test vector initialization patterns
TEST_F(Bf16VectorTest, VectorInitialization)
{
    // Default initialization
    {
        bf16x2_t vec2;
        bf16x4_t vec4;
        bf16x8_t vec8;
        // Default initialized vectors have undefined values, so we just check they exist
        EXPECT_EQ(sizeof(vec2), 4);
        EXPECT_EQ(sizeof(vec4), 8);
        EXPECT_EQ(sizeof(vec8), 16);
    }

    // Brace initialization
    {
        bf16x2_t vec2 = {float_to_bf16(1.0f), float_to_bf16(2.0f)};
        EXPECT_EQ(static_cast<float>(vec2.x), 1.0f);
        EXPECT_EQ(static_cast<float>(vec2.y), 2.0f);

        bf16x4_t vec4 = {
            float_to_bf16(1.0f), float_to_bf16(2.0f), float_to_bf16(3.0f), float_to_bf16(4.0f)};
        EXPECT_EQ(static_cast<float>(vec4.x), 1.0f);
        EXPECT_EQ(static_cast<float>(vec4.y), 2.0f);
        EXPECT_EQ(static_cast<float>(vec4.z), 3.0f);
        EXPECT_EQ(static_cast<float>(vec4.w), 4.0f);
    }
}

// Test vector operations on device
TEST_F(Bf16VectorTest, VectorOperationsDevice)
{
    const size_t n = 256;
    float* d_float_input;
    bf16_t* d_bf16_output;

    hip_check_error(hipMalloc(&d_float_input, n * sizeof(float)));
    hip_check_error(hipMalloc(&d_bf16_output, n * sizeof(bf16_t)));

    // Generate test data
    std::vector<float> h_float_input(n);
    std::vector<bf16_t> h_bf16_output(n);

    for(size_t i = 0; i < n; i++)
    {
        h_float_input[i] = static_cast<float>(i) * 0.1f - 12.8f;
    }

    // Test vector conversion on device
    {
        hip_check_error(hipMemcpy(
            d_float_input, h_float_input.data(), n * sizeof(float), hipMemcpyHostToDevice));

        dim3 block(128);
        dim3 grid((n / 2 + block.x - 1) / block.x);
        test_vector_conversion_kernel<<<grid, block>>>(d_float_input, d_bf16_output, n);

        hip_check_error(hipMemcpy(
            h_bf16_output.data(), d_bf16_output, n * sizeof(bf16_t), hipMemcpyDeviceToHost));

        // Verify results
        for(size_t i = 0; i < n; i++)
        {
            float expected = h_float_input[i];
            float actual   = static_cast<float>(h_bf16_output[i]);
            EXPECT_NEAR(actual, expected, std::abs(expected) * 0.01f + 0.01f)
                << "Mismatch at index " << i;
        }
    }

    // Test element access on device
    {
        // Initialize with pattern
        for(size_t i = 0; i < n; i++)
        {
            h_bf16_output[i] = float_to_bf16(static_cast<float>(i % 4));
        }

        hip_check_error(hipMemcpy(
            d_bf16_output, h_bf16_output.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));

        dim3 block(64);
        dim3 grid((n / 4 + block.x - 1) / block.x);
        test_vector_element_access_kernel<<<grid, block>>>(d_bf16_output, n);

        hip_check_error(hipMemcpy(
            h_bf16_output.data(), d_bf16_output, n * sizeof(bf16_t), hipMemcpyDeviceToHost));

        // Verify swapping worked correctly
        for(size_t i = 0; i < n; i += 4)
        {
            // Original: [0, 1, 2, 3]
            // After swap: [2, 3, 0, 1]
            EXPECT_EQ(static_cast<float>(h_bf16_output[i]), 2.0f);
            EXPECT_EQ(static_cast<float>(h_bf16_output[i + 1]), 3.0f);
            EXPECT_EQ(static_cast<float>(h_bf16_output[i + 2]), 0.0f);
            EXPECT_EQ(static_cast<float>(h_bf16_output[i + 3]), 1.0f);
        }
    }

    hip_check_error(hipFree(d_float_input));
    hip_check_error(hipFree(d_bf16_output));
}

// Test vector type traits
TEST_F(Bf16VectorTest, VectorTypeTraits)
{
    // Verify vector types are trivially copyable
    EXPECT_TRUE(std::is_trivially_copyable<bf16x2_t>::value);
    EXPECT_TRUE(std::is_trivially_copyable<bf16x4_t>::value);
    EXPECT_TRUE(std::is_trivially_copyable<bf16x8_t>::value);
    EXPECT_TRUE(std::is_trivially_copyable<bf16x16_t>::value);
    EXPECT_TRUE(std::is_trivially_copyable<bf16x32_t>::value);
    EXPECT_TRUE(std::is_trivially_copyable<bf16x64_t>::value);

    // Verify POD nature
    EXPECT_TRUE(std::is_standard_layout<bf16x2_t>::value);
    EXPECT_TRUE(std::is_standard_layout<bf16x4_t>::value);
    EXPECT_TRUE(std::is_standard_layout<bf16x8_t>::value);
}

// Test edge cases with vector operations
TEST_F(Bf16VectorTest, VectorEdgeCases)
{
    // Vector with all special values
    {
        bf16x4_t vec;
        vec.x = numeric<bf16_t>::infinity();
        vec.y = bits_to_bf16(0xFF80); // -infinity
        vec.z = numeric<bf16_t>::quiet_NaN();
        vec.w = float_to_bf16(0.0f);

        // Verify each element
        EXPECT_TRUE(std::isinf(static_cast<float>(vec.x)) && static_cast<float>(vec.x) > 0);
        EXPECT_TRUE(std::isinf(static_cast<float>(vec.y)) && static_cast<float>(vec.y) < 0);
        EXPECT_TRUE(isnan(vec.z));
        EXPECT_EQ(static_cast<float>(vec.w), 0.0f);
    }

    // Vector conversion with float max (IEEE RTN rounding to infinity)
    {
        fp32x2_t f32_vec;
        f32_vec.x = std::numeric_limits<float>::max();
        f32_vec.y = -std::numeric_limits<float>::max();

        bf16x2_t bf16_vec = fp32x2_to_bf16x2(f32_vec);

        // BF16 has same 8-bit exponent but only 7 mantissa bits (vs 23 for float32).
        // So bf16::max < float::max.
        // Hardware behavior differs by architecture:
        // - gfx950: RTN rounding -> float::max rounds to infinity (IEEE-754 compliant)
        // - gfx9 (gfx90a, gfx908, gfx942): Saturates -> float::max clamps to bf16::max
        // - gfx12/gfx1250: Saturates -> float::max clamps to bf16::max (faster, non-IEEE)
        float result_x = bf16_to_float(bf16_vec.x);
        float result_y = bf16_to_float(bf16_vec.y);

#ifdef CK_TILE_BF16_OVERFLOW_SATURATES
        // gfx9/gfx11/gfx12: Hardware saturates to bf16::max
        EXPECT_FALSE(std::isinf(result_x))
            << "gfx9/gfx11/gfx12: float::max should saturate to bf16::max";
        EXPECT_FALSE(std::isinf(result_y))
            << "gfx9/gfx11/gfx12: -float::max should saturate to -bf16::max";
#else
        // gfx950 and software: RTN rounding to infinity (IEEE-754 behavior)
        EXPECT_TRUE(std::isinf(result_x) && result_x > 0)
            << "Float max should overflow to bf16 +infinity with RTN rounding";
        EXPECT_TRUE(std::isinf(result_y) && result_y < 0)
            << "Negative float max should overflow to bf16 -infinity with RTN rounding";
#endif
    }

    // Vector conversion with denormals
    {
        fp32x2_t f32_vec;
        f32_vec.x = std::numeric_limits<float>::denorm_min();
        f32_vec.y = -std::numeric_limits<float>::denorm_min();

        bf16x2_t bf16_vec = fp32x2_to_bf16x2(f32_vec);

        // bf16 doesn't support denormals, should flush to zero
        float result_x = static_cast<float>(bf16_vec.x);
        float result_y = static_cast<float>(bf16_vec.y);
        EXPECT_EQ(result_x, 0.0f);
        EXPECT_FALSE(std::signbit(result_x)); // Positive zero
        EXPECT_EQ(result_y, 0.0f);
        EXPECT_TRUE(std::signbit(result_y)); // Negative zero
    }
}
