// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core.hpp"

#include "gtest/gtest.h"

template <typename T>
static ck_tile::uint32_t to_bits(T x)
{
    return ck_tile::bit_cast<ck_tile::uint32_t>(x);
}

#if CK_TILE_FLOAT_TO_TF32_DEFAULT == CK_TILE_FLOAT_TO_TF32_RNE

static ck_tile::tf32_t from_bits(ck_tile::uint32_t i)
{
    return ck_tile::type_convert<ck_tile::tf32_t>(ck_tile::bit_cast<float>(i));
}

#endif

TEST(ConvertTest, NumericTraits)
{
    using ck_tile::numeric_traits;
    using ck_tile::tf32_t;

    EXPECT_EQ(numeric_traits<tf32_t>::exp, 8);
    EXPECT_EQ(numeric_traits<tf32_t>::mant, 10);
    EXPECT_EQ(numeric_traits<tf32_t>::bias, 127);
    EXPECT_EQ(numeric_traits<tf32_t>::PackedSize, 1);
}

#if CK_TILE_FLOAT_TO_TF32_DEFAULT == CK_TILE_FLOAT_TO_TF32_TRUNC

TEST(ConvertTest, ToTf32Trunc)
{
    using ck_tile::isnan;
    using ck_tile::numeric;
    using ck_tile::tf32_t;
    using ck_tile::type_convert;
    using ck_tile::uint32_t;

    // exact values (low 13 bits already zero)
    EXPECT_EQ(to_bits(type_convert<tf32_t>(1.0f)), 0x3F800000u);  // 1.0f
    EXPECT_EQ(to_bits(type_convert<tf32_t>(-1.0f)), 0xBF800000u); // -1.0f
    EXPECT_EQ(to_bits(type_convert<tf32_t>(0.0f)), 0x00000000u);  // +0.0f
    EXPECT_EQ(to_bits(type_convert<tf32_t>(-0.0f)), 0x80000000u); // -0.0f
    EXPECT_EQ(to_bits(type_convert<tf32_t>(2.0f)), 0x40000000u);  // 2.0f
    EXPECT_EQ(to_bits(type_convert<tf32_t>(0.5f)), 0x3F000000u);  // 0.5f

    // truncation zeros the low 13 mantissa bits
    EXPECT_EQ(to_bits(type_convert<tf32_t>(1.1f)), 0x3F8CC000u); // 1.1f (0x3F8CCCCD)
    EXPECT_EQ(to_bits(type_convert<tf32_t>(3.14159265358979323846f)),
              0x40490000u); // pi (0x40490FDB)
    EXPECT_EQ(to_bits(type_convert<tf32_t>(123.456f)),
              0x42F6E000u);                                        // 123.456f (0x42F6E979)
    EXPECT_EQ(to_bits(type_convert<tf32_t>(-3.14f)), 0xC048E000u); // -3.14f (0xC048F5C3)

    // special values
    EXPECT_EQ(to_bits(numeric<tf32_t>::infinity()), 0x7F800000u);
    EXPECT_EQ(to_bits(-numeric<tf32_t>::infinity()), 0xFF800000u);
    EXPECT_TRUE(isnan(numeric<tf32_t>::quiet_NaN()));
    EXPECT_EQ(to_bits(numeric<tf32_t>::denorm_min()), 0x00002000u);

    // property: low 13 bits must be zero, top 19 bits preserved
    for(float val : {1.0f, 1.5f, 2.0f, 0.1f, 100.0f, -42.5f, 1e10f, 1e-10f})
    {
        uint32_t orig = to_bits(val);
        uint32_t tf32 = to_bits(type_convert<tf32_t>(val));

        EXPECT_EQ(tf32 & 0xFFFFE000u, tf32) << "val=" << val;
        EXPECT_EQ(orig & 0xFFFFE000u, tf32) << "val=" << val;
    }
}

#elif CK_TILE_FLOAT_TO_TF32_DEFAULT == CK_TILE_FLOAT_TO_TF32_RNE

TEST(ConvertTest, ToTf32Rne)
{
    using ck_tile::isnan;
    using ck_tile::numeric;
    using ck_tile::tf32_t;
    using ck_tile::type_convert;

    // exact values (low 13 bits already zero)
    EXPECT_EQ(to_bits(type_convert<tf32_t>(1.0f)),
              0x3F800000u); // 1.0f
    EXPECT_EQ(to_bits(type_convert<tf32_t>(-1.0f)),
              0xBF800000u); // -1.0f
    EXPECT_EQ(to_bits(type_convert<tf32_t>(0.0f)),
              0x00000000u); // +0.0f

    // past midpoint (bit12 + bit11 set) -> rounds up
    EXPECT_EQ(to_bits(from_bits(0x3F801800u)), 0x3F802000u);

    // special values (keep the same as float)
    EXPECT_EQ(to_bits(numeric<tf32_t>::infinity()),
              0x7F800000u); // infinity in float is 0x7F800000
    EXPECT_EQ(to_bits(-numeric<tf32_t>::infinity()),
              0xFF800000u);                           // negative infinity in float is 0xFF800000
    EXPECT_TRUE(isnan(numeric<tf32_t>::quiet_NaN())); // quiet NaN in float is 0x7FC00000
}

#endif
