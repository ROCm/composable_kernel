// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"

#include "ck_tile/core.hpp"

template <typename T>
class ConvertTest : public ::testing::Test
{
};

using TestTypes = ::testing::Types<float, ck_tile::fp16_t>;

TYPED_TEST_SUITE(ConvertTest, TestTypes);

TYPED_TEST(ConvertTest, ToFp8)
{
    using SrcT = TypeParam;
    using DstT = ck_tile::fp8_t;

    auto c = [](SrcT f) {
        return static_cast<unsigned int>(
            ck_tile::bit_cast<uint8_t>(ck_tile::impl::run_cast_to_f8<SrcT, DstT, true>(f)));
    };

    auto c_nosat = [](SrcT f) {
        return static_cast<unsigned int>(
            ck_tile::bit_cast<uint8_t>(ck_tile::impl::run_cast_to_f8<SrcT, DstT, false>(f)));
    };

#if CK_TILE_USE_OCP_FP8
    EXPECT_EQ(c(+1.0f), 0b0'0111'000);
    EXPECT_EQ(c(-1.0f), 0b1'0111'000);
    // max f8 normal
    EXPECT_EQ(c(+448.0f), 0b0'1111'110);
    EXPECT_EQ(c(-448.0f), 0b1'1111'110);
    // min f8 normal
    EXPECT_EQ(c(+0.015625f), 0b0'0001'000);
    EXPECT_EQ(c(-0.015625f), 0b1'0001'000);
    // max f8 subnormal
    EXPECT_EQ(c(+0.013671875f), 0b0'0000'111);
    EXPECT_EQ(c(-0.013671875f), 0b1'0000'111);
    // min f8 subnormal
    EXPECT_EQ(c(+0.001953125f), 0b0'0000'001);
    EXPECT_EQ(c(-0.001953125f), 0b1'0000'001);

    // saturating mode -> max f8 normal
    // max f32/f16 normal -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::max()), 0b0'1111'110);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::max()), 0b1'1111'110);
    // f32/f16 infinity -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::infinity()), 0b0'1111'110);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::infinity()), 0b1'1111'110);
    // large f32/f16 -> max f8 normal
    EXPECT_EQ(c(+1.23e9f), 0b0'1111'110);
    EXPECT_EQ(c(-1.23e9f), 0b1'1111'110);

    constexpr unsigned int nan_mask = 0b0'1111'111;

    // non-saturating mode -> f8 NaN (because OCP e4m3 has no infinity)
    // max f32/f16 normal -> f8 NaN
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::max()) & nan_mask, nan_mask);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::max()) & nan_mask, nan_mask);
    // f32/f16 infinity -> f8 NaN
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::infinity()) & nan_mask, nan_mask);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::infinity()) & nan_mask, nan_mask);
    // large f32/f16 -> f8 NaN
    EXPECT_EQ(c_nosat(+1.23e9f) & nan_mask, nan_mask);
    EXPECT_EQ(c_nosat(-1.23e9f) & nan_mask, nan_mask);

    // f32/f16 NaN -> f8 NaN
    EXPECT_EQ(c(ck_tile::numeric<SrcT>::quiet_NaN()) & nan_mask, nan_mask);
    EXPECT_EQ(c(ck_tile::numeric<SrcT>::signaling_NaN()) & nan_mask, nan_mask);

    // f32/f16 zero -> f8 zero
    EXPECT_EQ(c(+0.0f), 0b0'0000'000);
    EXPECT_EQ(c(-0.0f), 0b1'0000'000);
    // min f32/f16 normal -> f8 zero
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::min()), 0b0'0000'000);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::min()), 0b1'0000'000);
    // min f32/f16 subnormal -> f8 zero
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::denorm_min()), 0b0'0000'000);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::denorm_min()), 0b1'0000'000);

    // All values smaller than min f8 subnormal must be converted to f8 zero
    constexpr int start =
        ck_tile::numeric_traits<SrcT>::bias + ck_tile::numeric_traits<SrcT>::mant - 1;
    constexpr int end =
        ck_tile::numeric_traits<DstT>::bias + ck_tile::numeric_traits<DstT>::mant - 1;
    for(int n = -start; n < -end; ++n)
    {
        const float f = std::ldexp(1.0, n);
        EXPECT_EQ(c(+f), 0b0'0000'000) << "+f = 2^" << n << " = " << +f;
        EXPECT_EQ(c(-f), 0b1'0000'000) << "-f = 2^" << n << " = " << -f;
    }
#else // FNUZ
    EXPECT_EQ(c(+1.0f), 0b0'1000'000);
    EXPECT_EQ(c(-1.0f), 0b1'1000'000);
    // max f8 normal
    EXPECT_EQ(c(+240.0f), 0b0'1111'111);
    EXPECT_EQ(c(-240.0f), 0b1'1111'111);
    // min f8 normal
    EXPECT_EQ(c(+0.0078125f), 0b0'0001'000);
    EXPECT_EQ(c(-0.0078125f), 0b1'0001'000);
    // max f8 subnormal
    EXPECT_EQ(c(+0.0068359375f), 0b0'0000'111);
    EXPECT_EQ(c(-0.0068359375f), 0b1'0000'111);
    // min f8 subnormal
    EXPECT_EQ(c(+0.0009765625f), 0b0'0000'001);
    EXPECT_EQ(c(-0.0009765625f), 0b1'0000'001);

    // saturating mode -> max f8 normal
    // max f32/f16 normal -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::max()), 0b0'1111'111);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::max()), 0b1'1111'111);
    // f32/f16 infinity -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::infinity()), 0b0'1111'111);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::infinity()), 0b1'1111'111);
    // large f32/f16 -> max f8 normal
    EXPECT_EQ(c(+1.23e9f), 0b0'1111'111);
    EXPECT_EQ(c(-1.23e9f), 0b1'1111'111);

    constexpr unsigned int nan_value = 0b1'0000'000;

    // non-saturating mode -> f8 NaN (FN means "finite", so no infinity)
    // max f32/f16 normal -> f8 NaN
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::max()), nan_value);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::max()), nan_value);
    // f32/f16 infinity -> f8 NaN
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::infinity()), nan_value);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::infinity()), nan_value);
    // large f32/f16 -> f8 NaN
    EXPECT_EQ(c_nosat(+1.23e9f), nan_value);
    EXPECT_EQ(c_nosat(-1.23e9f), nan_value);

    // f32/f16 NaN -> f8 NaN
    EXPECT_EQ(c(ck_tile::numeric<SrcT>::quiet_NaN()), nan_value);
    EXPECT_EQ(c(ck_tile::numeric<SrcT>::signaling_NaN()), nan_value);

    // UZ means "unsigned zero" (0b1'0000'000 is NaN)
    // f32/f16 +-zero -> f8 +zero
    EXPECT_EQ(c(+0.0f), 0b0'0000'000);
    EXPECT_EQ(c(-0.0f), 0b0'0000'000);
    // min f32/f16 normal -> f8 +zero
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::min()), 0b0'0000'000);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::min()), 0b0'0000'000);
    // min f32/f16 subnormal -> f8 +zero
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::denorm_min()), 0b0'0000'000);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::denorm_min()), 0b0'0000'000);

    // All values smaller than min f8 subnormal must be converted to f8 zero
    constexpr int start =
        ck_tile::numeric_traits<SrcT>::bias + ck_tile::numeric_traits<SrcT>::mant - 1;
    constexpr int end =
        ck_tile::numeric_traits<DstT>::bias + ck_tile::numeric_traits<DstT>::mant - 1;
    for(int n = -start; n < -end; ++n)
    {
        const float f = std::ldexp(1.0, n);
        EXPECT_EQ(c(+f), 0b0'0000'000) << "+f = 2^" << n << " = " << +f;
        EXPECT_EQ(c(-f), 0b0'0000'000) << "-f = 2^" << n << " = " << -f;
    }
#endif
}

TYPED_TEST(ConvertTest, ToBf8)
{
    using SrcT = TypeParam;
    using DstT = ck_tile::bf8_t;

    auto c = [](SrcT f) {
        return static_cast<unsigned int>(
            ck_tile::bit_cast<uint8_t>(ck_tile::impl::run_cast_to_f8<SrcT, DstT, true>(f)));
    };

    auto c_nosat = [](SrcT f) {
        return static_cast<unsigned int>(
            ck_tile::bit_cast<uint8_t>(ck_tile::impl::run_cast_to_f8<SrcT, DstT, false>(f)));
    };

#if CK_TILE_USE_OCP_FP8
    EXPECT_EQ(c(+1.0f), 0b0'01111'00);
    EXPECT_EQ(c(-1.0f), 0b1'01111'00);
    // max f8 normal
    EXPECT_EQ(c(+57344.0f), 0b0'11110'11);
    EXPECT_EQ(c(-57344.0f), 0b1'11110'11);
    // min f8 normal
    EXPECT_EQ(c(+6.103515625e-05f), 0b0'00001'00);
    EXPECT_EQ(c(-6.103515625e-05f), 0b1'00001'00);
    // max f8 subnormal
    EXPECT_EQ(c(+4.57763671875e-05f), 0b0'00000'11);
    EXPECT_EQ(c(-4.57763671875e-05f), 0b1'00000'11);
    // min f8 subnormal
    EXPECT_EQ(c(+1.52587890625e-05f), 0b0'00000'01);
    EXPECT_EQ(c(-1.52587890625e-05f), 0b1'00000'01);

    // saturating mode -> max f8 normal
    // max f32/f16 normal -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::max()), 0b0'11110'11);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::max()), 0b1'11110'11);
    // f32/f16 infinity -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::infinity()), 0b0'11110'11);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::infinity()), 0b1'11110'11);
    // large f32/f16 -> max f8 normal
    EXPECT_EQ(c(+1.23e9f), 0b0'11110'11);
    EXPECT_EQ(c(-1.23e9f), 0b1'11110'11);

    // non-saturating mode -> f8 infinity
    // max f32/f16 normal -> f8 infinity
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::max()), 0b0'11111'00);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::max()), 0b1'11111'00);
    // f32/f16 infinity -> f8 infinity
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::infinity()), 0b0'11111'00);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::infinity()), 0b1'11111'00);
    // large f32/f16 -> f8 infinity
    EXPECT_EQ(c_nosat(+1.23e9f), 0b0'11111'00);
    EXPECT_EQ(c_nosat(-1.23e9f), 0b1'11111'00);

    // f32/f16 NaN -> f8 NaN
    EXPECT_TRUE((c(ck_tile::numeric<SrcT>::quiet_NaN()) & 0b0'11111'11) != 0b0'11111'00);
    EXPECT_TRUE((c(ck_tile::numeric<SrcT>::signaling_NaN()) & 0b0'11111'11) != 0b0'11111'00);

    // f32/f16 zero -> f8 zero
    EXPECT_EQ(c(+0.0f), 0b0'00000'00);
    EXPECT_EQ(c(-0.0f), 0b1'00000'00);
    if constexpr(std::is_same_v<SrcT, float>)
    {
        // min f32 normal -> f8 zero
        EXPECT_EQ(c(+ck_tile::numeric<SrcT>::min()), 0b0'00000'00);
        EXPECT_EQ(c(-ck_tile::numeric<SrcT>::min()), 0b1'00000'00);
    }
    else
    {
        // min f16 normal -> min f8 normal (they are equal)
        EXPECT_EQ(c(+ck_tile::numeric<SrcT>::min()), 0b0'00001'00);
        EXPECT_EQ(c(-ck_tile::numeric<SrcT>::min()), 0b1'00001'00);
    }
    // min f32/f16 subnormal -> f8 zero
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::denorm_min()), 0b0'00000'00);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::denorm_min()), 0b1'00000'00);

    // All values smaller than min f8 subnormal must be converted to f8 zero
    constexpr int start =
        ck_tile::numeric_traits<SrcT>::bias + ck_tile::numeric_traits<SrcT>::mant - 1;
    constexpr int end =
        ck_tile::numeric_traits<DstT>::bias + ck_tile::numeric_traits<DstT>::mant - 1;
    for(int n = -start; n < -end; ++n)
    {
        const float f = std::ldexp(1.0, n);
        EXPECT_EQ(c(+f), 0b0'00000'00) << "+f = 2^" << n << " = " << +f;
        EXPECT_EQ(c(-f), 0b1'00000'00) << "-f = 2^" << n << " = " << -f;
    }
#else // FNUZ
    EXPECT_EQ(c(+1.0f), 0b0'10000'00);
    EXPECT_EQ(c(-1.0f), 0b1'10000'00);
    // max f8 normal
    EXPECT_EQ(c(+57344.0f), 0b0'11111'11);
    EXPECT_EQ(c(-57344.0f), 0b1'11111'11);
    // min f8 normal
    EXPECT_EQ(c(+3.0517578125e-05f), 0b0'00001'00);
    EXPECT_EQ(c(-3.0517578125e-05f), 0b1'00001'00);
    // max f8 subnormal
    EXPECT_EQ(c(+2.288818e-05f), 0b0'00000'11);
    EXPECT_EQ(c(-2.288818e-05f), 0b1'00000'11);
    // min f8 subnormal
    EXPECT_EQ(c(+7.62939453125e-06f), 0b0'00000'01);
    EXPECT_EQ(c(-7.62939453125e-06f), 0b1'00000'01);

    // saturating mode -> max f8 normal
    // max f32/f16 normal -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::max()), 0b0'11111'11);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::max()), 0b1'1111'111);
    // f32/f16 infinity -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::infinity()), 0b0'11111'11);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::infinity()), 0b1'1111'111);
    // large f32/f16 -> max f8 normal
    EXPECT_EQ(c(+1.23e9f), 0b0'11111'11);
    EXPECT_EQ(c(-1.23e9f), 0b1'1111'111);

    constexpr unsigned int nan_value = 0b1'00000'00;

    // non-saturating mode -> f8 NaN (FN means "finite", so no infinity)
    // max f32/f16 normal -> f8 NaN
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::max()), nan_value);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::max()), nan_value);
    // f32/f16 infinity -> f8 NaN
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::infinity()), nan_value);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::infinity()), nan_value);
    // large f32/f16 -> f8 NaN
    EXPECT_EQ(c_nosat(+1.23e9f), nan_value);
    EXPECT_EQ(c_nosat(-1.23e9f), nan_value);

    // f32/f16 NaN -> f8 NaN
    EXPECT_EQ(c(ck_tile::numeric<SrcT>::quiet_NaN()), nan_value);
    EXPECT_EQ(c(ck_tile::numeric<SrcT>::signaling_NaN()), nan_value);

    // UZ means "unsigned zero" (0b1'00000'00 is NaN)
    // f32/f16 +-zero -> f8 +zero
    EXPECT_EQ(c(+0.0f), 0b0'00000'00);
    EXPECT_EQ(c(-0.0f), 0b0'00000'00);
    if constexpr(std::is_same_v<SrcT, float>)
    {
        // min f32 normal -> f8 +zero
        EXPECT_EQ(c(+ck_tile::numeric<SrcT>::min()), 0b0'00000'00);
        EXPECT_EQ(c(-ck_tile::numeric<SrcT>::min()), 0b0'00000'00);
    }
    else
    {
        // min f16 normal -> f8 normal
        EXPECT_EQ(c(+ck_tile::numeric<SrcT>::min()), 0b0'00010'00);
        EXPECT_EQ(c(-ck_tile::numeric<SrcT>::min()), 0b1'00010'00);
    }
    // min f32/f16 subnormal -> f8 +zero
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::denorm_min()), 0b0'00000'00);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::denorm_min()), 0b0'00000'00);

    // All values smaller than min f8 subnormal must be converted to f8 zero
    constexpr int start =
        ck_tile::numeric_traits<SrcT>::bias + ck_tile::numeric_traits<SrcT>::mant - 1;
    constexpr int end =
        ck_tile::numeric_traits<DstT>::bias + ck_tile::numeric_traits<DstT>::mant - 1;
    for(int n = -start; n < -end; ++n)
    {
        const float f = std::ldexp(1.0, n);
        EXPECT_EQ(c(+f), 0b0'00000'00) << "+f = 2^" << n << " = " << +f;
        EXPECT_EQ(c(-f), 0b0'00000'00) << "-f = 2^" << n << " = " << -f;
    }
#endif
}
