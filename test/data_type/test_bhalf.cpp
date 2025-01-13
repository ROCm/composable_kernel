// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using ck::bhalf_t;
using ck::type_convert;

TEST(BHALF_T, Nan)
{
    const uint16_t binary_bhalf_nan = 0x7FC0;
    const bhalf_t bhalf_nan         = *(&binary_bhalf_nan);
    EXPECT_EQ(bhalf_nan, type_convert<bhalf_t>(ck::NumericLimits<float>::QuietNaN()));
}

TEST(BHALF_T, MantisaOverflow)
{
    const float abs_tol   = std::pow(2, -7);
    const uint32_t val    = 0x81FFFFFF;
    const float float_val = *(&val);

    ASSERT_NEAR(float_val, type_convert<float>(type_convert<bhalf_t>(float_val)), abs_tol);
}
