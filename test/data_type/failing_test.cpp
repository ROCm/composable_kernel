// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using ck::bf8_t;
using ck::f8_convert_rne;
using ck::f8_t;
using ck::type_convert;

TEST(FailingTest, FP8)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // convert 0 float to fp8 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f8_convert_rne<f8_t>(0.0f)), abs_tol);
    // convert minimal float to fp8 and back, check if holds
    ASSERT_NEAR(std::numeric_limits<float>::min(),
                type_convert<float>(f8_convert_rne<f8_t>(std::numeric_limits<float>::min())),
                abs_tol);
}

TEST(FailingTest, BF8)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // convert 0 float to bf8 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f8_convert_rne<bf8_t>(0.0f)), abs_tol);
    // convert minimal float to bf8 and back, check if holds
    ASSERT_NEAR(std::numeric_limits<float>::min(),
                type_convert<float>(f8_convert_rne<bf8_t>(std::numeric_limits<float>::min())),
                abs_tol);
}
