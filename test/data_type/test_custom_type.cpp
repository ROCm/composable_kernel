// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"

using ck::f8_t;
using ck::half_t;

struct custom_f8_t
{
    uint8_t data;
};

TEST(Custom_FP8, Test)
{
    ASSERT_EQ(sizeof(f8_t), 1);
    ASSERT_EQ(sizeof(ck::f8x2_t), 2);
    ASSERT_EQ(sizeof(ck::f8x4_t), 4);

    ASSERT_EQ(sizeof(custom_f8_t), 1);
    ASSERT_EQ(sizeof(ck::non_native_vector_type<custom_f8_t, 1>), 1);
    ASSERT_EQ(sizeof(ck::non_native_vector_type<custom_f8_t, 2>), 2);
    ASSERT_EQ(sizeof(ck::non_native_vector_type<custom_f8_t, 4>), 4);
}
