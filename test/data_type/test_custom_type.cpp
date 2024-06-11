// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"

using ck::bf8_t;
using ck::bhalf_t;
using ck::f8_t;
using ck::half_t;
using ck::vector_type;

TEST(Custom_bool, Test)
{
    struct custom_bool_t
    {
        bool data;
    };
    ASSERT_EQ(sizeof(bool), 1);
    ASSERT_EQ(sizeof(custom_bool_t), 1);
    ASSERT_EQ(sizeof(vector_type<bool, 2>), 2);
    ASSERT_EQ(sizeof(vector_type<custom_bool_t, 2>), 2);
    ASSERT_EQ(sizeof(vector_type<bool, 4>), 4);
    ASSERT_EQ(sizeof(vector_type<custom_bool_t, 4>), 4);
    ASSERT_EQ(sizeof(vector_type<bool, 8>), 8);
    ASSERT_EQ(sizeof(vector_type<custom_bool_t, 8>), 8);
    ASSERT_EQ(sizeof(vector_type<bool, 16>), 16);
    ASSERT_EQ(sizeof(vector_type<custom_bool_t, 16>), 16);
    ASSERT_EQ(sizeof(vector_type<bool, 32>), 32);
    ASSERT_EQ(sizeof(vector_type<custom_bool_t, 32>), 32);
    ASSERT_EQ(sizeof(vector_type<bool, 64>), 64);
    ASSERT_EQ(sizeof(vector_type<custom_bool_t, 64>), 64);
}

TEST(Custom_int8_t, Test)
{
    struct custom_int8_t
    {
        int8_t data;
    };
    ASSERT_EQ(sizeof(int8_t), 1);
    ASSERT_EQ(sizeof(custom_int8_t), 1);
    ASSERT_EQ(sizeof(vector_type<int8_t, 2>), 2);
    ASSERT_EQ(sizeof(vector_type<custom_int8_t, 2>), 2);
    ASSERT_EQ(sizeof(vector_type<int8_t, 4>), 4);
    ASSERT_EQ(sizeof(vector_type<custom_int8_t, 4>), 4);
    ASSERT_EQ(sizeof(vector_type<int8_t, 8>), 8);
    ASSERT_EQ(sizeof(vector_type<custom_int8_t, 8>), 8);
    ASSERT_EQ(sizeof(vector_type<int8_t, 16>), 16);
    ASSERT_EQ(sizeof(vector_type<custom_int8_t, 16>), 16);
    ASSERT_EQ(sizeof(vector_type<int8_t, 32>), 32);
    ASSERT_EQ(sizeof(vector_type<custom_int8_t, 32>), 32);
    ASSERT_EQ(sizeof(vector_type<int8_t, 64>), 64);
    ASSERT_EQ(sizeof(vector_type<custom_int8_t, 64>), 64);
}

TEST(Custom_uint8_t, Test)
{
    struct custom_uint8_t
    {
        uint8_t data;
    };
    ASSERT_EQ(sizeof(uint8_t), 1);
    ASSERT_EQ(sizeof(custom_uint8_t), 1);
    ASSERT_EQ(sizeof(vector_type<uint8_t, 2>), 2);
    ASSERT_EQ(sizeof(vector_type<custom_uint8_t, 2>), 2);
    ASSERT_EQ(sizeof(vector_type<uint8_t, 4>), 4);
    ASSERT_EQ(sizeof(vector_type<custom_uint8_t, 4>), 4);
    ASSERT_EQ(sizeof(vector_type<uint8_t, 8>), 8);
    ASSERT_EQ(sizeof(vector_type<custom_uint8_t, 8>), 8);
    ASSERT_EQ(sizeof(vector_type<uint8_t, 16>), 16);
    ASSERT_EQ(sizeof(vector_type<custom_uint8_t, 16>), 16);
    ASSERT_EQ(sizeof(vector_type<uint8_t, 32>), 32);
    ASSERT_EQ(sizeof(vector_type<custom_uint8_t, 32>), 32);
    ASSERT_EQ(sizeof(vector_type<uint8_t, 64>), 64);
    ASSERT_EQ(sizeof(vector_type<custom_uint8_t, 64>), 64);
}

TEST(Custom_f8, Test)
{
    struct custom_f8_t
    {
        _BitInt(8) data;
    };
    ASSERT_EQ(sizeof(f8_t), 1);
    ASSERT_EQ(sizeof(custom_f8_t), 1);
    ASSERT_EQ(sizeof(vector_type<f8_t, 2>), 2);
    ASSERT_EQ(sizeof(vector_type<custom_f8_t, 2>), 2);
    ASSERT_EQ(sizeof(vector_type<f8_t, 4>), 4);
    ASSERT_EQ(sizeof(vector_type<custom_f8_t, 4>), 4);
    ASSERT_EQ(sizeof(vector_type<f8_t, 8>), 8);
    ASSERT_EQ(sizeof(vector_type<custom_f8_t, 8>), 8);
    ASSERT_EQ(sizeof(vector_type<f8_t, 16>), 16);
    ASSERT_EQ(sizeof(vector_type<custom_f8_t, 16>), 16);
    ASSERT_EQ(sizeof(vector_type<f8_t, 32>), 32);
    ASSERT_EQ(sizeof(vector_type<custom_f8_t, 32>), 32);
    ASSERT_EQ(sizeof(vector_type<f8_t, 64>), 64);
    ASSERT_EQ(sizeof(vector_type<custom_f8_t, 64>), 64);
}

TEST(Custom_bf8, Test)
{
    struct custom_bf8_t
    {
        unsigned _BitInt(8) data;
    };
    ASSERT_EQ(sizeof(bf8_t), 1);
    ASSERT_EQ(sizeof(custom_bf8_t), 1);
    ASSERT_EQ(sizeof(vector_type<bf8_t, 2>), 2);
    ASSERT_EQ(sizeof(vector_type<custom_bf8_t, 2>), 2);
    ASSERT_EQ(sizeof(vector_type<bf8_t, 4>), 4);
    ASSERT_EQ(sizeof(vector_type<custom_bf8_t, 4>), 4);
    ASSERT_EQ(sizeof(vector_type<bf8_t, 8>), 8);
    ASSERT_EQ(sizeof(vector_type<custom_bf8_t, 8>), 8);
    ASSERT_EQ(sizeof(vector_type<bf8_t, 16>), 16);
    ASSERT_EQ(sizeof(vector_type<custom_bf8_t, 16>), 16);
    ASSERT_EQ(sizeof(vector_type<bf8_t, 32>), 32);
    ASSERT_EQ(sizeof(vector_type<custom_bf8_t, 32>), 32);
    ASSERT_EQ(sizeof(vector_type<bf8_t, 64>), 64);
    ASSERT_EQ(sizeof(vector_type<custom_bf8_t, 64>), 64);
}

TEST(Custom_half, Test)
{
    struct custom_half_t
    {
        half_t data;
    };
    ASSERT_EQ(sizeof(half_t), 2);
    ASSERT_EQ(sizeof(custom_half_t), 2);
    ASSERT_EQ(sizeof(vector_type<half_t, 2>), 4);
    ASSERT_EQ(sizeof(vector_type<custom_half_t, 2>), 4);
    ASSERT_EQ(sizeof(vector_type<half_t, 4>), 8);
    ASSERT_EQ(sizeof(vector_type<custom_half_t, 4>), 8);
    ASSERT_EQ(sizeof(vector_type<half_t, 8>), 16);
    ASSERT_EQ(sizeof(vector_type<custom_half_t, 8>), 16);
    ASSERT_EQ(sizeof(vector_type<half_t, 16>), 32);
    ASSERT_EQ(sizeof(vector_type<custom_half_t, 16>), 32);
    ASSERT_EQ(sizeof(vector_type<half_t, 32>), 64);
    ASSERT_EQ(sizeof(vector_type<custom_half_t, 32>), 64);
    ASSERT_EQ(sizeof(vector_type<half_t, 64>), 128);
    ASSERT_EQ(sizeof(vector_type<custom_half_t, 64>), 128);
}

TEST(Custom_bhalf, Test)
{
    struct custom_bhalf_t
    {
        bhalf_t data;
    };
    ASSERT_EQ(sizeof(bhalf_t), 2);
    ASSERT_EQ(sizeof(custom_bhalf_t), 2);
    ASSERT_EQ(sizeof(vector_type<bhalf_t, 2>), 4);
    ASSERT_EQ(sizeof(vector_type<custom_bhalf_t, 2>), 4);
    ASSERT_EQ(sizeof(vector_type<bhalf_t, 4>), 8);
    ASSERT_EQ(sizeof(vector_type<custom_bhalf_t, 4>), 8);
    ASSERT_EQ(sizeof(vector_type<bhalf_t, 8>), 16);
    ASSERT_EQ(sizeof(vector_type<custom_bhalf_t, 8>), 16);
    ASSERT_EQ(sizeof(vector_type<bhalf_t, 16>), 32);
    ASSERT_EQ(sizeof(vector_type<custom_bhalf_t, 16>), 32);
    ASSERT_EQ(sizeof(vector_type<bhalf_t, 32>), 64);
    ASSERT_EQ(sizeof(vector_type<custom_bhalf_t, 32>), 64);
    ASSERT_EQ(sizeof(vector_type<bhalf_t, 64>), 128);
    ASSERT_EQ(sizeof(vector_type<custom_bhalf_t, 64>), 128);
}

TEST(Custom_float, Test)
{
    struct custom_float_t
    {
        float data;
    };
    ASSERT_EQ(sizeof(float), 4);
    ASSERT_EQ(sizeof(custom_float_t), 4);
    ASSERT_EQ(sizeof(vector_type<float, 2>), 8);
    ASSERT_EQ(sizeof(vector_type<custom_float_t, 2>), 8);
    ASSERT_EQ(sizeof(vector_type<float, 4>), 16);
    ASSERT_EQ(sizeof(vector_type<custom_float_t, 4>), 16);
    ASSERT_EQ(sizeof(vector_type<float, 8>), 32);
    ASSERT_EQ(sizeof(vector_type<custom_float_t, 8>), 32);
    ASSERT_EQ(sizeof(vector_type<float, 16>), 64);
    ASSERT_EQ(sizeof(vector_type<custom_float_t, 16>), 64);
    ASSERT_EQ(sizeof(vector_type<float, 32>), 128);
    ASSERT_EQ(sizeof(vector_type<custom_float_t, 32>), 128);
    ASSERT_EQ(sizeof(vector_type<float, 64>), 256);
    ASSERT_EQ(sizeof(vector_type<custom_float_t, 64>), 256);
}

TEST(Custom_double, Test)
{
    struct custom_double_t
    {
        double data;
    };
    ASSERT_EQ(sizeof(double), 8);
    ASSERT_EQ(sizeof(custom_double_t), 8);
    ASSERT_EQ(sizeof(vector_type<double, 2>), 16);
    ASSERT_EQ(sizeof(vector_type<custom_double_t, 2>), 16);
    ASSERT_EQ(sizeof(vector_type<double, 4>), 32);
    ASSERT_EQ(sizeof(vector_type<custom_double_t, 4>), 32);
    ASSERT_EQ(sizeof(vector_type<double, 8>), 64);
    ASSERT_EQ(sizeof(vector_type<custom_double_t, 8>), 64);
    ASSERT_EQ(sizeof(vector_type<double, 16>), 128);
    ASSERT_EQ(sizeof(vector_type<custom_double_t, 16>), 128);
    ASSERT_EQ(sizeof(vector_type<double, 32>), 256);
    ASSERT_EQ(sizeof(vector_type<custom_double_t, 32>), 256);
    ASSERT_EQ(sizeof(vector_type<double, 64>), 512);
    ASSERT_EQ(sizeof(vector_type<custom_double_t, 64>), 512);
}
