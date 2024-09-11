// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "profiler/profile_avg_pool2d_bwd_impl.hpp"
#include "test_pool_fwd_common.hpp"

template <typename T>
class AvgPool2dBWDTest : public ::testing::Test
{
    protected:
    using InDataType  = std::tuple_element_t<0, T>;
    using OutDataType = std::tuple_element_t<1, T>;

    static std::vector<PoolingParam> params;

    void Run()
    {
        for(auto param : this->params)
        {
            bool success =
                ck::profiler::profile_avg_pool2d_bwd_impl<InDataType, OutDataType, NHWC, NHWC>(
                    true,
                    1,
                    false,
                    false,
                    param.length_,
                    param.window_spatial_lengths_,
                    param.window_strides_,
                    param.window_dilations_,
                    param.input_left_pads_,
                    param.input_right_pads_);
            EXPECT_TRUE(success);
        }
    }
};

template <typename T>
std::vector<PoolingParam> AvgPool2dBWDTest<T>::params = {
    {{1, 1, 1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
    {{1, 1, 64, 64}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
    {{1, 5, 7, 7}, {2, 2}, {2, 2}, {1, 1}, {2, 2}, {0, 0}},
    {{1, 1, 8, 8}, {2, 2}, {2, 2}, {1, 1}, {2, 2}, {0, 0}},
    {{1, 1, 8, 8}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, {0, 0}},
    {{1, 2, 30, 30}, {2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0}}};

using Avg_Pool_2D_f32_types        = ::testing::Types<std::tuple<F32, F32>>;
using Avg_Pool_2D_int8_types       = ::testing::Types<std::tuple<I8, I8>>;
using Avg_Pool_2D_f16_types        = ::testing::Types<std::tuple<F16, F16>>;
using Avg_Pool_2D_bf16_float_types = ::testing::Types<std::tuple<BF16, BF16>>;
using Avg_Pool_2D_f8_bitint_types  = ::testing::Types<std::tuple<F8, F8>>;

template <typename TType>
class AvgPool2D_f32 : public AvgPool2dBWDTest<TType>
{
};

template <typename TType>
class AvgPool2D_int8 : public AvgPool2dBWDTest<TType>
{
};

template <typename TType>
class AvgPool2D_f16 : public AvgPool2dBWDTest<TType>
{
};

template <typename TType>
class AvgPool2D_bf16 : public AvgPool2dBWDTest<TType>
{
};

template <typename TType>
class AvgPool2D_f8 : public AvgPool2dBWDTest<TType>
{
};

TYPED_TEST_SUITE(AvgPool2D_f32, Avg_Pool_2D_f32_types);
TYPED_TEST_SUITE(AvgPool2D_int8, Avg_Pool_2D_int8_types);
TYPED_TEST_SUITE(AvgPool2D_f16, Avg_Pool_2D_f16_types);
TYPED_TEST_SUITE(AvgPool2D_bf16, Avg_Pool_2D_bf16_float_types);
TYPED_TEST_SUITE(AvgPool2D_f8, Avg_Pool_2D_f8_bitint_types);

TYPED_TEST(AvgPool2D_f32, AvgPool2DTest_f32)
{
    // trigger Run()
    this->Run();
}

TYPED_TEST(AvgPool2D_int8, AvgPool2DTest_int8)
{
    // trigger Run()
    this->Run();
}

TYPED_TEST(AvgPool2D_f16, AvgPool2DTest_f16)
{
    // trigger Run()
    this->Run();
}

TYPED_TEST(AvgPool2D_bf16, AvgPool2DTest_bf16)
{
    // trigger Run()
    this->Run();
}

TYPED_TEST(AvgPool2D_f8, AvgPool2DTest_f8)
{
    // trigger Run()
    this->Run();
}
