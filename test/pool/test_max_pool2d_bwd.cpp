// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "profiler/profile_max_pool2d_bwd_impl.hpp"
#include "test_pool_fwd_common.hpp"

template <typename T>
class MaxPool2dBWDTest : public ::testing::Test
{
    protected:
    using DOutDataType  = std::tuple_element_t<0, T>;
    using DInDataType   = std::tuple_element_t<1, T>;
    using IndexDataType = std::tuple_element_t<2, T>;

    using InDataType  = DInDataType;
    using OutDataType = DOutDataType;

    static std::vector<PoolingParam> params;

    void Run()
    {
        for(auto param : this->params)
        {
            bool success =
                ck::profiler::profile_max_pool2d_bwd_impl<InDataType,
                                                          OutDataType,
                                                          IndexDataType,
                                                          DOutDataType,
                                                          DInDataType,
                                                          false>(true,
                                                                 2,
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
std::vector<PoolingParam> MaxPool2dBWDTest<T>::params = {
    {{1, 1, 1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
    {{2, 16, 64, 64}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
    {{2, 16, 64, 64}, {4, 4}, {4, 4}, {2, 2}, {0, 0}, {0, 0}},
    {{2, 2, 30, 30}, {2, 2}, {2, 2}, {1, 1}, {1, 1}, {1, 1}}};

using Max_Pool_2D_f32_types  = ::testing::Types<std::tuple<F32, F32, I32>>;
using Max_Pool_2D_int8_types = ::testing::Types<std::tuple<I8, I8, I32>>;
using Max_Pool_2D_f16_types  = ::testing::Types<std::tuple<F16, F16, I32>>;
using Max_Pool_2D_bf16_types = ::testing::Types<std::tuple<BF16, BF16, I32>>;

template <typename TType>
class MaxPool2D_f32 : public MaxPool2dBWDTest<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP32)
        {
            GTEST_SKIP() << "Skipping MaxPool2D_f32 tests because CK_ENABLE_FP32 is not enabled";
        }
    }
};

template <typename TType>
class MaxPool2D_int8 : public MaxPool2dBWDTest<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_INT8)
        {
            GTEST_SKIP() << "Skipping MaxPool2D_int8 tests because CK_ENABLE_INT8 is not enabled";
        }
    }
};

template <typename TType>
class MaxPool2D_f16 : public MaxPool2dBWDTest<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP16)
        {
            GTEST_SKIP() << "Skipping MaxPool2D_f16 because CK_ENABLE_FP16 is not enabled";
        }
    }
};

template <typename TType>
class MaxPool2D_bf16 : public MaxPool2dBWDTest<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_BF16)
        {
            GTEST_SKIP() << "Skipping MaxPool2D_bf16 tests because CK_ENABLE_BF16 is not enabled";
        }
    }
};

TYPED_TEST_SUITE(MaxPool2D_f32, Max_Pool_2D_f32_types);
TYPED_TEST_SUITE(MaxPool2D_int8, Max_Pool_2D_int8_types);
TYPED_TEST_SUITE(MaxPool2D_f16, Max_Pool_2D_f16_types);
TYPED_TEST_SUITE(MaxPool2D_bf16, Max_Pool_2D_bf16_types);

TYPED_TEST(MaxPool2D_f32, MaxPool2DTest_f32)
{
    // trigger Run()
    this->Run();
}

TYPED_TEST(MaxPool2D_int8, MaxPool2DTest_int8)
{
    // trigger Run()
    this->Run();
}

TYPED_TEST(MaxPool2D_f16, MaxPool2DTest_f16)
{
    // trigger Run()
    this->Run();
}

TYPED_TEST(MaxPool2D_bf16, MaxPool2DTest_bf16)
{
    // trigger Run()
    this->Run();
}
