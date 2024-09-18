// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "profiler/profile_pool2d_fwd_impl.hpp"
#include "test_pool_fwd_common.hpp"

template <typename Tuple>
class TestAvgPool2dFwd : public ::testing::Test
{
    protected:
    using InDataType      = std::tuple_element_t<0, Tuple>;
    using OutDataType     = std::tuple_element_t<1, Tuple>;
    using ComputeDataType = std::tuple_element_t<2, Tuple>;
    using IndexDataType   = std::tuple_element_t<3, Tuple>;

    static std::vector<PoolingParam> params;

    void Run()
    {
        for(auto param : params)
        {
            bool success =
                ck::profiler::profile_pool2d_fwd_impl<InDataType,
                                                      OutDataType,
                                                      ComputeDataType,
                                                      IndexDataType,
                                                      ck::tensor_layout::convolution::NHWC,
                                                      ck::tensor_layout::convolution::NHWC,
                                                      ck::ReduceTensorOp::AVG,
                                                      false,
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
std::vector<PoolingParam> TestAvgPool2dFwd<T>::params = {
    {{{1, 1, 1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
     {{2, 16, 64, 64}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
     {{2, 16, 64, 64}, {4, 4}, {4, 4}, {2, 2}, {0, 0}, {0, 0}},
     {{2, 32, 30, 30}, {2, 2}, {2, 2}, {1, 1}, {1, 1}, {1, 1}}}};

using AvgPool2D_F32_Types =
    ::testing::Types<std::tuple<F32, F32, F32, I32>, std::tuple<F32, F32, F32, I32>>;
using AvgPool2D_F16_Types =
    ::testing::Types<std::tuple<F16, F16, F32, I32>, std::tuple<F16, F16, F32, I32>>;
using AvgPool2D_BF16_Types =
    ::testing::Types<std::tuple<I8, I8, F32, I32>, std::tuple<BF16, BF16, F32, I32>>;
using AvgPool2D_I8_Types =
    ::testing::Types<std::tuple<I8, I8, F32, I32>, std::tuple<I8, I8, F32, I32>>;
using AvgPool2D_F8_Types =
    ::testing::Types<std::tuple<F8, F8, F32, I32>, std::tuple<F8, F8, F32, I32>>;

template <typename TType>
class AvgPool2D_F32 : public TestAvgPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP32)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_F32 tests because CK_ENABLE_FP32 is "
                            "not enabled";
        }
    }
};

template <typename TType>
class AvgPool2D_F16 : public TestAvgPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP16)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_F16 tests because CK_ENABLE_FP16 is "
                            "not enabled";
        }
    }
};

template <typename TType>
class AvgPool2D_BF16 : public TestAvgPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_BF16)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_BF16 tests because CK_ENABLE_BF16 is "
                            "not enabled";
        }
    }
};

template <typename TType>
class AvgPool2D_I8 : public TestAvgPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_INT8)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_I8 tests because CK_ENABLE_INT8 is "
                            "not enabled";
        }
    }
};

template <typename TType>
class AvgPool2D_F8 : public TestAvgPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP8)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_F8 tests because CK_ENABLE_FP8 is "
                            "not enabled";
        }
    }
};

TYPED_TEST_SUITE(AvgPool2D_F32, AvgPool2D_F32_Types);
TYPED_TEST_SUITE(AvgPool2D_F16, AvgPool2D_F16_Types);
TYPED_TEST_SUITE(AvgPool2D_BF16, AvgPool2D_BF16_Types);
TYPED_TEST_SUITE(AvgPool2D_I8, AvgPool2D_I8_Types);
TYPED_TEST_SUITE(AvgPool2D_F8, AvgPool2D_F8_Types);

TYPED_TEST(AvgPool2D_F32, AvgPool2D_I8_Test) { this->Run(); }
TYPED_TEST(AvgPool2D_F16, AvgPool2D_F16_Test) { this->Run(); }
TYPED_TEST(AvgPool2D_BF16, AvgPool2D_BF16_Test) { this->Run(); }
TYPED_TEST(AvgPool2D_I8, AvgPool2D_I8_Test) { this->Run(); }
TYPED_TEST(AvgPool2D_F8, AvgPool2D_F8_Test) { this->Run(); }
