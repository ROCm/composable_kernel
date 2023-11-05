// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "profiler/profile_image_to_column_impl.hpp"

template <typename Tuple>
class TestImageToColumn : public ::testing::Test
{
    protected:
    using InDataType  = std::tuple_element_t<0, Tuple>;
    using OutDataType = std::tuple_element_t<1, Tuple>;
    using InLayout    = std::tuple_element_t<2, Tuple>;

    std::vector<ck::utils::conv::ConvParam> conv_params;

    template <ck::index_t NDimSpatial>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;
        for(auto& param : conv_params)
        {
            pass = pass && ck::profiler::profile_image_to_column_impl<NDimSpatial,
                                                                      InLayout,
                                                                      InDataType,
                                                                      OutDataType>(
                               true,  // do_verification
                               1,     // init_method: integer value
                               false, // do_log
                               false, // time_kernel
                               param);
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;

using KernelTypes1d = ::testing::Types<std::tuple<float, float, GNWC>,
                                       std::tuple<ck::bhalf_t, ck::bhalf_t, GNWC>,
                                       std::tuple<ck::half_t, ck::half_t, GNWC>,
                                       std::tuple<int8_t, int8_t, GNWC>>;

using KernelTypes2d = ::testing::Types<std::tuple<float, float, GNHWC>,
                                       std::tuple<ck::bhalf_t, ck::bhalf_t, GNHWC>,
                                       std::tuple<ck::half_t, ck::half_t, GNHWC>,
                                       std::tuple<int8_t, int8_t, GNHWC>>;

using KernelTypes3d = ::testing::Types<std::tuple<float, float, GNDHWC>,
                                       std::tuple<ck::bhalf_t, ck::bhalf_t, GNDHWC>,
                                       std::tuple<ck::half_t, ck::half_t, GNDHWC>,
                                       std::tuple<int8_t, int8_t, GNDHWC>>;

template <typename Tuple>
class TestImageToColumn1d : public TestImageToColumn<Tuple>
{
};

template <typename Tuple>
class TestImageToColumn2d : public TestImageToColumn<Tuple>
{
};

template <typename Tuple>
class TestImageToColumn3d : public TestImageToColumn<Tuple>
{
};

TYPED_TEST_SUITE(TestImageToColumn1d, KernelTypes1d);
TYPED_TEST_SUITE(TestImageToColumn2d, KernelTypes2d);
TYPED_TEST_SUITE(TestImageToColumn3d, KernelTypes3d);

TYPED_TEST(TestImageToColumn1d, Test1D)
{
    this->conv_params.clear();

    this->conv_params.push_back({1, 1, 4, 1, 192, {3}, {28}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 1, 64, 1, 64, {3}, {14}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 1, 64, 1, 64, {1}, {7}, {2}, {1}, {0}, {0}});
    this->conv_params.push_back({1, 1, 64, 1, 64, {1}, {3}, {1}, {1}, {0}, {0}});
    // ScalarPerVector should be 1
    this->conv_params.push_back({1, 1, 4, 1, 1, {3}, {28}, {1}, {1}, {1}, {1}});
    // stride != 1
    this->conv_params.push_back({1, 1, 1, 1, 4, {3}, {28}, {2}, {1}, {1}, {1}});
    // dilation != 1
    this->conv_params.push_back({1, 1, 1, 1, 4, {3}, {28}, {1}, {2}, {1}, {1}});
    this->template Run<1>();
}

TYPED_TEST(TestImageToColumn2d, Test2D)
{
    this->conv_params.clear();

    this->conv_params.push_back(
        {2, 1, 4, 1, 192, {3, 3}, {28, 28}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 1, 64, 1, 64, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 64, 1, 64, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back({2, 1, 64, 1, 64, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->template Run<2>();
}

TYPED_TEST(TestImageToColumn3d, Test3D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {3, 1, 16, 1, 64, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 2, 1, 64, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 32, 1, 64, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->template Run<3>();
}
