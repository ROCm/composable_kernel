// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "profiler/profile_grouped_conv_fwd_impl.hpp"

template <typename Tuple>
class TestGroupedConvndFwd : public ::testing::Test
{
    protected:
    using DataType  = std::tuple_element_t<0, Tuple>;
    using InLayout  = std::tuple_element_t<1, Tuple>;
    using WeiLayout = std::tuple_element_t<2, Tuple>;
    using OutLayout = std::tuple_element_t<3, Tuple>;
    using IndexType = std::tuple_element_t<4, Tuple>;

    std::vector<ck::utils::conv::ConvParam> conv_params;

    template <ck::index_t NDimSpatial>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;
        for(auto& param : conv_params)
        {
            pass = pass && ck::profiler::profile_grouped_conv_fwd_impl<NDimSpatial,
                                                                       InLayout,
                                                                       WeiLayout,
                                                                       OutLayout,
                                                                       DataType,
                                                                       DataType,
                                                                       DataType,
                                                                       DataType,
                                                                       DataType,
                                                                       IndexType>(
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

using KernelTypes1d = ::testing::Types<std::tuple<float, GNWC, GKXC, GNWK, ck::index_t>,
                                       std::tuple<ck::half_t, GNWC, GKXC, GNWK, ck::index_t>,
                                       std::tuple<ck::bhalf_t, GNWC, GKXC, GNWK, ck::index_t>,
                                       std::tuple<int8_t, GNWC, GKXC, GNWK, ck::index_t>>;

using KernelTypes2d = ::testing::Types<std::tuple<float, GNHWC, GKYXC, GNHWK, ck::index_t>,
                                       std::tuple<ck::half_t, GNHWC, GKYXC, GNHWK, ck::index_t>,
                                       std::tuple<ck::bhalf_t, GNHWC, GKYXC, GNHWK, ck::index_t>,
                                       std::tuple<int8_t, GNHWC, GKYXC, GNHWK, ck::index_t>,
                                       std::tuple<float, NHWGC, GKYXC, NHWGK, ck::index_t>,
                                       std::tuple<ck::half_t, NHWGC, GKYXC, NHWGK, ck::index_t>,
                                       std::tuple<ck::bhalf_t, NHWGC, GKYXC, NHWGK, ck::index_t>,
                                       std::tuple<int8_t, NHWGC, GKYXC, NHWGK, ck::index_t>>;

using KernelTypes3d = ::testing::Types<std::tuple<float, GNDHWC, GKZYXC, GNDHWK, ck::index_t>,
                                       std::tuple<ck::half_t, GNDHWC, GKZYXC, GNDHWK, ck::index_t>,
                                       std::tuple<ck::bhalf_t, GNDHWC, GKZYXC, GNDHWK, ck::index_t>,
                                       std::tuple<int8_t, GNDHWC, GKZYXC, GNDHWK, ck::index_t>,
                                       std::tuple<float, NDHWGC, GKZYXC, NDHWGK, ck::index_t>,
                                       std::tuple<ck::half_t, NDHWGC, GKZYXC, NDHWGK, ck::index_t>,
                                       std::tuple<ck::bhalf_t, NDHWGC, GKZYXC, NDHWGK, ck::index_t>,
                                       std::tuple<int8_t, NDHWGC, GKZYXC, NDHWGK, ck::index_t>>;

using KernelTypes2dLargeCases =
    ::testing::Types<std::tuple<float, NHWGC, GKYXC, NHWGK, ck::long_index_t>>;

template <typename Tuple>
class TestGroupedConvndFwd1d : public TestGroupedConvndFwd<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndFwd2d : public TestGroupedConvndFwd<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndFwd3d : public TestGroupedConvndFwd<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndFwd2dLargeCases : public TestGroupedConvndFwd<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndFwd1d, KernelTypes1d);
TYPED_TEST_SUITE(TestGroupedConvndFwd2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndFwd3d, KernelTypes3d);
TYPED_TEST_SUITE(TestGroupedConvndFwd2dLargeCases, KernelTypes2dLargeCases);

TYPED_TEST(TestGroupedConvndFwd1d, Test1D)
{
    this->conv_params.clear();
    this->conv_params.push_back({1, 2, 32, 128, 256, {1}, {14}, {2}, {1}, {0}, {0}});
    this->conv_params.push_back({1, 2, 32, 128, 256, {3}, {28}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 2, 32, 128, 256, {1}, {3}, {1}, {1}, {0}, {0}});
    this->conv_params.push_back({1, 1, 1, 1, 32, {3}, {32}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 1, 1, 64, 3, {3}, {32}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 96, 1, 1, 1, {3}, {512}, {1}, {1}, {1}, {1}});
    this->template Run<1>();
}

TYPED_TEST(TestGroupedConvndFwd2d, Test2D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {2, 2, 32, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 2, 32, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 2, 32, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back({2, 1, 1, 1, 32, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 1, 64, 3, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 1, 1, 1, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 96, 1, 1, 1, {3, 3}, {120, 160}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->template Run<2>();
}

TYPED_TEST(TestGroupedConvndFwd3d, Test3D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 1, 32, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 64, 3, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 1, 1, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 96, 1, 1, 1, {3, 3, 3}, {4, 30, 160}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->template Run<3>();
}

TYPED_TEST(TestGroupedConvndFwd2dLargeCases, Test2DLargeCases)
{
    // Case larger than 2GB
    this->conv_params.push_back(
        {2, 1, 64, 4, 192, {2, 2}, {224, 224}, {224, 224}, {1, 1}, {0, 0}, {0, 0}});
    // With supported NumGroupsToMerge > 1
    this->conv_params.push_back(
        {2, 32, 64, 1, 1, {2, 2}, {672, 672}, {672, 672}, {1, 1}, {0, 0}, {0, 0}});
    this->template Run<2>();
}
