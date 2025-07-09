// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

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
    using IndexType = ck::index_t;

    std::vector<ck::utils::conv::ConvParam> conv_params;

    template <ck::index_t NDimSpatial, ck::index_t SplitK = 1>
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
                               param,
                               SplitK);
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;

using KernelTypes1d = ::testing::Types<std::tuple<float, GNWC, GKXC, GNWK>,
                                       std::tuple<ck::half_t, GNWC, GKXC, GNWK>,
                                       std::tuple<ck::bhalf_t, GNWC, GKXC, GNWK>,
                                       std::tuple<int8_t, GNWC, GKXC, GNWK>>;

using KernelTypes2d = ::testing::Types<std::tuple<float, GNHWC, GKYXC, GNHWK>,
                                       std::tuple<ck::half_t, GNHWC, GKYXC, GNHWK>,
                                       std::tuple<ck::bhalf_t, GNHWC, GKYXC, GNHWK>,
                                       std::tuple<float, NHWGC, GKYXC, NHWGK>,
                                       std::tuple<ck::half_t, NHWGC, GKYXC, NHWGK>,
                                       std::tuple<ck::bhalf_t, NHWGC, GKYXC, NHWGK>,
                                       std::tuple<int8_t, NHWGC, GKYXC, NHWGK>,
                                       std::tuple<float, NGCHW, GKYXC, NGKHW>,
                                       std::tuple<ck::half_t, NGCHW, GKYXC, NGKHW>,
                                       std::tuple<ck::bhalf_t, NGCHW, GKYXC, NGKHW>,
                                       std::tuple<int8_t, NGCHW, GKYXC, NGKHW>,
                                       std::tuple<float, NGCHW, GKCYX, NGKHW>,
                                       std::tuple<ck::half_t, NGCHW, GKCYX, NGKHW>,
                                       std::tuple<ck::bhalf_t, NGCHW, GKCYX, NGKHW>>;

using KernelTypes3d = ::testing::Types<std::tuple<float, GNDHWC, GKZYXC, GNDHWK>,
                                       std::tuple<ck::half_t, GNDHWC, GKZYXC, GNDHWK>,
                                       std::tuple<ck::bhalf_t, GNDHWC, GKZYXC, GNDHWK>,
                                       std::tuple<int8_t, GNDHWC, GKZYXC, GNDHWK>,
                                       std::tuple<float, NDHWGC, GKZYXC, NDHWGK>,
                                       std::tuple<ck::half_t, NDHWGC, GKZYXC, NDHWGK>,
                                       std::tuple<ck::bhalf_t, NDHWGC, GKZYXC, NDHWGK>,
                                       std::tuple<float, NGCDHW, GKCZYX, NGKDHW>,
                                       std::tuple<ck::half_t, NGCDHW, GKCZYX, NGKDHW>,
                                       std::tuple<ck::bhalf_t, NGCDHW, GKCZYX, NGKDHW>>;

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

TYPED_TEST_SUITE(TestGroupedConvndFwd1d, KernelTypes1d);
TYPED_TEST_SUITE(TestGroupedConvndFwd2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndFwd3d, KernelTypes3d);

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

TYPED_TEST(TestGroupedConvndFwd1d, Test1DSplitK2)
{
    this->conv_params.clear();
    // Convert 3D shapes to 1D (take first spatial dimension)
    this->conv_params.push_back({1, 2, 1, 3, 128, {3}, {11}, {1}, {1}, {1}, {0}});
    this->conv_params.push_back({1, 1, 1, 512, 512, {3}, {5}, {1}, {1}, {1}, {0}});
    this->conv_params.push_back({1, 1, 1, 512, 512, {3}, {5}, {1}, {1}, {1}, {0}});
    this->conv_params.push_back({1, 2, 1, 3, 128, {3}, {11}, {1}, {1}, {1}, {0}});
    this->conv_params.push_back({1, 1, 1, 512, 512, {3}, {5}, {1}, {1}, {1}, {0}});
    this->template Run<1, 2>();
}

TYPED_TEST(TestGroupedConvndFwd1d, Test1DSplitK4)
{
    this->conv_params.clear();
    this->conv_params.push_back({1, 1, 1, 512, 512, {3}, {5}, {1}, {1}, {1}, {0}});
    this->conv_params.push_back({1, 1, 1, 512, 512, {3}, {7}, {1}, {1}, {1}, {0}});
    this->template Run<1, 4>();
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

TYPED_TEST(TestGroupedConvndFwd2d, Test2DSplitK2)
{
    this->conv_params.clear();
    // Convert 3D shapes to 2D (take first two spatial dimensions)
    this->conv_params.push_back(
        {2, 1, 1, 512, 512, {3, 3}, {7, 20}, {1, 1}, {1, 1}, {1, 1}, {0, 0}});
    this->conv_params.push_back(
        {2, 2, 1, 512, 512, {3, 3}, {7, 34}, {1, 1}, {1, 1}, {1, 1}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 1, 512, 512, {3, 3}, {7, 38}, {1, 1}, {1, 1}, {1, 1}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 1, 256, 256, {3, 3}, {11, 74}, {1, 1}, {1, 1}, {1, 1}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 1, 256, 512, {1, 1}, {9, 72}, {1, 1}, {1, 1}, {1, 1}, {0, 0}});
    this->template Run<2, 2>();
}

TYPED_TEST(TestGroupedConvndFwd2d, Test2DSplitK4)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {2, 1, 1, 256, 256, {3, 3}, {7, 130}, {1, 1}, {1, 1}, {1, 1}, {0, 0}});
    this->conv_params.push_back(
        {2, 2, 1, 512, 512, {3, 3}, {7, 66}, {1, 1}, {1, 1}, {1, 1}, {0, 0}});
    this->template Run<2, 4>();
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

TYPED_TEST(TestGroupedConvndFwd3d, Test3DSplitK2)
{
    this->conv_params.clear();
    // Use original 3D shapes
    this->conv_params.push_back(
        {3, 1, 1, 3, 128, {3, 3, 3}, {11, 258, 258}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 1, 512, 512, {3, 3, 3}, {5, 20, 18}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 512, 512, {3, 3, 3}, {7, 20, 34}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 256, 256, {3, 3, 3}, {7, 74, 66}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 256, 512, {3, 3, 3}, {7, 130, 66}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
    this->template Run<3, 2>();
}

TYPED_TEST(TestGroupedConvndFwd3d, Test3DSplitK4)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {3, 2, 1, 3, 128, {3, 3, 3}, {11, 146, 258}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 512, 512, {3, 3, 3}, {5, 66, 34}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
    this->template Run<3, 4>();
}
