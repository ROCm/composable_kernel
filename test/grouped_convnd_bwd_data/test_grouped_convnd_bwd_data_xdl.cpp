// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "profiler/profile_grouped_conv_bwd_data_impl.hpp"

template <typename Tuple>
class TestGroupedConvndBwdDataXdl : public ::testing::Test
{
    protected:
    using DataType  = std::tuple_element_t<0, Tuple>;
    using OutLayout = std::tuple_element_t<1, Tuple>;
    using WeiLayout = std::tuple_element_t<2, Tuple>;
    using InLayout  = std::tuple_element_t<3, Tuple>;

    std::vector<ck::utils::conv::ConvParam> conv_params;
    std::vector<ck::index_t> split_ks{1, 2};

    template <ck::index_t NDimSpatial>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;
        for(auto split_k : split_ks)
        {
            for(auto& param : conv_params)
            {
                pass = pass && ck::profiler::profile_grouped_conv_bwd_data_impl<NDimSpatial,
                                                                                OutLayout,
                                                                                WeiLayout,
                                                                                InLayout,
                                                                                DataType,
                                                                                DataType,
                                                                                DataType>(
                                   true,  // do_verification
                                   1,     // init_method: integer value
                                   false, // do_log
                                   false, // time_kernel
                                   param,
                                   split_k);
            }
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;

using KernelTypes2d = ::testing::Types<std::tuple<float, GNHWK, GKYXC, GNHWC>,
                                       std::tuple<ck::half_t, GNHWK, GKYXC, GNHWC>,
                                       std::tuple<ck::bhalf_t, GNHWK, GKYXC, GNHWC>,
                                       std::tuple<float, NGKHW, GKYXC, NGCHW>,
                                       std::tuple<ck::half_t, NGKHW, GKYXC, NGCHW>,
                                       std::tuple<ck::bhalf_t, NGKHW, GKYXC, NGCHW>,
                                       std::tuple<float, NGKHW, GKCYX, NGCHW>,
                                       std::tuple<ck::half_t, NGKHW, GKCYX, NGCHW>,
                                       std::tuple<ck::bhalf_t, NGKHW, GKCYX, NGCHW>,
                                       std::tuple<float, NHWGK, GKYXC, NHWGC>,
                                       std::tuple<ck::half_t, NHWGK, GKYXC, NHWGC>,
                                       std::tuple<ck::bhalf_t, NHWGK, GKYXC, NHWGC>>;

using KernelTypes3d = ::testing::Types<std::tuple<float, GNDHWK, GKZYXC, GNDHWC>,
                                       std::tuple<ck::half_t, GNDHWK, GKZYXC, GNDHWC>,
                                       std::tuple<ck::bhalf_t, GNDHWK, GKZYXC, GNDHWC>,
                                       std::tuple<float, NGKDHW, GKZYXC, NGCDHW>,
                                       std::tuple<ck::half_t, NGKDHW, GKZYXC, NGCDHW>,
                                       std::tuple<ck::bhalf_t, NGKDHW, GKZYXC, NGCDHW>,
                                       std::tuple<float, NGKDHW, GKCZYX, NGCDHW>,
                                       std::tuple<ck::half_t, NGKDHW, GKCZYX, NGCDHW>,
                                       std::tuple<ck::bhalf_t, NGKDHW, GKCZYX, NGCDHW>,
                                       std::tuple<float, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<ck::half_t, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<ck::bhalf_t, NDHWGK, GKZYXC, NDHWGC>>;

template <typename Tuple>
class TestGroupedConvndBwdDataXdl2d : public TestGroupedConvndBwdDataXdl<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndBwdDataXdl3d : public TestGroupedConvndBwdDataXdl<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndBwdDataXdl2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndBwdDataXdl3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndBwdDataXdl2d, Test2D)
{
    this->conv_params.clear();

    // GroupedGemmGroupsNum = 4, ZTilde * YTilde * XTilde = 4, MaxGroupedGemmGroupsNum = 32
    this->conv_params.push_back(
        {2, 2, 2, 16, 16, {3, 3}, {28, 28}, {2, 2}, {1, 1}, {1, 1}, {1, 1}});
    // GroupedGemmGroupsNum = 9, ZTilde * YTilde * XTilde = 36, MaxGroupedGemmGroupsNum = 32
    this->conv_params.push_back(
        {2, 2, 2, 16, 16, {3, 3}, {28, 28}, {6, 6}, {1, 1}, {1, 1}, {1, 1}});
    // GroupedGemmGroupsNum = 36, ZTilde * YTilde * XTilde = 36, MaxGroupedGemmGroupsNum = 32
    this->conv_params.push_back(
        {2, 2, 2, 16, 16, {6, 6}, {28, 28}, {6, 6}, {1, 1}, {1, 1}, {1, 1}});
    // GroupedGemmGroupsNum = 32, ZTilde * YTilde * XTilde = 32, MaxGroupedGemmGroupsNum = 32
    this->conv_params.push_back(
        {2, 2, 2, 16, 16, {4, 8}, {28, 28}, {4, 8}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 2, 2, 192, 192, {3, 3}, {28, 28}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 2, 2, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 2, 2, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 2, 2, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 2, 2, 32, 32, {2, 2}, {12, 12}, {3, 3}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 2, 2, 32, 32, {2, 2}, {12, 12}, {2, 2}, {2, 2}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 6, 448, 896, {1, 1}, {118, 182}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back({2, 1, 1, 1, 32, {8, 8}, {16, 16}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 1, 64, 3, {8, 8}, {16, 16}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 1, 1, 1, {8, 8}, {16, 16}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->template Run<2>();
}

TYPED_TEST(TestGroupedConvndBwdDataXdl3d, Test3D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {3, 2, 2, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 2, 128, 256, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 2, 2, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 2, 32, 32, {1, 2, 2}, {1, 12, 12}, {1, 3, 3}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 2, 32, 32, {1, 2, 2}, {1, 12, 12}, {1, 2, 2}, {1, 2, 2}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 1, 32, {3, 3, 3}, {4, 16, 16}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 64, 3, {3, 3, 3}, {4, 16, 16}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 1, 1, {3, 3, 3}, {4, 16, 16}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->template Run<3>();
}
