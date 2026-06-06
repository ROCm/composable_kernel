// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "profiler/profile_grouped_conv_bwd_data_impl.hpp"

template <typename Tuple>
class TestGroupedConvndBwdData : public ::testing::Test
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
                                   2,     // do_verification
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

using KernelTypes2d = ::testing::Types<std::tuple<float, NHWGK, GKYXC, NHWGC>,
                                       std::tuple<ck::half_t, NHWGK, GKYXC, NHWGC>,
                                       std::tuple<ck::bhalf_t, NHWGK, GKYXC, NHWGC>>;

using KernelTypes3d = ::testing::Types<std::tuple<float, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<ck::half_t, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<ck::bhalf_t, NDHWGK, GKZYXC, NDHWGC>>;

template <typename Tuple>
class TestGroupedConvndBwdData2d : public TestGroupedConvndBwdData<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndBwdData3d : public TestGroupedConvndBwdData<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndBwdData2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndBwdData3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndBwdData2d, Test2D)
{
    this->conv_params.clear();
    // Case larger than 2GB
    this->conv_params.push_back(
        {2, 1, 128, 4, 192, {2, 2}, {224, 224}, {224, 224}, {1, 1}, {0, 0}, {0, 0}});
    // With supported NumGroupsToMerge > 1
    this->conv_params.push_back(
        {2, 32, 64, 1, 1, {2, 2}, {672, 672}, {672, 672}, {1, 1}, {0, 0}, {0, 0}});
    // When image is larger than 2GB
    this->conv_params.push_back(
        {2, 2, 2, 128, 128, {3, 3}, {4096, 2048}, {300, 300}, {3, 3}, {1, 1}, {1, 1}});
    // Split N and G > 1
    this->conv_params.push_back(
        {2, 4, 112, 8, 8, {3, 3}, {469, 724}, {2, 2}, {2, 2}, {1, 1}, {1, 1}});
    this->template Run<2>();
}

TYPED_TEST(TestGroupedConvndBwdData3d, Test3D)
{
    this->conv_params.clear();
    // Case larger than 2GB
    this->conv_params.push_back({3,
                                 1,
                                 128,
                                 4,
                                 192,
                                 {2, 2, 2},
                                 {2, 224, 224},
                                 {1, 224, 224},
                                 {1, 1, 1},
                                 {0, 0, 0},
                                 {0, 0, 0}});
    // With supported NumGroupsToMerge > 1
    this->conv_params.push_back({3,
                                 32,
                                 64,
                                 1,
                                 1,
                                 {2, 2, 2},
                                 {360, 2, 672},
                                 {360, 2, 672},
                                 {1, 1, 1},
                                 {0, 0, 0},
                                 {0, 0, 0}});
    // When image is larger than 2GB
    this->conv_params.push_back({3,
                                 1,
                                 2,
                                 128,
                                 128,
                                 {3, 1, 3},
                                 {900, 2, 2048},
                                 {300, 1, 300},
                                 {3, 2, 3},
                                 {1, 1, 1},
                                 {1, 1, 1}});
    this->template Run<3>();
}
