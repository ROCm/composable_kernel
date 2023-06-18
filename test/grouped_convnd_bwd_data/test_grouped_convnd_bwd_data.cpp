// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

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

    template <ck::index_t NDimSpatial>
    void Run()
    {
        for(auto& param : conv_params)
        {
            bool pass;
            EXPECT_FALSE(conv_params.empty());
            pass = ck::profiler::profile_grouped_conv_bwd_data_impl<NDimSpatial,
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
                param);
            EXPECT_TRUE(pass);
        }
    }
};

using GNHWC = ck::tensor_layout::convolution::GNHWC;
using NHWGC = ck::tensor_layout::convolution::NHWGC;

using GKYXC = ck::tensor_layout::convolution::GKYXC;

using GNHWK = ck::tensor_layout::convolution::GNHWK;
using NHWGK = ck::tensor_layout::convolution::NHWGK;

using KernelTypes = ::testing::Types<std::tuple<float, GNHWK, GKYXC, GNHWC>,
                                     std::tuple<ck::half_t, GNHWK, GKYXC, GNHWC>,
                                     std::tuple<ck::bhalf_t, GNHWK, GKYXC, GNHWC>,
                                     std::tuple<float, NHWGK, GKYXC, NHWGC>,
                                     std::tuple<ck::half_t, NHWGK, GKYXC, NHWGC>,
                                     std::tuple<ck::bhalf_t, NHWGK, GKYXC, NHWGC>>;
TYPED_TEST_SUITE(TestGroupedConvndBwdData, KernelTypes);

TYPED_TEST(TestGroupedConvndBwdData, Test2D)
{
    this->conv_params.clear();

    this->conv_params.push_back(
        {2, 2, 4, 192, 192, {3, 3}, {28, 28}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 2, 128, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 2, 128, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 2, 128, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->template Run<2>();
}
