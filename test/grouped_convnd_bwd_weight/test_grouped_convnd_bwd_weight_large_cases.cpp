// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "profiler/profile_grouped_conv_bwd_weight_impl.hpp"

using namespace ck::tensor_layout::convolution;

template <typename Tuple>
class TestGroupedConvndBwdWeight : public ::testing::Test
{
    protected:
    using DataType  = std::tuple_element_t<0, Tuple>;
    using InLayout  = std::tuple_element_t<1, Tuple>;
    using WeiLayout = std::tuple_element_t<2, Tuple>;
    using OutLayout = std::tuple_element_t<3, Tuple>;

    std::vector<ck::utils::conv::ConvParam> conv_params;
    std::vector<ck::index_t> split_ks{1, 2};

    bool skip_case(const ck::index_t split_k)
    {
        // 1d NWGC is only supported by DL kernel
        // DL kernel is only supported for split_k=1
        if constexpr(std::is_same_v<InLayout, NWGC> && std::is_same_v<OutLayout, NWGK>)
        {
            if(split_k != 1)
            {
                return true;
            }
        }

        return false;
    }

    template <ck::index_t NDimSpatial>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;

        for(auto split_k : split_ks)
        {
            for(size_t i = 0; i < conv_params.size(); i++)
            {
                auto& param = conv_params[i];
                if(!skip_case(split_k))
                {
                    const bool success =
                        ck::profiler::profile_grouped_conv_bwd_weight_impl<NDimSpatial,
                                                                           InLayout,
                                                                           WeiLayout,
                                                                           OutLayout,
                                                                           DataType,
                                                                           DataType,
                                                                           DataType>(
                            2,     // do_verification
                            2,     // init_method: integer value
                            false, // do_log
                            false, // time_kernel
                            param,
                            std::to_string(split_k),
                            -1);
                    pass = pass && success;
                    if(!success)
                        std::cout << "Case " << param << " failed!" << std::endl;
                }
            }
        }
        EXPECT_TRUE(pass);
    }
};

template <typename Tuple>
class TestGroupedConvndBwdWeight2d : public TestGroupedConvndBwdWeight<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndBwdWeight3d : public TestGroupedConvndBwdWeight<Tuple>
{
};

using KernelTypes2d = ::testing::Types<std::tuple<float, NHWGC, GKYXC, NHWGK>,
                                       std::tuple<ck::half_t, NHWGC, GKYXC, NHWGK>,
                                       std::tuple<ck::bhalf_t, NHWGC, GKYXC, NHWGK>>;

using KernelTypes3d = ::testing::Types<std::tuple<float, NDHWGC, GKZYXC, NDHWGK>,
                                       std::tuple<ck::half_t, NDHWGC, GKZYXC, NDHWGK>,
                                       std::tuple<ck::bhalf_t, NDHWGC, GKZYXC, NDHWGK>>;

TYPED_TEST_SUITE(TestGroupedConvndBwdWeight2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndBwdWeight3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndBwdWeight2d, Test2D)
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

TYPED_TEST(TestGroupedConvndBwdWeight3d, Test3D)
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
