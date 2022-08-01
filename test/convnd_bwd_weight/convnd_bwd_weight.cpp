// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "profiler/include/profile_conv_bwd_weight_impl.hpp"

class TestConvndBwdWeight : public ::testing::Test
{
    protected:
    std::vector<ck::utils::conv::ConvParam> conv_params;
};

// 1d
TEST_F(TestConvndBwdWeight, Conv1dBwdWeight)
{
    conv_params.clear();
    conv_params.push_back({1, 1, 128, 128, 256, {1}, {14}, {2}, {1}, {0}, {0}});
    conv_params.push_back({1, 1, 128, 128, 256, {3}, {28}, {1}, {1}, {1}, {1}});
    conv_params.push_back({1, 1, 128, 128, 256, {1}, {3}, {1}, {1}, {0}, {0}});

    for(auto& param : conv_params)
    {
        bool pass;

        // fp32
        pass = ck::profiler::profile_conv_bwd_weight_impl<1,
                                                          ck::tensor_layout::convolution::NWC,
                                                          ck::tensor_layout::convolution::KXC,
                                                          ck::tensor_layout::convolution::NWK,
                                                          float,
                                                          float,
                                                          float>(true,  // do_verification
                                                                 1,     // init_method
                                                                 false, // do_log
                                                                 false, // time_kernel
                                                                 param,
                                                                 2);

        EXPECT_TRUE(pass);

        // fp16
        pass = ck::profiler::profile_conv_bwd_weight_impl<1,
                                                          ck::tensor_layout::convolution::NWC,
                                                          ck::tensor_layout::convolution::KXC,
                                                          ck::tensor_layout::convolution::NWK,
                                                          ck::half_t,
                                                          ck::half_t,
                                                          ck::half_t>(true,  // do_verification
                                                                      1,     // init_method
                                                                      false, // do_log
                                                                      false, // time_kernel
                                                                      param,
                                                                      2);

        EXPECT_TRUE(pass);

        // bf16
        pass = ck::profiler::profile_conv_bwd_weight_impl<1,
                                                          ck::tensor_layout::convolution::NWC,
                                                          ck::tensor_layout::convolution::KXC,
                                                          ck::tensor_layout::convolution::NWK,
                                                          ck::bhalf_t,
                                                          ck::bhalf_t,
                                                          ck::bhalf_t>(true,  // do_verification
                                                                       1,     // init_method
                                                                       false, // do_log
                                                                       false, // time_kernel
                                                                       param,
                                                                       2);

        EXPECT_TRUE(pass);
    }
}

// 2d
TEST_F(TestConvndBwdWeight, Conv2dBwdWeight)
{
    conv_params.clear();
    conv_params.push_back({2, 1, 128, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    conv_params.push_back({2, 1, 32, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    conv_params.push_back({2, 1, 128, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});

    for(auto& param : conv_params)
    {
        bool pass;

        // fp32
        pass = ck::profiler::profile_conv_bwd_weight_impl<2,
                                                          ck::tensor_layout::convolution::NHWC,
                                                          ck::tensor_layout::convolution::KYXC,
                                                          ck::tensor_layout::convolution::NHWK,
                                                          float,
                                                          float,
                                                          float>(true,  // do_verification
                                                                 1,     // init_method
                                                                 false, // do_log
                                                                 false, // time_kernel
                                                                 param,
                                                                 2);

        EXPECT_TRUE(pass);

        // fp16
        pass = ck::profiler::profile_conv_bwd_weight_impl<2,
                                                          ck::tensor_layout::convolution::NHWC,
                                                          ck::tensor_layout::convolution::KYXC,
                                                          ck::tensor_layout::convolution::NHWK,
                                                          ck::half_t,
                                                          ck::half_t,
                                                          ck::half_t>(true,  // do_verification
                                                                      1,     // init_method
                                                                      false, // do_log
                                                                      false, // time_kernel
                                                                      param,
                                                                      2);

        EXPECT_TRUE(pass);

        // bf16
        pass = ck::profiler::profile_conv_bwd_weight_impl<2,
                                                          ck::tensor_layout::convolution::NHWC,
                                                          ck::tensor_layout::convolution::KYXC,
                                                          ck::tensor_layout::convolution::NHWK,
                                                          ck::bhalf_t,
                                                          ck::bhalf_t,
                                                          ck::bhalf_t>(true,  // do_verification
                                                                       1,     // init_method
                                                                       false, // do_log
                                                                       false, // time_kernel
                                                                       param,
                                                                       2);

        EXPECT_TRUE(pass);
    }
}

// 3d
TEST_F(TestConvndBwdWeight, Conv3dBwdWeight)
{
    conv_params.clear();
    conv_params.push_back(
        {3, 1, 128, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    conv_params.push_back(
        {3, 1, 32, 128, 256, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    conv_params.push_back(
        {3, 1, 128, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});

    for(auto& param : conv_params)
    {
        bool pass;

        // fp32
        pass = ck::profiler::profile_conv_bwd_weight_impl<3,
                                                          ck::tensor_layout::convolution::NDHWC,
                                                          ck::tensor_layout::convolution::KZYXC,
                                                          ck::tensor_layout::convolution::NDHWK,
                                                          float,
                                                          float,
                                                          float>(true,  // do_verification
                                                                 1,     // init_method
                                                                 false, // do_log
                                                                 false, // time_kernel
                                                                 param,
                                                                 2);

        EXPECT_TRUE(pass);

        // fp16
        pass = ck::profiler::profile_conv_bwd_weight_impl<3,
                                                          ck::tensor_layout::convolution::NDHWC,
                                                          ck::tensor_layout::convolution::KZYXC,
                                                          ck::tensor_layout::convolution::NDHWK,
                                                          ck::half_t,
                                                          ck::half_t,
                                                          ck::half_t>(true,  // do_verification
                                                                      1,     // init_method
                                                                      false, // do_log
                                                                      false, // time_kernel
                                                                      param,
                                                                      2);

        EXPECT_TRUE(pass);

        // bf16
        pass = ck::profiler::profile_conv_bwd_weight_impl<3,
                                                          ck::tensor_layout::convolution::NDHWC,
                                                          ck::tensor_layout::convolution::KZYXC,
                                                          ck::tensor_layout::convolution::NDHWK,
                                                          ck::bhalf_t,
                                                          ck::bhalf_t,
                                                          ck::bhalf_t>(true,  // do_verification
                                                                       1,     // init_method
                                                                       false, // do_log
                                                                       false, // time_kernel
                                                                       param,
                                                                       2);

        EXPECT_TRUE(pass);
    }
}
