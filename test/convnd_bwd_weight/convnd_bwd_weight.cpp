// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <vector>

#include "test/convnd_fwd/conv_util.hpp"
#include "profiler/include/profile_convnd_bwd_weight_impl.hpp"

int test_self()
{
    bool pass = true;
    std::vector<ck::utils::conv::ConvParams> params;

    params.push_back({1, 128, 256, 256, {1}, {7}, {2}, {1}, {0}, {0}});
    params.push_back({1, 128, 256, 256, {3}, {14}, {1}, {1}, {1}, {1}});
    params.push_back({1, 128, 256, 256, {1}, {3}, {1}, {1}, {0}, {0}});

    for(auto& param : params)
    {
        // f32
        pass &= ck::profiler::profile_convnd_bwd_weight_impl<1,
                                                             float,
                                                             float,
                                                             float,
                                                             ck::tensor_layout::convolution::NWC,
                                                             ck::tensor_layout::convolution::KXC,
                                                             ck::tensor_layout::convolution::NWK>(
            true,  // do_verification
            1,     // init_method
            false, // do_log
            true,  // time_kernel
            param.N_,
            param.K_,
            param.C_,
            param.input_spatial_lengths_,
            param.filter_spatial_lengths_,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides_,
            param.conv_filter_dilations_,
            param.input_left_pads_,
            param.input_right_pads_,
            2);

        // fp16
        pass &= ck::profiler::profile_convnd_bwd_weight_impl<1,
                                                             ck::half_t,
                                                             ck::half_t,
                                                             ck::half_t,
                                                             ck::tensor_layout::convolution::NWC,
                                                             ck::tensor_layout::convolution::KXC,
                                                             ck::tensor_layout::convolution::NWK>(
            true,  // do_verification
            1,     // init_method
            false, // do_log
            true,  // time_kernel
            param.N_,
            param.K_,
            param.C_,
            param.input_spatial_lengths_,
            param.filter_spatial_lengths_,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides_,
            param.conv_filter_dilations_,
            param.input_left_pads_,
            param.input_right_pads_,
            2);

        // bf16
        pass &= ck::profiler::profile_convnd_bwd_weight_impl<1,
                                                             ck::bhalf_t,
                                                             ck::bhalf_t,
                                                             ck::bhalf_t,
                                                             ck::tensor_layout::convolution::NWC,
                                                             ck::tensor_layout::convolution::KXC,
                                                             ck::tensor_layout::convolution::NWK>(
            true,  // do_verification
            1,     // init_method
            false, // do_log
            true,  // time_kernel
            param.N_,
            param.K_,
            param.C_,
            param.input_spatial_lengths_,
            param.filter_spatial_lengths_,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides_,
            param.conv_filter_dilations_,
            param.input_left_pads_,
            param.input_right_pads_,
            2);
    }

    // check 2d
    params.clear();
    params.push_back({2, 128, 256, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    params.push_back({2, 128, 256, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    params.push_back({2, 128, 256, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});

    for(auto& param : params)
    {
        // f32
        pass &= ck::profiler::profile_convnd_bwd_weight_impl<2,
                                                             float,
                                                             float,
                                                             float,
                                                             ck::tensor_layout::convolution::NHWC,
                                                             ck::tensor_layout::convolution::KYXC,
                                                             ck::tensor_layout::convolution::NHWK>(
            true,  // do_verification
            1,     // init_method
            false, // do_log
            true,  // time_kernel
            param.N_,
            param.K_,
            param.C_,
            param.input_spatial_lengths_,
            param.filter_spatial_lengths_,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides_,
            param.conv_filter_dilations_,
            param.input_left_pads_,
            param.input_right_pads_,
            2);

        // fp16
        pass &= ck::profiler::profile_convnd_bwd_weight_impl<2,
                                                             ck::half_t,
                                                             ck::half_t,
                                                             ck::half_t,
                                                             ck::tensor_layout::convolution::NHWC,
                                                             ck::tensor_layout::convolution::KYXC,
                                                             ck::tensor_layout::convolution::NHWK>(
            true,  // do_verification
            1,     // init_method
            false, // do_log
            true,  // time_kernel
            param.N_,
            param.K_,
            param.C_,
            param.input_spatial_lengths_,
            param.filter_spatial_lengths_,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides_,
            param.conv_filter_dilations_,
            param.input_left_pads_,
            param.input_right_pads_,
            2);

        // bf16
        pass &= ck::profiler::profile_convnd_bwd_weight_impl<2,
                                                             ck::bhalf_t,
                                                             ck::bhalf_t,
                                                             ck::bhalf_t,
                                                             ck::tensor_layout::convolution::NHWC,
                                                             ck::tensor_layout::convolution::KYXC,
                                                             ck::tensor_layout::convolution::NHWK>(
            true,  // do_verification
            1,     // init_method
            false, // do_log
            true,  // time_kernel
            param.N_,
            param.K_,
            param.C_,
            param.input_spatial_lengths_,
            param.filter_spatial_lengths_,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides_,
            param.conv_filter_dilations_,
            param.input_left_pads_,
            param.input_right_pads_,
            2);
    }

    // check 2d
    params.clear();
    params.push_back(
        {3, 128, 256, 256, {1, 1, 1}, {4, 4, 4}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    params.push_back(
        {3, 128, 256, 256, {3, 3, 3}, {4, 4, 8}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    params.push_back(
        {3, 128, 256, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});

    for(auto& param : params)
    {
        // f32
        pass &= ck::profiler::profile_convnd_bwd_weight_impl<3,
                                                             float,
                                                             float,
                                                             float,
                                                             ck::tensor_layout::convolution::NDHWC,
                                                             ck::tensor_layout::convolution::KZYXC,
                                                             ck::tensor_layout::convolution::NDHWK>(
            true,  // do_verification
            1,     // init_method
            false, // do_log
            true,  // time_kernel
            param.N_,
            param.K_,
            param.C_,
            param.input_spatial_lengths_,
            param.filter_spatial_lengths_,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides_,
            param.conv_filter_dilations_,
            param.input_left_pads_,
            param.input_right_pads_,
            2);

        // fp16
        pass &= ck::profiler::profile_convnd_bwd_weight_impl<3,
                                                             ck::half_t,
                                                             ck::half_t,
                                                             ck::half_t,
                                                             ck::tensor_layout::convolution::NDHWC,
                                                             ck::tensor_layout::convolution::KZYXC,
                                                             ck::tensor_layout::convolution::NDHWK>(
            true,  // do_verification
            1,     // init_method
            false, // do_log
            true,  // time_kernel
            param.N_,
            param.K_,
            param.C_,
            param.input_spatial_lengths_,
            param.filter_spatial_lengths_,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides_,
            param.conv_filter_dilations_,
            param.input_left_pads_,
            param.input_right_pads_,
            2);

        // bf16
        pass &= ck::profiler::profile_convnd_bwd_weight_impl<3,
                                                             ck::bhalf_t,
                                                             ck::bhalf_t,
                                                             ck::bhalf_t,
                                                             ck::tensor_layout::convolution::NDHWC,
                                                             ck::tensor_layout::convolution::KZYXC,
                                                             ck::tensor_layout::convolution::NDHWK>(
            true,  // do_verification
            1,     // init_method
            false, // do_log
            true,  // time_kernel
            param.N_,
            param.K_,
            param.C_,
            param.input_spatial_lengths_,
            param.filter_spatial_lengths_,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides_,
            param.conv_filter_dilations_,
            param.input_left_pads_,
            param.input_right_pads_,
            2);
    }

    return pass;
}
int main()
{
    // int data_type   = 1;
    // int init_method = 1;

    bool pass = true;

    pass = test_self();

    if(pass)
    {
        std::cout << "test conv2d bwd weight : Pass" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "test conv2d bwd weight: Fail " << std::endl;
        return -1;
    }
}
