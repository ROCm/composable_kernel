#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include <vector>

#include "profile_convnd_bwd_data_impl.hpp"

int main()
{
    bool pass = true;
    // check 1d
    std::vector<ck::utils::conv::ConvParams> params;
    params.push_back({1, 128, 128, 256, {1}, {14}, {2}, {1}, {0}, {0}});
    params.push_back({1, 128, 128, 256, {3}, {28}, {1}, {1}, {1}, {1}});
    params.push_back({1, 128, 128, 256, {1}, {3}, {1}, {1}, {0}, {0}});

    for(auto& param : params)
    {
        pass &= ck::profiler::profile_convnd_bwd_data_impl<1,
                                                           float,
                                                           float,
                                                           float,
                                                           float,
                                                           ck::tensor_layout::convolution::NWC,
                                                           ck::tensor_layout::convolution::KXC,
                                                           ck::tensor_layout::convolution::NWK>(
            1, // do_verification,
            1, // init_method,
            0, // do_log,
            1, // nrepeat,
            param.N,
            param.K,
            param.C,
            param.input_spatial_lengths,
            param.filter_spatial_lengths,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides,
            param.conv_filter_dilations,
            param.input_left_pads,
            param.input_right_pads);

        pass &= ck::profiler::profile_convnd_bwd_data_impl<1,
                                                           ck::half_t,
                                                           ck::half_t,
                                                           ck::half_t,
                                                           float,
                                                           ck::tensor_layout::convolution::NWC,
                                                           ck::tensor_layout::convolution::KXC,
                                                           ck::tensor_layout::convolution::NWK>(
            1, // do_verification,
            1, // init_method,
            0, // do_log,
            1, // nrepeat,
            param.N,
            param.K,
            param.C,
            param.input_spatial_lengths,
            param.filter_spatial_lengths,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides,
            param.conv_filter_dilations,
            param.input_left_pads,
            param.input_right_pads);

        pass &= ck::profiler::profile_convnd_bwd_data_impl<1,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t,
                                                           float,
                                                           ck::tensor_layout::convolution::NWC,
                                                           ck::tensor_layout::convolution::KXC,
                                                           ck::tensor_layout::convolution::NWK>(
            1, // do_verification,
            1, // init_method,
            0, // do_log,
            1, // nrepeat,
            param.N,
            param.K,
            param.C,
            param.input_spatial_lengths,
            param.filter_spatial_lengths,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides,
            param.conv_filter_dilations,
            param.input_left_pads,
            param.input_right_pads);

        pass &= ck::profiler::profile_convnd_bwd_data_impl<1,
                                                           int8_t,
                                                           int8_t,
                                                           int8_t,
                                                           int,
                                                           ck::tensor_layout::convolution::NWC,
                                                           ck::tensor_layout::convolution::KXC,
                                                           ck::tensor_layout::convolution::NWK>(
            1, // do_verification,
            1, // init_method,
            0, // do_log,
            1, // nrepeat,
            param.N,
            param.K,
            param.C,
            param.input_spatial_lengths,
            param.filter_spatial_lengths,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides,
            param.conv_filter_dilations,
            param.input_left_pads,
            param.input_right_pads);
    }

    // check 2d
    params.clear();
    params.push_back({2, 128, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    params.push_back({2, 128, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    params.push_back({2, 128, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});

    for(auto& param : params)
    {
        pass &= ck::profiler::profile_convnd_bwd_data_impl<2,
                                                           float,
                                                           float,
                                                           float,
                                                           float,
                                                           ck::tensor_layout::convolution::NHWC,
                                                           ck::tensor_layout::convolution::KYXC,
                                                           ck::tensor_layout::convolution::NHWK>(
            1, // do_verification,
            1, // init_method,
            0, // do_log,
            1, // nrepeat,
            param.N,
            param.K,
            param.C,
            param.input_spatial_lengths,
            param.filter_spatial_lengths,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides,
            param.conv_filter_dilations,
            param.input_left_pads,
            param.input_right_pads);

        pass &= ck::profiler::profile_convnd_bwd_data_impl<2,
                                                           ck::half_t,
                                                           ck::half_t,
                                                           ck::half_t,
                                                           float,
                                                           ck::tensor_layout::convolution::NHWC,
                                                           ck::tensor_layout::convolution::KYXC,
                                                           ck::tensor_layout::convolution::NHWK>(
            1, // do_verification,
            1, // init_method,
            0, // do_log,
            1, // nrepeat,
            param.N,
            param.K,
            param.C,
            param.input_spatial_lengths,
            param.filter_spatial_lengths,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides,
            param.conv_filter_dilations,
            param.input_left_pads,
            param.input_right_pads);

        pass &= ck::profiler::profile_convnd_bwd_data_impl<2,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t,
                                                           float,
                                                           ck::tensor_layout::convolution::NHWC,
                                                           ck::tensor_layout::convolution::KYXC,
                                                           ck::tensor_layout::convolution::NHWK>(
            1, // do_verification,
            1, // init_method,
            0, // do_log,
            1, // nrepeat,
            param.N,
            param.K,
            param.C,
            param.input_spatial_lengths,
            param.filter_spatial_lengths,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides,
            param.conv_filter_dilations,
            param.input_left_pads,
            param.input_right_pads);

        pass &= ck::profiler::profile_convnd_bwd_data_impl<2,
                                                           int8_t,
                                                           int8_t,
                                                           int8_t,
                                                           int,
                                                           ck::tensor_layout::convolution::NHWC,
                                                           ck::tensor_layout::convolution::KYXC,
                                                           ck::tensor_layout::convolution::NHWK>(
            1, // do_verification,
            1, // init_method,
            0, // do_log,
            1, // nrepeat,
            param.N,
            param.K,
            param.C,
            param.input_spatial_lengths,
            param.filter_spatial_lengths,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides,
            param.conv_filter_dilations,
            param.input_left_pads,
            param.input_right_pads);
    }

    // check 3d
    params.clear();
    params.push_back(
        {3, 128, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    params.push_back(
        {3, 128, 128, 256, {3, 3, 3}, {14, 14, 14}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    params.push_back(
        {3, 128, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});

    for(auto& param : params)
    {
        pass &= ck::profiler::profile_convnd_bwd_data_impl<3,
                                                           float,
                                                           float,
                                                           float,
                                                           float,
                                                           ck::tensor_layout::convolution::NDHWC,
                                                           ck::tensor_layout::convolution::KZYXC,
                                                           ck::tensor_layout::convolution::NDHWK>(
            1, // do_verification,
            1, // init_method,
            0, // do_log,
            1, // nrepeat,
            param.N,
            param.K,
            param.C,
            param.input_spatial_lengths,
            param.filter_spatial_lengths,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides,
            param.conv_filter_dilations,
            param.input_left_pads,
            param.input_right_pads);

        pass &= ck::profiler::profile_convnd_bwd_data_impl<3,
                                                           ck::half_t,
                                                           ck::half_t,
                                                           ck::half_t,
                                                           float,
                                                           ck::tensor_layout::convolution::NDHWC,
                                                           ck::tensor_layout::convolution::KZYXC,
                                                           ck::tensor_layout::convolution::NDHWK>(
            1, // do_verification,
            1, // init_method,
            0, // do_log,
            1, // nrepeat,
            param.N,
            param.K,
            param.C,
            param.input_spatial_lengths,
            param.filter_spatial_lengths,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides,
            param.conv_filter_dilations,
            param.input_left_pads,
            param.input_right_pads);

        pass &= ck::profiler::profile_convnd_bwd_data_impl<3,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t,
                                                           float,
                                                           ck::tensor_layout::convolution::NDHWC,
                                                           ck::tensor_layout::convolution::KZYXC,
                                                           ck::tensor_layout::convolution::NDHWK>(
            1, // do_verification,
            1, // init_method,
            0, // do_log,
            1, // nrepeat,
            param.N,
            param.K,
            param.C,
            param.input_spatial_lengths,
            param.filter_spatial_lengths,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides,
            param.conv_filter_dilations,
            param.input_left_pads,
            param.input_right_pads);

        pass &= ck::profiler::profile_convnd_bwd_data_impl<3,
                                                           int8_t,
                                                           int8_t,
                                                           int8_t,
                                                           int,
                                                           ck::tensor_layout::convolution::NDHWC,
                                                           ck::tensor_layout::convolution::KZYXC,
                                                           ck::tensor_layout::convolution::NDHWK>(
            1, // do_verification,
            1, // init_method,
            0, // do_log,
            1, // nrepeat,
            param.N,
            param.K,
            param.C,
            param.input_spatial_lengths,
            param.filter_spatial_lengths,
            param.GetOutputSpatialLengths(),
            param.conv_filter_strides,
            param.conv_filter_dilations,
            param.input_left_pads,
            param.input_right_pads);
    }

    if(pass)
    {
        std::cout << "test convnd bwd : Pass" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "test convnd bwd: Fail " << std::endl;
        return -1;
    }
}
