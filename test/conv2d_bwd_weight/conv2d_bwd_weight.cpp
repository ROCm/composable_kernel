#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include <vector>

#include "conv_fwd_util.hpp"
#include "profile_conv_bwd_weight_impl.hpp"

int test_self()
{
    bool pass = true;
    std::vector<ck::utils::conv::ConvParams> params;

    params.push_back({2, 128, 256, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    params.push_back({2, 128, 256, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    params.push_back({2, 128, 256, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});

    for(auto& param : params)
    {
        // f32
        pass &= ck::profiler::profile_conv_bwd_weight_impl<2,
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
            param.input_right_pads,
            2);

        // fp16
        pass &= ck::profiler::profile_conv_bwd_weight_impl<2,
                                                           ck::half_t,
                                                           ck::half_t,
                                                           ck::half_t,
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
            param.input_right_pads,
            2);
    }
    return pass;
}
int main(int argc, char* argv[])
{
    int data_type   = 0;
    int init_method = 0;

    // Conv shape
    ck::index_t N               = 128;
    ck::index_t K               = 256;
    ck::index_t C               = 192;
    ck::index_t Y               = 3;
    ck::index_t X               = 3;
    ck::index_t Hi              = 71;
    ck::index_t Wi              = 71;
    ck::index_t conv_stride_h   = 2;
    ck::index_t conv_stride_w   = 2;
    ck::index_t conv_dilation_h = 1;
    ck::index_t conv_dilation_w = 1;
    ck::index_t in_left_pad_h   = 1;
    ck::index_t in_left_pad_w   = 1;
    ck::index_t in_right_pad_h  = 1;
    ck::index_t in_right_pad_w  = 1;
    ck::index_t split_k         = 1;

    bool pass = true;
    if(argc == 1)
    {
        pass = test_self();
    }
    else
    {
        if(argc == 3)
        {
            data_type   = std::stoi(argv[1]);
            init_method = std::stoi(argv[2]);
        }
        else if(argc == 19)
        {
            data_type   = std::stoi(argv[1]);
            init_method = std::stoi(argv[2]);

            N               = std::stoi(argv[3]);
            K               = std::stoi(argv[4]);
            C               = std::stoi(argv[5]);
            Y               = std::stoi(argv[6]);
            X               = std::stoi(argv[7]);
            Hi              = std::stoi(argv[8]);
            Wi              = std::stoi(argv[9]);
            conv_stride_h   = std::stoi(argv[10]);
            conv_stride_w   = std::stoi(argv[11]);
            conv_dilation_h = std::stoi(argv[12]);
            conv_dilation_w = std::stoi(argv[13]);
            in_left_pad_h   = std::stoi(argv[14]);
            in_left_pad_w   = std::stoi(argv[15]);
            in_right_pad_h  = std::stoi(argv[16]);
            in_right_pad_w  = std::stoi(argv[17]);
            split_k         = std::stoi(argv[18]);
        }
        else
        {
            printf("arg1: data type (0=fp32, 1=fp16, 2= bfp16, 3= int8_t )\n");
            printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
            printf("arg3 to 17: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
                   "RightPx\n");
            exit(1);
        }

        ck::utils::conv::ConvParams param{2,
                                          N,
                                          K,
                                          C,
                                          {Y, X},
                                          {Hi, Wi},
                                          {conv_stride_h, conv_stride_w},
                                          {conv_dilation_h, conv_dilation_w},
                                          {in_left_pad_h, in_left_pad_w},
                                          {in_right_pad_h, in_right_pad_w}};
        if(data_type == 0)
        {
            pass = ck::profiler::profile_conv_bwd_weight_impl<2,
                                                              float,
                                                              float,
                                                              float,
                                                              ck::tensor_layout::convolution::NHWC,
                                                              ck::tensor_layout::convolution::KYXC,
                                                              ck::tensor_layout::convolution::NHWK>(
                1,
                init_method,
                0,
                1,
                param.N,
                param.K,
                param.C,
                param.input_spatial_lengths,
                param.filter_spatial_lengths,
                param.GetOutputSpatialLengths(),
                param.conv_filter_strides,
                param.conv_filter_dilations,
                param.input_left_pads,
                param.input_right_pads,
                split_k);
        }
        else if(data_type == 1)
        {
            pass = ck::profiler::profile_conv_bwd_weight_impl<2,
                                                              ck::half_t,
                                                              ck::half_t,
                                                              ck::half_t,
                                                              ck::tensor_layout::convolution::NHWC,
                                                              ck::tensor_layout::convolution::KYXC,
                                                              ck::tensor_layout::convolution::NHWK>(
                1,
                init_method,
                0,
                1,
                param.N,
                param.K,
                param.C,
                param.input_spatial_lengths,
                param.filter_spatial_lengths,
                param.GetOutputSpatialLengths(),
                param.conv_filter_strides,
                param.conv_filter_dilations,
                param.input_left_pads,
                param.input_right_pads,
                split_k);
        }
        else
        {
            std::cout << "Not support data type" << std::endl;
            return 1;
        }
    }

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
