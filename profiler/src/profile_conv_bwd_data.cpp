#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "profile_conv_bwd_data_impl.hpp"

enum struct ConvDataType
{
    F32_F32_F32,    // 0
    F16_F16_F16,    // 1
    BF16_BF16_BF16, // 2
    INT8_INT8_INT8, // 3
};

enum struct ConvInputLayout
{
    NCHW, // 0
    NHWC, // 1
};

enum struct ConvWeightLayout
{
    KCYX, // 0
    KYXC, // 1
};

enum struct ConvOutputLayout
{
    NKHW, // 0
    NHWK, // 1
};

int profile_conv_bwd_data(int argc, char* argv[])
{
    if(argc != 25)
    {
        printf("arg1: tensor operation (conv_bwd: BackwardConvolution)\n");
        printf("arg2: data type (0: fp32; 1: fp16)\n");
        printf("arg3: input tensor layout (0: NCHW; 1: NHWC)\n");
        printf("arg4: weight tensor layout (0: KCYX; 1: KYXC)\n");
        printf("arg5: output tensor layout (0: NKHW; 1: NHWK)\n");
        printf("arg6: verification (0: no; 1: yes)\n");
        printf("arg7: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg8: print tensor value (0: no; 1: yes)\n");
        printf("arg9: run kernel # of times (>1)\n");
        printf("arg10 to 24: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
               "RightPx\n");
        exit(1);
    }

    const auto data_type       = static_cast<ConvDataType>(std::stoi(argv[2]));
    const auto in_layout       = static_cast<ConvInputLayout>(std::stoi(argv[3]));
    const auto wei_layout      = static_cast<ConvWeightLayout>(std::stoi(argv[4]));
    const auto out_layout      = static_cast<ConvOutputLayout>(std::stoi(argv[5]));
    const bool do_verification = std::stoi(argv[6]);
    const int init_method      = std::stoi(argv[7]);
    const bool do_log          = std::stoi(argv[8]);
    const int nrepeat          = std::stoi(argv[9]);

    const ck::index_t N  = std::stoi(argv[10]);
    const ck::index_t K  = std::stoi(argv[11]);
    const ck::index_t C  = std::stoi(argv[12]);
    const ck::index_t Y  = std::stoi(argv[13]);
    const ck::index_t X  = std::stoi(argv[14]);
    const ck::index_t Hi = std::stoi(argv[15]);
    const ck::index_t Wi = std::stoi(argv[16]);

    const ck::index_t conv_stride_h   = std::stoi(argv[17]);
    const ck::index_t conv_stride_w   = std::stoi(argv[18]);
    const ck::index_t conv_dilation_h = std::stoi(argv[19]);
    const ck::index_t conv_dilation_w = std::stoi(argv[20]);
    const ck::index_t in_left_pad_h   = std::stoi(argv[21]);
    const ck::index_t in_left_pad_w   = std::stoi(argv[22]);
    const ck::index_t in_right_pad_h  = std::stoi(argv[23]);
    const ck::index_t in_right_pad_w  = std::stoi(argv[24]);

    const ck::index_t YEff = (Y - 1) * conv_dilation_h + 1;
    const ck::index_t XEff = (X - 1) * conv_dilation_w + 1;

    const ck::index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - YEff) / conv_stride_h + 1;
    const ck::index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;

    if(data_type == ConvDataType::F32_F32_F32 && in_layout == ConvInputLayout::NHWC &&
       wei_layout == ConvWeightLayout::KYXC && out_layout == ConvOutputLayout::NHWK)
    {
        ck::profiler::profile_conv_bwd_data_impl<2,
                                                 float,
                                                 float,
                                                 float,
                                                 float,
                                                 ck::tensor_layout::convolution::NHWC,
                                                 ck::tensor_layout::convolution::KYXC,
                                                 ck::tensor_layout::convolution::NHWK>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            N,
            K,
            C,
            std::vector<ck::index_t>{Hi, Wi},
            std::vector<ck::index_t>{Y, X},
            std::vector<ck::index_t>{Ho, Wo},
            std::vector<ck::index_t>{conv_stride_h, conv_stride_w},
            std::vector<ck::index_t>{conv_dilation_h, conv_dilation_w},
            std::vector<ck::index_t>{in_left_pad_h, in_left_pad_w},
            std::vector<ck::index_t>{in_right_pad_h, in_right_pad_w});
    }
    else if(data_type == ConvDataType::F16_F16_F16 && in_layout == ConvInputLayout::NHWC &&
            wei_layout == ConvWeightLayout::KYXC && out_layout == ConvOutputLayout::NHWK)
    {
        ck::profiler::profile_conv_bwd_data_impl<2,
                                                 ck::half_t,
                                                 ck::half_t,
                                                 ck::half_t,
                                                 float,
                                                 ck::tensor_layout::convolution::NHWC,
                                                 ck::tensor_layout::convolution::KYXC,
                                                 ck::tensor_layout::convolution::NHWK>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            N,
            K,
            C,
            std::vector<ck::index_t>{Hi, Wi},
            std::vector<ck::index_t>{Y, X},
            std::vector<ck::index_t>{Ho, Wo},
            std::vector<ck::index_t>{conv_stride_h, conv_stride_w},
            std::vector<ck::index_t>{conv_dilation_h, conv_dilation_w},
            std::vector<ck::index_t>{in_left_pad_h, in_left_pad_w},
            std::vector<ck::index_t>{in_right_pad_h, in_right_pad_w});
    }
    else if(data_type == ConvDataType::BF16_BF16_BF16 && in_layout == ConvInputLayout::NHWC &&
            wei_layout == ConvWeightLayout::KYXC && out_layout == ConvOutputLayout::NHWK)
    {
        ck::profiler::profile_conv_bwd_data_impl<2,
                                                 uint16_t,
                                                 uint16_t,
                                                 uint16_t,
                                                 float,
                                                 ck::tensor_layout::convolution::NHWC,
                                                 ck::tensor_layout::convolution::KYXC,
                                                 ck::tensor_layout::convolution::NHWK>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            N,
            K,
            C,
            std::vector<ck::index_t>{Hi, Wi},
            std::vector<ck::index_t>{Y, X},
            std::vector<ck::index_t>{Ho, Wo},
            std::vector<ck::index_t>{conv_stride_h, conv_stride_w},
            std::vector<ck::index_t>{conv_dilation_h, conv_dilation_w},
            std::vector<ck::index_t>{in_left_pad_h, in_left_pad_w},
            std::vector<ck::index_t>{in_right_pad_h, in_right_pad_w});
    }
    else if(data_type == ConvDataType::INT8_INT8_INT8 && in_layout == ConvInputLayout::NHWC &&
            wei_layout == ConvWeightLayout::KYXC && out_layout == ConvOutputLayout::NHWK)
    {
        ck::profiler::profile_conv_bwd_data_impl<2,
                                                 int8_t,
                                                 int8_t,
                                                 int8_t,
                                                 int32_t,
                                                 ck::tensor_layout::convolution::NHWC,
                                                 ck::tensor_layout::convolution::KYXC,
                                                 ck::tensor_layout::convolution::NHWK>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            N,
            K,
            C,
            std::vector<ck::index_t>{Hi, Wi},
            std::vector<ck::index_t>{Y, X},
            std::vector<ck::index_t>{Ho, Wo},
            std::vector<ck::index_t>{conv_stride_h, conv_stride_w},
            std::vector<ck::index_t>{conv_dilation_h, conv_dilation_w},
            std::vector<ck::index_t>{in_left_pad_h, in_left_pad_w},
            std::vector<ck::index_t>{in_right_pad_h, in_right_pad_w});
    }
    else
    {
        throw std::runtime_error("wrong! this Conv data_type & layout is not implemented");
    }

    return 1;
}
