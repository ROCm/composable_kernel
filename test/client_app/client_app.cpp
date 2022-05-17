#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include <vector>

#include "client_app_impl.hpp"

int main(int argc, char* argv[])
{
    if(argc != 25)
    {
        printf("arg1: tensor operation (conv_fwd: ForwardConvolution)\n");
        printf("arg2: data type (0: fp32; 1: fp16)\n");
        printf("arg3: input tensor layout (0: NCHW; 1: NHWC)\n");
        printf("arg4: weight tensor layout (0: KCYX; 1: KYXC)\n");
        printf("arg5: output tensor layout (0: NKHW; 1: NHWK)\n");
        printf("arg6: verification (0: no; 1: yes)\n");
        printf("arg7: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg8: print tensor value (0: no; 1: yes)\n");
        printf("arg9: time kernel (0=n0, 1=yes)\n");
        printf("arg10 to 24: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
               "RightPx\n");
        exit(1);
    }

    const ConvDataType data_type = static_cast<ConvDataType>(std::stoi(argv[2]));
    const int in_layout          = static_cast<ConvInputLayout>(std::stoi(argv[3]));
    const int wei_layout         = static_cast<ConvWeightLayout>(std::stoi(argv[4]));
    const int out_layout         = static_cast<ConvOutputLayout>(std::stoi(argv[5]));
    const bool do_verification   = std::stoi(argv[6]);
    const int init_method        = std::stoi(argv[7]);
    const bool do_log            = std::stoi(argv[8]);
    const bool time_kernel       = std::stoi(argv[9]);

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

    ck::app::profile_conv_fwd_impl(do_verification,
                                   init_method,
                                   do_log,
                                   time_kernel,
                                   data_type,
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
    return 1;
}
