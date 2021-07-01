#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor_generator.hpp"
#include "conv_common.hpp"
#include "host_conv.hpp"
#include "device_tensor.hpp"
#include "device_convolution_forward_implicit_gemm_v4r1_nchw_kcyx_nkhw.hpp"
#include "device_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw.hpp"
#include "device_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk.hpp"

int main(int argc, char* argv[])
{
    using namespace ck;

    if(argc != 5)
    {
        printf("arg1: do_verification, arg2: do_log, arg3: init_method, arg4: nrepeat\n");
        exit(1);
    }

    const bool do_verification = atoi(argv[1]);
    const bool do_log          = atoi(argv[2]);
    const int init_method      = atoi(argv[3]);
    const int nrepeat          = atoi(argv[4]);

#if 0
    constexpr index_t N  = 256;
    constexpr index_t C  = 256;
    constexpr index_t HI = 16;
    constexpr index_t WI = 16;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 16;
    constexpr index_t HI = 1080;
    constexpr index_t WI = 1920;
    constexpr index_t K  = 16;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 1>;
    using InRightPads = Sequence<1, 1>;
#elif 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 16;
    constexpr index_t Hi = 540;
    constexpr index_t Wi = 960;
    constexpr index_t K  = 16;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 16;
    constexpr index_t Hi = 270;
    constexpr index_t Wi = 480;
    constexpr index_t K  = 16;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 16;
    constexpr index_t Hi = 1080;
    constexpr index_t Wi = 1920;
    constexpr index_t K  = 16;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 1>;
    using InRightPads = Sequence<1, 1>;
#elif 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 1;
    constexpr index_t Hi = 1024;
    constexpr index_t Wi = 2048;
    constexpr index_t K  = 4;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 1>;
    using InRightPads = Sequence<1, 1>;
#elif 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 16;
    constexpr index_t Hi = 540;
    constexpr index_t Wi = 960;
    constexpr index_t K  = 16;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 1>;
    using InRightPads = Sequence<1, 1>;
#elif 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 16;
    constexpr index_t Hi = 270;
    constexpr index_t Wi = 480;
    constexpr index_t K  = 16;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 1>;
    using InRightPads = Sequence<1, 1>;
#elif 0
    // 3x3, 36x36, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t Hi = 37;
    constexpr index_t Wi = 37;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 35x35, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t Hi = 35;
    constexpr index_t Wi = 35;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 71x71
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t HI = 71;
    constexpr index_t WI = 71;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 1>;
    using InRightPads = Sequence<1, 1>;
#elif 0
    // 1x1, 8x8
    constexpr index_t N  = 128;
    constexpr index_t C  = 1536;
    constexpr index_t Hi = 8;
    constexpr index_t Wi = 8;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 73x73
    constexpr index_t N  = 128;
    constexpr index_t C  = 160;
    constexpr index_t Hi = 73;
    constexpr index_t Wi = 73;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 35x35
    constexpr index_t N  = 128;
    constexpr index_t C  = 96;
    constexpr index_t Hi = 35;
    constexpr index_t Wi = 35;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 1>;
    using InRightPads = Sequence<1, 1>;
#elif 0
    // 3x3, 71x71
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t Hi = 71;
    constexpr index_t Wi = 71;
    constexpr index_t K  = 192;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 1>;
    using InRightPads = Sequence<1, 1>;
#elif 0
    // 7x1, 17x17
    constexpr index_t N  = 128;
    constexpr index_t C  = 128;
    constexpr index_t Hi = 17;
    constexpr index_t Wi = 17;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 7;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<3, 0>;
    using InRightPads = Sequence<3, 0>;
#elif 1
    // 1x7, 17x17
    constexpr index_t N  = 128;
    constexpr index_t C  = 128;
    constexpr index_t Hi = 17;
    constexpr index_t Wi = 17;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 7;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 3>;
    using InRightPads = Sequence<0, 3>;
#elif 0
    // 3x3, 299x299 stride=2
    constexpr index_t N  = 128;
    constexpr index_t C  = 3;
    constexpr index_t Hi = 299;
    constexpr index_t Wi = 299;
    constexpr index_t K  = 32;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 147x147
    constexpr index_t N  = 128;
    constexpr index_t C  = 128;
    constexpr index_t Hi = 147;
    constexpr index_t Wi = 147;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 1>;
    using InRightPads = Sequence<1, 1>;
#elif 0
    // 3x3, 149x149
    constexpr index_t N  = 128;
    constexpr index_t C  = 32;
    constexpr index_t Hi = 149;
    constexpr index_t Wi = 149;
    constexpr index_t K  = 32;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 17x17, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t Hi = 17;
    constexpr index_t Wi = 17;
    constexpr index_t K  = 192;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 35x35
    constexpr index_t N  = 128;
    constexpr index_t C  = 384;
    constexpr index_t Hi = 35;
    constexpr index_t Wi = 35;
    constexpr index_t K  = 96;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 35x35, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 288;
    constexpr index_t Hi = 35;
    constexpr index_t Wi = 35;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 1x3, 8x8
    constexpr index_t N  = 128;
    constexpr index_t C  = 384;
    constexpr index_t Hi = 8;
    constexpr index_t Wi = 8;
    constexpr index_t K  = 448;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 1>;
    using InRightPads = Sequence<0, 1>;
#elif 0
    // 3x1, 8x8
    constexpr index_t N  = 128;
    constexpr index_t C  = 448;
    constexpr index_t Hi = 8;
    constexpr index_t Wi = 8;
    constexpr index_t K  = 512;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 0>;
    using InRightPads = Sequence<1, 0>;
#elif 0
    // 3x3, 147x147
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t Hi = 147;
    constexpr index_t Wi = 147;
    constexpr index_t K  = 96;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 7x1, 73x73
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t Hi = 73;
    constexpr index_t Wi = 73;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 7;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<3, 0>;
    using InRightPads = Sequence<3, 0>;
#elif 0
    // 3x3, 73x73
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t Hi = 73;
    constexpr index_t Wi = 73;
    constexpr index_t K  = 96;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 14x14, stride 2
    constexpr index_t N  = 256;
    constexpr index_t C  = 1024;
    constexpr index_t Hi = 14;
    constexpr index_t Wi = 14;
    constexpr index_t K  = 2048;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 14x14
    constexpr index_t N  = 256;
    constexpr index_t C  = 1024;
    constexpr index_t Hi = 14;
    constexpr index_t Wi = 14;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 14x14, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 1024;
    constexpr index_t Hi = 14;
    constexpr index_t Wi = 14;
    constexpr index_t K  = 512;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 1
    // 3x3, 28x28
    constexpr index_t N  = 128;
    constexpr index_t C  = 128;
    constexpr index_t Hi = 28;
    constexpr index_t Wi = 28;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 1>;
    using InRightPads = Sequence<1, 1>;
#elif 1
    // 3x3, 14x14
    constexpr index_t N  = 128;
    constexpr index_t C  = 256;
    constexpr index_t Hi = 14;
    constexpr index_t Wi = 14;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 1>;
    using InRightPads = Sequence<1, 1>;
#elif 0
    // 1x1, 56x56, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 256;
    constexpr index_t Hi = 56;
    constexpr index_t Wi = 56;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 7x7, 230x230 stride=2
    constexpr index_t N  = 128;
    constexpr index_t C  = 3;
    constexpr index_t Hi = 230;
    constexpr index_t Wi = 230;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 7;
    constexpr index_t X  = 7;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 28x28, stride = 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t Hi = 28;
    constexpr index_t Wi = 28;
    constexpr index_t K  = 1024;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 28x28, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t Hi = 28;
    constexpr index_t Wi = 28;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 1
    // 1x1, 7x7
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t Hi = 7;
    constexpr index_t Wi = 7;
    constexpr index_t K  = 2048;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 7x7
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t Hi = 7;
    constexpr index_t Wi = 7;
    constexpr index_t K  = 512;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 1>;
    using InRightPads = Sequence<1, 1>;
#elif 0
    // 1x1, 56x56
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t Hi = 56;
    constexpr index_t Wi = 56;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<0, 0>;
    using InRightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 56x56
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t Hi = 56;
    constexpr index_t Wi = 56;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using InLeftPads  = Sequence<1, 1>;
    using InRightPads = Sequence<1, 1>;
#endif

    constexpr index_t YEff = (Y - 1) * ConvDilations{}[0] + 1;
    constexpr index_t XEff = (X - 1) * ConvDilations{}[1] + 1;

    constexpr index_t Ho = (Hi + InLeftPads{}[0] + InRightPads{}[0] - YEff) / ConvStrides{}[0] + 1;
    constexpr index_t Wo = (Wi + InLeftPads{}[1] + InRightPads{}[1] - XEff) / ConvStrides{}[1] + 1;

#if 1
    constexpr index_t in_vector_size = 1;
    using in_data_t                  = typename vector_type<float, in_vector_size>::type;
    using acc_data_t                 = float;
    using out_data_t                 = float;
#elif 1
    using in_data_t                  = half_t;
    constexpr index_t in_vector_size = 1;
    using acc_data_t                 = float;
    using out_data_t                 = half_t;
#elif 0
    constexpr index_t in_vector_size = 1;
    using in_data_t                  = typename vector_type<float, in_vector_size>::type;
    using acc_data_t                 = float;
    using out_data_t                 = int8_t;
#elif 1
    constexpr index_t in_vector_size = 16;
    using in_data_t                  = typename vector_type<int8_t, in_vector_size>::type;
    using acc_data_t                 = int32_t;
    using out_data_t                 = int8_t;
#endif

    Tensor<in_data_t> in_nchw(HostTensorDescriptor(std::initializer_list<index_t>{N, C, Hi, Wi}));
    Tensor<in_data_t> wei_kcyx(HostTensorDescriptor(std::initializer_list<index_t>{K, C, Y, X}));
    Tensor<out_data_t> out_nkhw_host(
        HostTensorDescriptor(std::initializer_list<index_t>{N, K, Ho, Wo}));
    Tensor<out_data_t> out_nkhw_device(
        HostTensorDescriptor(std::initializer_list<index_t>{N, K, Ho, Wo}));

    ostream_HostTensorDescriptor(in_nchw.mDesc, std::cout << "in_nchw_desc: ");
    ostream_HostTensorDescriptor(wei_kcyx.mDesc, std::cout << "wei_kcyx_desc: ");
    ostream_HostTensorDescriptor(out_nkhw_host.mDesc, std::cout << "out_nkhw_desc: ");

    print_array("InLeftPads", InLeftPads{});
    print_array("InRightPads", InRightPads{});
    print_array("ConvStrides", ConvStrides{});
    print_array("ConvDilations", ConvDilations{});

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(do_verification)
    {
        switch(init_method)
        {
        case 0:
            in_nchw.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
            wei_kcyx.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
            break;
        case 1:
            in_nchw.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
            wei_kcyx.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
            break;
        case 2:
            in_nchw.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
            wei_kcyx.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
            break;
        case 3:
            in_nchw.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
            wei_kcyx.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
            break;
        default:
            in_nchw.GenerateTensorValue(GeneratorTensor_2{1, 5}, num_thread);

            auto gen_wei = [](auto... is) {
                return GeneratorTensor_2{1, 5}(is...) * GeneratorTensor_Checkboard{}(is...);
            };
            wei_kcyx.GenerateTensorValue(gen_wei, num_thread);
        }
    }

    constexpr auto in_nchw_desc  = make_native_tensor_descriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto wei_kcyx_desc = make_native_tensor_descriptor_packed(Sequence<K, C, Y, X>{});
    constexpr auto out_nkhw_desc = make_native_tensor_descriptor_packed(Sequence<N, K, Ho, Wo>{});

#if 1
    device_convolution_forward_implicit_gemm_v4r1_nchw_kcyx_nkhw(in_nchw_desc,
                                                                 in_nchw,
                                                                 wei_kcyx_desc,
                                                                 wei_kcyx,
                                                                 out_nkhw_desc,
                                                                 out_nkhw_device,
                                                                 ConvStrides{},
                                                                 ConvDilations{},
                                                                 InLeftPads{},
                                                                 InRightPads{},
                                                                 nrepeat);
#elif 0
    device_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw(in_nchw_desc,
                                                                 in_nchw,
                                                                 wei_kcyx_desc,
                                                                 wei_kcyx,
                                                                 out_nkhw_desc,
                                                                 out_nkhw_device,
                                                                 ConvStrides{},
                                                                 ConvDilations{},
                                                                 InLeftPads{},
                                                                 InRightPads{},
                                                                 nrepeat);
#elif 0
    device_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk(in_nchw_desc,
                                                                 in_nchw,
                                                                 wei_kcyx_desc,
                                                                 wei_kcyx,
                                                                 out_nkhw_desc,
                                                                 out_nkhw_device,
                                                                 ConvStrides{},
                                                                 ConvDilations{},
                                                                 InLeftPads{},
                                                                 InRightPads{},
                                                                 nrepeat);
#endif

    if(do_verification)
    {
        host_direct_convolution(in_nchw,
                                wei_kcyx,
                                out_nkhw_host,
                                ConvStrides{},
                                ConvDilations{},
                                InLeftPads{},
                                InRightPads{});

        check_error(out_nkhw_host, out_nkhw_device);

        if(do_log)
        {
            LogRange(std::cout << "in_nchw : ", in_nchw.mData, ",") << std::endl;
            LogRange(std::cout << "wei_kcyx: ", wei_kcyx.mData, ",") << std::endl;
            LogRange(std::cout << "out_nkhw_host  : ", out_nkhw_host.mData, ",") << std::endl;
            LogRange(std::cout << "out_nkhw_device: ", out_nkhw_device.mData, ",") << std::endl;
        }
    }
}
