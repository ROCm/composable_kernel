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
#include "device_dynamic_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw.hpp"
#include "device_dynamic_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk.hpp"
#include "device_dynamic_convolution_forward_implicit_gemm_v5r1_nchw_kcyx_nkhw.hpp"

int main(int argc, char* argv[])
{
    using namespace ck;

#if 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 16;
    constexpr index_t HI = 1080;
    constexpr index_t WI = 1920;
    constexpr index_t K  = 16;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 16;
    constexpr index_t HI = 540;
    constexpr index_t WI = 960;
    constexpr index_t K  = 16;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 16;
    constexpr index_t HI = 270;
    constexpr index_t WI = 480;
    constexpr index_t K  = 16;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 16;
    constexpr index_t HI = 1080;
    constexpr index_t WI = 1920;
    constexpr index_t K  = 16;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 1;
    constexpr index_t HI = 1024;
    constexpr index_t WI = 2048;
    constexpr index_t K  = 4;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 16;
    constexpr index_t HI = 540;
    constexpr index_t WI = 960;
    constexpr index_t K  = 16;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    constexpr index_t N  = 1;
    constexpr index_t C  = 16;
    constexpr index_t HI = 270;
    constexpr index_t WI = 480;
    constexpr index_t K  = 16;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    // 3x3, 36x36, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t HI = 37;
    constexpr index_t WI = 37;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 35x35, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t HI = 35;
    constexpr index_t WI = 35;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 1
    // 3x3, 71x71
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t HI = 71;
    constexpr index_t WI = 71;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 1
    // 1x1, 8x8
    constexpr index_t N  = 128;
    constexpr index_t C  = 1536;
    constexpr index_t HI = 8;
    constexpr index_t WI = 8;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 73x73
    constexpr index_t N  = 128;
    constexpr index_t C  = 160;
    constexpr index_t HI = 73;
    constexpr index_t WI = 73;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 35x35
    constexpr index_t N  = 128;
    constexpr index_t C  = 96;
    constexpr index_t HI = 35;
    constexpr index_t WI = 35;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 1
    // 3x3, 71x71
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t HI = 71;
    constexpr index_t WI = 71;
    constexpr index_t K  = 192;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    // 7x1, 17x17
    constexpr index_t N  = 128;
    constexpr index_t C  = 128;
    constexpr index_t HI = 17;
    constexpr index_t WI = 17;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 7;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<3, 0>;
    using RightPads = Sequence<3, 0>;
#elif 0
    // 1x7, 17x17
    constexpr index_t N  = 128;
    constexpr index_t C  = 128;
    constexpr index_t HI = 17;
    constexpr index_t WI = 17;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 7;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 3>;
    using RightPads = Sequence<0, 3>;
#elif 0
    // 3x3, 299x299 stride=2
    constexpr index_t N  = 128;
    constexpr index_t C  = 3;
    constexpr index_t HI = 299;
    constexpr index_t WI = 299;
    constexpr index_t K  = 32;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 147x147
    constexpr index_t N  = 128;
    constexpr index_t C  = 128;
    constexpr index_t HI = 147;
    constexpr index_t WI = 147;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    // 3x3, 149x149
    constexpr index_t N  = 128;
    constexpr index_t C  = 32;
    constexpr index_t HI = 149;
    constexpr index_t WI = 149;
    constexpr index_t K  = 32;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 17x17, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t HI = 17;
    constexpr index_t WI = 17;
    constexpr index_t K  = 192;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 35x35
    constexpr index_t N  = 128;
    constexpr index_t C  = 384;
    constexpr index_t HI = 35;
    constexpr index_t WI = 35;
    constexpr index_t K  = 96;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 35x35, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 288;
    constexpr index_t HI = 35;
    constexpr index_t WI = 35;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x3, 8x8
    constexpr index_t N  = 128;
    constexpr index_t C  = 384;
    constexpr index_t HI = 8;
    constexpr index_t WI = 8;
    constexpr index_t K  = 448;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 1>;
    using RightPads = Sequence<0, 1>;
#elif 0
    // 3x1, 8x8
    constexpr index_t N  = 128;
    constexpr index_t C  = 448;
    constexpr index_t HI = 8;
    constexpr index_t WI = 8;
    constexpr index_t K  = 512;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 0>;
    using RightPads = Sequence<1, 0>;
#elif 0
    // 3x3, 147x147
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t HI = 147;
    constexpr index_t WI = 147;
    constexpr index_t K  = 96;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 7x1, 73x73
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t HI = 73;
    constexpr index_t WI = 73;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 7;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<3, 0>;
    using RightPads = Sequence<3, 0>;
#elif 0
    // 3x3, 73x73
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t HI = 73;
    constexpr index_t WI = 73;
    constexpr index_t K  = 96;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 14x14, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 1024;
    constexpr index_t HI = 14;
    constexpr index_t WI = 14;
    constexpr index_t K  = 2048;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 14x14
    constexpr index_t N  = 128;
    constexpr index_t C  = 1024;
    constexpr index_t HI = 14;
    constexpr index_t WI = 14;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 14x14, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 1024;
    constexpr index_t HI = 14;
    constexpr index_t WI = 14;
    constexpr index_t K  = 512;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 1
    // 3x3, 28x28
    constexpr index_t N  = 128;
    constexpr index_t C  = 128;
    constexpr index_t HI = 28;
    constexpr index_t WI = 28;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 1
    // 3x3, 14x14
    constexpr index_t N  = 128;
    constexpr index_t C  = 256;
    constexpr index_t HI = 14;
    constexpr index_t WI = 14;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    // 1x1, 56x56, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 256;
    constexpr index_t HI = 56;
    constexpr index_t WI = 56;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 7x7, 230x230 stride=2
    constexpr index_t N  = 128;
    constexpr index_t C  = 3;
    constexpr index_t HI = 230;
    constexpr index_t WI = 230;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 7;
    constexpr index_t X  = 7;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 28x28, stride = 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t HI = 28;
    constexpr index_t WI = 28;
    constexpr index_t K  = 1024;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 28x28, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t HI = 28;
    constexpr index_t WI = 28;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 1
    // 1x1, 7x7
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t HI = 7;
    constexpr index_t WI = 7;
    constexpr index_t K  = 2048;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 7x7
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t HI = 7;
    constexpr index_t WI = 7;
    constexpr index_t K  = 512;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    // 1x1, 56x56
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t HI = 56;
    constexpr index_t WI = 56;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 56x56
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t HI = 56;
    constexpr index_t WI = 56;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#endif

    auto in_nchw_desc  = make_native_tensor_descriptor_packed(Sequence<N, C, HI, WI>{});
    auto wei_kcyx_desc = make_native_tensor_descriptor_packed(Sequence<K, C, Y, X>{});
    auto out_nkhw_desc = get_convolution_output_default_4d_tensor_descriptor(
        in_nchw_desc, wei_kcyx_desc, ConvStrides{}, ConvDilations{}, LeftPads{}, RightPads{});

    ostream_tensor_descriptor(in_nchw_desc, std::cout << "in_nchw_desc: ");
    ostream_tensor_descriptor(wei_kcyx_desc, std::cout << "wei_kcyx_desc: ");
    ostream_tensor_descriptor(out_nkhw_desc, std::cout << "out_nkhw_desc: ");
    print_array("LeftPads", to_multi_index(LeftPads{}));
    print_array("RightPads", to_multi_index(RightPads{}));
    print_array("ConvStrides", to_multi_index(ConvStrides{}));
    print_array("ConvDilations", to_multi_index(ConvDilations{}));

#if 1
    using in_data_t                  = float;
    constexpr index_t in_vector_size = 1;
    using acc_data_t                 = float;
    using out_data_t                 = float;
#elif 0
    using in_data_t                  = float;
    constexpr index_t in_vector_size = 1;
    using acc_data_t                 = float;
    using out_data_t                 = int8_t;
#elif 1
    using in_data_t                  = int8_t;
    constexpr index_t in_vector_size = 16;
    using acc_data_t                 = int32_t;
    using out_data_t                 = int8_t;
#endif

    Tensor<in_data_t> in_nchw(make_HostTensorDescriptor(in_nchw_desc));
    Tensor<in_data_t> wei_kcyx(make_HostTensorDescriptor(wei_kcyx_desc));
    Tensor<out_data_t> out_nkhw_host(make_HostTensorDescriptor(out_nkhw_desc));
    Tensor<out_data_t> out_nkhw_device(make_HostTensorDescriptor(out_nkhw_desc));

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(argc != 4)
    {
        printf("arg1: do_verification, arg2: do_log, arg3: nrepeat\n");
        exit(1);
    }

    bool do_verification = atoi(argv[1]);
    bool do_log          = atoi(argv[2]);
    index_t nrepeat      = atoi(argv[3]);

    if(do_verification)
    {
#if 0
        in_nchw.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        wei_kcyx.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
#elif 0
        in_nchw.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        wei_kcyx.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
#elif 0
        in_nchw.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        wei_kcyx.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
#elif 1
        in_nchw.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        wei_kcyx.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
#elif 0
        in_nchw.GenerateTensorValue(GeneratorTensor_2{1, 5}, num_thread);

        auto gen_wei = [](auto... is) {
            return GeneratorTensor_2{1, 5}(is...) * GeneratorTensor_Checkboard{}(is...);
        };
        wei_kcyx.GenerateTensorValue(gen_wei, num_thread);
#endif
    }

#if 0
    device_convolution_forward_implicit_gemm_v4r1_nchw_kcyx_nkhw(in_nchw_desc,
                                                                 in_nchw,
                                                                 wei_kcyx_desc,
                                                                 wei_kcyx,
                                                                 out_nkhw_desc,
                                                                 out_nkhw_device,
                                                                 ConvStrides{},
                                                                 ConvDilations{},
                                                                 LeftPads{},
                                                                 RightPads{},
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
                                                                 LeftPads{},
                                                                 RightPads{},
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
                                                                 LeftPads{},
                                                                 RightPads{},
                                                                 nrepeat);
#elif 0
    device_dynamic_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw<in_data_t,
                                                                         in_vector_size,
                                                                         acc_data_t,
                                                                         out_data_t>(
        in_nchw_desc,
        in_nchw,
        wei_kcyx_desc,
        wei_kcyx,
        out_nkhw_desc,
        out_nkhw_device,
        ConvStrides{},
        ConvDilations{},
        LeftPads{},
        RightPads{},
        nrepeat);
#elif 1
    device_dynamic_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk<in_data_t,
                                                                         in_vector_size,
                                                                         acc_data_t,
                                                                         out_data_t>

        (in_nchw_desc,
         in_nchw,
         wei_kcyx_desc,
         wei_kcyx,
         out_nkhw_desc,
         out_nkhw_device,
         ConvStrides{},
         ConvDilations{},
         LeftPads{},
         RightPads{},
         nrepeat);
#elif 1
    device_dynamic_convolution_forward_implicit_gemm_v5r1_nchw_kcyx_nkhw<in_data_t,
                                                                         in_vector_size,
                                                                         acc_data_t,
                                                                         out_data_t>(
        in_nchw_desc,
        in_nchw,
        wei_kcyx_desc,
        wei_kcyx,
        out_nkhw_desc,
        out_nkhw_device,
        ConvStrides{},
        ConvDilations{},
        LeftPads{},
        RightPads{},
        nrepeat);
#endif

    if(do_verification)
    {
        host_direct_convolution(in_nchw,
                                wei_kcyx,
                                out_nkhw_host,
                                ConvStrides{},
                                ConvDilations{},
                                LeftPads{},
                                RightPads{});

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
