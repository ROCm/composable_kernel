#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "config.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "print_array.hpp"
#include "print_sequence.hpp"
#include "device.hpp"
#include "tensor_generator.hpp"
#include "device_tensor.hpp"
#include "conv_common.hpp"
#include "host_conv_bwd_data.hpp"
#include "device_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw.hpp"
#include "device_convolution_backward_data_implicit_gemm_v1r2_nchw_kcyx_nkhw.hpp"
#include "device_convolution_backward_data_implicit_gemm_v2r1_nchw_kcyx_nkhw.hpp"

int main(int argc, char* argv[])
{
    using namespace ck;

#if 0
    constexpr index_t N  = 8;
    constexpr index_t C  = 128;
    constexpr index_t HI = 16;
    constexpr index_t WI = 16;
    constexpr index_t K  = 8;
    constexpr index_t Y  = 2;
    constexpr index_t X  = 2;

    using ConvStrides   = Sequence<4, 4>;
    using ConvDilations = Sequence<2, 2>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 34x34
    constexpr index_t N  = 64;
    constexpr index_t C  = 256;
    constexpr index_t HI = 34;
    constexpr index_t WI = 34;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 8x8 image
    constexpr index_t N  = 64;
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
    // 1x1 filter, 8x8 image
    constexpr index_t N  = 128;
    constexpr index_t C  = 2048;
    constexpr index_t HI = 8;
    constexpr index_t WI = 8;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 7x7 image
    constexpr index_t N  = 128;
    constexpr index_t C  = 832;
    constexpr index_t HI = 7;
    constexpr index_t WI = 7;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 8x8 image
    constexpr index_t N  = 128;
    constexpr index_t C  = 1280;
    constexpr index_t HI = 8;
    constexpr index_t WI = 8;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 14x14 image
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t HI = 14;
    constexpr index_t WI = 14;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 8x8 image
    constexpr index_t N  = 64;
    constexpr index_t C  = 1536;
    constexpr index_t HI = 8;
    constexpr index_t WI = 8;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 28x28 image
    constexpr index_t N  = 128;
    constexpr index_t C  = 256;
    constexpr index_t HI = 28;
    constexpr index_t WI = 28;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 7x7 image
    constexpr index_t N  = 128;
    constexpr index_t C  = 832;
    constexpr index_t HI = 7;
    constexpr index_t WI = 7;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 17x17 input
    constexpr index_t N  = 128;
    constexpr index_t C  = 768;
    constexpr index_t HI = 17;
    constexpr index_t WI = 17;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 14x14 image
    constexpr index_t N  = 128;
    constexpr index_t C  = 528;
    constexpr index_t HI = 14;
    constexpr index_t WI = 14;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 14x14 image
    constexpr index_t N  = 128;
    constexpr index_t C  = 528;
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
    // 1x1 filter, 7x7 image
    constexpr index_t N  = 128;
    constexpr index_t C  = 832;
    constexpr index_t HI = 7;
    constexpr index_t WI = 7;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3 filter, 2x2 stride, 35x35 input, 17x17 output
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
    // 5x5 filter, 2x2 pad, 7x7 input
    constexpr index_t N  = 128;
    constexpr index_t C  = 48;
    constexpr index_t HI = 7;
    constexpr index_t WI = 7;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 5;
    constexpr index_t X  = 5;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<2, 2>;
    using RightPads = Sequence<2, 2>;
#elif 0
    // 7x1 filter, 3x0 pad, 17x17 input
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
#elif 1
    // 1x7 filter, 0x3 pad, 17x17 input
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
#endif

    constexpr auto in_nchw_desc  = make_native_tensor_descriptor_packed(Sequence<N, C, HI, WI>{});
    constexpr auto wei_kcyx_desc = make_native_tensor_descriptor_packed(Sequence<K, C, Y, X>{});
    constexpr auto out_nkhw_desc = get_convolution_output_default_4d_tensor_descriptor(
        in_nchw_desc, wei_kcyx_desc, ConvStrides{}, ConvDilations{}, LeftPads{}, RightPads{});

    ostream_ConstantTensorDescriptor(in_nchw_desc, std::cout << "in_nchw_desc: ");
    ostream_ConstantTensorDescriptor(wei_kcyx_desc, std::cout << "wei_kcyx_desc: ");
    ostream_ConstantTensorDescriptor(out_nkhw_desc, std::cout << "out_nkhw_desc: ");
    print_sequence("LeftPads", LeftPads{});
    print_sequence("LeftPads", LeftPads{});
    print_sequence("RightPads", RightPads{});
    print_sequence("ConvStrides", ConvStrides{});
    print_sequence("ConvDilations", ConvDilations{});

    Tensor<float> in_nchw_device(make_TensorDescriptor(in_nchw_desc));
    Tensor<float> in_nchw_host(make_TensorDescriptor(in_nchw_desc));
    Tensor<float> wei_kcyx(make_TensorDescriptor(wei_kcyx_desc));
    Tensor<float> out_nkhw(make_TensorDescriptor(out_nkhw_desc));

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(argc != 3)
    {
        printf("arg1: do_verification, arg2: nrepeat\n");
        exit(1);
    }

    bool do_verification = atoi(argv[1]);
    std::size_t nrepeat  = atoi(argv[2]);

    if(do_verification)
    {
#if 0
        wei_kcyx.GenerateTensorValue(GeneratorTensor_1{1}, num_thread);
        out_nkhw.GenerateTensorValue(GeneratorTensor_1{1}, num_thread);
#else
        wei_kcyx.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        out_nkhw.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
#endif
    }

#if 0
    device_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw
#elif 0
    device_convolution_backward_data_implicit_gemm_v1r2_nchw_kcyx_nkhw
#else
    device_convolution_backward_data_implicit_gemm_v2r1_nchw_kcyx_nkhw
#endif
    (in_nchw_desc,
     in_nchw_device,
     wei_kcyx_desc,
     wei_kcyx,
     out_nkhw_desc,
     out_nkhw,
     ConvStrides{},
     ConvDilations{},
     LeftPads{},
     RightPads{},
     nrepeat);

    if(do_verification)
    {
        host_direct_convolution_backward_data(in_nchw_host,
                                              wei_kcyx,
                                              out_nkhw,
                                              ConvStrides{},
                                              ConvDilations{},
                                              LeftPads{},
                                              RightPads{});

        check_error(in_nchw_host, in_nchw_device);

#if 0
        LogRange(std::cout << "out_nkhw : ", out_nkhw.mData, ",") << std::endl;
        LogRange(std::cout << "wei_kcyx : ", wei_kcyx.mData, ",") << std::endl;
        LogRange(std::cout << "in_nchw_host : ", in_nchw_host.mData, ",") << std::endl;
        LogRange(std::cout << "in_nchw_device : ", in_nchw_device.mData, ",") << std::endl;
#endif
    }
}
