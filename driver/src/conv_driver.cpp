#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "config.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "print_array.hpp"
#include "print_sequence.hpp"
#include "device.hpp"
#include "tensor_generator.hpp"
#include "conv_common.hpp"
#include "host_conv.hpp"
#include "device_tensor.hpp"
//#include "device_convolution_direct_v2_nchw_kcyx_nkhw.hpp"
//#include "device_convolution_implicit_gemm_v1_chwn_cyxk_khwn.hpp"
//#include "device_convolution_implicit_gemm_v1_chwn_cyxk_khwn_padded.hpp"
//#include "device_convolution_implicit_gemm_v1_nchw_cyxk_nkhw.hpp"
//#include "device_convolution_implicit_gemm_v2_chwn_cyxk_khwn.hpp"
//#include "device_convolution_implicit_gemm_v3_nchw_cyxk_nkhw.hpp"
#include "device_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_deprecated.hpp"
#include "device_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw.hpp"
//#include "device_convolution_implicit_gemm_v4r2_nchw_kcyx_nkhw.hpp"
//#include "device_convolution_implicit_gemm_v4r3_nchw_kcyx_nkhw.hpp"
#include "device_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw_deprecated.hpp"
#include "device_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw.hpp"

int main(int argc, char* argv[])
{
    using namespace ck;

#if 0
    // 1x1
    constexpr index_t N  = 256;
    constexpr index_t C  = 1024;
    constexpr index_t HI = 8;
    constexpr index_t WI = 8;
    constexpr index_t K  = 1024;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x7
    constexpr index_t N  = 128;
    constexpr index_t C  = 1024;
    constexpr index_t HI = 17;
    constexpr index_t WI = 17;
    constexpr index_t K  = 1024;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 7;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 3>;
    using RightPads = Sequence<0, 3>;
#elif 0
    // 3x3, 34x34
    constexpr index_t N  = 64;
    constexpr index_t C  = 256;
    constexpr index_t HI = 34;
    constexpr index_t WI = 34;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3 filter, 2x2 stride, 35x35 input, 17x17 output
    constexpr index_t N  = 128;
    constexpr index_t C  = 128;
    constexpr index_t HI = 35;
    constexpr index_t WI = 35;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 8x8 image
    // cudnn@V100 68%, ck@V100 72%, ck@P100 52%, ck@VII 42%
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
    // cudnn@V100 77%, ck@V100 76%, ck@P100 79%, ck@VII 51%
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
    // cudnn@V100 82%, ck@V100 76%, ck@P100 67%, ck@VII 64%
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
    // cudnn@V100 83%, ck@V100 75%, ck@P100 78%, ck@VII 65%
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
    // cudnn@V100 62%, ck@V100 68%, ck@P100 70%, ck@VII 50%
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
    // cudnn@V100 74%, ck@V100 57%, ck@P100 78%, ck@VII 61%
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
    // cudnn@V100 86%, ck@V100 84%, ck@P100 80%, ck@VII 69%
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
    // cudnn@V100 71%, ck@V100 55%, ck@P100 70%, ck@VII 62%
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
    // cudnn@V100 81%, ck@V100 76%, ck@P100 70%, ck@VII 76%
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
    // cudnn@V100 73%, ck@V100 71%, ck@P100 70%, ck@VII 64%
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
    // cudnn@V100 73%, ck@V100 72%, ck@P100 79%, ck@VII 75%
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
    // cudnn@V100 49%, ck@V100 50%, ck@P100 61%, ck@VII 52%
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
    // cudnn@V100 90%, ck@V100 93%, ck@P100 83%, ck@VII 81%
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
#elif 1
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
#endif

    auto in_nchw_desc  = make_ConstantTensorDescriptor_packed(Sequence<N, C, HI, WI>{});
    auto wei_kcyx_desc = make_ConstantTensorDescriptor_packed(Sequence<K, C, Y, X>{});
    auto out_nkhw_desc = get_convolution_output_default_4d_tensor_descriptor_deprecated(
        in_nchw_desc, wei_kcyx_desc, ConvStrides{}, ConvDilations{}, LeftPads{}, RightPads{});

    ostream_ConstantTensorDescriptor(in_nchw_desc, std::cout << "in_nchw_desc: ");
    ostream_ConstantTensorDescriptor(wei_kcyx_desc, std::cout << "wei_kcyx_desc: ");
    ostream_ConstantTensorDescriptor(out_nkhw_desc, std::cout << "out_nkhw_desc: ");
    print_sequence("LeftPads", LeftPads{});
    print_sequence("RightPads", RightPads{});
    print_sequence("ConvStrides", ConvStrides{});
    print_sequence("ConvDilations", ConvDilations{});

    using in_data_t  = float;
    using out_data_t = float;
    Tensor<in_data_t> in_nchw(make_TensorDescriptor(in_nchw_desc));
    Tensor<in_data_t> wei_kcyx(make_TensorDescriptor(wei_kcyx_desc));
    Tensor<out_data_t> out_nkhw_host(make_TensorDescriptor(out_nkhw_desc));
    Tensor<out_data_t> out_nkhw_device(make_TensorDescriptor(out_nkhw_desc));

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(argc != 3)
    {
        printf("arg1: do_verification, arg2: nrepeat\n");
        exit(1);
    }

    bool do_verification = atoi(argv[1]);
    index_t nrepeat      = atoi(argv[2]);

    if(do_verification)
    {
#if 0
        in_nchw.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        wei_kcyx.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
#elif 0
        in_nchw.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        wei_kcyx.GenerateTensorValue(GeneratorTensor_3{}, num_thread);
#elif 0
        in_nchw.GenerateTensorValue(GeneratorTensor_3{}, num_thread);
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
    device_convolution_direct_v2_nchw_kcyx_nkhw
        (in_nchw_desc, in_nchw, wei_kcyx_desc, wei_kcyx, out_nkhw_desc, out_nkhw_device, nrepeat);
#elif 0
    device_convolution_implicit_gemm_v1_chwn_cyxk_khwn(
        in_nchw_desc, in_nchw, wei_kcyx_desc, wei_kcyx, out_nkhw_desc, out_nkhw_device, nrepeat);
#elif 0
    device_convolution_implicit_gemm_v1_chwn_cyxk_khwn_padded(in_nchw_desc,
                                                              in_nchw,
                                                              wei_kcyx_desc,
                                                              wei_kcyx,
                                                              out_nkhw_desc,
                                                              out_nkhw_device,
                                                              LeftPads{},
                                                              RightPads{},
                                                              nrepeat);
#elif 0
    device_convolution_implicit_gemm_v1_nchw_cyxk_nkhw(
        in_nchw_desc, in_nchw, wei_kcyx_desc, wei_kcyx, out_nkhw_desc, out_nkhw_device, nrepeat);
#elif 0
    device_convolution_implicit_gemm_v2_chwn_cyxk_khwn(
        in_nchw_desc, in_nchw, wei_kcyx_desc, wei_kcyx, out_nkhw_desc, out_nkhw_device, nrepeat);
#elif 0
    device_convolution_implicit_gemm_v3_nchw_cyxk_nkhw(
        (in_nchw_desc, in_nchw, wei_kcyx_desc, wei_kcyx, out_nkhw_desc, out_nkhw_device, nrepeat);
#elif 0
    device_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_deprecated(in_nchw_desc,
                                                                    in_nchw,
                                                                    wei_kcyx_desc,
                                                                    wei_kcyx,
                                                                    out_nkhw_desc,
                                                                    out_nkhw_device,
                                                                    ConvStrides{},
                                                                    ConvDilations{},
                                                                    nrepeat);
#elif 0
    device_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw(in_nchw_desc,
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
    device_convolution_implicit_gemm_v4r2_nchw_kcyx_nkhw(in_nchw_desc,
                                                         in_nchw,
                                                         wei_kcyx_desc,
                                                         wei_kcyx,
                                                         out_nkhw_desc,
                                                         out_nkhw_device,
                                                         ConvStrides{},
                                                         ConvDilations{},
                                                         nrepeat);
#elif 0
    device_convolution_implicit_gemm_v4r3_nchw_kcyx_nkhw(in_nchw_desc,
                                                         in_nchw,
                                                         wei_kcyx_desc,
                                                         wei_kcyx,
                                                         out_nkhw_desc,
                                                         out_nkhw_device,
                                                         ConvStrides{},
                                                         ConvDilations{},
                                                         nrepeat);
#elif 0
    device_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw_deprecated(in_nchw_desc,
                                                                    in_nchw,
                                                                    wei_kcyx_desc,
                                                                    wei_kcyx,
                                                                    out_nkhw_desc,
                                                                    out_nkhw_device,
                                                                    ConvStrides{},
                                                                    ConvDilations{},
                                                                    nrepeat);
#elif 1
    device_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw(in_nchw_desc,
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
#if 1
        if(Y == 3 && X == 3 && ConvStrides{}[0] == 1 && ConvStrides{}[1] == 1 &&
           ConvDilations{}[0] == 1 && ConvDilations{}[1] == 1)
        {
            host_winograd_3x3_convolution(
                in_nchw, wei_kcyx, out_nkhw_host, LeftPads{}, RightPads{});
        }
        else
#endif
        {
            host_direct_convolution(in_nchw,
                                    wei_kcyx,
                                    out_nkhw_host,
                                    ConvStrides{},
                                    ConvDilations{},
                                    LeftPads{},
                                    RightPads{});
        }
        check_error(out_nkhw_host, out_nkhw_device);

#if 0
        LogRange(std::cout << "in_nchw : ", in_nchw.mData, ",") << std::endl;
        LogRange(std::cout << "wei_kcyx: ", wei_kcyx.mData, ",") << std::endl;
        LogRange(std::cout << "out_nkhw_host  : ", out_nkhw_host.mData, ",") << std::endl;
        LogRange(std::cout << "out_nkhw_device: ", out_nkhw_device.mData, ",") << std::endl;
#endif
    }
}
