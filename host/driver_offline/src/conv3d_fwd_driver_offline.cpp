#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
//#include <half.hpp>
#include "config.hpp"
#include "debug.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "conv_common.hpp"
#include "host_conv.hpp"
#include "device_tensor.hpp"
#include "device_convolution3d_forward_implicit_gemm_v4r4r4_xdlops_ndhwc_kzyxc_ndhwk.hpp"

#define USE_DYNAMIC_MODE 1
#define USE_CONV3D_FWD_V4R4R4_XDL_NHWC 1

enum Conv3dTensorLayout
{
    NDHWC = 0
};

enum ConvForwardAlgo
{
    V4R4R4XDLNDHWC = 0 // 0
};

int main(int argc, char* argv[])
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};

#if USE_DYNAMIC_MODE
    // dynamic mode
    if(argc != 28)
    {
        printf("argc = %d\n", argc);
        printf("arg1 to 6: layout, algo, do_verification, init_method, do_log, nrepeat\n");
        printf("rest: N, K, C, Z, Y, X, Di, Hi, Wi, Sz, Sy, Sx, Dz, Dy, Dx, LeftPz, LeftPy, "
               "LeftPx, RightPz, RightPy, RightPx\n");
        exit(1);
    }

    const Conv3dTensorLayout layout = static_cast<Conv3dTensorLayout>(std::stoi(argv[1]));
    const ConvForwardAlgo algo      = static_cast<ConvForwardAlgo>(std::stoi(argv[2]));
    const bool do_verification      = std::stoi(argv[3]);
    const int init_method           = std::stoi(argv[4]);
    const bool do_log               = std::stoi(argv[5]);
    const int nrepeat               = std::stoi(argv[6]);

    const index_t N  = std::stoi(argv[7]);
    const index_t K  = std::stoi(argv[8]);
    const index_t C  = std::stoi(argv[9]);
    const index_t Z  = std::stoi(argv[10]);
    const index_t Y  = std::stoi(argv[11]);
    const index_t X  = std::stoi(argv[12]);
    const index_t Di = std::stoi(argv[13]);
    const index_t Hi = std::stoi(argv[14]);
    const index_t Wi = std::stoi(argv[15]);

    const index_t conv_stride_d   = std::stoi(argv[16]);
    const index_t conv_stride_h   = std::stoi(argv[17]);
    const index_t conv_stride_w   = std::stoi(argv[18]);
    const index_t conv_dilation_d = std::stoi(argv[19]);
    const index_t conv_dilation_h = std::stoi(argv[20]);
    const index_t conv_dilation_w = std::stoi(argv[21]);
    const index_t in_left_pad_d   = std::stoi(argv[22]);
    const index_t in_left_pad_h   = std::stoi(argv[23]);
    const index_t in_left_pad_w   = std::stoi(argv[24]);
    const index_t in_right_pad_d  = std::stoi(argv[25]);
    const index_t in_right_pad_h  = std::stoi(argv[26]);
    const index_t in_right_pad_w  = std::stoi(argv[27]);

    const index_t ZEff = (Z - 1) * conv_dilation_d + 1;
    const index_t YEff = (Y - 1) * conv_dilation_h + 1;
    const index_t XEff = (X - 1) * conv_dilation_w + 1;

    const index_t Do = (Di + in_left_pad_d + in_right_pad_d - ZEff) / conv_stride_d + 1;
    const index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - YEff) / conv_stride_h + 1;
    const index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;
#else
    // static mode
    if(argc < 7)
    {
        printf("arg1 to 6: layout, algo, do_verification, init_method, do_log, nrepeat\n");
        exit(1);
    }

    const Conv3dTensorLayout layout = static_cast<Conv3dTensorLayout>(std::stoi(argv[1]));
    const ConvForwardAlgo algo      = static_cast<ConvForwardAlgo>(std::stoi(argv[2]));
    const bool do_verification      = std::stoi(argv[3]);
    const int init_method           = std::stoi(argv[4]);
    const bool do_log               = std::stoi(argv[5]);
    const int nrepeat               = std::stoi(argv[6]);

    constexpr auto N  = Number<128>{};
    constexpr auto C  = Number<192>{};
    constexpr auto Di = Number<71>{};
    constexpr auto Hi = Number<71>{};
    constexpr auto Wi = Number<71>{};
    constexpr auto K  = Number<256>{};
    constexpr auto Z  = Number<3>{};
    constexpr auto Y  = Number<3>{};
    constexpr auto X  = Number<3>{};

    constexpr auto conv_stride_d   = I2;
    constexpr auto conv_stride_h   = I2;
    constexpr auto conv_stride_w   = I2;
    constexpr auto conv_dilation_d = I1;
    constexpr auto conv_dilation_h = I1;
    constexpr auto conv_dilation_w = I1;
    constexpr auto in_left_pad_d   = I1;
    constexpr auto in_left_pad_h   = I1;
    constexpr auto in_left_pad_w   = I1;
    constexpr auto in_right_pad_d  = I1;
    constexpr auto in_right_pad_h  = I1;
    constexpr auto in_right_pad_w  = I1;

    constexpr auto ZEff = (Z - I1) * conv_dilation_d + I1;
    constexpr auto YEff = (Y - I1) * conv_dilation_h + I1;
    constexpr auto XEff = (X - I1) * conv_dilation_w + I1;

    constexpr auto Do = (Di + in_left_pad_d + in_right_pad_d - ZEff) / conv_stride_d + I1;
    constexpr auto Ho = (Hi + in_left_pad_h + in_right_pad_h - YEff) / conv_stride_h + I1;
    constexpr auto Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + I1;
#endif

#if 0
    using in_data_t  = float;
    using acc_data_t = float;
    using out_data_t = float;
#elif 1
    using in_data_t   = half_t;
    using acc_data_t  = float;
    using out_data_t  = half_t;
#elif 1
    using in_data_t  = int8_t;
    using acc_data_t = int32_t;
    using out_data_t = int8_t;
#endif

    std::vector<std::size_t> in_lengths_host(5), wei_lengths_host(5), out_lengths_host(5);

    if(layout == Conv3dTensorLayout::NDHWC)
    {
        in_lengths_host[0]  = static_cast<std::size_t>(N);
        in_lengths_host[1]  = static_cast<std::size_t>(Di);
        in_lengths_host[2]  = static_cast<std::size_t>(Hi);
        in_lengths_host[3]  = static_cast<std::size_t>(Wi);
        in_lengths_host[4]  = static_cast<std::size_t>(C);
        wei_lengths_host[0] = static_cast<std::size_t>(K);
        wei_lengths_host[1] = static_cast<std::size_t>(Z);
        wei_lengths_host[2] = static_cast<std::size_t>(Y);
        wei_lengths_host[3] = static_cast<std::size_t>(X);
        wei_lengths_host[4] = static_cast<std::size_t>(C);
        out_lengths_host[0] = static_cast<std::size_t>(N);
        out_lengths_host[1] = static_cast<std::size_t>(Do);
        out_lengths_host[2] = static_cast<std::size_t>(Ho);
        out_lengths_host[3] = static_cast<std::size_t>(Wo);
        out_lengths_host[4] = static_cast<std::size_t>(K);
    }
    else
    {
        std::runtime_error("wrong! not implemented");
    }

    Tensor<in_data_t> in(in_lengths_host);
    Tensor<in_data_t> wei(wei_lengths_host);
    Tensor<out_data_t> out_host(out_lengths_host);
    Tensor<out_data_t> out_device(out_lengths_host);

    std::cout << "layout: " << layout << std::endl;
    ostream_HostTensorDescriptor(in.mDesc, std::cout << "in: ");
    ostream_HostTensorDescriptor(wei.mDesc, std::cout << "wei: ");
    ostream_HostTensorDescriptor(out_host.mDesc, std::cout << "out: ");
    print_array("InLeftPads", make_tuple(in_left_pad_d, in_left_pad_h, in_left_pad_w));
    print_array("InRightPads", make_tuple(in_right_pad_d, in_right_pad_h, in_right_pad_w));
    print_array("ConvStrides", make_tuple(conv_stride_d, conv_stride_h, conv_stride_w));
    print_array("ConvDilations", make_tuple(conv_dilation_d, conv_dilation_h, conv_dilation_w));

    std::size_t num_thread = std::thread::hardware_concurrency();

    switch(init_method)
    {
    case 0:
        // no initialization
        break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_1<in_data_t>{}, num_thread);
        wei.GenerateTensorValue(GeneratorTensor_1<in_data_t>{}, num_thread);
        break;
    case 2:
        in.GenerateTensorValue(GeneratorTensor_1<in_data_t>{}, num_thread);
        wei.GenerateTensorValue(GeneratorTensor_2<in_data_t>{-5, 5}, num_thread);
        break;
    case 3:
        in.GenerateTensorValue(GeneratorTensor_2<in_data_t>{-5, 5}, num_thread);
        wei.GenerateTensorValue(GeneratorTensor_1<in_data_t>{}, num_thread);
        break;
    case 4:
        in.GenerateTensorValue(GeneratorTensor_2<in_data_t>{-5, 5}, num_thread);
        wei.GenerateTensorValue(GeneratorTensor_2<in_data_t>{-5, 5}, num_thread);
        break;
    case 5:
        in.GenerateTensorValue(GeneratorTensor_3<in_data_t>{0.0, 1.0}, num_thread);
        wei.GenerateTensorValue(GeneratorTensor_3<in_data_t>{-0.5, 0.5}, num_thread);
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_2<in_data_t>{1, 5}, num_thread);

        auto gen_wei = [](auto... is) {
            return GeneratorTensor_2<in_data_t>{1, 5}(is...) * GeneratorTensor_Checkboard{}(is...);
        };
        wei.GenerateTensorValue(gen_wei, num_thread);
    }

    auto f_make_for_device_ndhwc = [&]() {
        const auto in_lengths_dev   = make_tuple(N, Di, Hi, Wi, C);
        const auto wei_lengths_dev  = make_tuple(K, Z, Y, X, C);
        const auto out_lengths_dev  = make_tuple(N, Do, Ho, Wo, K);
        const auto conv_strides_dev = make_tuple(conv_stride_d, conv_stride_h, conv_stride_w);
        const auto conv_dilations_dev =
            make_tuple(conv_dilation_d, conv_dilation_h, conv_dilation_w);
        const auto in_left_pads_dev  = make_tuple(in_left_pad_d, in_left_pad_h, in_left_pad_w);
        const auto in_right_pads_dev = make_tuple(in_right_pad_d, in_right_pad_h, in_right_pad_w);

        return make_tuple(in_lengths_dev,
                          wei_lengths_dev,
                          out_lengths_dev,
                          conv_strides_dev,
                          conv_dilations_dev,
                          in_left_pads_dev,
                          in_right_pads_dev);
    };

#if USE_CONV3D_FWD_V4R4R4_XDL_NHWC
    if(algo == ConvForwardAlgo::V4R4R4XDLNDHWC)
    {
        if(layout != Conv3dTensorLayout::NDHWC)
        {
            throw std::runtime_error("wrong! layout");
        }

        const auto tmp = f_make_for_device_ndhwc();

        device_convolution3d_forward_implicit_gemm_v4r4r4_xdlops_ndhwc_kzyxc_ndhwk<in_data_t,
                                                                                   acc_data_t,
                                                                                   out_data_t>(
            tmp[I0],
            tmp[I1],
            tmp[I2],
            tmp[I3],
            tmp[I4],
            tmp[I5],
            tmp[I6],
            in,
            wei,
            out_device,
            nrepeat);
    }

    if(do_verification)
    {
        host_conv3d_ndhwc_kzyxc_ndhwk(in,
                                      wei,
                                      out_host,
                                      make_tuple(conv_stride_d, conv_stride_h, conv_stride_w),
                                      make_tuple(conv_dilation_d, conv_dilation_h, conv_dilation_w),
                                      make_tuple(in_left_pad_d, in_left_pad_h, in_left_pad_w),
                                      make_tuple(in_right_pad_d, in_right_pad_h, in_right_pad_w));

        check_error(out_host, out_device);

        if(do_log)
        {
            LogRangeAsType<float>(std::cout << "in : ", in.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "wei: ", wei.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "out_host  : ", out_host.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "out_device: ", out_device.mData, ",") << std::endl;
        }
    }
#endif
}
