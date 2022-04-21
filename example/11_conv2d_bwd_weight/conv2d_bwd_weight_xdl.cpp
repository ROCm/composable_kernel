#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

#include "check_err.hpp"
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "tensor_layout.hpp"
#include "element_wise_operation.hpp"
#include "device_conv2d_backward_weight_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp"
#include "reference_conv_backward_weight.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InLayout  = ck::tensor_layout::convolution::NHWC;
using WeiLayout = ck::tensor_layout::convolution::KYXC;
using OutLayout = ck::tensor_layout::convolution::NHWK;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

// clang-format off
using DeviceConvBwdWeightInstance = ck::tensor_operation::device::
    DeviceConv2dBwdWeightXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        InDataType,                       // InDataType
        WeiDataType,                      // WeiDataType
        OutDataType,                      // OutDataType
        AccDataType,                      // AccDataType
        InElementOp,                      // InElementwiseOperation
        WeiElementOp,                     // WeiElementwiseOperation
        OutElementOp,                     // OutElementwiseOperation
        256,                              // BlockSize
        128,                              // MPerBlock
        128,                              // NPerBlock
        4,                                // K0PerBlock
        8,                                // K1
        32,                               // MPerXdl
        32,                               // NPerXdl
        2,                                // MXdlPerWave
        2,                                // NXdlPerWave
        S<1, 4, 16, 4>,                   // ABlockTransferThreadClusterLengths_K0_M_K1
        S<0, 3, 1, 2>,                    // ABlockTransferThreadClusterArrangeOrder
        S<0, 2, 1, 3>,                    // ABlockTransferSrcAccessOrder
        2,                                // ABlockTransferSrcVectorDim
        8,                                // ABlockTransferSrcScalarPerVector
        2,                                // ABlockTransferDstScalarPerVector_K1
        true,                             // ABlockLdsAddExtraM
        S<1, 4, 16, 4>,                   // BBlockTransferThreadClusterLengths_K0_N_K1
        S<0, 3, 1, 2>,                    // BBlockTransferThreadClusterArrangeOrder
        S<0, 2, 1, 3>,                    // BBlockTransferSrcAccessOrder
        2,                                // BBlockTransferSrcVectorDim
        8,                                // BBlockTransferSrcScalarPerVector
        2,                                // BBlockTransferDstScalarPerVector_K1
        true,                             // BBlockLdsAddExtraN
        1,                                // CShuffleMXdlPerWavePerShuffle
        1,                                // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 4>,                   // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8>;                               // CBlockTransferScalarPerVector_NWaveNPerXdl
// clang-format on

using ReferenceConvBwdWeightInstance =
    ck::tensor_operation::host::ReferenceConvBwdWeight<InDataType,
                                                       WeiDataType,
                                                       OutDataType,
                                                       InElementOp,
                                                       WeiElementOp,
                                                       OutElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = 0;
    int init_method      = 0;
    int nrepeat          = 5;
    int do_log           = 0;
    int split_k          = 4;

    // Conv shape
    ck::index_t N               = 128;
    ck::index_t K               = 256;
    ck::index_t C               = 1024;
    ck::index_t Y               = 3;
    ck::index_t X               = 3;
    ck::index_t Hi              = 14;
    ck::index_t Wi              = 14;
    ck::index_t conv_stride_h   = 2;
    ck::index_t conv_stride_w   = 2;
    ck::index_t conv_dilation_h = 1;
    ck::index_t conv_dilation_w = 1;
    ck::index_t in_left_pad_h   = 0;
    ck::index_t in_left_pad_w   = 0;
    ck::index_t in_right_pad_h  = 0;
    ck::index_t in_right_pad_w  = 0;

    if(argc == 6)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
        do_log          = std::stoi(argv[4]);
        split_k         = std::stoi(argv[5]);
    }
    else if(argc == 21)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
        do_log          = std::stoi(argv[4]);
        split_k         = std::stoi(argv[5]);

        N               = std::stoi(argv[6]);
        K               = std::stoi(argv[7]);
        C               = std::stoi(argv[8]);
        Y               = std::stoi(argv[9]);
        X               = std::stoi(argv[10]);
        Hi              = std::stoi(argv[11]);
        Wi              = std::stoi(argv[12]);
        conv_stride_h   = std::stoi(argv[13]);
        conv_stride_w   = std::stoi(argv[14]);
        conv_dilation_h = std::stoi(argv[15]);
        conv_dilation_w = std::stoi(argv[16]);
        in_left_pad_h   = std::stoi(argv[17]);
        in_left_pad_w   = std::stoi(argv[18]);
        in_right_pad_h  = std::stoi(argv[19]);
        in_right_pad_w  = std::stoi(argv[20]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: run kernel # of times (>1)\n");
        printf("arg4: is show log (0=no, 1=yes)\n");
        printf("arg5: split-k \n");
        printf("arg6 to 19: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
               "RightPx\n");
        exit(0);
    }

    const ck::index_t YEff = (Y - 1) * conv_dilation_h + 1;
    const ck::index_t XEff = (X - 1) * conv_dilation_w + 1;

    const ck::index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - YEff) / conv_stride_h + 1;
    const ck::index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;

    const std::vector<ck::index_t> conv_filter_strides{{conv_stride_h, conv_stride_w}};
    const std::vector<ck::index_t> conv_filter_dilations{{conv_dilation_h, conv_dilation_w}};
    const std::vector<ck::index_t> input_left_pads{{in_left_pad_h, in_left_pad_w}};
    const std::vector<ck::index_t> input_right_pads{{in_right_pad_h, in_right_pad_w}};

    // tensor layout
    auto f_host_tensor_descriptor = [](std::size_t N_,
                                       std::size_t C_,
                                       std::size_t H,
                                       std::size_t W,
                                       auto layout) {
        if constexpr(ck::is_same<decltype(layout), ck::tensor_layout::convolution::NCHW>::value ||
                     ck::is_same<decltype(layout), ck::tensor_layout::convolution::KCYX>::value ||
                     ck::is_same<decltype(layout), ck::tensor_layout::convolution::NKHW>::value)
        {
            return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                        std::vector<std::size_t>({C_ * H * W, H * W, W, 1}));
        }
        else if constexpr(ck::is_same<decltype(layout),
                                      ck::tensor_layout::convolution::NHWC>::value ||
                          ck::is_same<decltype(layout),
                                      ck::tensor_layout::convolution::KYXC>::value ||
                          ck::is_same<decltype(layout),
                                      ck::tensor_layout::convolution::NHWK>::value)
        {
            return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                        std::vector<std::size_t>({C_ * H * W, 1, W * C_, C_}));
        }
    };

    Tensor<InDataType> in_n_c_hi_wi(f_host_tensor_descriptor(N, C, Hi, Wi, InLayout{}));
    Tensor<WeiDataType> wei_k_c_y_x_host_result(f_host_tensor_descriptor(K, C, Y, X, WeiLayout{}));
    Tensor<WeiDataType> wei_k_c_y_x_device_result(
        f_host_tensor_descriptor(K, C, Y, X, WeiLayout{}));
    Tensor<OutDataType> out_n_k_ho_wo(f_host_tensor_descriptor(N, K, Ho, Wo, OutLayout{}));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi.mDesc << std::endl;
    std::cout << "wei_k_c_y_x: " << wei_k_c_y_x_host_result.mDesc << std::endl;
    std::cout << "out_n_k_ho_wo: " << out_n_k_ho_wo.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        out_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        break;
    default:
        in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_1<InDataType>{1});
        out_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_1<OutDataType>{1});
    }
    wei_k_c_y_x_device_result.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{0});

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) *
                             wei_k_c_y_x_device_result.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_n_k_ho_wo.mDesc.GetElementSpace());

    in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
    out_device_buf.ToDevice(out_n_k_ho_wo.mData.data());
    wei_device_buf.ToDevice(wei_k_c_y_x_device_result.mData.data());

    // do GEMM
    auto conv     = DeviceConvBwdWeightInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                      static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                      static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                      N,
                                      K,
                                      C,
                                      std::vector<ck::index_t>{{Hi, Wi}},
                                      std::vector<ck::index_t>{{Y, X}},
                                      std::vector<ck::index_t>{{Ho, Wo}},
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      InElementOp{},
                                      WeiElementOp{},
                                      OutElementOp{},
                                      split_k);

    if(!conv.IsSupportedArgument(argument))
    {
        std::cout << "wrong! device_conv with the specified compilation parameters does "
                     "not support this Conv problem"
                  << std::endl;
        return 1;
    }

    float ave_time = invoker.Run(argument, nrepeat);

    std::size_t flop = std::size_t(2) * N * K * Ho * Wo * C * Y * X;

    std::size_t num_btype = sizeof(InDataType) * (N * C * Hi * Wi) +
                            sizeof(WeiDataType) * (K * C * Y * X) +
                            sizeof(OutDataType) * (N * K * Ho * Wo);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        auto ref_conv    = ReferenceConvBwdWeightInstance{};
        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(in_n_c_hi_wi,
                                                  wei_k_c_y_x_host_result,
                                                  out_n_k_ho_wo,
                                                  conv_filter_strides,
                                                  conv_filter_dilations,
                                                  input_left_pads,
                                                  input_right_pads,
                                                  InElementOp{},
                                                  WeiElementOp{},
                                                  OutElementOp{});

        ref_invoker.Run(ref_argument);

        wei_device_buf.FromDevice(wei_k_c_y_x_device_result.mData.data());

        if(do_log)
        {
            LogRangeAsType<float>(std::cout << "out: ", out_n_k_ho_wo.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "in : ", in_n_c_hi_wi.mData, ",") << std::endl;
            LogRangeAsType<float>(
                std::cout << "wei_device(after): ", wei_k_c_y_x_device_result.mData, ",")
                << std::endl;
            LogRangeAsType<float>(std::cout << "wei_host  : ", wei_k_c_y_x_host_result.mData, ",")
                << std::endl;
        }
        ck::utils::check_err(wei_k_c_y_x_device_result.mData, wei_k_c_y_x_host_result.mData);
    }
}
