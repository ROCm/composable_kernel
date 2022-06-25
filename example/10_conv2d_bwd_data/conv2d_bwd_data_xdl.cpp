// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/conv_util.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;
static constexpr auto ConvBwdDefault =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default;

using DeviceConvBwdDataInstance = ck::tensor_operation::device::
    DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        InDataType,     // InDataType
        WeiDataType,    // WeiDataType
        OutDataType,    // OutDataType
        AccDataType,    // AccDataType
        InElementOp,    // InElementwiseOperation
        WeiElementOp,   // WeiElementwiseOperation
        OutElementOp,   // OutElementwiseOperation
        ConvBwdDefault, // ConvolutionBackwardDataSpecialization
        256,            // BlockSize
        128,            // MPerBlock
        128,            // NPerBlock
        4,              // K0PerBlock
        8,              // K1
        32,             // MPerXdl
        32,             // NPerXdl
        2,              // MXdlPerWave
        2,              // NXdlPerWave
        S<4, 64, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
        S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
        2,              // ABlockTransferSrcVectorDim
        8,              // ABlockTransferSrcScalarPerVector
        8,              // ABlockTransferDstScalarPerVector_K1
        true,           // ABlockLdsAddExtraM
        S<4, 64, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
        S<2, 0, 1>,     // BBlockTransferThreadClusterArrangeOrder
        S<0, 2, 1>,     // BBlockTransferSrcAccessOrder
        1,              // BBlockTransferSrcVectorDim
        2,              // BBlockTransferSrcScalarPerVector
        8,              // BBlockTransferDstScalarPerVector_K1
        true,           // BBlockLdsAddExtraN
        7,
        1>; // GemmCThreadTransferDstScalarPerVector

using ReferenceConvBwdInstance = ck::tensor_operation::host::ReferenceConvBwdData<InDataType,
                                                                                  WeiDataType,
                                                                                  OutDataType,
                                                                                  AccDataType,
                                                                                  InElementOp,
                                                                                  WeiElementOp,
                                                                                  OutElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // Conv shape
    ck::index_t N               = 128;
    ck::index_t K               = 256;
    ck::index_t C               = 256;
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

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 19)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        N               = std::stoi(argv[4]);
        K               = std::stoi(argv[5]);
        C               = std::stoi(argv[6]);
        Y               = std::stoi(argv[7]);
        X               = std::stoi(argv[8]);
        Hi              = std::stoi(argv[9]);
        Wi              = std::stoi(argv[10]);
        conv_stride_h   = std::stoi(argv[11]);
        conv_stride_w   = std::stoi(argv[12]);
        conv_dilation_h = std::stoi(argv[13]);
        conv_dilation_w = std::stoi(argv[14]);
        in_left_pad_h   = std::stoi(argv[15]);
        in_left_pad_w   = std::stoi(argv[16]);
        in_right_pad_h  = std::stoi(argv[17]);
        in_right_pad_w  = std::stoi(argv[18]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        printf("arg4 to 18: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
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
    auto f_host_tensor_descriptor =
        [](std::size_t N_, std::size_t C_, std::size_t H, std::size_t W) {
            return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                        std::vector<std::size_t>({C_ * H * W, 1, W * C_, C_}));
        };

    Tensor<OutDataType> out_n_k_ho_wo(f_host_tensor_descriptor(N, K, Ho, Wo));
    Tensor<WeiDataType> wei_k_c_y_x(f_host_tensor_descriptor(K, C, Y, X));
    Tensor<InDataType> in_n_c_hi_wi_host_result(f_host_tensor_descriptor(N, C, Hi, Wi));
    Tensor<InDataType> in_n_c_hi_wi_device_result(f_host_tensor_descriptor(N, C, Hi, Wi));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi_host_result.mDesc << std::endl;
    std::cout << "wei_k_c_y_x: " << wei_k_c_y_x.mDesc << std::endl;
    std::cout << "out_n_k_ho_wo: " << out_n_k_ho_wo.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        out_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        out_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_1<OutDataType>{1});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{1});
    }

    DeviceMem in_device_buf(sizeof(InDataType) *
                            in_n_c_hi_wi_device_result.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_n_k_ho_wo.mDesc.GetElementSpace());

    out_device_buf.ToDevice(out_n_k_ho_wo.mData.data());
    wei_device_buf.ToDevice(wei_k_c_y_x.mData.data());

    // reset input to zero
    in_n_c_hi_wi_device_result.GenerateTensorValue(GeneratorTensor_1<InDataType>{0});
    in_device_buf.ToDevice(in_n_c_hi_wi_device_result.mData.data());

    // do GEMM
    auto conv     = DeviceConvBwdDataInstance{};
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
                                      OutElementOp{});

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

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
        auto ref_conv    = ReferenceConvBwdInstance{};
        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(in_n_c_hi_wi_host_result,
                                                  wei_k_c_y_x,
                                                  out_n_k_ho_wo,
                                                  conv_filter_strides,
                                                  conv_filter_dilations,
                                                  input_left_pads,
                                                  input_right_pads,
                                                  InElementOp{},
                                                  WeiElementOp{},
                                                  OutElementOp{});

        ref_invoker.Run(ref_argument);

        in_device_buf.FromDevice(in_n_c_hi_wi_device_result.mData.data());

        return ck::utils::check_err(in_n_c_hi_wi_device_result.mData,
                                    in_n_c_hi_wi_host_result.mData)
                   ? 0
                   : 1;
    }
    return 0;
}
