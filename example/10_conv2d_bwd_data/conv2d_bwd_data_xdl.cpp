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
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
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

using ReferenceConvBwdInstance =
    ck::tensor_operation::host::ReferenceConvBwdData<2,
                                                     ck::tensor_layout::convolution::NHWC,
                                                     ck::tensor_layout::convolution::KYXC,
                                                     ck::tensor_layout::convolution::NHWK,
                                                     InDataType,
                                                     WeiDataType,
                                                     OutDataType,
                                                     InElementOp,
                                                     WeiElementOp,
                                                     OutElementOp>;

ck::tensor_operation::device::ConvParams
parse_conv_params(int num_dim_spatial, int arg_idx, char* const argv[])
{
    const ck::index_t N = std::stoi(argv[arg_idx++]);
    const ck::index_t K = std::stoi(argv[arg_idx++]);
    const ck::index_t C = std::stoi(argv[arg_idx++]);

    std::vector<ck::index_t> filter_spatial_lengths(num_dim_spatial);
    std::vector<ck::index_t> input_spatial_lengths(num_dim_spatial);
    std::vector<ck::index_t> conv_filter_strides(num_dim_spatial);
    std::vector<ck::index_t> conv_filter_dilations(num_dim_spatial);
    std::vector<ck::index_t> input_left_pads(num_dim_spatial);
    std::vector<ck::index_t> input_right_pads(num_dim_spatial);

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        filter_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        conv_filter_strides[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        conv_filter_dilations[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_left_pads[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_right_pads[i] = std::stoi(argv[arg_idx++]);
    }

    return ck::tensor_operation::device::ConvParams{num_dim_spatial,
                                                    N,
                                                    K,
                                                    C,
                                                    filter_spatial_lengths,
                                                    input_spatial_lengths,
                                                    conv_filter_strides,
                                                    conv_filter_dilations,
                                                    input_left_pads,
                                                    input_right_pads};
}

void print_helper_msg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << "arg4: N spatial dimensions (default 2)\n"
              << "Following arguments (depending on number of spatial dims):\n"
              << " N, K, C, \n"
              << " <filter spatial dimensions>, (ie Y, X for 2D)\n"
              << " <input image spatial dimensions>, (ie Hi, Wi for 2D)\n"
              << " <strides>, (ie Sy, Sx for 2D)\n"
              << " <dilations>, (ie Dy, Dx for 2D)\n"
              << " <left padding>, (ie LeftPy, LeftPx for 2D)\n"
              << " <right padding>, (ie RightPy, RightPx for 2D)\n"
              << std::endl;
}

int main(int argc, char* argv[])
{
    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    int num_dim_spatial  = 2;

    ck::tensor_operation::device::ConvParams params{
        2, 128, 256, 256, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};

    if(argc == 1)
    {
        // use default
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
        num_dim_spatial = std::stoi(argv[4]);

        params = parse_conv_params(num_dim_spatial, 5, argv);
    }

    auto f_nhwc_host_tensor_descriptor =
        [](ck::index_t n, ck::index_t c, std::vector<ck::index_t> spatial_lengths) {
            std::vector<std::size_t> nhwc_lengths{static_cast<std::size_t>(n),
                                                  static_cast<std::size_t>(c)};
            nhwc_lengths.insert(
                nhwc_lengths.begin() + 1, spatial_lengths.begin(), spatial_lengths.end());

            return HostTensorDescriptor(nhwc_lengths);
        };

    Tensor<InDataType> in_n_hi_wi_c_host_result(
        f_nhwc_host_tensor_descriptor(params.N_, params.C_, params.input_spatial_lengths_));
    Tensor<InDataType> in_n_hi_wi_c_device_result(
        f_nhwc_host_tensor_descriptor(params.N_, params.C_, params.input_spatial_lengths_));
    Tensor<WeiDataType> wei_k_y_x_c(
        f_nhwc_host_tensor_descriptor(params.K_, params.C_, params.filter_spatial_lengths_));
    Tensor<OutDataType> out_n_ho_wo_k(
        f_nhwc_host_tensor_descriptor(params.N_, params.K_, params.GetOutputSpatialLengths()));

    std::cout << "in_n_hi_wi_c: " << in_n_hi_wi_c_host_result.mDesc << std::endl;
    std::cout << "wei_k_y_x_c: " << wei_k_y_x_c.mDesc << std::endl;
    std::cout << "out_n_ho_wo_k: " << out_n_ho_wo_k.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        out_n_ho_wo_k.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        wei_k_y_x_c.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        out_n_ho_wo_k.GenerateTensorValue(GeneratorTensor_1<OutDataType>{1});
        wei_k_y_x_c.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{1});
    }

    DeviceMem in_device_buf(sizeof(InDataType) *
                            in_n_hi_wi_c_device_result.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_k_y_x_c.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_n_ho_wo_k.mDesc.GetElementSpace());

    out_device_buf.ToDevice(out_n_ho_wo_k.mData.data());
    wei_device_buf.ToDevice(wei_k_y_x_c.mData.data());

    // reset input to zero
    in_device_buf.SetZero();

    // do GEMM
    auto conv     = DeviceConvBwdDataInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                      static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                      static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                      params.N_,
                                      params.K_,
                                      params.C_,
                                      params.input_spatial_lengths_,
                                      params.filter_spatial_lengths_,
                                      params.GetOutputSpatialLengths(),
                                      params.conv_filter_strides_,
                                      params.conv_filter_dilations_,
                                      params.input_left_pads_,
                                      params.input_right_pads_,
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

    std::size_t flop      = params.GetFlops();
    std::size_t num_btype = params.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        auto ref_conv    = ReferenceConvBwdInstance{};
        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(in_n_hi_wi_c_host_result,
                                                  wei_k_y_x_c,
                                                  out_n_ho_wo_k,
                                                  params.conv_filter_strides_,
                                                  params.conv_filter_dilations_,
                                                  params.input_left_pads_,
                                                  params.input_right_pads_,
                                                  InElementOp{},
                                                  WeiElementOp{},
                                                  OutElementOp{});

        ref_invoker.Run(ref_argument);

        in_device_buf.FromDevice(in_n_hi_wi_c_device_result.mData.data());

        return ck::utils::check_err(in_n_hi_wi_c_device_result.mData,
                                    in_n_hi_wi_c_host_result.mData)
                   ? 0
                   : 1;
    }

    return 0;
}
