// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_convnd_backward_weight_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/conv_util.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_backward_weight.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdWeightDefault =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;

using DeviceConvBwdWeightBasePtr =
    ck::tensor_operation::device::DeviceConvBwdWeightPtr<InElementOp, WeiElementOp, OutElementOp>;

// clang-format off
template <ck::index_t NumDimSpatial>
using DeviceConvndBwdWeightInstance = ck::tensor_operation::device::
    DeviceConvndBwdWeightXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        InDataType,                       // InDataType
        WeiDataType,                      // WeiDataType
        OutDataType,                      // OutDataType
        AccDataType,                      // AccDataType
        InElementOp,                      // InElementwiseOperation
        WeiElementOp,                     // WeiElementwiseOperation
        OutElementOp,                     // OutElementwiseOperation
        ConvBwdWeightDefault,             // ConvolutionBackwardWeightSpecialization
        NumDimSpatial,                    // NumDimSpatial
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

template <ck::index_t NumDimSpatial>
using ReferenceConvBwdWeightInstance =
    ck::tensor_operation::host::ReferenceConvBwdWeight<InDataType,
                                                       WeiDataType,
                                                       OutDataType,
                                                       InElementOp,
                                                       WeiElementOp,
                                                       OutElementOp,
                                                       NumDimSpatial>;

void print_use_msg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=random value, 2= init to 1 )\n"
              << "arg3: time kernel (0=n0, 1=yes)\n"
              << "arg4: is show log (0=no, 1=yes)\n"
              << "arg5: split-k \n"
              << "arg6: N spatial dimensions (default 2)\n"
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

ck::utils::conv::ConvParams parse_conv_params(int num_dim_spatial, char* argv[])
{
    // (N, K, C) + num_dim_spatial * 6 (filter, input, strides, dilations, pad left, pad right)
    ck::utils::conv::ConvParams params;
    int arg_idx = 7;

    params.num_dim_spatial_ = num_dim_spatial;
    params.N_               = std::stoi(argv[arg_idx++]);
    params.K_               = std::stoi(argv[arg_idx++]);
    params.C_               = std::stoi(argv[arg_idx++]);

    params.filter_spatial_lengths_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.filter_spatial_lengths_[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_spatial_lengths_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_spatial_lengths_[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_strides_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_strides_[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_dilations_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_dilations_[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_left_pads_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_left_pads_[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_right_pads_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_right_pads_[i] = std::stoi(argv[arg_idx++]);
    }

    return params;
}

DeviceConvBwdWeightBasePtr get_conv_instance(int num_dim_spatial)
{
    switch(num_dim_spatial)
    {
    case 3: {
        return std::make_unique<DeviceConvndBwdWeightInstance<3>>();
    }
    case 2: {
        return std::make_unique<DeviceConvndBwdWeightInstance<2>>();
    }
    case 1: {
        return std::make_unique<DeviceConvndBwdWeightInstance<1>>();
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    int num_dim_spatial  = 2;
    int do_log           = 0;
    int split_k          = 1;

    ck::utils::conv::ConvParams params;
    params.C_ = 128;

    if(argc == 6)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
        do_log          = std::stoi(argv[4]);
        split_k         = std::stoi(argv[5]);
    }
    else if(argc > 6)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
        do_log          = std::stoi(argv[4]);
        split_k         = std::stoi(argv[5]);
        num_dim_spatial = std::stoi(argv[6]);
        // check args number
        int conv_args     = 3 + num_dim_spatial * 6;
        int cmdline_nargs = conv_args + 7;
        if(cmdline_nargs != argc)
        {
            print_use_msg();
            exit(1);
        }

        params = parse_conv_params(num_dim_spatial, argv);
    }
    else if(argc != 1)
    {
        print_use_msg();
        exit(1);
    }

    std::vector<std::size_t> input_dims{static_cast<std::size_t>(params.N_),
                                        static_cast<std::size_t>(params.C_)};
    input_dims.insert(std::end(input_dims),
                      std::begin(params.input_spatial_lengths_),
                      std::end(params.input_spatial_lengths_));

    std::vector<std::size_t> filter_dims{static_cast<std::size_t>(params.K_),
                                         static_cast<std::size_t>(params.C_)};
    filter_dims.insert(std::end(filter_dims),
                       std::begin(params.filter_spatial_lengths_),
                       std::end(params.filter_spatial_lengths_));

    const std::vector<ck::index_t>& output_spatial_lengths = params.GetOutputSpatialLengths();
    std::vector<std::size_t> output_dims{static_cast<std::size_t>(params.N_),
                                         static_cast<std::size_t>(params.K_)};
    output_dims.insert(std::end(output_dims),
                       std::begin(output_spatial_lengths),
                       std::end(output_spatial_lengths));

    Tensor<InDataType> in_n_c_hi_wi(
        ck::utils::conv::get_input_host_tensor_descriptor(input_dims, num_dim_spatial));
    Tensor<WeiDataType> wei_k_c_y_x_host_result(
        ck::utils::conv::get_filters_host_tensor_descriptor(filter_dims, num_dim_spatial));
    Tensor<WeiDataType> wei_k_c_y_x_device_result(
        ck::utils::conv::get_filters_host_tensor_descriptor(filter_dims, num_dim_spatial));
    Tensor<OutDataType> out_n_k_ho_wo(
        ck::utils::conv::get_output_host_tensor_descriptor(output_dims, num_dim_spatial));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi.mDesc << std::endl;
    std::cout << "wei_k_c_y_x: " << wei_k_c_y_x_device_result.mDesc << std::endl;
    std::cout << "out_n_k_ho_wo: " << out_n_k_ho_wo.mDesc << std::endl;

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi.mDesc << std::endl;
    std::cout << "wei_k_c_y_x: " << wei_k_c_y_x_host_result.mDesc << std::endl;
    std::cout << "out_n_k_ho_wo: " << out_n_k_ho_wo.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        out_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-2, 2});
        in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_2<InDataType>{-2, 2});
        break;
    default:
        out_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_1<OutDataType>{1});
        in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_1<InDataType>{1});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) *
                             wei_k_c_y_x_device_result.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_n_k_ho_wo.mDesc.GetElementSpace());

    in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
    out_device_buf.ToDevice(out_n_k_ho_wo.mData.data());
    // reset input to zero
    wei_device_buf.SetZero();

    // do GEMM
    auto conv    = get_conv_instance(num_dim_spatial);
    auto invoker = conv->MakeInvokerPointer();
    auto argument =
        conv->MakeArgumentPointer(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                  static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                  static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                  params.N_,
                                  params.K_,
                                  params.C_,
                                  params.input_spatial_lengths_,
                                  params.filter_spatial_lengths_,
                                  output_spatial_lengths,
                                  params.conv_filter_strides_,
                                  params.conv_filter_dilations_,
                                  params.input_left_pads_,
                                  params.input_right_pads_,
                                  InElementOp{},
                                  WeiElementOp{},
                                  OutElementOp{},
                                  split_k);

    // alloc work space
    float ave_time = 0.f;
    if(!conv->IsSupportedArgument(argument.get()))
    {
        std::cout << "wrong! device_conv with the specified compilation parameters does "
                     "not support this Conv problem"
                  << std::endl;
        return 1;
    }
    ave_time = invoker->Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::size_t flop = ck::utils::conv::get_flops(
        params.N_, params.C_, params.K_, params.filter_spatial_lengths_, output_spatial_lengths);
    std::size_t num_btype = ck::utils::conv::get_btype<InDataType, WeiDataType, OutDataType>(
        params.N_,
        params.C_,
        params.K_,
        params.input_spatial_lengths_,
        params.filter_spatial_lengths_,
        output_spatial_lengths);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        auto verify_f = [&](const auto& ref_conv) {
            auto ref_invoker = ref_conv.MakeInvoker();

            auto ref_argument = ref_conv.MakeArgument(in_n_c_hi_wi,
                                                      wei_k_c_y_x_host_result,
                                                      out_n_k_ho_wo,
                                                      params.conv_filter_strides_,
                                                      params.conv_filter_dilations_,
                                                      params.input_left_pads_,
                                                      params.input_right_pads_,
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
                LogRangeAsType<float>(
                    std::cout << "wei_host  : ", wei_k_c_y_x_host_result.mData, ",")
                    << std::endl;
            }

            return ck::utils::check_err(wei_k_c_y_x_device_result.mData,
                                        wei_k_c_y_x_host_result.mData)
                       ? 0
                       : 1;
        };

        switch(num_dim_spatial)
        {
        case 3: {
            auto ref_conv = ReferenceConvBwdWeightInstance<3>();
            return verify_f(ref_conv);
        }
        case 2: {
            auto ref_conv = ReferenceConvBwdWeightInstance<2>();
            return verify_f(ref_conv);
        }
        case 1: {
            auto ref_conv = ReferenceConvBwdWeightInstance<1>();
            return verify_f(ref_conv);
        }
        default: {
            throw std::runtime_error("Unsupported number of spatial dimensions provided!");
        }
        }
    }
    return 0;
}
