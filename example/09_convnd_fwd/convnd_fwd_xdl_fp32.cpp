// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_convnd_fwd_xdl_nhwc_kyxc_nhwk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/conv_util.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

namespace {

using InDataType  = float;
using WeiDataType = float;
using OutDataType = float;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

using DeviceConvFwdBasePtr =
    ck::tensor_operation::device::DeviceConvFwdPtr<InElementOp, WeiElementOp, OutElementOp>;

template <ck::index_t NumDimSpatial>
using DeviceConvNDFwdInstance = ck::tensor_operation::device::
    DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        // clang-format off
        InDataType,         //
        WeiDataType,        //
        OutDataType,        //
        AccDataType,        //
        InElementOp,        // Input Elementwise Operation
        WeiElementOp,       // Weights Elementwise Operation
        OutElementOp,       // Output Elementwise Operation
        ConvFwdDefault,     // ConvForwardSpecialization
        NumDimSpatial,      // NumDimSpatial
        256,                // BlockSize
        256,                // MPerBlock
        128,                // NPerBlock
        4,                  // K0PerBlock
        4,                  // K1
        32,                 // MPerXDL
        32,                 // NPerXDL
        4,                  // MXdlPerWave
        2,                  // NXdlPerWave
        S<4, 64, 1>,        // ABlockTransferThreadClusterLengths_K0_M_K1
        S<1, 0, 2>,         // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,         // ABlockTransferSrcAccessOrder
        2,                  // ABlockTransferSrcVectorDim
        4,                  // ABlockTransferSrcScalarPerVector
        4,                  // ABlockTransferDstScalarPerVector_K1
        true,               // ABlockLdsAddExtraM
        S<4, 64, 1>,        // BBlockTransferThreadClusterLengths_K0_N_K1
        S<1, 0, 2>,         // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,         // BBlockTransferSrcAccessOrder
        2,                  // BBlockTransferSrcVectorDim
        4,                  // BBlockTransferSrcScalarPerVector
        4,                  // BBlockTransferDstScalarPerVector_K1
        true,               // BBlockTransferAddExtraN
        7,                  // CThreadTransferSrcDstVectorDim
        1>;                 // CThreadTransferDstScalarPerVector
// clang-format on

template <ck::index_t NumDimSpatial>
using ReferenceConvNDFwdInstance = ck::tensor_operation::host::ReferenceConvFwd<InDataType,
                                                                                WeiDataType,
                                                                                OutDataType,
                                                                                InElementOp,
                                                                                WeiElementOp,
                                                                                OutElementOp,
                                                                                NumDimSpatial>;

DeviceConvFwdBasePtr get_conv_instance(int num_dim_spatial)
{
    switch(num_dim_spatial)
    {
    case 3: {
        return std::make_unique<DeviceConvNDFwdInstance<3>>();
    }
    case 2: {
        return std::make_unique<DeviceConvNDFwdInstance<2>>();
    }
    case 1: {
        return std::make_unique<DeviceConvNDFwdInstance<1>>();
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}

void print_use_msg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=n0, 1=yes)\n"
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

ck::utils::conv::ConvParams parse_conv_params(int num_dim_spatial, int argc, char* argv[])
{
    // (N, K, C) + num_dim_spatial * 6 (filter, input, strides, dilations, pad left, pad right)
    int conv_args     = 3 + num_dim_spatial * 6;
    int cmdline_nargs = conv_args + 5;
    if(cmdline_nargs != argc)
    {
        print_use_msg();
        exit(0);
    }

    ck::utils::conv::ConvParams params;
    int arg_idx = 5;

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

} // anonymous namespace

int main(int argc, char* argv[])
{
    using namespace ck::utils::conv;

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    int num_dim_spatial  = 2;

    ck::utils::conv::ConvParams params;

    if(argc >= 5)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
        num_dim_spatial = std::stoi(argv[4]);
    }

    if(argc >= 6)
    {
        params = parse_conv_params(num_dim_spatial, argc, argv);
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

    Tensor<InDataType> input(get_input_host_tensor_descriptor(input_dims, num_dim_spatial));
    Tensor<WeiDataType> weights(get_filters_host_tensor_descriptor(filter_dims, num_dim_spatial));
    Tensor<OutDataType> host_output(
        get_output_host_tensor_descriptor(output_dims, num_dim_spatial));
    Tensor<OutDataType> device_output(
        get_output_host_tensor_descriptor(output_dims, num_dim_spatial));

    std::cout << "input: " << input.mDesc << std::endl;
    std::cout << "weights: " << weights.mDesc << std::endl;
    std::cout << "output: " << host_output.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        input.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        weights.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        input.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        weights.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * input.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weights.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * device_output.mDesc.GetElementSpace());

    in_device_buf.ToDevice(input.mData.data());
    wei_device_buf.ToDevice(weights.mData.data());

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
                                  OutElementOp{});

    if(!conv->IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float ave_time = invoker->Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::size_t flop = get_flops(
        params.N_, params.C_, params.K_, params.filter_spatial_lengths_, output_spatial_lengths);
    std::size_t num_btype =
        get_btype<InDataType, WeiDataType, OutDataType>(params.N_,
                                                        params.C_,
                                                        params.K_,
                                                        params.input_spatial_lengths_,
                                                        params.filter_spatial_lengths_,
                                                        output_spatial_lengths);

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;
    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        auto verify_f = [&input, &weights, &host_output, &params, &out_device_buf, &device_output](
                            const auto& ref_conv) {
            auto ref_invoker  = ref_conv.MakeInvoker();
            auto ref_argument = ref_conv.MakeArgument(input,
                                                      weights,
                                                      host_output,
                                                      params.conv_filter_strides_,
                                                      params.conv_filter_dilations_,
                                                      params.input_left_pads_,
                                                      params.input_right_pads_,
                                                      InElementOp{},
                                                      WeiElementOp{},
                                                      OutElementOp{});

            ref_invoker.Run(ref_argument);
            out_device_buf.FromDevice(device_output.mData.data());
            return ck::utils::check_err(device_output.mData,
                                        host_output.mData,
                                        "Error: incorrect results!",
                                        1e-5f,
                                        1e-4f)
                       ? 0
                       : 1;
        };

        switch(num_dim_spatial)
        {
        case 3: {
            auto ref_conv = ReferenceConvNDFwdInstance<3>();
            return verify_f(ref_conv);
        }
        case 2: {
            auto ref_conv = ReferenceConvNDFwdInstance<2>();
            return verify_f(ref_conv);
        }
        case 1: {
            auto ref_conv = ReferenceConvNDFwdInstance<1>();
            return verify_f(ref_conv);
        }
        default: {
            throw std::runtime_error("Unsupported number of spatial dimensions provided!");
        }
        }
    }
    return 0;
}
