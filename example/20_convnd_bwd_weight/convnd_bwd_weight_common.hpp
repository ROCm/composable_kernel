// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"

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
              << "split_k" << std::endl;
}

ck::utils::conv::ConvParam parse_conv_params(int num_dim_spatial, int arg_idx, char* const argv[])
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

    return ck::utils::conv::ConvParam{num_dim_spatial,
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

// FIXME: current implementation only support NCHW/NHWC layout
template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp,
          typename DeviceConvBwdWeightInstance>
int run_conv_bwd_weight(bool do_verification,
                        int init_method,
                        bool time_kernel,
                        const ck::utils::conv::ConvParam& conv_param,
                        const InElementOp& in_element_op,
                        const WeiElementOp& wei_element_op,
                        const OutElementOp& out_element_op,
                        ck::index_t split_k)
{
    const auto in_desc  = ck::utils::conv::get_input_host_tensor_descriptor<InLayout>(conv_param);
    const auto wei_desc = ck::utils::conv::get_weight_host_tensor_descriptor<WeiLayout>(conv_param);
    const auto out_desc = ck::utils::conv::get_output_host_tensor_descriptor<OutLayout>(conv_param);

    Tensor<InDataType> in(in_desc);
    Tensor<WeiDataType> wei_host_result(wei_desc);
    Tensor<WeiDataType> wei_device_result(wei_desc);
    Tensor<OutDataType> out(out_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei_host_result.mDesc << std::endl;
    std::cout << "out: " << out.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        out.GenerateTensorValue(GeneratorTensor_3<OutDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_device_result.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    out_device_buf.ToDevice(out.mData.data());

    // init to 0
    wei_device_buf.SetZero();

    // do GEMM
    auto conv     = DeviceConvBwdWeightInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                      static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                      static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                      conv_param.N_,
                                      conv_param.K_,
                                      conv_param.C_,
                                      conv_param.input_spatial_lengths_,
                                      conv_param.filter_spatial_lengths_,
                                      conv_param.output_spatial_lengths_,
                                      conv_param.conv_filter_strides_,
                                      conv_param.conv_filter_dilations_,
                                      conv_param.input_left_pads_,
                                      conv_param.input_right_pads_,
                                      in_element_op,
                                      wei_element_op,
                                      out_element_op,
                                      split_k);

    if(!conv.IsSupportedArgument(argument))
    {
        std::cout << "wrong! device_conv with the specified compilation parameters does "
                     "not support this Conv problem"
                  << std::endl;
        return 1;
    }

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = conv_param.GetFlops();
    std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

    float gb_per_sec = num_btype / 1.E6 / avg_time;

    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << conv.GetTypeString() << std::endl;

    if(do_verification)
    {
        auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                           InLayout,
                                                                           WeiLayout,
                                                                           OutLayout,
                                                                           InDataType,
                                                                           WeiDataType,
                                                                           OutDataType,
                                                                           InElementOp,
                                                                           WeiElementOp,
                                                                           OutElementOp>{};

        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei_host_result,
                                                  out,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  InElementOp{},
                                                  WeiElementOp{},
                                                  OutElementOp{});

        ref_invoker.Run(ref_argument);

        wei_device_buf.FromDevice(wei_device_result.mData.data());

        return ck::utils::check_err(wei_device_result.mData, wei_host_result.mData) ? 0 : 1;
    }

    return 0;
}
