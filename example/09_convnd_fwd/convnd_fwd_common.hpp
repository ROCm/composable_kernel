// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/library/utility/array.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

void print_helper_msg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp,
          typename DeviceConvNDFwdInstance>
bool run_grouped_conv_fwd(bool do_verification,
                          int init_method,
                          bool time_kernel,
                          const ck::utils::conv::ConvParam& conv_param,
                          const HostTensorDescriptor& in_g_n_c_wis_desc,
                          const HostTensorDescriptor& wei_g_k_c_xs_desc,
                          const HostTensorDescriptor& out_g_n_k_wos_desc,
                          const InElementOp& in_element_op,
                          const WeiElementOp& wei_element_op,
                          const OutElementOp& out_element_op)
{
    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<OutDataType> out_host(out_g_n_k_wos_desc);
    Tensor<OutDataType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.GetDesc() << std::endl;
    std::cout << "wei: " << wei.GetDesc() << std::endl;
    std::cout << "out: " << out_host.GetDesc() << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(in.GetMemorySize());
    DeviceMem wei_device_buf(wei.GetMemorySize());
    DeviceMem out_device_buf(out_device.GetMemorySize());

    in_device_buf.ToDevice(in.data());
    wei_device_buf.ToDevice(wei.data());

    using ck::utils::empty_array, ck::utils::to_array;

    // do Conv
    auto conv     = DeviceConvNDFwdInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                                      wei_device_buf.GetDeviceBuffer(),
                                      empty_array(),
                                      out_device_buf.GetDeviceBuffer(),
                                      to_array(in_g_n_c_wis_desc.GetLengths()),
                                      to_array(in_g_n_c_wis_desc.GetStrides()),
                                      to_array(wei_g_k_c_xs_desc.GetLengths()),
                                      to_array(wei_g_k_c_xs_desc.GetStrides()),
                                      empty_array(),
                                      empty_array(),
                                      to_array(out_g_n_k_wos_desc.GetLengths()),
                                      to_array(out_g_n_k_wos_desc.GetStrides()),
                                      to_array(conv_param.conv_filter_strides_),
                                      to_array(conv_param.conv_filter_dilations_),
                                      to_array(conv_param.input_left_pads_),
                                      to_array(conv_param.input_right_pads_),
                                      in_element_op,
                                      wei_element_op,
                                      out_element_op);

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = conv_param.GetFlops();
    std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    if(do_verification)
    {
        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     OutDataType,
                                                                     InElementOp,
                                                                     WeiElementOp,
                                                                     OutElementOp>();

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei,
                                                  out_host,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);

        ref_invoker.Run(ref_argument);

        out_device_buf.FromDevice(out_device.data());

        return ck::utils::check_err(
            out_device, out_host, "Error: incorrect results!", 1e-5f, 1e-4f);
    }

    return true;
}
