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
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data_bias_activation.hpp"

#if 0
using BF16 = ck::bhalf_t;
using FP16 = ck::half_t;
using FP32 = float;
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
using I4 = ck::int4_t;
#endif
using I8  = std::int8_t;
using I32 = std::int32_t;

template <typename InLay, typename WeiLay, typename OutLay, typename DELay>
struct LayoutSetting
{
    using InLayout  = InLay;
    using WeiLayout  = WeiLay;
    using OutLayout  = OutLay;
    using DELayout = DELay;
};

template <ck::index_t NDimSpatial>
struct LayoutSettingSelector;

namespace ctl = ck::tensor_layout::convolution;

template <>
struct LayoutSettingSelector<1> final : LayoutSetting<ctl::NWC, ctl::KXC, ctl::NWK, ctl::NWC>
{
};

template <>
struct LayoutSettingSelector<2> final : LayoutSetting<ctl::NHWC, ctl::KYXC, ctl::NHWK, ctl::NHWC>
{
};

template <>
struct LayoutSettingSelector<3> final
    : LayoutSetting<ctl::NDHWC, ctl::KZYXC, ctl::NDHWK, ctl::NDHWC>
{
};

template <ck::index_t NDimSpatial>
using InLayout = typename LayoutSettingSelector<NDimSpatial>::InLayout;

template <ck::index_t NDimSpatial>
using WeiLayout = typename LayoutSettingSelector<NDimSpatial>::WeiLayout;

template <ck::index_t NDimSpatial>
using OutLayout = typename LayoutSettingSelector<NDimSpatial>::OutLayout;

template <ck::index_t NDimSpatial>
using DELayout = typename LayoutSettingSelector<NDimSpatial>::DELayout;

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};
#endif

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
          typename DeviceConvNdBwdDataBiasReluInstance>
int run_conv_bwd_data_bias_relu(bool do_verification,
                      int init_method,
                      bool time_kernel,
                      const ck::utils::conv::ConvParam& conv_param,
                      const HostTensorDescriptor& in_g_n_c_wis_desc,
                      const HostTensorDescriptor& wei_g_k_c_xs_desc,
                      const HostTensorDescriptor& out_g_n_k_wos_desc,
                      const HostTensorDescriptor& bias_c_desc,
                      const InElementOp& in_element_op,
                      const WeiElementOp& wei_element_op,
                      const OutElementOp& out_element_op)
{
    Tensor<InDataType> in_host(in_g_n_c_wis_desc);
    Tensor<InDataType> in_device(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<OutDataType> out(out_g_n_k_wos_desc);
    Tensor<InDataType> bias(bias_c_desc);

    std::cout << "in: " << in_host.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out.mDesc << std::endl;
    std::cout << "bias: " << bias.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        out.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in_device.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());
    DeviceMem bias_device_buf(sizeof(InDataType) * bias.mDesc.GetElementSpaceSize());

    out_device_buf.ToDevice(out.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    bias_device_buf.ToDevice(bias.mData.data());

    // reset input to zero
    in_device_buf.SetZero();

    // do GEMM
    auto conv     = DeviceConvNdBwdDataBiasReluInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                      static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                      static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                      static_cast<InDataType*>(bias_device_buf.GetDeviceBuffer()),
                                      conv_param.N_,
                                      conv_param.K_,
                                      conv_param.C_,
                                      conv_param.input_spatial_lengths_,
                                      conv_param.filter_spatial_lengths_,
                                      conv_param.GetOutputSpatialLengths(),
                                      conv_param.conv_filter_strides_,
                                      conv_param.conv_filter_dilations_,
                                      conv_param.input_left_pads_,
                                      conv_param.input_right_pads_,
                                      in_element_op,
                                      wei_element_op,
                                      out_element_op);

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = conv_param.GetFlops();
    std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdDataBiasActivation<NDimSpatial,
                                                                         InDataType,
                                                                         WeiDataType,
                                                                         OutDataType,
                                                                         InElementOp,
                                                                         WeiElementOp,
                                                                         OutElementOp>();

        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(in_host,
                                                  wei,
                                                  out,
                                                  bias,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);

        ref_invoker.Run(ref_argument);

        in_device_buf.FromDevice(in_device.mData.data());

        return ck::utils::check_err(in_device.mData, in_host.mData) ? 0 : 1;
    }

    return 0;
}
