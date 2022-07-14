// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_bwd_data.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"

using F16  = ck::half_t;
using F32  = float;
using BF16 = ck::bhalf_t;
using INT8 = int8_t;

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using DeviceConvBwdDataNoOpPtr =
    DeviceConvBwdDataPtr<ck::tensor_operation::element_wise::PassThrough,
                         ck::tensor_operation::element_wise::PassThrough,
                         ck::tensor_operation::element_wise::PassThrough>;
void add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_f32_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_f16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_bf16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_int8_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);

void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f32_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_bf16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_int8_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);

void add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_f32_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_f16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_bf16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_int8_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace profiler {
using DeviceConvBwdDataNoOpPtr = ck::tensor_operation::device::instance::DeviceConvBwdDataNoOpPtr;

template <typename InDataType, typename WeiDataType, typename OutDataType>
void get_device_conv_bwd_data_op_ptr(
    InDataType, WeiDataType, OutDataType, std::vector<DeviceConvBwdDataNoOpPtr>&, int)
{
    std::cout << "can not find device conv bwd data" << std::endl;
    exit(1);
}
template <>
void get_device_conv_bwd_data_op_ptr(
    F32, F32, F32, std::vector<DeviceConvBwdDataNoOpPtr>& conv_ptrs, int num_dim_spatial)
{
    switch(num_dim_spatial)
    {
    case 1:
        ck::tensor_operation::device::instance::
            add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_f32_instances(conv_ptrs);
        break;
    case 2:
        ck::tensor_operation::device::instance::
            add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f32_instances(conv_ptrs);
        break;
    case 3:
        ck::tensor_operation::device::instance::
            add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_f32_instances(conv_ptrs);
        break;
    default: break;
    }
}
template <>
void get_device_conv_bwd_data_op_ptr(
    F16, F16, F16, std::vector<DeviceConvBwdDataNoOpPtr>& conv_ptrs, int num_dim_spatial)
{
    switch(num_dim_spatial)
    {
    case 1:
        ck::tensor_operation::device::instance::
            add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_f16_instances(conv_ptrs);
        break;
    case 2:
        ck::tensor_operation::device::instance::
            add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);
        break;
    case 3:
        ck::tensor_operation::device::instance::
            add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_f16_instances(conv_ptrs);
        break;
    default: break;
    }
}
template <>
void get_device_conv_bwd_data_op_ptr(
    BF16, BF16, BF16, std::vector<DeviceConvBwdDataNoOpPtr>& conv_ptrs, int num_dim_spatial)
{
    switch(num_dim_spatial)
    {
    case 1:
        ck::tensor_operation::device::instance::
            add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_bf16_instances(conv_ptrs);
        break;
    case 2:
        ck::tensor_operation::device::instance::
            add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_bf16_instances(conv_ptrs);
        break;
    case 3:
        ck::tensor_operation::device::instance::
            add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_bf16_instances(conv_ptrs);
        break;
    default: break;
    }
}
template <>
void get_device_conv_bwd_data_op_ptr(
    INT8, INT8, INT8, std::vector<DeviceConvBwdDataNoOpPtr>& conv_ptrs, int num_dim_spatial)
{
    switch(num_dim_spatial)
    {
    case 1:
        ck::tensor_operation::device::instance::
            add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_int8_instances(conv_ptrs);
        break;
    case 2:
        ck::tensor_operation::device::instance::
            add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_int8_instances(conv_ptrs);
        break;
    case 3:
        ck::tensor_operation::device::instance::
            add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_int8_instances(conv_ptrs);
        break;
    default: break;
    }
}

template <typename DataType>
void show_data_nhwc_layout(Tensor<DataType>& nhwc)
{
    std::cout << "[";
    for(int n = 0; n < ck::type_convert<int>(nhwc.mDesc.GetLengths()[0]); n++)
    {
        std::cout << "[";
        for(int hi = 0; hi < ck::type_convert<int>(nhwc.mDesc.GetLengths()[2]); hi++)
        {
            std::cout << "[";
            for(int wi = 0; wi < ck::type_convert<int>(nhwc.mDesc.GetLengths()[3]); wi++)
            {
                std::cout << "[";
                for(int c = 0; c < ck::type_convert<int>(nhwc.mDesc.GetLengths()[1]); c++)
                {
                    std::cout << static_cast<float>(nhwc(n, c, hi, wi)) << "  ";
                }
                std::cout << "]";
            }
            std::cout << "]";
        }
        std::cout << "]";
    }
    std::cout << "]";
}

template <int NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType>
bool profile_conv_bwd_data_impl(int do_verification,
                                int init_method,
                                bool do_log,
                                bool time_kernel,
                                const ck::tensor_operation::device::ConvParams& params)
{
    using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
    using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    // make host tensor descritpor
    auto f_nhwc_host_tensor_descriptor =
        [](ck::index_t n, ck::index_t c, std::vector<ck::index_t> spatial_lengths) {
            std::vector<std::size_t> nhwc_lengths{static_cast<std::size_t>(n),
                                                  static_cast<std::size_t>(c)};
            nhwc_lengths.insert(
                nhwc_lengths.begin() + 1, spatial_lengths.begin(), spatial_lengths.end());

            return HostTensorDescriptor(nhwc_lengths);
        };

    auto f_nchw_host_tensor_descriptor =
        [](ck::index_t n, ck::index_t c, std::vector<ck::index_t> spatial_lengths) {
            std::vector<std::size_t> nchw_lengths{static_cast<std::size_t>(n),
                                                  static_cast<std::size_t>(c)};
            nchw_lengths.insert(nchw_lengths.end(), spatial_lengths.begin(), spatial_lengths.end());

            return HostTensorDescriptor(nchw_lengths);
        };

    HostTensorDescriptor in_desc, wei_desc, out_desc;

    // FIXME: properly implement "make host descriptor" for different layout
    if constexpr(is_same_v<InLayout, ck::tensor_layout::convolution::NWC> ||
                 is_same_v<InLayout, ck::tensor_layout::convolution::NHWC> ||
                 is_same_v<InLayout, ck::tensor_layout::convolution::NDHWC>)
    {
        in_desc =
            f_nhwc_host_tensor_descriptor(params.N_, params.C_, params.input_spatial_lengths_);
    }
    else if constexpr(is_same_v<InLayout, ck::tensor_layout::convolution::NCW> ||
                      is_same_v<InLayout, ck::tensor_layout::convolution::NCHW> ||
                      is_same_v<InLayout, ck::tensor_layout::convolution::NCDHW>)
    {
        in_desc =
            f_nchw_host_tensor_descriptor(params.N_, params.C_, params.input_spatial_lengths_);
    }

    // FIXME: properly implement "make host descriptor" for different layout
    if constexpr(is_same_v<WeiLayout, ck::tensor_layout::convolution::KXC> ||
                 is_same_v<WeiLayout, ck::tensor_layout::convolution::KYXC> ||
                 is_same_v<WeiLayout, ck::tensor_layout::convolution::KZYXC>)
    {
        wei_desc =
            f_nhwc_host_tensor_descriptor(params.K_, params.C_, params.filter_spatial_lengths_);
    }
    else if constexpr(is_same_v<WeiLayout, ck::tensor_layout::convolution::KCX> ||
                      is_same_v<WeiLayout, ck::tensor_layout::convolution::KCYX> ||
                      is_same_v<WeiLayout, ck::tensor_layout::convolution::KCZYX>)
    {
        wei_desc =
            f_nchw_host_tensor_descriptor(params.K_, params.C_, params.filter_spatial_lengths_);
    }

    // FIXME: properly implement "make host descriptor" for different layout
    if constexpr(is_same_v<OutLayout, ck::tensor_layout::convolution::NWK> ||
                 is_same_v<OutLayout, ck::tensor_layout::convolution::NHWK> ||
                 is_same_v<OutLayout, ck::tensor_layout::convolution::NDHWK>)
    {
        out_desc =
            f_nhwc_host_tensor_descriptor(params.N_, params.K_, params.GetOutputSpatialLengths());
    }
    else if constexpr(is_same_v<OutLayout, ck::tensor_layout::convolution::NKW> ||
                      is_same_v<OutLayout, ck::tensor_layout::convolution::NKHW> ||
                      is_same_v<OutLayout, ck::tensor_layout::convolution::NKDHW>)
    {
        out_desc =
            f_nchw_host_tensor_descriptor(params.N_, params.K_, params.GetOutputSpatialLengths());
    }

    Tensor<InDataType> input_host_result(in_desc);
    Tensor<InDataType> input_device_result(in_desc);
    Tensor<WeiDataType> weight(wei_desc);
    Tensor<OutDataType> output(out_desc);

    std::cout << "input: " << input_host_result.mDesc << std::endl;
    std::cout << "weight: " << weight.mDesc << std::endl;
    std::cout << "output: " << output.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        output.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        weight.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        output.GenerateTensorValue(GeneratorTensor_1<OutDataType>{1});
        weight.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{1});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * input_device_result.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weight.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * output.mDesc.GetElementSpace());

    out_device_buf.ToDevice(output.mData.data());
    wei_device_buf.ToDevice(weight.mData.data());

    if(do_verification)
    {
        auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
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

        auto ref_argument = ref_conv.MakeArgument(input_host_result,
                                                  weight,
                                                  output,
                                                  params.conv_filter_strides_,
                                                  params.conv_filter_dilations_,
                                                  params.input_left_pads_,
                                                  params.input_right_pads_,
                                                  InElementOp{},
                                                  WeiElementOp{},
                                                  OutElementOp{});
        ref_invoker.Run(ref_argument);
    }

    // add device Conv instances
    std::vector<DeviceConvBwdDataNoOpPtr> conv_ptrs;
    get_device_conv_bwd_data_op_ptr(
        InDataType{}, WeiDataType{}, OutDataType{}, conv_ptrs, NDimSpatial);

    if(conv_ptrs.size() <= 0)
    {
        throw std::runtime_error("wrong! no device Conv instance found");
    }

    std::string best_op_name;
    float best_avg_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device Conv instances
    bool pass = true;

    for(auto& conv_ptr : conv_ptrs)
    {
        auto argument_ptr = conv_ptr->MakeArgumentPointer(
            static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
            static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
            static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
            params.N_,
            params.K_,
            params.C_,
            params.input_spatial_lengths_,
            params.filter_spatial_lengths_,
            params.output_spatial_lengths_,
            params.conv_filter_strides_,
            params.conv_filter_dilations_,
            params.input_left_pads_,
            params.input_right_pads_,
            in_element_op,
            wei_element_op,
            out_element_op);

        auto invoker_ptr = conv_ptr->MakeInvokerPointer();

        if(conv_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            // reset input to zero
            in_device_buf.SetZero();

            std::string op_name = conv_ptr->GetTypeString();

            float avg_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop      = params.GetFlops();
            std::size_t num_btype = params.GetByte<InDataType, WeiDataType, OutDataType>();

            float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
            float gb_per_sec = num_btype / 1.E6 / avg_time;

            std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s" << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                in_device_buf.FromDevice(input_device_result.mData.data());

                pass =
                    pass & ck::utils::check_err(input_device_result.mData, input_host_result.mData);

                if(do_log)
                {
                    std::cout << "in : ";
                    show_data_nhwc_layout(output);
                    std::cout << std::endl;

                    std::cout << "wei: ";
                    show_data_nhwc_layout(weight);
                    std::cout << std::endl;

                    std::cout << "out_host  : ";
                    show_data_nhwc_layout(input_host_result);
                    std::cout << std::endl;

                    std::cout << "out_device: ";
                    show_data_nhwc_layout(input_device_result);
                    std::cout << std::endl;
                }
            }
        }
    }

    std::cout << "Best Perf: " << best_avg_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
