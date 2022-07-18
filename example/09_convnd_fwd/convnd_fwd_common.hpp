// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

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
          typename DeviceConvNDFwdInstance>
int run_conv_fwd(bool do_verification,
                 int init_method,
                 bool time_kernel,
                 const ck::tensor_operation::device::ConvParams& params,
                 const InElementOp& in_element_op,
                 const WeiElementOp& wei_element_op,
                 const OutElementOp& out_element_op)
{
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
    if constexpr(ck::is_same_v<InLayout, ck::tensor_layout::convolution::NWC> ||
                 ck::is_same_v<InLayout, ck::tensor_layout::convolution::NHWC> ||
                 ck::is_same_v<InLayout, ck::tensor_layout::convolution::NDHWC>)
    {
        in_desc =
            f_nhwc_host_tensor_descriptor(params.N_, params.C_, params.input_spatial_lengths_);
    }
    else if constexpr(ck::is_same_v<InLayout, ck::tensor_layout::convolution::NCW> ||
                      ck::is_same_v<InLayout, ck::tensor_layout::convolution::NCHW> ||
                      ck::is_same_v<InLayout, ck::tensor_layout::convolution::NCDHW>)
    {
        in_desc =
            f_nchw_host_tensor_descriptor(params.N_, params.C_, params.input_spatial_lengths_);
    }

    // FIXME: properly implement "make host descriptor" for different layout
    if constexpr(ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KXC> ||
                 ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KYXC> ||
                 ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KZYXC>)
    {
        wei_desc =
            f_nhwc_host_tensor_descriptor(params.K_, params.C_, params.filter_spatial_lengths_);
    }
    else if constexpr(ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KCX> ||
                      ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KCYX> ||
                      ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KCZYX>)
    {
        wei_desc =
            f_nchw_host_tensor_descriptor(params.K_, params.C_, params.filter_spatial_lengths_);
    }

    // FIXME: properly implement "make host descriptor" for different layout
    if constexpr(ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NWK> ||
                 ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NHWK> ||
                 ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NDHWK>)
    {
        out_desc =
            f_nhwc_host_tensor_descriptor(params.N_, params.K_, params.GetOutputSpatialLengths());
    }
    else if constexpr(ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NKW> ||
                      ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NKHW> ||
                      ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NKDHW>)
    {
        out_desc =
            f_nchw_host_tensor_descriptor(params.N_, params.K_, params.GetOutputSpatialLengths());
    }

    Tensor<InDataType> in(in_desc);
    Tensor<WeiDataType> wei(wei_desc);
    Tensor<OutDataType> out_host(out_desc);
    Tensor<OutDataType> out_device(out_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

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

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_device.mDesc.GetElementSpace());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    // do GEMM
    auto conv     = DeviceConvNDFwdInstance{};
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

    std::size_t flop      = params.GetFlops();
    std::size_t num_btype = params.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    if(do_verification)
    {
        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                     InLayout,
                                                                     WeiLayout,
                                                                     OutLayout,
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
                                                  params.conv_filter_strides_,
                                                  params.conv_filter_dilations_,
                                                  params.input_left_pads_,
                                                  params.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);

        ref_invoker.Run(ref_argument);

        out_device_buf.FromDevice(out_device.mData.data());

        return ck::utils::check_err(
                   out_host.mData, out_device.mData, "Error: incorrect results!", 1e-5f, 1e-4f)
                   ? 0
                   : 1;
    }

    return 0;
}
