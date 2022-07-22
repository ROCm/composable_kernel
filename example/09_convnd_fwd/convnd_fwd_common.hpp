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
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
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
          typename DeviceConvNDFwdInstance>
int run_conv_fwd(bool do_verification,
                 int init_method,
                 bool time_kernel,
                 const ck::utils::conv::ConvParam& conv_param,
                 const InElementOp& in_element_op,
                 const WeiElementOp& wei_element_op,
                 const OutElementOp& out_element_op)
{
    const auto in_desc  = ck::utils::conv::get_input_host_tensor_descriptor<InLayout>(conv_param);
    const auto wei_desc = ck::utils::conv::get_weight_host_tensor_descriptor<WeiLayout>(conv_param);
    const auto out_desc = ck::utils::conv::get_output_host_tensor_descriptor<OutLayout>(conv_param);

    // hacky, hardcoded for 2d NHWK
    const auto bias_desc = HostTensorDescriptor(
        std::vector<std::size_t>{static_cast<std::size_t>(conv_param.N_),
                                 static_cast<std::size_t>(conv_param.output_spatial_lengths_[0]),
                                 static_cast<std::size_t>(conv_param.output_spatial_lengths_[1]),
                                 static_cast<std::size_t>(conv_param.K_)},
        std::vector<std::size_t>{0, 0, 0, 1});

    Tensor<InDataType> in(in_desc);
    Tensor<WeiDataType> wei(wei_desc);
    Tensor<OutDataType> bias(bias_desc);
    Tensor<OutDataType> out_host(out_desc);
    Tensor<OutDataType> out_device(out_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "bias: " << bias.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        bias.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        bias.GenerateTensorValue(GeneratorTensor_3<OutDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpace());
    DeviceMem bias_device_buf(sizeof(OutDataType) * bias.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_device.mDesc.GetElementSpace());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    bias_device_buf.ToDevice(bias.mData.data());

    // tensor descriptor in NCHW/KXYC/NKHW dimensional order
    HostTensorDescriptor in_n_c_wis_desc   = in_desc;
    HostTensorDescriptor wei_k_c_xs_desc   = wei_desc;
    HostTensorDescriptor bias_n_k_wos_desc = bias_desc;
    HostTensorDescriptor out_n_k_wos_desc  = out_desc;

    // input
    if constexpr(ck::is_same_v<InLayout, ck::tensor_layout::convolution::NWC>)
    {
        in_n_c_wis_desc = transpose_host_tensor_descriptor_given_new2old(
            in_desc, std::vector<std::size_t>{0, 2, 1});
    }
    else if constexpr(ck::is_same_v<InLayout, ck::tensor_layout::convolution::NHWC>)
    {
        in_n_c_wis_desc = transpose_host_tensor_descriptor_given_new2old(
            in_desc, std::vector<std::size_t>{0, 3, 1, 2});
    }
    else if constexpr(ck::is_same_v<InLayout, ck::tensor_layout::convolution::NDHWC>)
    {
        in_n_c_wis_desc = transpose_host_tensor_descriptor_given_new2old(
            in_desc, std::vector<std::size_t>{0, 4, 1, 2, 3});
    }

    // weight
    if constexpr(ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KXC>)
    {
        wei_k_c_xs_desc = transpose_host_tensor_descriptor_given_new2old(
            wei_desc, std::vector<std::size_t>{0, 2, 1});
    }
    else if constexpr(ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KYXC>)
    {
        wei_k_c_xs_desc = transpose_host_tensor_descriptor_given_new2old(
            wei_desc, std::vector<std::size_t>{0, 3, 1, 2});
    }
    else if constexpr(ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KZYXC>)
    {
        wei_k_c_xs_desc = transpose_host_tensor_descriptor_given_new2old(
            wei_desc, std::vector<std::size_t>{0, 4, 1, 2, 3});
    }

    // output
    if constexpr(ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NWK>)
    {
        out_n_k_wos_desc = transpose_host_tensor_descriptor_given_new2old(
            out_desc, std::vector<std::size_t>{0, 2, 1});

        bias_n_k_wos_desc = transpose_host_tensor_descriptor_given_new2old(
            bias_desc, std::vector<std::size_t>{0, 2, 1});
    }
    else if constexpr(ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NHWK>)
    {
        out_n_k_wos_desc = transpose_host_tensor_descriptor_given_new2old(
            out_desc, std::vector<std::size_t>{0, 3, 1, 2});

        bias_n_k_wos_desc = transpose_host_tensor_descriptor_given_new2old(
            bias_desc, std::vector<std::size_t>{0, 3, 1, 2});
    }
    else if constexpr(ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NDHWK>)
    {
        out_n_k_wos_desc = transpose_host_tensor_descriptor_given_new2old(
            out_desc, std::vector<std::size_t>{0, 4, 1, 2, 3});

        bias_n_k_wos_desc = transpose_host_tensor_descriptor_given_new2old(
            bias_desc, std::vector<std::size_t>{0, 4, 1, 2, 3});
    }

    std::array<ck::index_t, NDimSpatial + 2> a_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 2> a_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 2> b_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 2> b_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 2> d_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 2> d_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 2> e_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 2> e_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](auto& x, auto& y) { std::copy(x.begin(), x.end(), y.begin()); };

    copy(in_n_c_wis_desc.GetLengths(), a_n_c_wis_lengths);
    copy(in_n_c_wis_desc.GetStrides(), a_n_c_wis_strides);
    copy(wei_k_c_xs_desc.GetLengths(), b_k_c_xs_lengths);
    copy(wei_k_c_xs_desc.GetStrides(), b_k_c_xs_strides);
    copy(bias_n_k_wos_desc.GetLengths(), d_n_k_wos_lengths);
    copy(bias_n_k_wos_desc.GetStrides(), d_n_k_wos_strides);
    copy(out_n_k_wos_desc.GetLengths(), e_n_k_wos_lengths);
    copy(out_n_k_wos_desc.GetStrides(), e_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    // do GEMM
    auto conv     = DeviceConvNDFwdInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(
        in_device_buf.GetDeviceBuffer(),
        wei_device_buf.GetDeviceBuffer(),
        std::array<const void*, 1>{bias_device_buf.GetDeviceBuffer()},
        out_device_buf.GetDeviceBuffer(),
        a_n_c_wis_lengths,
        a_n_c_wis_strides,
        b_k_c_xs_lengths,
        b_k_c_xs_strides,
        std::array<std::array<ck::index_t, NDimSpatial + 2>, 1>{{d_n_k_wos_lengths}},
        std::array<std::array<ck::index_t, NDimSpatial + 2>, 1>{{d_n_k_wos_strides}},
        e_n_k_wos_lengths,
        e_n_k_wos_strides,
        conv_filter_strides,
        conv_filter_dilations,
        input_left_pads,
        input_right_pads,
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
        using PassThrough = ck::tensor_operation::element_wise::PassThrough;

        Tensor<OutDataType> c_host(out_desc);

        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                     InLayout,
                                                                     WeiLayout,
                                                                     OutLayout,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     OutDataType,
                                                                     InElementOp,
                                                                     WeiElementOp,
                                                                     PassThrough>();

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei,
                                                  c_host,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int n = 0; n < out_host.mDesc.GetLengths()[0]; n++)
        {
            for(int ho = 0; ho < out_host.mDesc.GetLengths()[1]; ho++)
            {
                for(int wo = 0; wo < out_host.mDesc.GetLengths()[2]; wo++)
                {
                    for(int k = 0; k < out_host.mDesc.GetLengths()[3]; k++)
                    {
                        out_element_op(
                            out_host(n, ho, wo, k), c_host(n, ho, wo, k), bias(n, ho, wo, k));
                    }
                }
            }
        }

        out_device_buf.FromDevice(out_device.mData.data());

        return ck::utils::check_err(
                   out_device.mData, out_host.mData, "Error: incorrect results!", 1e-5f, 1e-4f)
                   ? 0
                   : 1;
    }

    return 0;
}
