// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_nwc_kxc_nwk_dl.hpp"

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

template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
    ck::tensor_operation::device::DeviceConvNdBwdWeightNwcKxcNwk_Dl<
        NDimSpatial,          // NDimSpatial
        InDataType,           // InDataType
        WeiDataType,          // WeiDataType
        OutDataType,          // OutDataType
        AccDataType,          // AccDataType
        InElementOp,          // InElementwiseOperation
        WeiElementOp,         // WeiElementwiseOperation
        OutElementOp,         // OutElementwiseOperation
        ConvBwdWeightDefault, // ConvBackwardWeightSpecialization
        256,                  // BlockSize
        128,                  // MPerBlock
        128,                  // NPerBlock
        16,                   // K0PerBlock
        2,                    // K1
        4,                    // M1PerThread
        4,                    // N1PerThread
        1,                    // KPerThread
        S<8, 2>,              // M1N1ThreadClusterM1Xs
        S<8, 2>,              // M1N1ThreadClusterN1Xs
        S<8, 1, 1, 2>,        // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
        S<2, 1, 128, 1>,      // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
        S<1, 2, 0, 3>,        // ABlockTransferThreadClusterArrangeOrder
        S<1, 2, 0, 3>,        // ABlockTransferSrcAccessOrder
        S<4, 1, 1, 2>,        // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
        S<1, 2, 0, 3>,        // ABlockTransferSrcVectorTensorContiguousDimOrder
        S<1, 1, 1, 2>,        // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
        S<1, 1, 8, 2>,        // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
        S<16, 1, 16, 1>,      // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
        S<0, 3, 1, 2>,        // BBlockTransferThreadClusterArrangeOrder
        S<0, 3, 1, 2>,        // BBlockTransferSrcAccessOrder
        S<1, 1, 8, 1>,        // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
        S<0, 3, 1, 2>,        // BBlockTransferSrcVectorTensorContiguousDimOrder
        S<1, 1, 1, 2>,        // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
        S<0, 1, 2, 3, 4, 5>,  // CThreadTransferSrcDstAccessOrder
        5,                    // CThreadTransferSrcDstVectorDim
        4>;                   // CThreadTransferDstScalarPerVector

template <ck::index_t NDimSpatial>
using HostConvBwdWeightInstance = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                                     InDataType,
                                                                                     WeiDataType,
                                                                                     OutDataType,
                                                                                     InElementOp,
                                                                                     WeiElementOp,
                                                                                     OutElementOp>;

template <ck::index_t NDimSpatial,
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
                        const HostTensorDescriptor& in_g_n_c_wis_desc,
                        const HostTensorDescriptor& wei_g_k_c_xs_desc,
                        const HostTensorDescriptor& out_g_n_k_wos_desc,
                        const InElementOp& in_element_op,
                        const WeiElementOp& wei_element_op,
                        const OutElementOp& out_element_op,
                        ck::index_t split_k)
{
    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei_host_result(wei_g_k_c_xs_desc);
    Tensor<WeiDataType> wei_device_result(wei_g_k_c_xs_desc);
    Tensor<OutDataType> out(out_g_n_k_wos_desc);

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

    std::array<ck::index_t, NDimSpatial> input_spatial_lengths{};
    std::array<ck::index_t, NDimSpatial> filter_spatial_lengths{};
    std::array<ck::index_t, NDimSpatial> output_spatial_lengths{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};
    
    auto range_copy = [](const auto& from, auto to) { std::copy(begin(from), end(from), to); };

    range_copy(conv_param.input_spatial_lengths_, begin(input_spatial_lengths));
    range_copy(conv_param.filter_spatial_lengths_, begin(filter_spatial_lengths));
    range_copy(conv_param.output_spatial_lengths_, begin(output_spatial_lengths));
    range_copy(conv_param.conv_filter_strides_, begin(conv_filter_strides));
    range_copy(conv_param.conv_filter_dilations_, begin(conv_filter_dilations));
    range_copy(conv_param.input_left_pads_, begin(input_left_pads));
    range_copy(conv_param.input_right_pads_, begin(input_right_pads));

    // do GEMM
    auto conv     = DeviceConvBwdWeightInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                      static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                      static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                      conv_param.G_,
                                      conv_param.N_,
                                      conv_param.K_,
                                      conv_param.C_,
                                      input_spatial_lengths,
                                      filter_spatial_lengths,
                                      output_spatial_lengths,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
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
        auto ref_conv = HostConvBwdWeightInstance<NDimSpatial>{};

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

void print_helper_msg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

int main(int argc, char* argv[])
{
    namespace ctc = ck::tensor_layout::convolution;

    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    ck::utils::conv::ConvParam conv_param{
        2, 4, 1, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};

    ck::index_t split_k = 1;

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
        do_verification                   = std::stoi(argv[1]);
        init_method                       = std::stoi(argv[2]);
        time_kernel                       = std::stoi(argv[3]);
        const ck::index_t num_dim_spatial = std::stoi(argv[4]);

        conv_param = ck::utils::conv::parse_conv_param(num_dim_spatial, 5, argv);
    }

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    if(conv_param.num_dim_spatial_ == 1)
    {
        using InLayout  = ctc::GNWC;
        using WeiLayout = ctc::GKXC;
        using OutLayout = ctc::GNWK;

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        return run_conv_bwd_weight<1,
                                   InDataType,
                                   WeiDataType,
                                   OutDataType,
                                   InElementOp,
                                   WeiElementOp,
                                   OutElementOp,
                                   DeviceConvBwdWeightInstance<1>>(do_verification,
                                                                     init_method,
                                                                     time_kernel,
                                                                     conv_param,
                                                                     in_g_n_c_wis_desc,
                                                                     wei_g_k_c_xs_desc,
                                                                     out_g_n_k_wos_desc,
                                                                     in_element_op,
                                                                     wei_element_op,
                                                                     out_element_op,
                                                                     split_k);
    }
    else if(conv_param.num_dim_spatial_ == 2)
    {
        using InLayout  = ctc::GNHWC;
        using WeiLayout = ctc::GKYXC;
        using OutLayout = ctc::GNHWK;

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        return run_conv_bwd_weight<2,
                                   InDataType,
                                   WeiDataType,
                                   OutDataType,
                                   InElementOp,
                                   WeiElementOp,
                                   OutElementOp,
                                   DeviceConvBwdWeightInstance<2>>(do_verification,
                                                                     init_method,
                                                                     time_kernel,
                                                                     conv_param,
                                                                     in_g_n_c_wis_desc,
                                                                     wei_g_k_c_xs_desc,
                                                                     out_g_n_k_wos_desc,
                                                                     in_element_op,
                                                                     wei_element_op,
                                                                     out_element_op,
                                                                     split_k);
    }
    else if(conv_param.num_dim_spatial_ == 3)
    {
        using InLayout  = ctc::GNDHWC;
        using WeiLayout = ctc::GKZYXC;
        using OutLayout = ctc::GNDHWK;

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        return run_conv_bwd_weight<3,
                                   InDataType,
                                   WeiDataType,
                                   OutDataType,
                                   InElementOp,
                                   WeiElementOp,
                                   OutElementOp,
                                   DeviceConvBwdWeightInstance<3>>(do_verification,
                                                                     init_method,
                                                                     time_kernel,
                                                                     conv_param,
                                                                     in_g_n_c_wis_desc,
                                                                     wei_g_k_c_xs_desc,
                                                                     out_g_n_k_wos_desc,
                                                                     in_element_op,
                                                                     wei_element_op,
                                                                     out_element_op,
                                                                     split_k);
    }

    return 0;
}
