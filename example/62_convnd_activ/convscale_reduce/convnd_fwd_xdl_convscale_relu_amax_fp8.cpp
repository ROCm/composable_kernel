// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_fwd_convscale_reduce_common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"

using InDataType       = ck::f8_t;
using WeiDataType      = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = float;
using ConvOutDataType  = float;    // data type of convolution result
using OutDataType      = ck::f8_t; // data type of final result
using AComputeDataType = ck::f8_t;
using BComputeDataType = ck::f8_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = ConvScaleRelu;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::index_t NDimSpatial, typename InLayout, typename WeiLayout, typename OutLayout>
using DeviceGroupedConvNDFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        NDimSpatial,
        InLayout,
        WeiLayout,
        ck::Tuple<>,
        OutLayout,
        InDataType,
        WeiDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<>,
        ConvOutDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvSpec,    // ConvForwardSpecialization
        GemmSpec,    // GemmSpecialization
        1,           //
        256,         // BlockSize
        128,         // MPerBlock
        256,         // NPerBlock
        32,          // KPerBlock
        8,           // AK1
        8,           // BK1
        32,          // MPerXdl
        32,          // NPerXdl
        2,           // MXdlPerWave
        4,           // NXdlPerWave
        S<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,           // ABlockTransferSrcVectorDim
        8,           // ABlockTransferSrcScalarPerVector
        8,           // ABlockTransferDstScalarPerVector_AK1
        1,           // ABlockLdsExtraM
        S<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // BBlockTransferSrcAccessOrder
        2,           // BBlockTransferSrcVectorDim
        8,           // BBlockTransferSrcScalarPerVector
        8,           // BBlockTransferDstScalarPerVector_BK1
        1,           // BBlockLdsExtraN
        1,
        1,
        S<1, 32, 1, 8>,
        8,
        AComputeDataType,
        BComputeDataType>;

bool run_convnd_fwd_example(int argc, char* argv[])
{
    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

#if 1
    ck::utils::conv::ConvParam conv_param{
        2, 1, 128, 256, 192, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};
#else
    ck::utils::conv::ConvParam conv_param{
        2, 1, 2, 16, 16, {3, 3}, {21, 21}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};
#endif

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

    // instantiate in and wei element ops, will
    // instantiate out_element_op below for every iteration
    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};

    const auto run = [&](auto ndim_spatial, auto in_layout, auto wei_layout, auto out_layout) {
        constexpr ck::index_t ndim_spatial_value = ndim_spatial.value;

        using InLayout  = decltype(in_layout);
        using WeiLayout = decltype(wei_layout);
        using OutLayout = decltype(out_layout);

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        return run_grouped_conv_fwd<
            ndim_spatial_value,
            InDataType,
            WeiDataType,
            ConvOutDataType,
            OutDataType,
            InElementOp,
            WeiElementOp,
            OutElementOp,
            DeviceGroupedConvNDFwdInstance<ndim_spatial_value, InLayout, WeiLayout, OutLayout>>(
            do_verification,
            init_method,
            time_kernel,
            conv_param,
            in_g_n_c_wis_desc,
            wei_g_k_c_xs_desc,
            out_g_n_k_wos_desc,
            in_element_op,
            wei_element_op);
    };

    namespace ctc = ck::tensor_layout::convolution;

    if(conv_param.num_dim_spatial_ == 1)
    {
        return run(ck::Number<1>{}, ctc::GNWC{}, ctc::GKXC{}, ctc::GNWK{});
    }
    else if(conv_param.num_dim_spatial_ == 2)
    {
        return run(ck::Number<2>{}, ctc::GNHWC{}, ctc::GKYXC{}, ctc::GNHWK{});
    }
    else if(conv_param.num_dim_spatial_ == 3)
    {
        return run(ck::Number<3>{}, ctc::GNDHWC{}, ctc::GKZYXC{}, ctc::GNDHWK{});
    }

    return true;
}

int main(int argc, char* argv[]) { return run_convnd_fwd_example(argc, argv) ? 0 : 1; }
