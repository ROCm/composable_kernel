// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_fwd_common.hpp"

#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_d_xdl_cshuffle.hpp"

#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

using InDataType       = ck::bhalf_t;
using WeiDataType      = ck::bhalf_t;
using AccDataType      = float;
using CShuffleDataType = float;
using OutDataType      = ck::bhalf_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::index_t NDimSpatial, typename InLayout, typename WeiLayout, typename OutLayout>
using DeviceGroupedConvNDFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<
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
        OutDataType,
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
        8>;

int main(int argc, char* argv[])
{
    namespace ctc = ck::tensor_layout::convolution;

    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    ck::utils::conv::ConvParam conv_param{
        2, 1, 128, 256, 192, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};

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

        return run_grouped_conv_fwd<
            1,
            InDataType,
            WeiDataType,
            OutDataType,
            InElementOp,
            WeiElementOp,
            OutElementOp,
            DeviceGroupedConvNDFwdInstance<1, InLayout, WeiLayout, OutLayout>>(do_verification,
                                                                               init_method,
                                                                               time_kernel,
                                                                               conv_param,
                                                                               in_g_n_c_wis_desc,
                                                                               wei_g_k_c_xs_desc,
                                                                               out_g_n_k_wos_desc,
                                                                               in_element_op,
                                                                               wei_element_op,
                                                                               out_element_op);
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

        return run_grouped_conv_fwd<
            2,
            InDataType,
            WeiDataType,
            OutDataType,
            InElementOp,
            WeiElementOp,
            OutElementOp,
            DeviceGroupedConvNDFwdInstance<2, InLayout, WeiLayout, OutLayout>>(do_verification,
                                                                               init_method,
                                                                               time_kernel,
                                                                               conv_param,
                                                                               in_g_n_c_wis_desc,
                                                                               wei_g_k_c_xs_desc,
                                                                               out_g_n_k_wos_desc,
                                                                               in_element_op,
                                                                               wei_element_op,
                                                                               out_element_op);
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

        return run_grouped_conv_fwd<
            3,
            InDataType,
            WeiDataType,
            OutDataType,
            InElementOp,
            WeiElementOp,
            OutElementOp,
            DeviceGroupedConvNDFwdInstance<3, InLayout, WeiLayout, OutLayout>>(do_verification,
                                                                               init_method,
                                                                               time_kernel,
                                                                               conv_param,
                                                                               in_g_n_c_wis_desc,
                                                                               wei_g_k_c_xs_desc,
                                                                               out_g_n_k_wos_desc,
                                                                               in_element_op,
                                                                               wei_element_op,
                                                                               out_element_op);
    }

    return 0;
}
