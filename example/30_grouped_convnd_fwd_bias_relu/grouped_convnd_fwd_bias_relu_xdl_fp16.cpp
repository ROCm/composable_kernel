// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "grouped_convnd_fwd_bias_common.hpp"

#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_d_xdl_cshuffle.hpp"

#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

using InDataType       = ck::half_t;
using WeiDataType      = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using BiasDataType     = ck::half_t;
using OutDataType      = ck::half_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::AddRelu;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename BiasLayout,
          typename OutLayout>
using DeviceGroupedConvNDFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<
        NDimSpatial,
        InLayout,
        WeiLayout,
        ck::Tuple<BiasLayout>,
        OutLayout,
        InDataType,
        WeiDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<BiasDataType>,
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
        2, 2, 128, 256, 192, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};

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
        using InLayout   = ctc::G_NW_C;
        using WeiLayout  = ctc::G_K_X_C;
        using BiasLayout = ctc::G_NW_K;
        using OutLayout  = ctc::G_NW_K;

        const auto in_g_n_c_wis_desc = HostTensorDescriptor(
            {conv_param.G_, conv_param.N_, conv_param.C_, conv_param.input_spatial_lengths_[0]},
            {
                conv_param.C_,                                                        // g
                conv_param.input_spatial_lengths_[0] * conv_param.G_ * conv_param.C_, // n
                1,                                                                    // c
                conv_param.G_ * conv_param.C_                                         // wi
            });

        const auto wei_g_k_c_xs_desc = HostTensorDescriptor(
            {conv_param.G_, conv_param.K_, conv_param.C_, conv_param.filter_spatial_lengths_[0]},
            {
                conv_param.C_,                                                         // g
                conv_param.filter_spatial_lengths_[0] * conv_param.G_ * conv_param.C_, // k
                1,                                                                     // c
                conv_param.G_ * conv_param.C_                                          // x
            });

        const auto bias_g_n_k_wos_desc = HostTensorDescriptor(
            {conv_param.G_, conv_param.N_, conv_param.K_, conv_param.output_spatial_lengths_[0]},
            {
                conv_param.K_, // g
                0,             // k
                1,             // c
                0              // x
            });

        const auto out_g_n_k_wos_desc = HostTensorDescriptor(
            {conv_param.G_, conv_param.N_, conv_param.K_, conv_param.output_spatial_lengths_[0]},
            {
                conv_param.K_,                                                         // g
                conv_param.output_spatial_lengths_[0] * conv_param.G_ * conv_param.K_, // n
                1,                                                                     // k
                conv_param.G_ * conv_param.K_                                          // wo
            });

        return run_grouped_conv_fwd_bias<
            1,
            InDataType,
            WeiDataType,
            OutDataType,
            InElementOp,
            WeiElementOp,
            OutElementOp,
            DeviceGroupedConvNDFwdInstance<1, InLayout, WeiLayout, BiasLayout, OutLayout>>(
            do_verification,
            init_method,
            time_kernel,
            conv_param,
            in_g_n_c_wis_desc,
            wei_g_k_c_xs_desc,
            bias_g_n_k_wos_desc,
            out_g_n_k_wos_desc,
            in_element_op,
            wei_element_op,
            out_element_op);
    }
    else if(conv_param.num_dim_spatial_ == 2)
    {
        using InLayout   = ctc::G_NHW_C;
        using WeiLayout  = ctc::G_K_YX_C;
        using BiasLayout = ctc::G_NHW_K;
        using OutLayout  = ctc::G_NHW_K;

        const auto in_g_n_c_wis_desc = HostTensorDescriptor(
            {conv_param.G_,
             conv_param.N_,
             conv_param.C_,
             conv_param.input_spatial_lengths_[0],
             conv_param.input_spatial_lengths_[1]},
            {
                conv_param.output_spatial_lengths_[0] * conv_param.C_, // g
                conv_param.input_spatial_lengths_[0] * conv_param.input_spatial_lengths_[1] *
                    conv_param.G_ * conv_param.C_,                                    // n
                1,                                                                    // c
                conv_param.input_spatial_lengths_[1] * conv_param.G_ * conv_param.C_, // hi
                conv_param.G_ * conv_param.C_                                         // wi
            });

        const auto wei_g_k_c_xs_desc = HostTensorDescriptor(
            {conv_param.G_,
             conv_param.K_,
             conv_param.C_,
             conv_param.filter_spatial_lengths_[0],
             conv_param.filter_spatial_lengths_[1]},
            {
                conv_param.C_, // g
                conv_param.filter_spatial_lengths_[0] * conv_param.filter_spatial_lengths_[1] *
                    conv_param.G_ * conv_param.C_,                                     // k
                1,                                                                     // c
                conv_param.filter_spatial_lengths_[1] * conv_param.G_ * conv_param.C_, // y
                conv_param.G_ * conv_param.C_                                          // x
            });

        const auto bias_g_n_k_wos_desc =
            HostTensorDescriptor({conv_param.G_,
                                  conv_param.N_,
                                  conv_param.K_,
                                  conv_param.output_spatial_lengths_[0],
                                  conv_param.output_spatial_lengths_[1]},
                                 {
                                     conv_param.K_, // g
                                     0,             // n
                                     1,             // k
                                     0,             // ho
                                     0              // wo
                                 });

        const auto out_g_n_k_wos_desc = HostTensorDescriptor(
            {conv_param.G_,
             conv_param.N_,
             conv_param.K_,
             conv_param.output_spatial_lengths_[0],
             conv_param.output_spatial_lengths_[1]},
            {
                conv_param.K_, // g
                conv_param.output_spatial_lengths_[0] * conv_param.output_spatial_lengths_[1] *
                    conv_param.G_ * conv_param.K_,                                     // n
                1,                                                                     // k
                conv_param.output_spatial_lengths_[1] * conv_param.G_ * conv_param.K_, // ho
                conv_param.G_ * conv_param.K_                                          // wo
            });

        return run_grouped_conv_fwd_bias<
            2,
            InDataType,
            WeiDataType,
            OutDataType,
            InElementOp,
            WeiElementOp,
            OutElementOp,
            DeviceGroupedConvNDFwdInstance<2, InLayout, WeiLayout, BiasLayout, OutLayout>>(
            do_verification,
            init_method,
            time_kernel,
            conv_param,
            in_g_n_c_wis_desc,
            wei_g_k_c_xs_desc,
            bias_g_n_k_wos_desc,
            out_g_n_k_wos_desc,
            in_element_op,
            wei_element_op,
            out_element_op);
    }
    else if(conv_param.num_dim_spatial_ == 3)
    {
        using InLayout   = ctc::G_NDHW_C;
        using WeiLayout  = ctc::G_K_ZYX_C;
        using BiasLayout = ctc::G_NDHW_K;
        using OutLayout  = ctc::G_NDHW_K;

        const auto in_g_n_c_wis_desc = HostTensorDescriptor(
            {conv_param.G_,
             conv_param.N_,
             conv_param.C_,
             conv_param.input_spatial_lengths_[0],
             conv_param.input_spatial_lengths_[1],
             conv_param.input_spatial_lengths_[2]},
            {
                conv_param.output_spatial_lengths_[0] * conv_param.C_, // g
                conv_param.input_spatial_lengths_[0] * conv_param.input_spatial_lengths_[1] *
                    conv_param.input_spatial_lengths_[2] * conv_param.G_ * conv_param.C_, // n
                1,                                                                        // c
                conv_param.input_spatial_lengths_[1] * conv_param.input_spatial_lengths_[2] *
                    conv_param.G_ * conv_param.C_,                                    // di
                conv_param.input_spatial_lengths_[2] * conv_param.G_ * conv_param.C_, // hi
                conv_param.G_ * conv_param.C_                                         // wi
            });

        const auto wei_g_k_c_xs_desc = HostTensorDescriptor(
            {conv_param.G_,
             conv_param.K_,
             conv_param.C_,
             conv_param.filter_spatial_lengths_[0],
             conv_param.filter_spatial_lengths_[1],
             conv_param.filter_spatial_lengths_[2]},
            {
                conv_param.C_, // g
                conv_param.filter_spatial_lengths_[0] * conv_param.filter_spatial_lengths_[1] *
                    conv_param.filter_spatial_lengths_[2] * conv_param.G_ * conv_param.C_, // k
                1,                                                                         // c
                conv_param.filter_spatial_lengths_[1] * conv_param.filter_spatial_lengths_[2] *
                    conv_param.G_ * conv_param.C_,                                     // z
                conv_param.filter_spatial_lengths_[2] * conv_param.G_ * conv_param.C_, // y
                conv_param.G_ * conv_param.C_                                          // x
            });

        const auto bias_g_n_k_wos_desc =
            HostTensorDescriptor({conv_param.G_,
                                  conv_param.N_,
                                  conv_param.K_,
                                  conv_param.output_spatial_lengths_[0],
                                  conv_param.output_spatial_lengths_[1],
                                  conv_param.output_spatial_lengths_[2]},
                                 {
                                     conv_param.K_, // g
                                     0,             // n
                                     1,             // k
                                     0,             // z
                                     0,             // y
                                     0              // x
                                 });

        const auto out_g_n_k_wos_desc = HostTensorDescriptor(
            {conv_param.G_,
             conv_param.N_,
             conv_param.K_,
             conv_param.output_spatial_lengths_[0],
             conv_param.output_spatial_lengths_[1],
             conv_param.output_spatial_lengths_[2]},
            {
                conv_param.K_, // g
                conv_param.output_spatial_lengths_[0] * conv_param.output_spatial_lengths_[1] *
                    conv_param.output_spatial_lengths_[2] * conv_param.G_ * conv_param.K_, // n
                1,                                                                         // k
                conv_param.output_spatial_lengths_[1] * conv_param.output_spatial_lengths_[2] *
                    conv_param.G_ * conv_param.K_,                                     // do
                conv_param.output_spatial_lengths_[2] * conv_param.G_ * conv_param.K_, // ho
                conv_param.G_ * conv_param.K_                                          // wo
            });

        return run_grouped_conv_fwd_bias<
            3,
            InDataType,
            WeiDataType,
            OutDataType,
            InElementOp,
            WeiElementOp,
            OutElementOp,
            DeviceGroupedConvNDFwdInstance<3, InLayout, WeiLayout, BiasLayout, OutLayout>>(
            do_verification,
            init_method,
            time_kernel,
            conv_param,
            in_g_n_c_wis_desc,
            wei_g_k_c_xs_desc,
            bias_g_n_k_wos_desc,
            out_g_n_k_wos_desc,
            in_element_op,
            wei_element_op,
            out_element_op);
    }

    return 0;
}
