// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "grouped_conv_bwd_data_bias_relu_common.hpp"

#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using OutDataType      = ck::half_t;
using WeiDataType      = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using BiasDataType     = ck::half_t; // bias
using InDataType       = ck::half_t;

using OutLayout  = ck::tensor_layout::convolution::GNHWK;
using WeiLayout  = ck::tensor_layout::convolution::GKYXC;
using BiasLayout = ck::tensor_layout::convolution::G_C;
using InLayout   = ck::tensor_layout::convolution::GNHWC;

using OutElementOp     = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp     = ck::tensor_operation::element_wise::PassThrough;
using CBiasInElementOp = ck::tensor_operation::element_wise::AddRelu;

static constexpr auto ConvBwdDataDefault =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default;

template <ck::index_t NDimSpatial>
using DeviceConvNdBwdDataInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<
        NDimSpatial,
        OutLayout,
        WeiLayout,
        ck::Tuple<BiasLayout>,
        InLayout,
        OutDataType,
        WeiDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<BiasDataType>,
        InDataType,
        OutElementOp,
        WeiElementOp,
        CBiasInElementOp,
        ConvBwdDataDefault,
        true, // DoPadGemmM
        true, // DoPadGemmN
        1,
        256,
        128,
        256,
        32,
        8,
        2,
        32,
        32,
        2,
        4,
        S<4, 64, 1>,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        1,
        S<4, 64, 1>,
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        4,
        2,
        0,
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
        2, 2, 128, 256, 256, {3, 3}, {14, 14}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};

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

    const auto in_element_op  = CBiasInElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    if(conv_param.num_dim_spatial_ == 2)
    {
        // output image: GNHWK
        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        // weight: GKYXC
        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        // input image bias: G_C
        const auto bias_g_n_c_wis_desc =
            HostTensorDescriptor({conv_param.G_,
                                  conv_param.N_,
                                  conv_param.C_,
                                  conv_param.input_spatial_lengths_[0],
                                  conv_param.input_spatial_lengths_[1]},
                                 {
                                     conv_param.C_, // g
                                     0,             // n
                                     1,             // c
                                     0,             // hi
                                     0              // wi
                                 });

        // input image: GNHWC
        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        using DeviceInstance = DeviceConvNdBwdDataInstance<2>;

        run_conv_bwd_data_bias_relu<2,
                                    OutDataType,
                                    WeiDataType,
                                    BiasDataType,
                                    InDataType,
                                    OutElementOp,
                                    WeiElementOp,
                                    CBiasInElementOp,
                                    DeviceInstance>(do_verification,
                                                    init_method,
                                                    time_kernel,
                                                    conv_param,
                                                    out_g_n_k_wos_desc,
                                                    wei_g_k_c_xs_desc,
                                                    bias_g_n_c_wis_desc,
                                                    in_g_n_c_wis_desc,
                                                    wei_element_op,
                                                    out_element_op,
                                                    in_element_op);
    }

    return 0;
}
