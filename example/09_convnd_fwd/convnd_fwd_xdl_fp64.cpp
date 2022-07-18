// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_fwd_common.hpp"

#include "ck/tensor_operation/gpu/device/device_convnd_fwd_nwc_kxc_nwk_xdl.hpp"

using InDataType  = double;
using WeiDataType = double;
using OutDataType = double;
using AccDataType = double;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

template <ck::index_t NDimSpatial>
using DeviceConvNDFwdInstance = ck::tensor_operation::device::DeviceConvNdFwdNwcKxcNwk_Xdl<
    NDimSpatial,    // NDimSpatial
    InDataType,     //
    WeiDataType,    //
    OutDataType,    //
    AccDataType,    //
    InElementOp,    // Input Elementwise Operation
    WeiElementOp,   // Weights Elementwise Operation
    OutElementOp,   // Output Elementwise Operation
    ConvFwdDefault, // ConvForwardSpecialization
    256,            // BlockSize
    128,            // MPerBlock
    128,            // NPerBlock
    4,              // K0PerBlock
    2,              // K1
    16,             // MPerXDL
    16,             // NPerXDL
    4,              // MXdlPerWave
    4,              // NXdlPerWave
    S<4, 64, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    2,              // ABlockTransferSrcScalarPerVector
    2,              // ABlockTransferDstScalarPerVector_K1
    true,           // ABlockLdsAddExtraM
    S<4, 64, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
    2,              // BBlockTransferSrcVectorDim
    2,              // BBlockTransferSrcScalarPerVector
    2,              // BBlockTransferDstScalarPerVector_K1
    true,           // BBlockTransferAddExtraN
    7,              // CThreadTransferSrcDstVectorDim
    1>;             // CThreadTransferDstScalarPerVector

int main(int argc, char* argv[])
{
    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    int num_dim_spatial  = 2;

    ck::utils::conv::ConvParam params{
        2, 128, 256, 192, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};

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
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
        num_dim_spatial = std::stoi(argv[4]);

        params = parse_conv_params(num_dim_spatial, 5, argv);
    }

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    if(num_dim_spatial == 1)
    {
        return run_conv_fwd<1,
                            ck::tensor_layout::convolution::NWC,
                            ck::tensor_layout::convolution::KXC,
                            ck::tensor_layout::convolution::NWK,
                            InDataType,
                            WeiDataType,
                            OutDataType,
                            InElementOp,
                            WeiElementOp,
                            OutElementOp,
                            DeviceConvNDFwdInstance<1>>(do_verification,
                                                        init_method,
                                                        time_kernel,
                                                        params,
                                                        in_element_op,
                                                        wei_element_op,
                                                        out_element_op);
    }
    else if(num_dim_spatial == 2)
    {
        return run_conv_fwd<2,
                            ck::tensor_layout::convolution::NHWC,
                            ck::tensor_layout::convolution::KYXC,
                            ck::tensor_layout::convolution::NHWK,
                            InDataType,
                            WeiDataType,
                            OutDataType,
                            InElementOp,
                            WeiElementOp,
                            OutElementOp,
                            DeviceConvNDFwdInstance<2>>(do_verification,
                                                        init_method,
                                                        time_kernel,
                                                        params,
                                                        in_element_op,
                                                        wei_element_op,
                                                        out_element_op);
    }
    else if(num_dim_spatial == 3)
    {
        return run_conv_fwd<3,
                            ck::tensor_layout::convolution::NDHWC,
                            ck::tensor_layout::convolution::KZYXC,
                            ck::tensor_layout::convolution::NDHWK,
                            InDataType,
                            WeiDataType,
                            OutDataType,
                            InElementOp,
                            WeiElementOp,
                            OutElementOp,
                            DeviceConvNDFwdInstance<3>>(do_verification,
                                                        init_method,
                                                        time_kernel,
                                                        params,
                                                        in_element_op,
                                                        wei_element_op,
                                                        out_element_op);
    }

    return 0;
}
