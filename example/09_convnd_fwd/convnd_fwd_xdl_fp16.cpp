// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_fwd_common.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

template <ck::index_t NumDimSpatial>
using DeviceConvNDFwdInstance = ck::tensor_operation::device::DeviceConvNdFwdNwcKxcNwk_Xdl<
    NumDimSpatial,  // NumDimSpatial
    InDataType,     //
    WeiDataType,    //
    OutDataType,    //
    AccDataType,    //
    InElementOp,    // Input Elementwise Operation
    WeiElementOp,   // Weights Elementwise Operation                          =
    OutElementOp,   // Output Elementwise Operation
    ConvFwdDefault, // ConvForwardSpecialization
    256,            // BlockSize
    128,            // MPerBlock
    256,            // NPerBlock
    4,              // K0PerBlock
    8,              // K1
    32,             // MPerXdl
    32,             // NPerXdl
    2,              // MXdlPerWave
    4,              // NXdlPerWave
    S<4, 64, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    8,              // ABlockTransferSrcScalarPerVector
    8,              // ABlockTransferDstScalarPerVector_K1
    true,           // ABlockLdsAddExtraM
    S<4, 64, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
    2,              // BBlockTransferSrcVectorDim
    8,              // BBlockTransferSrcScalarPerVector
    8,              // BBlockTransferDstScalarPerVector_K1
    true,           // BBlockLdsAddExtraN
    7,              // CThreadTransferSrcDstVectorDim
    1>;             // CThreadTransferDstScalarPerVector

template <ck::index_t NumDimSpatial>
using ReferenceConvNDFwdInstance =
    ck::tensor_operation::host::ReferenceConvFwd<NumDimSpatial,
                                                 ck::tensor_layout::convolution::NHWC,
                                                 ck::tensor_layout::convolution::KYXC,
                                                 ck::tensor_layout::convolution::NHWK,
                                                 InDataType,
                                                 WeiDataType,
                                                 OutDataType,
                                                 InElementOp,
                                                 WeiElementOp,
                                                 OutElementOp>;

int main(int argc, char* argv[])
{
    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    int num_dim_spatial  = 2;

    ck::tensor_operation::device::ConvParams params{
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

    if(num_dim_spatial == 1)
    {
        return run_conv_fwd_nhwc<1,
                                 InDataType,
                                 WeiDataType,
                                 OutDataType,
                                 AccDataType,
                                 InElementOp,
                                 WeiElementOp,
                                 OutElementOp,
                                 DeviceConvNDFwdInstance<1>,
                                 ReferenceConvNDFwdInstance<1>>(
            params, do_verification, init_method, time_kernel);
    }
    else if(num_dim_spatial == 2)
    {
        return run_conv_fwd_nhwc<2,
                                 InDataType,
                                 WeiDataType,
                                 OutDataType,
                                 AccDataType,
                                 InElementOp,
                                 WeiElementOp,
                                 OutElementOp,
                                 DeviceConvNDFwdInstance<2>,
                                 ReferenceConvNDFwdInstance<2>>(
            params, do_verification, init_method, time_kernel);
    }
    else if(num_dim_spatial == 3)
    {
        return run_conv_fwd_nhwc<3,
                                 InDataType,
                                 WeiDataType,
                                 OutDataType,
                                 AccDataType,
                                 InElementOp,
                                 WeiElementOp,
                                 OutElementOp,
                                 DeviceConvNDFwdInstance<3>,
                                 ReferenceConvNDFwdInstance<3>>(
            params, do_verification, init_method, time_kernel);
    }

    return 0;
}
