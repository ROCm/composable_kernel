// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_fwd_common.hpp"

using InDataType  = float;
using WeiDataType = float;
using OutDataType = float;
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
    InDataType,     //
    WeiDataType,    //
    OutDataType,    //
    AccDataType,    //
    InElementOp,    // Input Elementwise Operation
    WeiElementOp,   // Weights Elementwise Operation
    OutElementOp,   // Output Elementwise Operation
    ConvFwdDefault, // ConvForwardSpecialization
    NumDimSpatial,  // NumDimSpatial
    256,            // BlockSize
    256,            // MPerBlock
    128,            // NPerBlock
    4,              // K0PerBlock
    4,              // K1
    32,             // MPerXDL
    32,             // NPerXDL
    4,              // MXdlPerWave
    2,              // NXdlPerWave
    S<4, 64, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    4,              // ABlockTransferSrcScalarPerVector
    4,              // ABlockTransferDstScalarPerVector_K1
    true,           // ABlockLdsAddExtraM
    S<4, 64, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
    2,              // BBlockTransferSrcVectorDim
    4,              // BBlockTransferSrcScalarPerVector
    4,              // BBlockTransferDstScalarPerVector_K1
    true,           // BBlockTransferAddExtraN
    7,              // CThreadTransferSrcDstVectorDim
    1>;             // CThreadTransferDstScalarPerVector

template <ck::index_t NumDimSpatial>
using ReferenceConvNDFwdInstance = ck::tensor_operation::host::ReferenceConvFwd<InDataType,
                                                                                WeiDataType,
                                                                                OutDataType,
                                                                                InElementOp,
                                                                                WeiElementOp,
                                                                                OutElementOp,
                                                                                NumDimSpatial>;

int main(int argc, char* argv[])
{
    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    int num_dim_spatial  = 2;

    ck::tensor_operation::device::ConvParams params;

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

        params = parse_conv_params(num_dim_spatial, argc, argv);
    }

    if(num_dim_spatial == 1)
    {
        return run_conv_fwd<1,
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
        return run_conv_fwd<2,
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
        return run_conv_fwd<3,
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
