// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_bwd_data_common.hpp"

#include "ck/tensor_operation/gpu/device/device_convnd_bwd_data_nwc_kxc_nwk_xdl.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdDefault =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default;

template <ck::index_t NDimSpatial>
using DeviceConvNdBwdDataInstance = ck::tensor_operation::device::DeviceConvNdBwdDataNwcKxcNwk_Xdl<
    NDimSpatial,    // NDimSpatial
    InDataType,     // InDataType
    WeiDataType,    // WeiDataType
    OutDataType,    // OutDataType
    AccDataType,    // AccDataType
    InElementOp,    // InElementwiseOperation
    WeiElementOp,   // WeiElementwiseOperation
    OutElementOp,   // OutElementwiseOperation
    ConvBwdDefault, // ConvolutionBackwardDataSpecialization
    256,            // BlockSize
    128,            // MPerBlock
    128,            // NPerBlock
    4,              // K0PerBlock
    8,              // K1
    32,             // MPerXdl
    32,             // NPerXdl
    2,              // MXdlPerWave
    2,              // NXdlPerWave
    S<4, 64, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    8,              // ABlockTransferSrcScalarPerVector
    8,              // ABlockTransferDstScalarPerVector_K1
    true,           // ABlockLdsAddExtraM
    S<4, 64, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
    S<2, 0, 1>,     // BBlockTransferThreadClusterArrangeOrder
    S<0, 2, 1>,     // BBlockTransferSrcAccessOrder
    1,              // BBlockTransferSrcVectorDim
    2,              // BBlockTransferSrcScalarPerVector
    8,              // BBlockTransferDstScalarPerVector_K1
    true,           // BBlockLdsAddExtraN
    7,
    1>; // GemmCThreadTransferDstScalarPerVector

int main(int argc, char* argv[])
{
    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    int num_dim_spatial  = 2;

    ck::tensor_operation::device::ConvParams params{
        2, 128, 256, 256, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};

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
        return run_conv_bwd_data_nhwc<1,
                                      InDataType,
                                      WeiDataType,
                                      OutDataType,
                                      AccDataType,
                                      InElementOp,
                                      WeiElementOp,
                                      OutElementOp,
                                      DeviceConvNdBwdDataInstance<1>>(do_verification,
                                                                      init_method,
                                                                      time_kernel,
                                                                      params,
                                                                      in_element_op,
                                                                      wei_element_op,
                                                                      out_element_op);
    }
    else if(num_dim_spatial == 2)
    {
        return run_conv_bwd_data_nhwc<2,
                                      InDataType,
                                      WeiDataType,
                                      OutDataType,
                                      AccDataType,
                                      InElementOp,
                                      WeiElementOp,
                                      OutElementOp,
                                      DeviceConvNdBwdDataInstance<2>>(do_verification,
                                                                      init_method,
                                                                      time_kernel,
                                                                      params,
                                                                      in_element_op,
                                                                      wei_element_op,
                                                                      out_element_op);
    }
    else if(num_dim_spatial == 3)
    {
        return run_conv_bwd_data_nhwc<3,
                                      InDataType,
                                      WeiDataType,
                                      OutDataType,
                                      AccDataType,
                                      InElementOp,
                                      WeiElementOp,
                                      OutElementOp,
                                      DeviceConvNdBwdDataInstance<3>>(do_verification,
                                                                      init_method,
                                                                      time_kernel,
                                                                      params,
                                                                      in_element_op,
                                                                      wei_element_op,
                                                                      out_element_op);
    }

    return 0;
}
