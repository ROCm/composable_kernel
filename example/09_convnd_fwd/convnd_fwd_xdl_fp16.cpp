// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_fwd_common.hpp"

#include "ck/tensor_operation/gpu/device/device_convnd_fwd_nwc_kxc_nwk_xdl.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_fwd_multiple_d_xdl_cshuffle.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
// using OutElementOp = ck::tensor_operation::element_wise::UnaryConvert;
using OutElementOp = ck::tensor_operation::element_wise::AddRelu;

#if 0
static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

template <ck::index_t NDimSpatial>
using DeviceConvNDFwdInstance = ck::tensor_operation::device::DeviceConvNdFwdNwcKxcNwk_Xdl<
    NDimSpatial,    //
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
    256,            // NPerBlock
    4,              // K0PerBlock
    8,              // K1
    32,             // MPerXdl
    32,             // NPerXdl
    2,              // MXdlPerWave
    4,              // NXdlPerWave
    S<4, 64, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>, 1    // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    8,              // ABlockTransferSrcScalarPerVector
    8,              // ABlockTransferDstScalarPerVector_K1
    true,           // ABlockLdsExtraM
    S<4, 64, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
    2,              // BBlockTransferSrcVectorDim
    8,              // BBlockTransferSrcScalarPerVector
    8,              // BBlockTransferDstScalarPerVector_K1
    true,           // BBlockLdsExtraN
    7,              // CThreadTransferSrcDstVectorDim
    1>;             // CThreadTransferDstScalarPerVector
#else
using CShuffleDataType = ck::half_t;
using DDataType        = ck::half_t;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::index_t NDimSpatial>
using DeviceConvNDFwdInstance = ck::tensor_operation::device::DeviceConvFwdMultipleD_Xdl_CShuffle<
    NDimSpatial,
#if 0
    ck::tuple_element_t<NDimSpatial - 1,
                        ck::Tuple<ck::tensor_layout::convolution::G_NW_C,
                                  ck::tensor_layout::convolution::G_NHW_C,
                                  ck::tensor_layout::convolution::G_NDHW_C>>,
    ck::tuple_element_t<NDimSpatial - 1,
                        ck::Tuple<ck::tensor_layout::convolution::G_K_X_C,
                                  ck::tensor_layout::convolution::G_K_YX_C,
                                  ck::tensor_layout::convolution::G_K_ZYX_C>>,
    ck::Tuple<ck::tuple_element_t<NDimSpatial - 1,
                                  ck::Tuple<ck::tensor_layout::convolution::G_NW_K,
                                            ck::tensor_layout::convolution::G_NHW_K,
                                            ck::tensor_layout::convolution::G_NDHW_K>>>,
    ck::tuple_element_t<NDimSpatial - 1,
                        ck::Tuple<ck::tensor_layout::convolution::G_NW_K,
                                  ck::tensor_layout::convolution::G_NHW_K,
                                  ck::tensor_layout::convolution::G_NDHW_K>>,
#else
    ck::tuple_element_t<NDimSpatial - 1,
                        ck::Tuple<ck::tensor_layout::convolution::NWGC,
                                  ck::tensor_layout::convolution::NHWGC,
                                  ck::tensor_layout::convolution::NDHWGC>>,
    ck::tuple_element_t<NDimSpatial - 1,
                        ck::Tuple<ck::tensor_layout::convolution::KXGC,
                                  ck::tensor_layout::convolution::KYXGC,
                                  ck::tensor_layout::convolution::KZYXGC>>,
    ck::Tuple<ck::tuple_element_t<NDimSpatial - 1,
                                  ck::Tuple<ck::tensor_layout::convolution::NWGK,
                                            ck::tensor_layout::convolution::NHWGK,
                                            ck::tensor_layout::convolution::NDHWGK>>>,
    ck::tuple_element_t<NDimSpatial - 1,
                        ck::Tuple<ck::tensor_layout::convolution::NWGK,
                                  ck::tensor_layout::convolution::NHWGK,
                                  ck::tensor_layout::convolution::NDHWGK>>,
#endif
    InDataType,
    WeiDataType,
    AccDataType,
    CShuffleDataType,
    ck::Tuple<DDataType>,
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
    8,           // K1
    32,          // MPerXdl
    32,          // NPerXdl
    2,           // MXdlPerWave
    4,           // NXdlPerWave
    S<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,  // ABlockTransferSrcAccessOrder
    2,           // ABlockTransferSrcVectorDim
    8,           // ABlockTransferSrcScalarPerVector
    8,           // ABlockTransferDstScalarPerVector_K1
    1,           // ABlockLdsExtraM
    S<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,  // BBlockTransferSrcAccessOrder
    2,           // BBlockTransferSrcVectorDim
    8,           // BBlockTransferSrcScalarPerVector
    8,           // BBlockTransferDstScalarPerVector_K1
    1,           // BBlockLdsExtraN
    1,
    1,
    S<1, 32, 1, 8>,
    8>;
#endif

int main(int argc, char* argv[])
{
    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    int num_dim_spatial  = 2;

    ck::utils::conv::ConvParam params{
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
                            ck::tensor_layout::convolution::NWGC,
                            ck::tensor_layout::convolution::KXGC,
                            ck::tensor_layout::convolution::NWGK,
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
                            ck::tensor_layout::convolution::NHWGC,
                            ck::tensor_layout::convolution::KYXGC,
                            ck::tensor_layout::convolution::NHWGK,
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
                            ck::tensor_layout::convolution::NDHWGC,
                            ck::tensor_layout::convolution::KZYXGC,
                            ck::tensor_layout::convolution::NDHWGK,
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
