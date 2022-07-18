// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_bwd_weight_common.hpp"

#include "ck/tensor_operation/gpu/device/device_convnd_bwd_weight_nwc_kxc_nwk_xdl_cshuffle.hpp"

using InDataType = ck::bhalf_t;
// bf16 kernel use fp32 atomic add to accumulate Weight tensor into global memory
using WeiDataType = float;
using OutDataType = ck::bhalf_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdWeightDefault =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;

template <ck::index_t NDimSpatial>
using DeviceConvndBwdWeightInstance =
    ck::tensor_operation::device::DeviceConvNdBwdWeightNwcKxcNwk_Xdl_CShuffle<
        NDimSpatial,          // NDimSpatial
        InDataType,           // InDataType
        WeiDataType,          // WeiDataType
        OutDataType,          // OutDataType
        AccDataType,          // AccDataType
        InElementOp,          // InElementwiseOperation
        WeiElementOp,         // WeiElementwiseOperation
        OutElementOp,         // OutElementwiseOperation
        ConvBwdWeightDefault, // ConvolutionBackwardWeightSpecialization
        256,                  // BlockSize
        128,                  // MPerBlock
        128,                  // NPerBlock
        4,                    // K0PerBlock
        8,                    // K1
        32,                   // MPerXdl
        32,                   // NPerXdl
        2,                    // MXdlPerWave
        2,                    // NXdlPerWave
        S<1, 4, 16, 4>,       // ABlockTransferThreadClusterLengths_K0_M_K1
        S<0, 3, 1, 2>,        // ABlockTransferThreadClusterArrangeOrder
        S<0, 2, 1, 3>,        // ABlockTransferSrcAccessOrder
        2,                    // ABlockTransferSrcVectorDim
        8,                    // ABlockTransferSrcScalarPerVector
        2,                    // ABlockTransferDstScalarPerVector_K1
        true,                 // ABlockLdsAddExtraM
        S<1, 4, 16, 4>,       // BBlockTransferThreadClusterLengths_K0_N_K1
        S<0, 3, 1, 2>,        // BBlockTransferThreadClusterArrangeOrder
        S<0, 2, 1, 3>,        // BBlockTransferSrcAccessOrder
        2,                    // BBlockTransferSrcVectorDim
        8,                    // BBlockTransferSrcScalarPerVector
        2,                    // BBlockTransferDstScalarPerVector_K1
        true,                 // BBlockLdsAddExtraN
        1,                    // CShuffleMXdlPerWavePerShuffle
        1,                    // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 4>,       // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        4>;                   // CBlockTransferScalarPerVector_NWaveNPerXdl

int main(int argc, char* argv[])
{
    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    int num_dim_spatial  = 2;

    ck::tensor_operation::device::ConvParams params{
        2, 32, 256, 1024, {3, 3}, {14, 14}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};

    ck::index_t split_k = 4;

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

        split_k = std::stoi(argv[5 + 3 + 6 * num_dim_spatial - 1]);
        split_k = std::max(1, split_k);
    }

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    if(num_dim_spatial == 1)
    {
        return run_conv_bwd_weight<1,
                                   ck::tensor_layout::convolution::NWC,
                                   ck::tensor_layout::convolution::KXC,
                                   ck::tensor_layout::convolution::NWK,
                                   InDataType,
                                   WeiDataType,
                                   OutDataType,
                                   InElementOp,
                                   WeiElementOp,
                                   OutElementOp,
                                   DeviceConvndBwdWeightInstance<1>>(do_verification,
                                                                     init_method,
                                                                     time_kernel,
                                                                     params,
                                                                     in_element_op,
                                                                     wei_element_op,
                                                                     out_element_op,
                                                                     split_k);
    }
    else if(num_dim_spatial == 2)
    {
        return run_conv_bwd_weight<2,
                                   ck::tensor_layout::convolution::NHWC,
                                   ck::tensor_layout::convolution::KYXC,
                                   ck::tensor_layout::convolution::NHWK,
                                   InDataType,
                                   WeiDataType,
                                   OutDataType,
                                   InElementOp,
                                   WeiElementOp,
                                   OutElementOp,
                                   DeviceConvndBwdWeightInstance<2>>(do_verification,
                                                                     init_method,
                                                                     time_kernel,
                                                                     params,
                                                                     in_element_op,
                                                                     wei_element_op,
                                                                     out_element_op,
                                                                     split_k);
    }
    else if(num_dim_spatial == 3)
    {
        return run_conv_bwd_weight<3,
                                   ck::tensor_layout::convolution::NDHWC,
                                   ck::tensor_layout::convolution::KZYXC,
                                   ck::tensor_layout::convolution::NDHWK,
                                   InDataType,
                                   WeiDataType,
                                   OutDataType,
                                   InElementOp,
                                   WeiElementOp,
                                   OutElementOp,
                                   DeviceConvndBwdWeightInstance<3>>(do_verification,
                                                                     init_method,
                                                                     time_kernel,
                                                                     params,
                                                                     in_element_op,
                                                                     wei_element_op,
                                                                     out_element_op,
                                                                     split_k);
    }

    return 0;
}
