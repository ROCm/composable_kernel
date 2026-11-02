// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "convnd_fwd_convscale_reduce_common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck/library/utility/run_fp8_example.hpp"

template <typename Fp8, typename Bf8>
struct Example
{

    using InDataType       = Fp8;
    using WeiDataType      = Fp8;
    using AccDataType      = float;
    using CShuffleDataType = float;
    using ConvOutDataType  = float; // data type of convolution result
    using OutDataType      = Fp8;   // data type of final result
    using AComputeDataType = Fp8;
    using BComputeDataType = Fp8;

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
            64,          // KPerBlock
            8,           // AK1
            8,           // BK1
            16,          // MPerXdl
            16,          // NPerXdl
            4,           // MXdlPerWave
            8,           // NXdlPerWave
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
            4,
            AComputeDataType,
            BComputeDataType>;

#include "run_convnd_fwd_example.inc"

    int Run(int argc, char* argv[]);
};

template <typename Fp8, typename Bf8>
int Example<Fp8, Bf8>::Run(int argc, char* argv[])
{
    return run_convnd_fwd_example(argc, argv) ? 0 : 1;
}

int main(int argc, char* argv[]) { return ck::utils::run_fp8_example<Example>(argc, argv); }
