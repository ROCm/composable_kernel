// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "convnd_fwd_common.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_waveletmodel_cshuffle_v3.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

using InDataType       = ck::half_t;
using WeiDataType      = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using OutDataType      = ck::half_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

template <ck::index_t NDimSpatial, typename InLayout, typename WeiLayout, typename OutLayout>
using DeviceGroupedConvNDFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<
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
        256,         // TileLoadThreadGroupSize
        256,         // TileMathThreadGroupSize
        128,         // MPerBlock
        64,          // NPerBlock
        64,          // KPerBlock
        8,           // K1
        32,          // MPerXdl
        32,          // NPerXdl
        1,           // MXdlPerWave
        2,           // NXdlPerWave
        S<8, 32, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,           // ABlockTransferSrcVectorDim
        8,           // ABlockTransferSrcScalarPerVector
        8,           // ABlockTransferDstScalarPerVector_AK1
        1,           // ABlockLdsExtraM
        S<8, 32, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // BBlockTransferSrcAccessOrder
        2,           // BBlockTransferSrcVectorDim
        8,           // BBlockTransferSrcScalarPerVector
        8,           // BBlockTransferDstScalarPerVector_BK1
        1,           // BBlockLdsExtraN
        1,
        1,
        S<1, 32, 1, 8>,
        4>;

#include "run_convnd_fwd_example.inc"

int main(int argc, char* argv[]) { return run_convnd_fwd_example(argc, argv) ? 0 : 1; }
