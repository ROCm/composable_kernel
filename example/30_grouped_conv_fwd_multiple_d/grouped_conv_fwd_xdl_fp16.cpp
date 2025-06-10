// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.
#include "device_grouped_conv_fwd_dl_v4.hpp"
#include "common.hpp"

// kernel data types
using InKernelDataType  = FP16;
using WeiKernelDataType = FP16;
using AccDataType       = FP32;
using CShuffleDataType  = FP16;
using OutKernelDataType = FP16;

// tensor data types
using InUserDataType  = InKernelDataType;
using WeiUserDataType = WeiKernelDataType;
using OutUserDataType = OutKernelDataType;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;


template <ck::index_t NDimSpatial>
using DeviceConvFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        NDimSpatial,
        InputLayout<NDimSpatial>,
        WeightLayout<NDimSpatial>,
        ck::Tuple<>,
        OutputLayout<NDimSpatial>,
        InKernelDataType,
        WeiKernelDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<>,
        OutKernelDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvSpec,    // ConvForwardSpecialization
        GemmSpec,    // GemmSpecialization
        1,           //
        256,         // BlockSize
        128,         // MPerBlock
        256,         // NPerBlock
        16,          // KPerBlock
        4,           // AK1
        4,           // BK1
        32,          // MPerXdl
        32,          // NPerXdl
        2,           // MXdlPerWave
        4,           // NXdlPerWave
        S<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,           // ABlockTransferSrcVectorDim
        4,           // ABlockTransferSrcScalarPerVector
        4,           // ABlockTransferDstScalarPerVector_AK1
        1,           // ABlockLdsExtraM
        S<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // BBlockTransferSrcAccessOrder
        2,           // BBlockTransferSrcVectorDim
        4,           // BBlockTransferSrcScalarPerVector
        4,           // BBlockTransferDstScalarPerVector_BK1
        1,           // BBlockLdsExtraN
        1,
        1,
        S<1, 16, 1, 16>,
        4>;

#include "run_grouped_conv_fwd_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_conv_fwd_example(argc, argv); }
