// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using OutDataType      = FP16;
using WeiDataType      = FP16;
using AccDataType      = FP32;
using CShuffleDataType = FP16;
using BiasDataType     = FP16; // bias
using InDataType       = FP16;

using OutLayout  = ck::tensor_layout::convolution::GNHWK;
using WeiLayout  = ck::tensor_layout::convolution::GKYXC;
using BiasLayout = ck::tensor_layout::convolution::G_C;
using InLayout   = ck::tensor_layout::convolution::GNHWC;

using OutElementOp = PassThrough;
using WeiElementOp = PassThrough;
using InElementOp  = ck::tensor_operation::element_wise::AddRelu;

using DeviceConvInstance =
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
        InElementOp,
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

#include "run_grouped_conv_bwd_data_bias_relu_example.inc"

int main(int argc, char* argv[]) { return run_grouped_conv_bwd_data_bias_relu_example(argc, argv); }
