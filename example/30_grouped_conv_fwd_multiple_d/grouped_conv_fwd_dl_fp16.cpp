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
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV4<
        NDimSpatial,
        64,
        InKernelDataType,
        WeiKernelDataType,
        AccDataType,
        OutKernelDataType,
        S<28, 28>,
        5,
        ck::Tuple<S<1,1>, S<1,1>, S<2,2>>,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        4,
        4,4,
        4,
        4,
        false>;

#include "run_grouped_conv_fwd_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_conv_fwd_example(argc, argv); }
